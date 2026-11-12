# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import sys
from itertools import product
from pathlib import Path

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

HEAD_DIM = 256
DEFAULT_HEADS = 4
REGION_SIZE = 8
SENTINEL = -313
GUARD_ELEMENTS = 8
QK_DTYPES = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
WEIGHT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
SCENARIOS = (
    pytest.param((5, 0, 3), 37, 7, 17, (0, 0, 16), id="under-k-partial-pages"),
    pytest.param((3, 0, 2), 16403, 16, 17, (16403, 0, 16390), id="streaming-topk"),
    pytest.param((2, 0, 1), 9000, 16, 512, (8999, 0, 8999), id="model-topk"),
)


@pytest.fixture(scope="module")
def region_score():
    if not torch.npu.is_available() or "950" not in torch.npu.get_device_name():
        pytest.skip("IndexerRegionScore requires Ascend950")
    if not hasattr(torch.ops._C_ascend, "npu_indexer_region_score"):
        # Lazy setup also permits validation with an already loaded native extension.
        assert importlib.import_module("vllm_ascend.utils").enable_custom_op()
    path = Path(__file__).resolve().parents[6] / "vllm_ascend/ops/indexer_region_score.py"
    spec = importlib.util.spec_from_file_location("paged_region_score_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.indexer_region_score


def make_inputs(
    dtype,
    weight_dtype,
    query_lengths,
    regions,
    page_size,
    histories,
    zero_weights=False,
    affine_pages=False,
    groups=1,
    heads=DEFAULT_HEADS,
):
    generator = torch.Generator().manual_seed(431)
    tokens = sum(query_lengths)
    columns = max(1, (regions + page_size - 1) // page_size)
    pages = len(query_lengths) * columns + 3
    # Binary fractions make the FP32 oracle exact even for FP8, including ties.
    query = (torch.randint(-2, 3, (tokens, groups, heads, HEAD_DIM), generator=generator).float() / 4).to(dtype)
    weights = (torch.randint(-4, 5, (tokens, groups, heads), generator=generator).float() / 8).to(weight_dtype)
    if zero_weights:
        weights.zero_()
    key = (torch.randint(-2, 3, (pages, page_size, 1, HEAD_DIM), generator=generator).float() / 4).to(dtype)
    # Poison unused layers to expose incorrect page stride or storage-offset handling.
    backing = torch.full((pages, 3, page_size, 1, HEAD_DIM), 31.0).to(dtype)
    backing[:, 1] = key
    if affine_pages:
        table = torch.arange(len(query_lengths) * columns, dtype=torch.int32).view(len(query_lengths), columns)
    else:
        table = torch.randint(0, pages, (len(query_lengths), columns), generator=generator, dtype=torch.int32)
        if columns > 2:
            table[:, 1] = -1
            table[:, 2] = pages
    cu = torch.tensor([0, *torch.tensor(query_lengths).cumsum(0).tolist()], dtype=torch.int32)
    lengths = torch.tensor(
        [query_length + history * REGION_SIZE for query_length, history in zip(query_lengths, histories)],
        dtype=torch.int32,
    )
    return query, weights, backing, lengths, table, cu


def reference(query, weights, backing, lengths, table, cu, regions, top_k):
    scores = torch.empty((query.shape[1], query.shape[0], regions), dtype=torch.float32)
    ids = torch.full((query.shape[1], query.shape[0], top_k), -1, dtype=torch.int32)
    visible_counts = torch.zeros(query.shape[0], dtype=torch.int64)
    page_size = backing.shape[2]
    for request in range(table.shape[0]):
        start, end = int(cu[request]), int(cu[request + 1])
        logical = torch.arange(regions)
        physical = table[request, logical // page_size].long()
        valid = (physical >= 0) & (physical < backing.shape[0])
        keys = backing[physical.clamp(0, backing.shape[0] - 1), 1, logical % page_size, 0].float()
        keys[~valid] = 0
        # All groups share the same keys, but use independent queries and weights.
        dots = torch.matmul(query[start:end].float(), keys.T).relu()
        values = (dots * weights[start:end, :, :, None].float() * 0.5).sum(dim=2)
        for token in range(start, end):
            visible = max(0, min(regions, (int(lengths[request]) - (end - start) + token - start) // REGION_SIZE))
            visible_counts[token] = visible
            scores[:, token, :visible] = values[token - start, :, :visible]
            # Stable CPU sorting resolves equal scores by the original logical ID.
            selected = torch.argsort(values[token - start, :, :visible], descending=True, stable=True)[:, :top_k]
            ids[:, token, : selected.shape[1]] = selected.sort().values.to(torch.int32)
    return scores, ids, visible_counts


def check_case(
    region_score,
    dtype,
    weight_dtype,
    query_lengths,
    regions,
    page_size,
    top_k,
    histories,
    zero_weights=False,
    affine_pages=False,
    groups=1,
    heads=DEFAULT_HEADS,
):
    query, weights, backing, lengths, table, cu = make_inputs(
        dtype, weight_dtype, query_lengths, regions, page_size, histories, zero_weights, affine_pages, groups, heads
    )
    expected_scores, expected_ids, visible_counts = reference(
        query, weights, backing, lengths, table, cu, regions, top_k
    )
    query, weights, backing, lengths, table, cu = (
        tensor.npu() for tensor in (query, weights, backing, lengths, table, cu)
    )
    scores = torch.full((groups, query.shape[0], regions + 9), SENTINEL, dtype=torch.float32, device=query.device)
    guarded_ids = torch.full(
        (groups * query.shape[0] * top_k + 2 * GUARD_ELEMENTS,), SENTINEL, dtype=torch.int32, device=query.device
    )
    ids = guarded_ids[GUARD_ELEMENTS:-GUARD_ELEMENTS].view(groups, query.shape[0], top_k)
    actual_scores, actual_ids = region_score(
        query,
        weights,
        backing[:, 1],
        lengths,
        table,
        cu,
        max_region_len=regions,
        top_k=top_k,
        head_scale=0.5,
        out=scores,
        selected_region_ids=ids,
    )
    assert actual_scores.data_ptr() == scores.data_ptr()
    assert actual_ids.data_ptr() == ids.data_ptr()
    actual_scores, actual_ids, guarded_ids = (tensor.cpu() for tensor in (actual_scores, actual_ids, guarded_ids))
    for token, visible in enumerate(visible_counts.tolist()):
        torch.testing.assert_close(
            actual_scores[:, token, :visible], expected_scores[:, token, :visible], rtol=0, atol=0
        )
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    assert torch.all(guarded_ids[:GUARD_ELEMENTS] == SENTINEL)
    assert torch.all(guarded_ids[-GUARD_ELEMENTS:] == SENTINEL)


@pytest.mark.parametrize("dtype,weight_dtype", tuple(product(QK_DTYPES, WEIGHT_DTYPES)))
@pytest.mark.parametrize("query_lengths,regions,page_size,top_k,histories", SCENARIOS)
def test_paged_region_score(region_score, dtype, weight_dtype, query_lengths, regions, page_size, top_k, histories):
    check_case(region_score, dtype, weight_dtype, query_lengths, regions, page_size, top_k, histories)


@pytest.mark.parametrize("dtype", QK_DTYPES)
def test_topk_only_matches_full_score_and_supports_preallocated_ids(region_score, dtype):
    query, weights, backing, lengths, table, cu = make_inputs(
        dtype, torch.float32, (4,), 65, 16, (65,), groups=4, heads=12
    )
    inputs = tuple(tensor.npu() for tensor in (query, weights, backing[:, 1], lengths, table, cu))
    out = torch.empty((4, 4, 65), dtype=torch.float32, device=inputs[0].device)
    scores, expected_ids = region_score(*inputs, max_region_len=65, top_k=17, head_scale=0.5, out=out)
    del scores
    torch.npu.synchronize()

    ids = region_score(*inputs, max_region_len=65, top_k=17, head_scale=0.5)
    torch.testing.assert_close(ids.cpu(), expected_ids.cpu(), rtol=0, atol=0)

    selected = torch.empty_like(ids)
    returned = region_score(*inputs, max_region_len=65, top_k=17, head_scale=0.5, selected_region_ids=selected)
    assert returned.data_ptr() == selected.data_ptr()
    torch.testing.assert_close(returned.cpu(), expected_ids.cpu(), rtol=0, atol=0)


def test_topk_only_meta_default_and_out_shapes(region_score):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        query = torch.empty((9, 4, 12, HEAD_DIM), dtype=torch.bfloat16)
        weights = torch.empty((9, 4, 12), dtype=torch.float32)
        key = torch.empty((8, 16, 1, HEAD_DIM), dtype=torch.bfloat16)
        lengths = torch.empty((1,), dtype=torch.int32)
        table = torch.empty((1, 8), dtype=torch.int32)
        cu = torch.empty((2,), dtype=torch.int32)
        ids = region_score(query, weights, key, lengths, table, cu, max_region_len=65, top_k=17)
        assert ids.shape == (4, 9, 17)
        assert ids.dtype == torch.int32
        selected = torch.empty((4, 9, 17), dtype=torch.int32)
        returned = region_score(
            query,
            weights,
            key,
            lengths,
            table,
            cu,
            max_region_len=65,
            top_k=17,
            selected_region_ids=selected,
        )
        assert returned is selected
        score = torch.empty((4, 9, 65), dtype=torch.float32)
        returned = region_score(
            query,
            weights,
            key,
            lengths,
            table,
            cu,
            max_region_len=65,
            top_k=17,
            out=score,
            selected_region_ids=selected,
        )
        assert returned[0] is score and returned[1] is selected
        operator = torch.ops._C_ascend.npu_indexer_region_score
        allocated_score, allocated_ids = operator.score(query, weights, key, lengths, table, cu, 65, 17)
        assert allocated_score.shape == (4, 9, 65) and allocated_score.dtype == torch.float32
        assert allocated_ids.shape == (4, 9, 17) and allocated_ids.dtype == torch.int32
        assert (
            operator.out(query, weights, key, lengths, table, cu, 65, 17, out=score, selected_region_ids=selected)
            is None
        )


def test_single_dispatcher_overloads_return_types_and_mutations(region_score):
    query, weights, backing, lengths, table, cu = make_inputs(
        torch.bfloat16, torch.float32, (3,), 65, 16, (65,), groups=4, heads=12
    )
    expected_scores, expected_ids, visible = reference(query, weights, backing, lengths, table, cu, 65, 17)
    inputs = tuple(tensor.npu() for tensor in (query, weights, backing[:, 1], lengths, table, cu))
    args = (*inputs, 65, 17, REGION_SIZE, 0.5, 1, 3, 1)
    operator = torch.ops._C_ascend.npu_indexer_region_score
    assert set(operator.overloads()) == {"default", "score", "ids_out", "out"}
    assert not hasattr(torch.ops._C_ascend, "npu_indexer_region_score_topk")
    ids = operator.default(*args)
    torch.testing.assert_close(ids.cpu(), expected_ids, rtol=0, atol=0)
    ids.fill_(SENTINEL)
    assert operator.ids_out(*args, selected_region_ids=ids) is None
    torch.testing.assert_close(ids.cpu(), expected_ids, rtol=0, atol=0)
    scores, allocated_ids = operator.score(*args)
    torch.testing.assert_close(allocated_ids.cpu(), expected_ids, rtol=0, atol=0)
    scores.fill_(SENTINEL)
    ids.fill_(SENTINEL)
    assert operator.out(*args, out=scores, selected_region_ids=ids) is None
    torch.testing.assert_close(ids.cpu(), expected_ids, rtol=0, atol=0)
    for token, count in enumerate(visible.tolist()):
        torch.testing.assert_close(scores[:, token, :count].cpu(), expected_scores[:, token, :count], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ("ids", "ids_out", "out"))
@pytest.mark.parametrize("transform", ("aot_eager", "functionalize"))
@pytest.mark.parametrize("layout", ("dense", "outer-views"))
def test_single_interface_functionalization(region_score, mode, transform, layout):
    query, weights, backing, lengths, table, cu = make_inputs(
        torch.bfloat16, torch.float32, (3,), 65, 16, (65,), groups=4, heads=12
    )
    expected_scores, expected_ids, visible = reference(query, weights, backing, lengths, table, cu, 65, 17)
    cpu_inputs = (query, weights, backing[:, 1], lengths, table, cu)
    inputs = tuple(device_input_view(tensor) if layout == "outer-views" else tensor.npu() for tensor in cpu_inputs)
    if layout == "outer-views":
        score_out, score_storage, score_untouched = guarded_output_view((4, 3, 65), torch.float32, inputs[0].device)
        ids_out, ids_storage, ids_untouched = guarded_output_view((4, 3, 17), torch.int32, inputs[0].device)
    else:
        score_out = torch.empty((4, 3, 65), dtype=torch.float32, device=inputs[0].device)
        ids_out = torch.empty((4, 3, 17), dtype=torch.int32, device=inputs[0].device)

    def execute(query, weights, key, lengths, table, cu, score_out, ids_out):
        kwargs = dict(max_region_len=65, top_k=17, head_scale=0.5)
        if mode != "ids":
            kwargs["selected_region_ids"] = ids_out
        if mode == "out":
            kwargs["out"] = score_out
        return region_score(query, weights, key, lengths, table, cu, **kwargs)

    transformed = (
        torch.compile(execute, backend="aot_eager", fullgraph=True)
        if transform == "aot_eager"
        else torch.func.functionalize(execute)
    )
    result = transformed(*inputs, score_out, ids_out)
    actual_ids = result[1] if mode == "out" else result
    torch.testing.assert_close(actual_ids.cpu(), expected_ids, rtol=0, atol=0)
    if mode != "ids":
        torch.testing.assert_close(ids_out.cpu(), expected_ids, rtol=0, atol=0)
        if transform == "aot_eager":
            assert actual_ids.data_ptr() == ids_out.data_ptr()
    if mode == "out":
        if transform == "aot_eager":
            assert result[0].data_ptr() == score_out.data_ptr()
        for token, count in enumerate(visible.tolist()):
            for scores in (result[0], score_out):
                torch.testing.assert_close(
                    scores[:, token, :count].cpu(), expected_scores[:, token, :count], rtol=0, atol=0
                )
    if layout == "outer-views":
        assert torch.all(score_storage.cpu()[score_untouched] == SENTINEL)
        assert torch.all(ids_storage.cpu()[ids_untouched] == SENTINEL)


@pytest.mark.parametrize("layout", ("extra-capacity", "strided"))
@pytest.mark.parametrize("transform", ("eager", "aot_eager", "functionalize"))
def test_score_output_views_in_all_execution_modes(region_score, layout, transform):
    query, weights, backing, lengths, table, cu = make_inputs(
        torch.bfloat16, torch.float32, (3,), 65, 16, (65,), groups=4, heads=12
    )
    expected_scores, expected_ids, visible = reference(query, weights, backing, lengths, table, cu, 65, 17)
    inputs = tuple(tensor.npu() for tensor in (query, weights, backing[:, 1], lengths, table, cu))
    capacity = 74
    if layout == "strided":
        score_out, score_storage, untouched = guarded_output_view((4, 3, capacity), torch.float32, inputs[0].device)
    else:
        score_storage = torch.full((4, 3, capacity), SENTINEL, dtype=torch.float32, device=inputs[0].device)
        score_out, untouched = score_storage, None
    ids_out, ids_storage, ids_untouched = guarded_output_view((4, 3, 17), torch.int32, inputs[0].device)

    def execute(query, weights, key, lengths, table, cu, score_out, ids_out):
        return region_score(
            query,
            weights,
            key,
            lengths,
            table,
            cu,
            max_region_len=65,
            top_k=17,
            head_scale=0.5,
            out=score_out,
            selected_region_ids=ids_out,
        )

    transformed = (
        torch.compile(execute, backend="aot_eager", fullgraph=True)
        if transform == "aot_eager"
        else torch.func.functionalize(execute)
        if transform == "functionalize"
        else execute
    )
    returned_score, returned_ids = transformed(*inputs, score_out, ids_out)
    torch.testing.assert_close(returned_ids.cpu(), expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(ids_out.cpu(), expected_ids, rtol=0, atol=0)
    if transform != "functionalize":
        assert returned_score.data_ptr() == score_out.data_ptr()
        assert returned_ids.data_ptr() == ids_out.data_ptr()
    for token, count in enumerate(visible.tolist()):
        for score in (returned_score, score_out):
            torch.testing.assert_close(score[:, token, :count].cpu(), expected_scores[:, token, :count], rtol=0, atol=0)
    if untouched is not None:
        assert torch.all(score_storage.cpu()[untouched] == SENTINEL)
    assert torch.all(ids_storage.cpu()[ids_untouched] == SENTINEL)
    if transform != "eager":
        # Functional output is R-wide and copies only this prefix into caller capacity.
        assert torch.all(score_out[..., 65:].cpu() == SENTINEL)


@pytest.mark.parametrize("dtype", QK_DTYPES)
@pytest.mark.parametrize(
    "input_layout",
    (
        "query",
        "query-offset",
        "weights",
        "weights-offset",
        "lengths",
        "table",
        "cu",
        "all",
        "interleaved",
        "key-transpose",
        "key-page-stride",
        "key-offset",
    ),
)
def test_ids_only_materializes_input_and_output_views(region_score, dtype, input_layout):
    check_layout_case(region_score, dtype, input_layout=input_layout, output_layout="ids", ids_only=True)


@pytest.mark.parametrize("dtype", QK_DTYPES)
@pytest.mark.parametrize(
    "tokens,groups,regions,heads,top_k",
    ((2049, 1, 32769, 4, 512), (1901, 4, 9000, 12, 37), (3777, 1, 32769, 4, 512)),
)
def test_topk_only_forces_bounded_workspace_batch(region_score, dtype, tokens, groups, regions, heads, top_k):
    dim = HEAD_DIM
    page_size = 8
    pages = (regions + page_size - 1) // page_size
    query = torch.zeros((tokens, groups, heads, dim), dtype=dtype, device="npu")
    weights = torch.zeros((tokens, groups, heads), dtype=torch.float32, device="npu")
    key = torch.zeros((pages, page_size, 1, dim), dtype=dtype, device="npu")
    table = torch.arange(pages, dtype=torch.int32, device="npu").expand(4, -1).contiguous()
    split = tokens // 2
    cu = torch.tensor((0, 0, split, split, tokens), dtype=torch.int32, device="npu")
    lengths = torch.tensor(
        (0, regions * REGION_SIZE + split, 0, tokens - split),
        dtype=torch.int32,
        device="npu",
    )
    dense_score_bytes = groups * tokens * regions * torch.empty((), dtype=torch.float32).element_size()
    # These inputs are aligned/dense; no extra outer materialization is needed.
    budget_bytes = 256 * 1024**2
    assert dense_score_bytes > budget_bytes

    start_allocated = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    ids = region_score(query, weights, key, lengths, table, cu, max_region_len=regions, top_k=top_k)
    torch.npu.synchronize()
    expected = (
        torch.arange(top_k, dtype=torch.int32, device="cpu").view(1, 1, top_k).expand(groups, tokens, top_k).clone()
    )
    visible = (torch.arange(tokens - split) // REGION_SIZE).clamp(max=regions)
    expected[:, split:, :] = torch.where(torch.arange(top_k)[None, :] < visible[:, None], expected[:, split:, :], -1)
    torch.testing.assert_close(ids.cpu(), expected, rtol=0, atol=0)
    peak_delta = torch.npu.max_memory_allocated() - start_allocated
    ids_bytes = groups * tokens * top_k * torch.empty((), dtype=torch.int32).element_size()
    assert peak_delta <= budget_bytes + ids_bytes + 4 * 1024**2, (peak_delta, budget_bytes, ids_bytes)


@pytest.mark.parametrize("dtype", QK_DTYPES)
@pytest.mark.parametrize("regions,top_k,history", ((0, 17, 0), (521, 3, 521)))
def test_empty_and_equal_score_candidates(region_score, dtype, regions, top_k, history):
    check_case(region_score, dtype, torch.float32, (9,), regions, 16, top_k, (history,), zero_weights=True)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("zero_weights", (False, True))
def test_multiple_topk_carries(region_score, dtype, zero_weights):
    check_case(region_score, dtype, torch.float32, (3,), 33001, 16, 17, (33001,), zero_weights=zero_weights)


@pytest.mark.parametrize("dtype", QK_DTYPES)
def test_affine_page_gather_with_layer_stride(region_score, dtype):
    check_case(region_score, dtype, torch.float32, (2,), 512, 16, 512, (512,), affine_pages=True)


@pytest.mark.parametrize("dtype", QK_DTYPES)
@pytest.mark.parametrize(
    "groups,heads,weight_dtype",
    (
        (1, 8, torch.float32),
        (4, 4, torch.float16),
        (4, 12, torch.bfloat16),
        (8, 16, torch.float32),
        (4, 32, torch.float16),
        (8, 64, torch.bfloat16),
        (4, 96, torch.float32),
        (4, 128, torch.float16),
    ),
)
def test_variable_groups_and_heads(region_score, dtype, groups, heads, weight_dtype):
    check_case(
        region_score,
        dtype,
        weight_dtype,
        (5, 0, 2),
        137,
        7,
        17,
        (130, 0, 0),
        groups=groups,
        heads=heads,
    )


@pytest.mark.parametrize("heads", range(4, 129, 4))
def test_all_supported_head_counts(region_score, heads):
    dtype_index = heads // 4 - 1
    check_case(
        region_score,
        QK_DTYPES[dtype_index % len(QK_DTYPES)],
        WEIGHT_DTYPES[(dtype_index + 1) % len(WEIGHT_DTYPES)],
        (3, 0, 2),
        137,
        7,
        3,
        (137, 0, 100),
        groups=4 if heads % 8 else 8,
        heads=heads,
    )


@pytest.mark.parametrize("dtype,weight_dtype", tuple(product(QK_DTYPES, WEIGHT_DTYPES)))
def test_grouped_weight_dtypes(region_score, dtype, weight_dtype):
    check_case(region_score, dtype, weight_dtype, (3,), 257, 16, 17, (257,), groups=4, heads=12)


@pytest.mark.parametrize("dtype", QK_DTYPES)
@pytest.mark.parametrize("top_k,zero_weights", ((1, False), (3, False), (7, False), (3, True)))
def test_small_topk_across_group_boundary(region_score, dtype, top_k, zero_weights):
    # With T=3, adjacent groups share 32-byte transactions for these small K.
    check_case(
        region_score,
        dtype,
        torch.float32,
        (3,),
        521,
        16,
        top_k,
        (521,),
        zero_weights=zero_weights,
        groups=4,
    )


@pytest.mark.parametrize("dtype", QK_DTYPES)
@pytest.mark.parametrize("zero_weights", (False, True))
def test_grouped_streaming_topk(region_score, dtype, zero_weights):
    check_case(
        region_score,
        dtype,
        torch.float32,
        (1,),
        16403,
        16,
        17,
        (16403,),
        zero_weights=zero_weights,
        groups=4,
        heads=12,
    )


def test_max_heads_streaming_topk(region_score):
    check_case(region_score, torch.float16, torch.float32, (1,), 16403, 16, 17, (16403,), groups=4, heads=128)


@pytest.mark.parametrize("dtype", QK_DTYPES)
def test_grouped_empty_score_output(region_score, dtype):
    check_case(region_score, dtype, torch.float32, (3,), 0, 16, 17, (0,), groups=4, heads=128)


@pytest.mark.parametrize("dtype", QK_DTYPES)
def test_grouped_affine_page_gather(region_score, dtype):
    check_case(
        region_score,
        dtype,
        torch.float32,
        (3,),
        512,
        16,
        512,
        (512,),
        affine_pages=True,
        groups=4,
        heads=16,
    )


@pytest.mark.parametrize("dtype", QK_DTYPES)
@pytest.mark.parametrize(
    "heads,tokens,weight_dtype",
    (
        (4, 65, torch.float32),
        (12, 21, torch.float16),
        (20, 13, torch.bfloat16),
        (68, 3, torch.float32),
        (96, 65, torch.float32),
        (128, 65, torch.float32),
    ),
)
def test_grouped_full_query_tiles(region_score, dtype, heads, tokens, weight_dtype):
    # Exercise full microtiles, padded weight rows and both resident Q microtiles.
    check_case(
        region_score,
        dtype,
        weight_dtype,
        (tokens,),
        512,
        16,
        17,
        (512,),
        groups=4,
        heads=heads,
    )


@pytest.mark.parametrize("groups,heads", ((0, 4), (2, 4), (3, 4), (1, 0), (1, 1), (1, 6), (1, 132)))
def test_invalid_groups_or_heads(region_score, groups, heads):
    query, weights, backing, lengths, table, cu = make_inputs(
        torch.float16, torch.float32, (1,), 16, 16, (16,), groups=groups, heads=heads
    )
    inputs = (tensor.npu() for tensor in (query, weights, backing[:, 1], lengths, table, cu))
    with pytest.raises(RuntimeError, match="query"):
        region_score(*inputs, max_region_len=16, top_k=3)


@pytest.mark.parametrize("dtype", QK_DTYPES)
@pytest.mark.parametrize("score_output", (False, True))
@pytest.mark.parametrize("input_index", (0, 2))
@pytest.mark.parametrize("aligned", (False, True))
def test_dense_offset_views_with_large_backing(region_score, dtype, score_output, input_index, aligned):
    query, weights, backing, lengths, table, cu = make_inputs(
        dtype, torch.float32, (3,), 65, 16, (65,), groups=4, heads=12
    )
    expected_scores, expected_ids, visible = reference(query, weights, backing, lengths, table, cu, 65, 17)
    cpu_inputs = (query, weights, backing[:, 1], lengths, table, cu)
    inputs = [tensor.npu() for tensor in cpu_inputs]
    tensor = cpu_inputs[input_index]
    alignment_elements = 32 // tensor.element_size()
    offset = tensor.numel() + (alignment_elements if aligned else 1)
    storage = torch.full((offset + tensor.numel() + 257,), 31.0).to(dtype)
    storage[offset : offset + tensor.numel()].view(tensor.shape).copy_(tensor)
    inputs[input_index] = storage.npu()[offset : offset + tensor.numel()].view(tensor.shape)
    assert inputs[input_index].is_contiguous() and inputs[input_index].storage_offset() > tensor.numel()
    assert (inputs[input_index].storage_offset() * tensor.element_size() % 32 == 0) == aligned
    count = expected_ids.numel()
    ids_storage = torch.full((2 * count + 258,), SENTINEL, dtype=torch.int32, device=inputs[0].device)
    ids = ids_storage[count + 1 : 2 * count + 1].view(expected_ids.shape)
    kwargs = dict(max_region_len=65, top_k=17, head_scale=0.5, selected_region_ids=ids)
    if score_output:
        kwargs["out"] = torch.empty((4, 3, 65), dtype=torch.float32, device=inputs[0].device)
    result = region_score(*inputs, **kwargs)
    actual_ids = result[1] if score_output else result
    assert actual_ids.data_ptr() == ids.data_ptr()
    torch.testing.assert_close(actual_ids.cpu(), expected_ids, rtol=0, atol=0)
    assert torch.all(ids_storage[: count + 1].cpu() == SENTINEL)
    assert torch.all(ids_storage[2 * count + 1 :].cpu() == SENTINEL)
    if score_output:
        for token, count_visible in enumerate(visible.tolist()):
            torch.testing.assert_close(
                result[0][:, token, :count_visible].cpu(), expected_scores[:, token, :count_visible], rtol=0, atol=0
            )


@pytest.mark.parametrize("dtype", QK_DTYPES)
@pytest.mark.parametrize("score_output", (False, True))
def test_dense_aligned_metadata_offsets_retain_original_capacity(region_score, dtype, score_output):
    query, weights, backing, lengths, table, cu = make_inputs(
        dtype, torch.float32, (3,), 65, 16, (65,), groups=4, heads=12
    )
    expected_scores, expected_ids, visible = reference(query, weights, backing, lengths, table, cu, 65, 17)
    inputs = [query.npu(), weights.npu(), backing.npu()[:, 1], lengths.npu(), table.npu(), cu.npu()]
    # Metadata can have a logical one-element shape but an aligned offset of eight.
    # The original backing capacity must survive a Contiguous-created alias view.
    for index, tensor in ((0, query), (1, weights), (3, lengths), (4, table), (5, cu)):
        offset = 1 if index == 0 else 32 // tensor.element_size()
        storage = torch.full((offset + tensor.numel() + 257,), 31.0).to(tensor.dtype)
        storage[offset : offset + tensor.numel()].view(tensor.shape).copy_(tensor)
        inputs[index] = storage.npu()[offset : offset + tensor.numel()].view(tensor.shape)
        assert inputs[index].is_contiguous() and inputs[index].storage_offset() == offset
    kwargs = dict(max_region_len=65, top_k=17, head_scale=0.5)
    if score_output:
        kwargs["out"] = torch.empty((4, 3, 65), dtype=torch.float32, device=inputs[0].device)
    result = region_score(*inputs, **kwargs)
    actual_ids = result[1] if score_output else result
    torch.testing.assert_close(actual_ids.cpu(), expected_ids, rtol=0, atol=0)
    if score_output:
        for token, count in enumerate(visible.tolist()):
            torch.testing.assert_close(
                result[0][:, token, :count].cpu(), expected_scores[:, token, :count], rtol=0, atol=0
            )


@pytest.mark.parametrize("invalid", ("score-dtype", "score-capacity", "ids-dtype", "ids-shape", "input-device"))
@pytest.mark.parametrize("transform", ("eager", "aot_eager", "functionalize"))
def test_invalid_buffers_still_rejected_after_layout_normalization(region_score, invalid, transform):
    query, weights, backing, lengths, table, cu = make_inputs(
        torch.bfloat16, torch.float32, (3,), 65, 16, (65,), groups=4, heads=12
    )
    inputs = [tensor.npu() for tensor in (query, weights, backing[:, 1], lengths, table, cu)]
    score = torch.empty((4, 3, 64 if invalid == "score-capacity" else 74), device=inputs[0].device)
    ids = torch.empty((4, 3, 18 if invalid == "ids-shape" else 17), dtype=torch.int32, device=inputs[0].device)
    if invalid == "score-dtype":
        score = score.to(torch.bfloat16)
    if invalid == "ids-dtype":
        ids = ids.float()
    if invalid == "input-device":
        inputs[1] = inputs[1].cpu()

    def execute(query, weights, key, lengths, table, cu, score, ids):
        return region_score(
            query,
            weights,
            key,
            lengths,
            table,
            cu,
            max_region_len=65,
            top_k=17,
            head_scale=0.5,
            out=score,
            selected_region_ids=ids,
        )

    transformed = (
        torch.compile(execute, backend="aot_eager", fullgraph=True)
        if transform == "aot_eager"
        else torch.func.functionalize(execute)
        if transform == "functionalize"
        else execute
    )
    with pytest.raises((RuntimeError, ValueError)):
        transformed(*inputs, score, ids)


def device_input_view(tensor, layout="interleaved"):
    if layout == "key-transpose":
        return tensor.transpose(1, 3).contiguous().npu().transpose(1, 3)
    if layout == "key-page-stride":
        pages, summaries, _, dim = tensor.shape
        page_stride = summaries * dim + 1
        size = (pages - 1) * page_stride + summaries * dim
        strides = (page_stride, dim, dim, 1)
        offset = 0
    else:
        # Interleave every logical element with poison, retaining a nonzero offset.
        offset = 1 if layout.endswith("-offset") else GUARD_ELEMENTS + 1
        step = 1 if layout.endswith("-offset") else 2
        size = offset + step * tensor.numel() + GUARD_ELEMENTS
        strides = tuple(stride * step for stride in tensor.contiguous().stride())
    backing = torch.full((size,), 31.0).to(tensor.dtype)
    backing.as_strided(tensor.shape, strides, offset).copy_(tensor)
    return backing.npu().as_strided(tensor.shape, strides, offset)


def guarded_output_view(shape, dtype, device):
    count = 1
    for size in shape:
        count *= size
    offset = GUARD_ELEMENTS + 1
    backing = torch.full((offset + 2 * count + GUARD_ELEMENTS,), SENTINEL, dtype=dtype, device=device)
    view = backing[offset : offset + 2 * count : 2].view(shape)
    untouched = torch.ones(backing.numel(), dtype=torch.bool)
    untouched[offset : offset + 2 * count : 2] = False
    return view, backing, untouched


def check_layout_case(
    region_score,
    dtype,
    input_layout=None,
    output_layout=None,
    regions=137,
    allocate_ids=False,
    single_key_page=False,
    ids_only=False,
):
    groups, heads, top_k = 4, 20, 3
    query, weights, backing, lengths, table, cu = make_inputs(
        dtype, torch.bfloat16, (5, 0, 2), regions, 7, (regions, 0, 100), groups=groups, heads=heads
    )
    if single_key_page:
        backing = backing[:1]
        table.zero_()
    expected_scores, expected_ids, visible_counts = reference(
        query, weights, backing, lengths, table, cu, regions, top_k
    )
    cpu_inputs = (query, weights, backing[:, 1], lengths, table, cu)
    inputs = [query.npu(), weights.npu(), backing.npu()[:, 1], lengths.npu(), table.npu(), cu.npu()]
    input_names = ("query", "weights", "key", "lengths", "table", "cu")
    for index, name in enumerate(input_names):
        if name != "key" and input_layout in (name, "all", f"{name}-offset"):
            layout = input_layout if input_layout.endswith("-offset") else "interleaved"
            inputs[index] = device_input_view(cpu_inputs[index], layout)
            if layout.endswith("-offset"):
                assert inputs[index].is_contiguous() and inputs[index].storage_offset() == 1
            else:
                assert not inputs[index].is_contiguous()
    key_layout = "interleaved" if input_layout == "all" else input_layout
    if key_layout in ("interleaved", "key-transpose", "key-page-stride", "key-offset"):
        inputs[2] = device_input_view(cpu_inputs[2], key_layout)
        if key_layout == "key-offset":
            assert inputs[2].is_contiguous() and inputs[2].storage_offset() == 1
        elif single_key_page:
            assert inputs[2].is_contiguous() and inputs[2].stride(0) % (32 // inputs[2].element_size()) != 0
        else:
            assert not inputs[2].is_contiguous()
    kwargs = dict(max_region_len=regions, top_k=top_k, head_scale=0.5, cache_mode=1, mask_mode=3, sort_mode=1)
    guards = []
    if not allocate_ids:
        outputs = {}
        for name, shape, output_dtype in (
            ("scores", (groups, query.shape[0], regions + 9), torch.float32),
            ("ids", (groups, query.shape[0], top_k), torch.int32),
        ):
            if ids_only and name == "scores":
                continue
            if output_layout in (name, "both"):
                view, storage, untouched = guarded_output_view(shape, output_dtype, inputs[0].device)
                assert not view.is_contiguous()
                guards.append((storage, untouched))
            else:
                view = torch.full(shape, SENTINEL, dtype=output_dtype, device=inputs[0].device)
            outputs[name] = view
        kwargs["selected_region_ids"] = outputs["ids"]
        if not ids_only:
            kwargs["out"] = outputs["scores"]
    else:
        # The unified interface requires explicit Score storage; IDs may be allocated.
        if not ids_only:
            kwargs["out"] = torch.empty((groups, query.shape[0], regions), dtype=torch.float32, device=inputs[0].device)
    result = region_score(*inputs, **kwargs)
    actual_scores, actual_ids = (None, result) if ids_only else result
    if not allocate_ids:
        if not ids_only:
            assert actual_scores.data_ptr() == outputs["scores"].data_ptr()
        assert actual_ids.data_ptr() == outputs["ids"].data_ptr()
    assert actual_ids.shape == expected_ids.shape
    actual_ids = actual_ids.cpu()
    if not ids_only:
        assert actual_scores.shape == (groups, query.shape[0], regions if allocate_ids else regions + 9)
        actual_scores = actual_scores.cpu()
        for token, visible in enumerate(visible_counts.tolist()):
            torch.testing.assert_close(
                actual_scores[:, token, :visible], expected_scores[:, token, :visible], rtol=0, atol=0
            )
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    for storage, untouched in guards:
        assert torch.all(storage.cpu()[untouched] == SENTINEL)


@pytest.mark.parametrize("dtype", QK_DTYPES)
@pytest.mark.parametrize(
    "input_layout",
    (
        "query",
        "query-offset",
        "weights",
        "lengths",
        "table",
        "cu",
        "interleaved",
        "key-transpose",
        "key-page-stride",
        "key-offset",
    ),
)
def test_noncontiguous_input_views(region_score, dtype, input_layout):
    check_layout_case(region_score, dtype, input_layout=input_layout)


@pytest.mark.parametrize("dtype", QK_DTYPES)
def test_ids_only_views_with_allocated_output(region_score, dtype):
    check_layout_case(region_score, dtype, input_layout="all", allocate_ids=True, ids_only=True)


@pytest.mark.parametrize("dtype", QK_DTYPES)
@pytest.mark.parametrize("output_layout", ("scores", "ids", "both"))
@pytest.mark.parametrize("regions", (0, 137))
def test_noncontiguous_output_views(region_score, dtype, output_layout, regions):
    check_layout_case(region_score, dtype, output_layout=output_layout, regions=regions)


@pytest.mark.parametrize("dtype", QK_DTYPES)
@pytest.mark.parametrize("allocate_ids", (False, True))
def test_mixed_noncontiguous_views(region_score, dtype, allocate_ids):
    check_layout_case(region_score, dtype, input_layout="all", output_layout="both", allocate_ids=allocate_ids)


@pytest.mark.parametrize("dtype", QK_DTYPES)
def test_noncontiguous_inputs_with_allocated_empty_scores(region_score, dtype):
    check_layout_case(region_score, dtype, input_layout="all", regions=0, allocate_ids=True)


@pytest.mark.parametrize("dtype", QK_DTYPES)
def test_single_page_noncanonical_stride(region_score, dtype):
    # P=1 makes this view logically contiguous despite its unaligned page stride.
    check_layout_case(region_score, dtype, input_layout="key-page-stride", regions=7, single_key_page=True)


@pytest.mark.parametrize(
    "parameter,value",
    (*product(("cache_mode", "mask_mode", "sort_mode"), (-1, 0, 2)), ("mask_mode", 1)),
)
@pytest.mark.parametrize("direct", (False, True))
def test_unsupported_modes(region_score, parameter, value, direct):
    query, weights, backing, lengths, table, cu = make_inputs(torch.float16, torch.float32, (1,), 16, 16, (16,))
    inputs = (tensor.npu() for tensor in (query, weights, backing[:, 1], lengths, table, cu))
    operation = torch.ops._C_ascend.npu_indexer_region_score if direct else region_score
    with pytest.raises(RuntimeError if direct else ValueError, match="cache_mode|mask_mode|sort_mode"):
        operation(*inputs, max_region_len=16, top_k=3, **{parameter: value})
