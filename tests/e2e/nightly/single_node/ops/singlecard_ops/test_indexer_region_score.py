# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
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
    spec.loader.exec_module(module)
    return module.indexer_region_score


@pytest.fixture(scope="module")
def region_score_topk(region_score):
    return region_score.__globals__["indexer_region_score_topk"]


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
def test_topk_only_matches_full_score_and_supports_preallocated_ids(region_score, region_score_topk, dtype):
    query, weights, backing, lengths, table, cu = make_inputs(
        dtype, torch.float32, (4,), 65, 16, (65,), groups=4, heads=12
    )
    inputs = tuple(tensor.npu() for tensor in (query, weights, backing[:, 1], lengths, table, cu))
    scores, expected_ids = region_score(*inputs, max_region_len=65, top_k=17, head_scale=0.5)
    del scores
    torch.npu.synchronize()

    ids = region_score_topk(*inputs, max_region_len=65, top_k=17, head_scale=0.5)
    torch.testing.assert_close(ids.cpu(), expected_ids.cpu(), rtol=0, atol=0)

    selected = torch.empty_like(ids)
    returned = region_score_topk(*inputs, max_region_len=65, top_k=17, head_scale=0.5, selected_region_ids=selected)
    assert returned.data_ptr() == selected.data_ptr()
    torch.testing.assert_close(returned.cpu(), expected_ids.cpu(), rtol=0, atol=0)


def test_topk_only_meta_default_and_out_shapes(region_score_topk):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        query = torch.empty((9, 4, 12, HEAD_DIM), dtype=torch.bfloat16)
        weights = torch.empty((9, 4, 12), dtype=torch.float32)
        key = torch.empty((8, 16, 1, HEAD_DIM), dtype=torch.bfloat16)
        lengths = torch.empty((1,), dtype=torch.int32)
        table = torch.empty((1, 8), dtype=torch.int32)
        cu = torch.empty((2,), dtype=torch.int32)
        ids = region_score_topk(query, weights, key, lengths, table, cu, max_region_len=65, top_k=17)
        assert ids.shape == (4, 9, 17)
        assert ids.dtype == torch.int32
        selected = torch.empty((4, 9, 17), dtype=torch.int32)
        returned = region_score_topk(
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


def test_topk_only_rejects_unsupported_layouts(region_score_topk):
    query = torch.empty((4, 1, 4, 512), dtype=torch.bfloat16, device="npu")[..., ::2]
    weights = torch.zeros((4, 1, 4), dtype=torch.float32, device="npu")
    key = torch.zeros((9, 8, 1, HEAD_DIM), dtype=torch.bfloat16, device="npu")
    lengths = torch.tensor((80,), dtype=torch.int32, device="npu")
    table = torch.arange(9, dtype=torch.int32, device="npu").view(1, 9)
    cu = torch.tensor((0, 4), dtype=torch.int32, device="npu")
    args = (query, weights, key, lengths, table, cu)
    with pytest.raises(RuntimeError, match="contiguous|TopK-only"):
        region_score_topk(*args, max_region_len=65, top_k=17)

    query = query.contiguous()
    ids_storage = torch.empty(4 * 17 + 1, dtype=torch.int32, device="npu")
    unaligned_ids = ids_storage[1:].view(1, 4, 17)
    assert unaligned_ids.is_contiguous() and unaligned_ids.storage_offset() % 8 != 0
    with pytest.raises(RuntimeError, match="aligned|TopK-only"):
        region_score_topk(query, *args[1:], max_region_len=65, top_k=17, selected_region_ids=unaligned_ids)


@pytest.mark.parametrize("dtype", QK_DTYPES)
@pytest.mark.parametrize(
    "tokens,groups,regions,heads,top_k",
    ((2049, 1, 32769, 4, 512), (1901, 4, 9000, 12, 37), (3777, 1, 32769, 4, 512)),
)
def test_topk_only_forces_bounded_workspace_batch(region_score_topk, dtype, tokens, groups, regions, heads, top_k):
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
    budget_bytes = 256 * 1024**2
    assert dense_score_bytes > budget_bytes

    start_allocated = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    ids = region_score_topk(query, weights, key, lengths, table, cu, max_region_len=regions, top_k=top_k)
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
    default_outputs=False,
    single_key_page=False,
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
    if not default_outputs:
        outputs = []
        for name, shape, output_dtype in (
            ("scores", (groups, query.shape[0], regions + 9), torch.float32),
            ("ids", (groups, query.shape[0], top_k), torch.int32),
        ):
            if output_layout in (name, "both"):
                view, storage, untouched = guarded_output_view(shape, output_dtype, inputs[0].device)
                assert not view.is_contiguous()
                guards.append((storage, untouched))
            else:
                view = torch.full(shape, SENTINEL, dtype=output_dtype, device=inputs[0].device)
            outputs.append(view)
        kwargs.update(out=outputs[0], selected_region_ids=outputs[1])
    actual_scores, actual_ids = region_score(*inputs, **kwargs)
    if not default_outputs:
        assert actual_scores.data_ptr() == outputs[0].data_ptr()
        assert actual_ids.data_ptr() == outputs[1].data_ptr()
    assert actual_scores.shape == (groups, query.shape[0], regions if default_outputs else regions + 9)
    assert actual_ids.shape == expected_ids.shape
    actual_scores, actual_ids = actual_scores.cpu(), actual_ids.cpu()
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
@pytest.mark.parametrize("output_layout", ("scores", "ids", "both"))
@pytest.mark.parametrize("regions", (0, 137))
def test_noncontiguous_output_views(region_score, dtype, output_layout, regions):
    check_layout_case(region_score, dtype, output_layout=output_layout, regions=regions)


@pytest.mark.parametrize("dtype", QK_DTYPES)
@pytest.mark.parametrize("default_outputs", (False, True))
def test_mixed_noncontiguous_views(region_score, dtype, default_outputs):
    check_layout_case(region_score, dtype, input_layout="all", output_layout="both", default_outputs=default_outputs)


@pytest.mark.parametrize("dtype", QK_DTYPES)
def test_noncontiguous_inputs_with_default_empty_scores(region_score, dtype):
    check_layout_case(region_score, dtype, input_layout="all", regions=0, default_outputs=True)


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
