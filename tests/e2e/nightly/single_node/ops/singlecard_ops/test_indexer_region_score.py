# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest
import torch

HEAD_DIM = 256
HEAD_SCALE = 0.5
SENTINEL = -313
GUARD_ELEMENTS = 8
QK_DTYPES = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
# Actual lengths include the packed query; causal visibility varies per token.
CASES = (
    [
        pytest.param(dtype, (5, 0, 3), 37, 7, 17, (4, 0, 129), 8, 1, 4, False, id=f"causal-{dtype}")
        for dtype in QK_DTYPES
    ]
    + [pytest.param(dtype, (2,), 19, 5, 7, (153,), 8, 4, 12, False, id=f"grouped-{dtype}") for dtype in QK_DTYPES]
    + [
        pytest.param(torch.bfloat16, (2,), 521, 16, 512, (4170,), 8, 4, 12, False, id="model-k512"),
        pytest.param(torch.float16, (2, 0, 1), 0, 7, 3, (100, 0, 100), 8, 4, 12, False, id="zero-regions"),
        pytest.param(torch.float8_e4m3fn, (3,), 19, 5, 7, (153,), 8, 1, 4, True, id="ties-smaller-id"),
    ]
)


@pytest.fixture(scope="module")
def region_score_op():
    # Keep collection usable without torch_npu, vLLM, or an initialized NPU.
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available() or "950" not in torch.npu.get_device_name():
        pytest.skip("IndexerRegionScore requires Ascend A5 (950)")
    if not hasattr(torch.ops._C_ascend, "npu_indexer_region_score"):
        assert importlib.import_module("vllm_ascend.utils").enable_custom_op(), "IRS extension must be installed on A5"
    op = torch.ops._C_ascend.npu_indexer_region_score
    assert op.overloads() == ["ids_out"]
    return op


def _call(op, inputs, regions, top_k, region_size, ids=None):
    if ids is None:
        query = inputs[0]
        ids = torch.empty((query.shape[1], query.shape[0], top_k), dtype=torch.int32, device=query.device)
    result = op.ids_out(
        *inputs,
        max_region_len=regions,
        top_k=top_k,
        region_size=region_size,
        head_scale=HEAD_SCALE,
        selected_region_ids=ids,
    )
    assert result is None
    return ids


def _inputs(dtype, query_lengths, regions, page_size, lengths, groups, heads, ties=False):
    generator = torch.Generator().manual_seed(431)
    tokens = sum(query_lengths)
    columns = max(1, (regions + page_size - 1) // page_size)
    pages = len(query_lengths) * columns + 3
    # Small binary fractions are exactly representable even in E4M3FN. Every
    # product and sum also fits FP32 exactly, so accumulation order is irrelevant.
    query = (torch.randint(-2, 3, (tokens, groups, heads, HEAD_DIM), generator=generator).float() / 4).to(dtype)
    weights = torch.randint(-4, 5, (tokens, groups, heads), generator=generator).float() / 8
    if ties:
        weights.zero_()
    keys = (torch.randint(-2, 3, (pages, page_size, 1, HEAD_DIM), generator=generator).float() / 4).to(dtype)
    # A layer view has both a storage offset and gaps between physical pages.
    backing = torch.full((pages, 2, page_size, 1, HEAD_DIM), 31.0).to(dtype)
    backing[:, 1] = keys
    # Permuted pages and an unused layer expose incorrect cache addressing.
    table = torch.randperm(pages, generator=generator)[: len(query_lengths) * columns]
    table = table.reshape(len(query_lengths), columns).to(torch.int32)
    cu = torch.tensor([0, *torch.tensor(query_lengths).cumsum(0).tolist()], dtype=torch.int32)
    return query, weights, backing, torch.tensor(lengths, dtype=torch.int32), table, cu


def _reference(query, weights, key, lengths, table, cu, regions, top_k, region_size):
    """Evaluate the mathematical contract on CPU, independently of NPU tiling."""
    groups, tokens = query.shape[1], query.shape[0]
    ids = torch.full((groups, tokens, top_k), -1, dtype=torch.int32)
    logical = torch.arange(regions)
    for request in range(table.shape[0]):
        start, end = cu[request : request + 2].tolist()
        if start == end:
            continue
        # Reconstruct the request's logical sequence before computing attention.
        physical_pages = table[request, logical // key.shape[1]].long()
        request_keys = key[physical_pages, logical % key.shape[1], 0].float()
        dots = torch.matmul(query[start:end].float(), request_keys.T).relu()
        values = (dots * weights[start:end, :, :, None].float() * HEAD_SCALE).sum(dim=2)
        history = int(lengths[request]) - (end - start)
        for position, token in enumerate(range(start, end)):
            visible = min(regions, max(0, (history + position) // region_size))
            # Stable sorting starts in logical order: smaller logical IDs win
            # equal scores. The public result then sorts the selected IDs.
            ranked = torch.argsort(values[position, :, :visible], descending=True, stable=True)[:, :top_k]
            ids[:, token, : ranked.shape[1]] = ranked.sort().values.to(torch.int32)
    return ids


@pytest.mark.parametrize("dtype,query_lengths,regions,page_size,top_k,lengths,region_size,groups,heads,ties", CASES)
@torch.inference_mode()
def test_indexer_region_score_precision(
    region_score_op, dtype, query_lengths, regions, page_size, top_k, lengths, region_size, groups, heads, ties
):
    query, weights, backing, lengths, table, cu = _inputs(
        dtype, query_lengths, regions, page_size, lengths, groups, heads, ties
    )
    expected = _reference(query, weights, backing[:, 1], lengths, table, cu, regions, top_k, region_size)
    inputs = (query.npu(), weights.npu(), backing.npu()[:, 1], lengths.npu(), table.npu(), cu.npu())
    ids = _call(region_score_op, inputs, regions, top_k, region_size)
    torch.testing.assert_close(ids.cpu(), expected, rtol=0, atol=0)


@torch.inference_mode()
def test_indexer_region_score_output_view(region_score_op):
    groups, regions, top_k, region_size = 4, 19, 7, 8
    query, weights, backing, lengths, table, cu = _inputs(
        torch.bfloat16, (2, 0, 1), regions, 5, (153, 0, 9), groups, 12
    )
    expected = _reference(query, weights, backing[:, 1], lengths, table, cu, regions, top_k, region_size)
    inputs = (query.npu(), weights.npu(), backing.npu()[:, 1], lengths.npu(), table.npu(), cu.npu())
    storage = torch.full(
        (groups * query.shape[0] * top_k * 2 + 2 * GUARD_ELEMENTS,), SENTINEL, dtype=torch.int32, device="npu"
    )
    ids = storage[GUARD_ELEMENTS:-GUARD_ELEMENTS:2].view(groups, query.shape[0], top_k)
    assert not ids.is_contiguous()
    assert _call(region_score_op, inputs, regions, top_k, region_size, ids=ids) is ids
    torch.testing.assert_close(ids.cpu(), expected, rtol=0, atol=0)
    actual = storage.cpu()
    assert torch.all(actual[:GUARD_ELEMENTS] == SENTINEL)
    assert torch.all(actual[-GUARD_ELEMENTS:] == SENTINEL)
    assert torch.all(actual[GUARD_ELEMENTS:-GUARD_ELEMENTS][1::2] == SENTINEL)


def test_indexer_region_score_schema(region_score_op):
    schema = region_score_op.ids_out._schema
    assert not schema.returns
    output = schema.arguments[-1]
    assert output.name == "selected_region_ids"
    assert output.kwarg_only and not output.has_default_value()
    assert output.alias_info.is_write
