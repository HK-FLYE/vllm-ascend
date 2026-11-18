# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Check the installed standalone dispatch contract without allocating NPU data."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
irs = pytest.importorskip("vllm_ascend_indexer")


def meta_inputs():
    return (
        torch.empty((33, 4, 4, 256), dtype=torch.bfloat16, device="meta"),
        torch.empty((33, 4, 4), dtype=torch.bfloat16, device="meta"),
        torch.empty((639, 1, 1, 256), dtype=torch.bfloat16, device="meta"),
        torch.empty((1,), dtype=torch.int32, device="meta"),
        torch.empty((1, 639), dtype=torch.int32, device="meta"),
        torch.empty((2,), dtype=torch.int32, device="meta"),
    )


def test_meta_ids_out_returns_none():
    inputs = meta_inputs()
    output = torch.empty((4, 33, 512), dtype=torch.int32, device="meta")
    assert irs.ids_out(*inputs, max_region_len=639, selected_region_ids=output) is None
    assert output.shape == (4, 33, 512)
    assert output.device.type == "meta"


def test_schema_preserves_output_mutation():
    schema = torch.ops.vllm_ascend_indexer.npu_indexer_region_score.ids_out._schema
    output = next(argument for argument in schema.arguments if argument.name == "selected_region_ids")
    assert output.alias_info is not None
    assert output.alias_info.is_write
    assert not schema.returns


@pytest.mark.parametrize("shape,dtype", [((4, 33, 511), torch.int32), ((4, 33, 512), torch.float32)])
def test_meta_rejects_invalid_output(shape, dtype):
    output = torch.empty(shape, dtype=dtype, device="meta")
    with pytest.raises(RuntimeError, match="selected_region_ids"):
        irs.ids_out(*meta_inputs(), max_region_len=639, selected_region_ids=output)


def test_meta_rejects_non_bf16_query():
    inputs = list(meta_inputs())
    inputs[0] = inputs[0].float()
    output = torch.empty((4, 33, 512), dtype=torch.int32, device="meta")
    with pytest.raises(RuntimeError, match="BF16"):
        irs.ids_out(*inputs, max_region_len=639, selected_region_ids=output)


def test_meta_rejects_rank_three_key():
    inputs = list(meta_inputs())
    inputs[2] = torch.empty((639, 1, 256), dtype=torch.bfloat16, device="meta")
    output = torch.empty((4, 33, 512), dtype=torch.int32, device="meta")
    with pytest.raises(RuntimeError, match="expected query"):
        irs.ids_out(*inputs, max_region_len=639, selected_region_ids=output)
