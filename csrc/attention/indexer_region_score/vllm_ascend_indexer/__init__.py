# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Standalone BF16 IndexerRegionScore, with its native CANN package bundled."""

import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401

VENDOR_PATH = Path(__file__).resolve().parent / "_cann_ops_custom" / "vendors" / "vllm_ascend_indexer"


def _activate_vendor() -> None:
    op_api = VENDOR_PATH / "op_api" / "lib" / "libcust_opapi.so"
    if not op_api.is_file():
        raise ImportError("IndexerRegionScore native package is missing; rebuild the standalone wheel with CANN loaded")
    entries = [entry for entry in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if entry]
    vendor = str(VENDOR_PATH)
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join([vendor, *(entry for entry in entries if entry != vendor)])


_activate_vendor()
# The extension snapshots custom OPP search paths when loaded, so bootstrap
# its bundled vendor before importing the native adapter.
from . import _C  # noqa: E402, F401


def ids_out(
    query,
    weights,
    key_summary,
    actual_kv_lengths,
    block_table,
    cu_seq_lens,
    max_region_len,
    top_k=512,
    region_size=8,
    head_scale=1.0,
    cache_mode=1,
    mask_mode=3,
    sort_mode=1,
    *,
    selected_region_ids,
) -> None:
    """Write ascending logical IDs to an INT32 ``[G,T,top_k]`` NPU buffer.

    Query/key must be BF16. Weights may be FP16, BF16 or FP32. No dense score
    output is allocated, and unused ID positions are filled with -1.
    """
    torch.ops.vllm_ascend_indexer.npu_indexer_region_score.ids_out(
        query,
        weights,
        key_summary,
        actual_kv_lengths,
        block_table,
        cu_seq_lens,
        max_region_len,
        top_k,
        region_size,
        head_scale,
        cache_mode,
        mask_mode,
        sort_mode,
        selected_region_ids=selected_region_ids,
    )


__all__ = ["ids_out"]
