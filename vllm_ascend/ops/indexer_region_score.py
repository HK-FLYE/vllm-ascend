# SPDX-License-Identifier: Apache-2.0
"""Paged D256 region scoring with grouped indexer heads."""

import torch


def indexer_region_score(
    query: torch.Tensor,
    weights: torch.Tensor,
    key_summary: torch.Tensor,
    actual_kv_lengths: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    *,
    max_region_len: int,
    top_k: int = 512,
    region_size: int = 8,
    head_scale: float = 1.0,
    cache_mode: int = 1,
    mask_mode: int = 3,
    sort_mode: int = 1,
    out: torch.Tensor | None = None,
    selected_region_ids: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return IDs, or ``(out, IDs)`` when a score output buffer is supplied.

    Ascend950 only. ``query`` is [T,G,H,256], ``weights`` is [T,G,H], and
    ``key_summary`` is [P,S,1,256], shared by all G groups. G is 1 or a
    positive multiple of 4; H is 4..128 in steps of 4. Query and key use FP16, BF16,
    or FP8 E4M3FN dtype; weights may also use FP32. With ``out=None``, IDs-only
    execution limits the IRS algorithm workspace to 256 MiB. Outer input
    materialization and output-view copies require additional workspace.
    Both modes convert noncontiguous or unaligned tensors as needed; aligned
    page-inner-contiguous Key views preserve zero-copy access.
    Only cache_mode=1, mask_mode=3 and sort_mode=1 are currently supported.

    ``block_table`` [N,C] maps logical pages to physical pages. Metadata stays on device:
    ``cu_seq_lens`` [N+1] partitions query tokens and
    ``actual_kv_lengths`` [N] includes the current query tokens. Right-down
    causal masking selects only complete historical regions; it does not
    append the current region, map IDs to physical pages, or pack the cache.

    A supplied ``out`` is FP32 [G,T,R_out] with R_out >= max_region_len. Only visible
    candidates have defined scores. IDs are INT32 [G,T,top_k], 1 <= top_k <= 512;
    ties prefer smaller IDs and missing candidates are padded with -1.
    Optional output buffers are reused and must not alias inputs or each other.
    """
    if (cache_mode, mask_mode, sort_mode) != (1, 3, 1):
        raise ValueError("only cache_mode=1, mask_mode=3 and sort_mode=1 are supported")
    if query.dim() != 4:
        raise RuntimeError("query must have shape [T,G,H,256]")
    if out is not None and (
        out.dtype != torch.float32
        or out.device != query.device
        or out.dim() != 3
        or out.shape[:2] != (query.shape[1], query.shape[0])
        or out.shape[2] < max_region_len
    ):
        raise RuntimeError("out must be FP32 [G,T,R_out] with R_out >= max_region_len")
    if selected_region_ids is not None and (
        selected_region_ids.dtype != torch.int32
        or selected_region_ids.device != query.device
        or selected_region_ids.shape != (query.shape[1], query.shape[0], top_k)
    ):
        raise RuntimeError("selected_region_ids must be INT32 [G,T,top_k]")
    inputs = (query, weights, key_summary, actual_kv_lengths, block_table, cu_seq_lens)
    attributes = (max_region_len, top_k, region_size, head_scale, cache_mode, mask_mode, sort_mode)
    args = (*inputs, *attributes)
    operator = torch.ops._C_ascend.npu_indexer_region_score
    if out is not None:
        if torch.compiler.is_compiling():
            score, ids = operator.score(*args)
            out[..., :max_region_len].copy_(score)
            if selected_region_ids is not None:
                selected_region_ids.copy_(ids)
            return out, ids if selected_region_ids is None else selected_region_ids
        if selected_region_ids is None:
            selected_region_ids = query.new_empty((query.shape[1], query.shape[0], top_k), dtype=torch.int32)
        operator.out(*args, out=out, selected_region_ids=selected_region_ids)
        return out, selected_region_ids
    if selected_region_ids is not None:
        operator.ids_out(*args, selected_region_ids=selected_region_ids)
        return selected_region_ids
    return operator.default(*args)
