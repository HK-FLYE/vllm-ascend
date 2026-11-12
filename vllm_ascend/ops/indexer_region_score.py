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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score paged regions and return selected logical region IDs in ascending order.

    Ascend950 only. ``query`` is [T,G,H,256], ``weights`` is [T,G,H], and
    ``key_summary`` is [P,S,1,256], shared by all G groups. G is 1 or a
    positive multiple of 4; H is 4..128 in steps of 4. Query and key use FP16, BF16,
    or FP8 E4M3FN dtype; weights may also use FP32. Noncontiguous tensors are
    converted as needed; aligned paged cache views preserve zero-copy access.
    Only cache_mode=1, mask_mode=3 and sort_mode=1 are currently supported.

    ``block_table`` [N,C] maps logical pages to physical pages. INT32 request
    metadata stays on device: ``cu_seq_lens`` [N+1] partitions query tokens and
    ``actual_kv_lengths`` [N] includes the current query tokens. Right-down
    causal masking selects only complete historical regions; it does not
    append the current region, map IDs to physical pages, or pack the cache.

    Scores are FP32 [G,T,R_out] with R_out >= max_region_len. Only visible
    candidates have defined scores. IDs are INT32 [G,T,top_k], 1 <= top_k <= 512;
    ties prefer smaller IDs and missing candidates are padded with -1.
    Optional output buffers are reused and must not alias inputs or each other.
    """
    if (cache_mode, mask_mode, sort_mode) != (1, 3, 1):
        raise ValueError("only cache_mode=1, mask_mode=3 and sort_mode=1 are supported")
    return torch.ops._C_ascend.npu_indexer_region_score(
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
        out,
        selected_region_ids,
    )


def indexer_region_score_topk(
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
    selected_region_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the same IDs without a dense score output, within 256 MiB workspace.

    Shape, dtype and masking follow :func:`indexer_region_score`. Q, weights,
    metadata and IDs must be contiguous; Key may use an aligned paged view.
    Caller IDs require a 32-byte aligned storage base and INT32 offset divisible
    by eight. Unsupported layouts and a minimum batch exceeding the budget are rejected.
    Omit ``selected_region_ids`` for the functional graph path.
    """
    if (cache_mode, mask_mode, sort_mode) != (1, 3, 1):
        raise ValueError("only cache_mode=1, mask_mode=3 and sort_mode=1 are supported")
    args = (
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
    )
    op = torch.ops._C_ascend.npu_indexer_region_score_topk
    if selected_region_ids is None:
        return op.default(*args)
    return op.out(*args, selected_region_ids=selected_region_ids)
