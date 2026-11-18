// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "ops_common.h"
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include "indexer_region_score_torch_adpt.h"

namespace {
void IdsOut(const at::Tensor &query, const at::Tensor &weights, const at::Tensor &key,
    const at::Tensor &lengths, const at::Tensor &table, const at::Tensor &cu,
    int64_t regions, int64_t top_k, int64_t region_size, double head_scale,
    int64_t cache_mode, int64_t mask_mode, int64_t sort_mode, const at::Tensor &selected_ids)
{
    TORCH_CHECK(query.scalar_type() == at::kBFloat16 && key.scalar_type() == at::kBFloat16,
        "the standalone arch22 IndexerRegionScore package requires BF16 query and key tensors");
    vllm_ascend::npu_indexer_region_score_ids_out(query, weights, key, lengths, table, cu,
        regions, top_k, region_size, head_scale, cache_mode, mask_mode, sort_mode, selected_ids);
}

void IdsOutMeta(const at::Tensor &query, const at::Tensor &weights, const at::Tensor &key,
    const at::Tensor &lengths, const at::Tensor &table, const at::Tensor &cu,
    int64_t regions, int64_t top_k, int64_t region_size, double head_scale,
    int64_t cache_mode, int64_t mask_mode, int64_t sort_mode, const at::Tensor &selected_ids)
{
    TORCH_CHECK(query.dim() == 4 && weights.dim() == 3 && key.dim() == 4,
        "expected query[T,G,H,256], weights[T,G,H], key[P,S,1,256]");
    TORCH_CHECK(query.scalar_type() == at::kBFloat16 && key.scalar_type() == at::kBFloat16,
        "the standalone arch22 package requires BF16 query and key tensors");
    TORCH_CHECK(selected_ids.scalar_type() == at::kInt && selected_ids.dim() == 3 &&
        selected_ids.size(0) == query.size(1) && selected_ids.size(1) == query.size(0) &&
        selected_ids.size(2) == top_k, "selected_region_ids must be INT32 [G,T,top_k]");
}
} // namespace

TORCH_LIBRARY(vllm_ascend_indexer, m)
{
    m.def("npu_indexer_region_score.ids_out(Tensor query, Tensor weights, Tensor key_summary, "
          "Tensor actual_kv_lengths, Tensor block_table, Tensor cu_seq_lens, int max_region_len, "
          "int top_k=512, int region_size=8, float head_scale=1.0, int cache_mode=1, int mask_mode=3, "
          "int sort_mode=1, *, Tensor(a!) selected_region_ids) -> ()");
}
TORCH_LIBRARY_IMPL(vllm_ascend_indexer, PrivateUse1, m)
{
    m.impl("npu_indexer_region_score.ids_out", &IdsOut);
}
TORCH_LIBRARY_IMPL(vllm_ascend_indexer, Meta, m)
{
    m.impl("npu_indexer_region_score.ids_out", &IdsOutMeta);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
