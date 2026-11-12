// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "aclnn_vllm_ascend_indexer_region_score.h"
#include <cstdint>
#include "../indexer_region_score_bounded_workspace.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/tensor_move.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
using namespace op;
namespace {
bool IsPagedKeyView(const aclTensor *key)
{
    const auto &shape = key->GetViewShape();
    const auto &strides = key->GetViewStrides();
    const int64_t bytes = key->GetDataType() == DataType::DT_FLOAT8_E4M3FN ? 1 : 2;
    const int64_t offset = key->GetViewOffset();
    constexpr int64_t DIM = 256, ALIGNMENT = 32, MAX_SUMMARIES = 1024;
    return shape.GetDimNum() == 4 && strides.size() == 4 && shape.GetDim(0) > 0 &&
        shape.GetDim(1) > 0 && shape.GetDim(1) <= MAX_SUMMARIES && shape.GetDim(2) == 1 && shape.GetDim(3) == DIM &&
        strides[3] == 1 && (shape.GetDim(1) == 1 || strides[1] == DIM) &&
        strides[0] >= shape.GetDim(1) * DIM && strides[0] % (ALIGNMENT / bytes) == 0 &&
        offset >= 0 && offset % (ALIGNMENT / bytes) == 0 &&
        offset <= INT64_MAX / bytes - shape.GetDim(1) * DIM &&
        shape.GetDim(0) - 1 <= (INT64_MAX / bytes - shape.GetDim(1) * DIM - offset) / strides[0];
}

const aclTensor *PrepareTensor(const aclTensor *tensor, bool pagedKey, bool aligned, aclOpExecutor *executor)
{
    if (pagedKey && IsPagedKeyView(tensor)) {
        return executor->CreateView(tensor, tensor->GetViewShape(), tensor->GetStorageShape(),
            tensor->GetViewStrides(), tensor->GetViewOffset());
    }
    const auto *contiguous = l0op::Contiguous(tensor, executor);
    if (!contiguous) return nullptr;
    const int64_t bytes = tensor->GetDataType() == DataType::DT_FLOAT8_E4M3FN ? 1 : 2;
    constexpr int64_t ALIGNMENT = 32;
    if ((!aligned || contiguous->IsFromWorkspace() || tensor->GetViewOffset() % (ALIGNMENT / bytes) == 0) &&
        (!pagedKey || IsPagedKeyView(contiguous))) return contiguous;
    // Contiguous can retain unaligned slices or noncanonical singleton strides.
    auto *copy = executor->AllocTensor(tensor->GetViewShape(), tensor->GetDataType());
    if (!copy || !l0op::TensorMove(contiguous, copy, executor)) return nullptr;
    return copy;
}
} // namespace
namespace l0op {
OP_TYPE_REGISTER(VllmAscendIndexerRegionScore);
aclnnStatus VllmAscendIndexerRegionScore(
    const aclTensor *query, const aclTensor *weights, const aclTensor *key,
    const aclTensor *lengths, const aclTensor *table, const aclTensor *cu,
    int64_t regions, int64_t topK, int64_t regionSize, double headScale,
    int64_t cacheMode, int64_t maskMode, int64_t sortMode,
    const aclTensor *output, const aclTensor *ids, aclOpExecutor *executor)
{
    L0_DFX(VllmAscendIndexerRegionScore, query, weights, key, lengths, table, cu,
        regions, topK, regionSize, headScale, cacheMode, maskMode, sortMode, output, ids);
    const bool returnScore = output != nullptr;
    if (!output) output = executor->AllocTensor(Shape{0}, DataType::DT_FLOAT);
    if (!output) return ACLNN_ERR_INNER_NULLPTR;
    return ADD_TO_LAUNCHER_LIST_AICORE(VllmAscendIndexerRegionScore,
        OP_INPUT(query, weights, key, lengths, table, cu), OP_OUTPUT(output, ids),
        OP_ATTR(regions, topK, regionSize, static_cast<float>(headScale), cacheMode, maskMode, sortMode, returnScore));
}
} // namespace l0op
extern "C" aclnnStatus aclnnVllmAscendIndexerRegionScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensor *weights, const aclTensor *key,
    const aclTensor *lengths, const aclTensor *table, const aclTensor *cu,
    int64_t regions, int64_t topK, int64_t regionSize, double headScale,
    int64_t cacheMode, int64_t maskMode, int64_t sortMode,
    const aclTensor *output, const aclTensor *ids, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnVllmAscendIndexerRegionScore,
        DFX_IN(query, weights, key, lengths, table, cu, regions, topK, regionSize, headScale,
            cacheMode, maskMode, sortMode), DFX_OUT(output, ids));
    if (!workspaceSize || !executor) return ACLNN_ERR_PARAM_NULLPTR;
    *workspaceSize = 0;
    *executor = nullptr;
    const bool returnScore = output != nullptr;
    if (cacheMode != 1 || maskMode != 3 || sortMode != 1) return ACLNN_ERR_PARAM_INVALID;
    for (const aclTensor *tensor : {query, weights, key, lengths, table, cu, ids}) {
        if (!tensor) return ACLNN_ERR_PARAM_NULLPTR;
        if (tensor->GetViewOffset() < 0) return ACLNN_ERR_PARAM_INVALID;
        tensor->SetOriginalShape(tensor->GetViewShape());
    }
    if (output) output->SetOriginalShape(output->GetViewShape());
    if (!returnScore) {
        for (const aclTensor *tensor : {query, weights, lengths, table, cu, ids}) {
            if (!IsContiguous(tensor)) return ACLNN_ERR_PARAM_INVALID;
        }
        if (!IsPagedKeyView(key) || ids->GetViewOffset() % 8 != 0 ||
            reinterpret_cast<uintptr_t>(ids->GetStorageAddr()) % 32 != 0) return ACLNN_ERR_PARAM_INVALID;
    } else if (output->GetViewOffset() < 0) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto exec = CREATE_EXECUTOR();
    if (!exec.get()) return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    const aclTensor *views[8]{};
    const aclTensor *tensors[] = {query, weights, key, lengths, table, cu, output, ids};
    for (uint32_t i = 0; i < 8; ++i) {
        if (!tensors[i]) continue;
        views[i] = returnScore ? PrepareTensor(tensors[i], i == 2, i == 0 || i == 2, exec.get()) :
            exec->CreateView(tensors[i], tensors[i]->GetViewShape(), tensors[i]->GetStorageShape(),
                tensors[i]->GetViewStrides(), tensors[i]->GetViewOffset());
        if (!views[i]) return ACLNN_ERR_INNER_NULLPTR;
    }
    auto ret = l0op::VllmAscendIndexerRegionScore(views[0], views[1], views[2], views[3], views[4], views[5],
        regions, topK, regionSize, headScale, cacheMode, maskMode, sortMode, views[6], views[7], exec.get());
    if (ret != ACLNN_SUCCESS) return ret;
    if (returnScore && (!l0op::ViewCopy(views[6], output, exec.get()) ||
        !l0op::ViewCopy(views[7], ids, exec.get()))) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    *workspaceSize = exec->GetWorkspaceSize();
    if (!returnScore && *workspaceSize > IndexerRegionScoreHost::kTopKOnlyWorkspaceBudgetBytes) {
        *workspaceSize = 0;
        return ACLNN_ERR_PARAM_INVALID;
    }
    exec.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
extern "C" aclnnStatus aclnnVllmAscendIndexerRegionScore(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnVllmAscendIndexerRegionScore);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
extern "C" aclnnStatus aclnnVllmAscendIndexerRegionScoreTopKGetWorkspaceSize(
    const aclTensor *query, const aclTensor *weights, const aclTensor *key,
    const aclTensor *lengths, const aclTensor *table, const aclTensor *cu,
    int64_t regions, int64_t topK, int64_t regionSize, double headScale,
    int64_t cacheMode, int64_t maskMode, int64_t sortMode,
    const aclTensor *ids, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnVllmAscendIndexerRegionScoreGetWorkspaceSize(query, weights, key, lengths, table, cu,
        regions, topK, regionSize, headScale, cacheMode, maskMode, sortMode, nullptr, ids, workspaceSize, executor);
}
extern "C" aclnnStatus aclnnVllmAscendIndexerRegionScoreTopK(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return aclnnVllmAscendIndexerRegionScore(workspace, workspaceSize, executor, stream);
}
