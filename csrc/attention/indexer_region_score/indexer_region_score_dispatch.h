// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <memory>

#include <ATen/ATen.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

// The including adapter supplies GetOpApiFuncAddr, ConvertType and Release.
// Both the repository adapter and the standalone PTA provide these helpers.
namespace vllm_ascend {
namespace indexer_region_score_detail {
using WorkspaceFunction = int (*)(const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, int64_t, int64_t, int64_t,
    double, int64_t, int64_t, int64_t, const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
using LaunchFunction = int (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using DestroyExecutorFunction = int (*)(aclOpExecutor *);

struct CallState {
    std::array<at::Tensor, 7> tensors;
    std::array<aclTensor *, 7> descriptors{};
    at::Tensor workspace;
    uint64_t workspace_bytes = 0;
    aclOpExecutor *executor = nullptr;
    DestroyExecutorFunction destroy_executor = nullptr;

    void ReleaseResources()
    {
        if (executor != nullptr && destroy_executor != nullptr) {
            destroy_executor(executor);
            executor = nullptr;
        }
        for (auto *&descriptor : descriptors) {
            if (descriptor != nullptr) ::Release(descriptor);
            descriptor = nullptr;
        }
        tensors = {};
        workspace = at::Tensor();
    }

    ~CallState() { ReleaseResources(); }
};

inline void Dispatch(const at::Tensor &query, const at::Tensor &weights, const at::Tensor &key,
    const at::Tensor &lengths, const at::Tensor &table, const at::Tensor &cu,
    int64_t regions, int64_t top_k, int64_t region_size, double head_scale,
    int64_t cache_mode, int64_t mask_mode, int64_t sort_mode, const at::Tensor &selected_ids)
{
    static const auto get_workspace = reinterpret_cast<WorkspaceFunction>(
        ::GetOpApiFuncAddr("aclnnIndexerRegionScoreGetWorkspaceSize"));
    static const auto launch = reinterpret_cast<LaunchFunction>(::GetOpApiFuncAddr("aclnnIndexerRegionScore"));
    static const auto destroy = reinterpret_cast<DestroyExecutorFunction>(::GetOpApiFuncAddr("aclDestroyAclOpExecutor"));
    TORCH_CHECK(get_workspace != nullptr && launch != nullptr && destroy != nullptr,
        "IndexerRegionScore ACLNN library is unavailable; load its matching custom OPP package before calling it");

    auto state = std::make_shared<CallState>();
    state->tensors = {query, weights, key, lengths, table, cu, selected_ids};
    state->destroy_executor = destroy;
    for (size_t index = 0; index < state->tensors.size(); ++index) {
        state->descriptors[index] = ::ConvertType(state->tensors[index]);
        TORCH_CHECK(state->descriptors[index] != nullptr, "failed to create IndexerRegionScore tensor metadata");
    }
    const auto &input = state->descriptors;
    const auto status = get_workspace(input[0], input[1], input[2], input[3], input[4], input[5],
        regions, top_k, region_size, head_scale, cache_mode, mask_mode, sort_mode, nullptr,
        input[6], &state->workspace_bytes, &state->executor);
    TORCH_CHECK(status == 0, "IndexerRegionScore GetWorkspaceSize failed with status ", status);
    TORCH_CHECK(state->executor != nullptr, "IndexerRegionScore returned an empty executor");
    TORCH_CHECK(state->workspace_bytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
        "IndexerRegionScore workspace exceeds the tensor allocation limit");
    if (state->workspace_bytes != 0) {
        state->workspace = at::empty({static_cast<int64_t>(state->workspace_bytes)}, query.options().dtype(at::kByte));
    }

    const auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : state->tensors) {
        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
    }
    if (state->workspace.defined()) {
        c10_npu::NPUCachingAllocator::recordStream(state->workspace.storage().data_ptr(), stream);
    }
    const auto acl_stream = stream.stream(false);
    auto operation = [state, acl_stream]() -> int {
        // The native launch consumes its executor. Before launch, state owns it
        // so an allocation or queueing exception cannot leak it.
        auto *executor = state->executor;
        state->executor = nullptr;
        void *workspace = state->workspace.defined() ? state->workspace.data_ptr() : nullptr;
        int launch_status;
        try {
            launch_status = launch(workspace, state->workspace_bytes, executor, acl_stream);
        } catch (...) {
            state->ReleaseResources();
            throw;
        }
        // recordStream protects the device allocations after launch. Release
        // host Tensor references now, before the queue retains this callback
        // for deferred destruction (which can outlive Python shutdown).
        state->ReleaseResources();
        TORCH_CHECK(launch_status == 0, "IndexerRegionScore launch failed with status ", launch_status);
        return launch_status;
    };
    // state retains inputs, output, ACL descriptors and workspace until the host
    // queue callback executes. recordStream retains their allocations through
    // asynchronous device execution, including inputs created on other streams.
    at_npu::native::OpCommand::RunOpApiV2("aclnnIndexerRegionScore", operation);
}
} // namespace indexer_region_score_detail
} // namespace vllm_ascend
