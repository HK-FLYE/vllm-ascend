// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include <cassert>
#include <cstdint>
#include <iostream>
#include "../op_host/indexer_region_score_bounded_workspace.h"

int main()
{
    using namespace IndexerRegionScoreHost;
    BoundedWorkspacePlan plan;
    constexpr uint64_t LIBRARY_BYTES = 16U * 1024U * 1024U;
    assert(!PlanBoundedWorkspace(0, 0, 1, 1, 1, plan));
    assert(!PlanBoundedWorkspace(0, 1, 0, 1, 1, plan));
    assert(!PlanBoundedWorkspace(0, 1, 1, 16777217U, 1, plan));
    assert(!PlanBoundedWorkspace(UINT64_MAX, 1, 1, 1, 1, plan));
    assert(!PlanBoundedWorkspace(LIBRARY_BYTES, 4, 1, 16777216U, 1, plan));
    uint32_t checked = 0;
    for (uint32_t groups : {1U, 4U, 8U, 1024U}) {
        for (uint32_t tokens : {1U, 31U, 2049U, 3777U, UINT32_MAX}) {
            for (uint32_t regions : {0U, 1U, 7U, 8U, 9000U, 32769U, 16777216U}) {
                for (uint32_t tile : {1U, 10U, 32U}) {
                    if (!PlanBoundedWorkspace(LIBRARY_BYTES + 3U, groups, tokens, regions, tile, plan)) continue;
                    assert(plan.batchTokens > 0 && plan.batchTokens <= tokens);
                    assert(plan.regionStride >= regions && plan.regionStride >= 8 && plan.regionStride % 8 == 0);
                    assert((plan.scorePoolOffset + LIBRARY_BYTES + 3U) % 32 == 0);
                    assert(plan.totalBytes == LIBRARY_BYTES + 3U + plan.scorePoolOffset +
                        uint64_t(groups) * plan.batchTokens * plan.regionStride * sizeof(float));
                    assert(plan.totalBytes <= kTopKOnlyWorkspaceBudgetBytes - kBoundedExecutorReserve);
                    if (plan.batchTokens < tokens && plan.batchTokens >= tile) assert(plan.batchTokens % tile == 0);
                    ++checked;
                }
            }
        }
    }
    std::cout << checked << " bounded workspace plans and rejection guards passed\n";
}
