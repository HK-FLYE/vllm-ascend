// 版权所有 (c) 2026 华为技术有限公司，遵循 CANN 开源软件许可协议 2.0。
#ifndef INDEXER_REGION_SCORE_PAGED_TILING_DATA_H
#define INDEXER_REGION_SCORE_PAGED_TILING_DATA_H

#include <cstdint>
#include <cstddef>

namespace IndexerRegionScorePaged {
constexpr uint64_t PAGE_LOAD_ALIGNED = 0U;
constexpr uint64_t PAGE_LOAD_WITH_OFFSET = 1U;
// Host 与设备端共用此 POD，所有步长均以元素计。
struct alignas(8) PagedTilingData {
    uint32_t totalTokens;
    uint32_t groups;
    uint32_t headsPerGroup;
    uint32_t proxyDim;
    uint32_t numRequests;
    uint32_t pageCols;
    uint32_t summariesPerPage;
    uint32_t seqK;                 // 逻辑 R 长度，不超过 pageCols*summariesPerPage，范围为 uint32。
    uint32_t qTile;                // Q 范围分片的 token 粒度。
    uint32_t kTile;
    uint32_t usedCubeCores;
    uint32_t maskMode;
    // headScale 不为 1.0 时，在加权 ReLU 归约前缩放所有 head 权重。
    float headScale;
    uint32_t keyPagesPerMacro;
    uint32_t qShardCount;          // 连续 Q 范围分片数。
    uint32_t kShardCount;          // 每个 Q 分片的连续 K 宏分片数。
    // Scalar 页段规划器要求正 block_table 页 ID 小于 paged_key_summary 的物理页数。
    uint32_t numPhysicalPages;
    // 类型编码为 0=FP32、1=BF16、2=FP16；AIV 路径将权重扩展为 FP32。
    uint32_t weightDtype;
    uint64_t qStrideToken;
    uint64_t qStrideGroup;
    uint64_t qStrideHead;
    uint64_t weightStrideToken;
    uint64_t weightStrideGroup;
    uint64_t cacheStridePage;
    uint64_t cacheStrideRegion;
    uint64_t blockTableStrideRequest;
    uint64_t outputStrideGroup;
    uint64_t outputStrideToken;
    uint64_t reservedStride0; // 相对 Kernel workspace 指针的 Scratch 字节偏移。
    // Host 推导的分页搬运模式，非用户属性；0=页对齐，1=支持页内起点。
    uint64_t pageLoadMode;
    uint32_t regionSize;
    // 位 [15:0] 为 topK，位 16 为 ID 排序，位 17 为 scratch 对齐。
    uint32_t selectionConfig;
    // Zero batchTokens selects the full-score path; coordinates stay global.
    uint32_t batchTokens;
    uint32_t batchBegin;
    uint32_t batchCount;
    uint32_t batchReserved;
    uint64_t scorePoolOffset; // bytes relative to the kernel workspace pointer
};
static_assert(offsetof(PagedTilingData, pageLoadMode) == 160, "page-load ABI offset");
static_assert(offsetof(PagedTilingData, regionSize) == 168, "region-size ABI offset");
static_assert(offsetof(PagedTilingData, selectionConfig) == 172, "selection ABI offset");
static_assert(sizeof(PagedTilingData) % 8 == 0, "paged tiling must be naturally aligned");
static_assert(offsetof(PagedTilingData, keyPagesPerMacro) == 52, "key-pages ABI offset");
static_assert(sizeof(PagedTilingData) == 200, "update every paged tiling-data packer");
} // 命名空间 IndexerRegionScorePaged

// CANN Tiling 注册宏要求标准 Host 命名空间；此别名供 Host 与设备入口共用。
namespace optiling {
using PagedTilingData = IndexerRegionScorePaged::PagedTilingData;
} // 命名空间 optiling

#endif
