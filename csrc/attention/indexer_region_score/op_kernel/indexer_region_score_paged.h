// 版权所有 (c) 2026 华为技术有限公司，遵循 CANN 开源软件许可协议 2.0。
#ifndef INDEXER_REGION_SCORE_PAGED_H
#define INDEXER_REGION_SCORE_PAGED_H

#include "indexer_region_score_common.h"
#include <cstdint>

// 本文件实现分页缓存的 Key 常驻 Scalar 元数据路径及 GM 到 L1 的 NZ 适配。
namespace IndexerRegionScorePaged {
using namespace AscendC;

template <bool PARTIAL_PAGES>
class ScalarPagedKeyResidentBringup final : public KeyResidentAdapterBase {
public:
    __aicore__ inline void Init(GM_ADDR pagedKeySummary, GM_ADDR blockTable,
                                GM_ADDR cuSeqLens, const PagedTilingData *tiling,
                                TPipe *pipe)
    {
        tiling_ = *tiling;
        InitPageLayout();
        pipe_ = pipe;
        pagedKeySummaryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ QkInputType *>(pagedKeySummary));
        blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTable));
        cuSeqLensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cuSeqLens));

        // Launch-invariant source bounds are checked once, not with a divide per macro.
        const uint64_t sourceElementLimit = (uint64_t{1} << 40U) / sizeof(QkInputType);
        affineMaxPageDelta_ = tiling_.cacheStridePage > 0U ?
            sourceElementLimit / tiling_.cacheStridePage : 0U;
        affineSourceBounds_ = tiling_.cacheStrideRegion == QkInputDim(tiling_) &&
            tiling_.cacheStridePage >= static_cast<uint64_t>(tiling_.summariesPerPage) *
                QkInputDim(tiling_) &&
            tiling_.numPhysicalPages < affineMaxPageDelta_;

        InitKeyMacroSlots();
    }

    // 此接口将分页 K 宏转换为 Key 常驻 L1 操作数，Cube 路径需等待就绪并在消费后释放槽位。
    __aicore__ inline uint32_t PrepareKeyMacroToL1(uint32_t request, uint32_t kStart,
                                                   uint32_t kCount, uint32_t slot,
                                                   uint32_t kRowsAlign)
    {
        if (request >= tiling_.numRequests || kCount == 0U || kCount > kPagedKeyMacro ||
            kStart >= tiling_.seqK || kCount > tiling_.seqK - kStart ||
            slot >= kPagedKeyL1Slots || kRowsAlign < kCount ||
            kRowsAlign > kPagedKeyMacro || (kRowsAlign % kPagedNAlignment) != 0U) {
            return 0U;
        }
        if (keySlotState_[slot] != kKeySlotFree) {
            return 0U;
        }
        WaitFlag<HardEvent::MTE1_MTE2>(keyFree_[slot]);
        const uint32_t runCount = ScanOneMacroPageRuns(request, kStart, kCount, slot,
                                                        kRowsAlign);
        if (runCount == 0U) {
            // Host Tiling 保证此错误不可达；仍填零 K 宏以免 AIC/AIV 遍历失步并在 C_READY 死锁。
            ZeroKeyL1Slot(slot);
            SetFlag<HardEvent::MTE2_MTE1>(keyReady_[slot]);
            keySlotState_[slot] = kKeySlotReady;
            return 1U;
        }
        SetFlag<HardEvent::MTE2_MTE1>(keyReady_[slot]);
        keySlotState_[slot] = kKeySlotReady;
        return runCount;
    }

    __aicore__ inline bool GetRequestTokenSpan(uint32_t request, uint32_t &begin, uint32_t &end)
    {
        return request < tiling_.numRequests && LoadRequestTokenSpan(request, begin, end);
    }

private:
    __aicore__ inline void InitPageLayout()
    {
        pagesContiguous_ = tiling_.cacheStridePage ==
            static_cast<uint64_t>(tiling_.summariesPerPage) * QkInputDim(tiling_);
    }

    __aicore__ inline uint32_t MinValue(uint32_t left, uint32_t right) const
    {
        return left < right ? left : right;
    }

    __aicore__ inline bool LoadRequestTokenSpan(uint32_t request, uint32_t &begin, uint32_t &end)
    {
        // 此路径运行于 AIC Scalar，直接读取 cu_seqlens 和 block-table，避免搬运标量元数据导致 AIC/AIV 工作列表不一致。
        const int32_t tokenBegin = cuSeqLensGm_.GetValue(request);
        const int32_t tokenEnd = cuSeqLensGm_.GetValue(request + 1U);
        // Host 仅能校验 cu_seq_lens.shape == N+1；此处继续校验数值以避免生成无效 Q GM 地址。
        if (tokenBegin < 0 || tokenEnd < tokenBegin ||
            static_cast<uint32_t>(tokenEnd) > tiling_.totalTokens) {
            return false;
        }
        begin = static_cast<uint32_t>(tokenBegin);
        end = static_cast<uint32_t>(tokenEnd);
        return true;
    }

    // 分页 L1 适配器共用页段扫描；FP8 NZ 目标行按 row*32 元素定位，而非 row*D。
    __aicore__ inline void CopyContiguousRunToL1(uint32_t slot, uint32_t logicalRow,
                                                 uint32_t runRows, int32_t physicalPage,
                                                 uint32_t kRowsAlign, uint32_t sourceInPageRow = 0U)
    {
        Nd2NzParams nd2nz;
        nd2nz.ndNum = 1U;
        nd2nz.nValue = runRows;
        nd2nz.dValue = QkInputDim(tiling_);
        nd2nz.srcNdMatrixStride = 0U;
        nd2nz.srcDValue = QkInputDim(tiling_);
        nd2nz.dstNzC0Stride = kRowsAlign;
        nd2nz.dstNzNStride = 1U;
        nd2nz.dstNzMatrixStride = 0U;
        const uint64_t sourceOffset = static_cast<uint64_t>(static_cast<uint32_t>(physicalPage)) *
            tiling_.cacheStridePage +
            static_cast<uint64_t>(sourceInPageRow) * tiling_.cacheStrideRegion;
        DataCopy(kL1_[slot][logicalRow * kPagedInputC0], pagedKeySummaryGm_[sourceOffset], nd2nz);
    }

    __aicore__ inline void ZeroKeyL1Slot(uint32_t slot)
    {
        // E4M3 零值为全零位模式；先在 MTE2 清零再覆盖有效页，使无效页等价于 Triton 的 other=0.0。
        InitConstValueParams<uint16_t> zeroParams(
            1U, static_cast<uint16_t>(kPagedKeyL1Bytes / 32U), 0U, 0U);
        InitConstValue(kL1_[slot].template ReinterpretCast<uint16_t>(), zeroParams);
        PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline uint32_t ScanOneMacroPageRuns(uint32_t request, uint32_t kStart,
                                                     uint32_t kCount, uint32_t slot,
                                                     uint32_t kRowsAlign)
    {
        // Use the whole-page scanner when macro boundaries are page-aligned.
        if constexpr (!PARTIAL_PAGES) {
            return ScanAlignedMacroPageRuns(request, kStart, kCount, slot, kRowsAlign);
        } else {
            return ScanPartialMacroPageRuns(request, kStart, kCount, slot, kRowsAlign);
        }
    }

    __aicore__ inline uint32_t ScanAlignedMacroPageRuns(uint32_t request, uint32_t kStart,
                                                     uint32_t kCount, uint32_t slot,
                                                     uint32_t kRowsAlign)
    {
        const uint32_t firstPage = kStart / tiling_.summariesPerPage;
        const uint32_t pageCount = (kCount + tiling_.summariesPerPage - 1U) /
            tiling_.summariesPerPage;
        if (pageCount == 0U || pageCount > kPagedMaxPagesPerMacro ||
            firstPage > tiling_.pageCols || pageCount > tiling_.pageCols - firstPage) {
            return 0U;
        }
        // Bound scalar metadata to 256 bytes; smaller pages retain the streaming scanner.
        constexpr uint32_t kCachedPageIds = 256U / sizeof(int32_t);
        int32_t pageIds[kCachedPageIds];
        // Only a full-slot overwrite can benefit from the metadata pre-scan.
        // Tail and partial-page slots retain the streaming scanner.
        const bool fullSlotCoverage = kCount == kPagedKeyMacro &&
            kRowsAlign == kPagedKeyMacro && kStart % tiling_.summariesPerPage == 0U &&
            kCount % tiling_.summariesPerPage == 0U &&
            static_cast<uint64_t>(kCount) * QkInputDim(tiling_) * sizeof(QkInputType) == kPagedKeyL1Bytes;
        const bool cachePages = fullSlotCoverage && pageCount <= kCachedPageIds;
        bool allPagesValid = cachePages;
        bool hasAdjacentRun = false;
        bool positiveAffine = cachePages && pageCount > 1U;
        int64_t pageDelta = 0;
        const uint64_t tableBase = static_cast<uint64_t>(request) *
            tiling_.blockTableStrideRequest + firstPage;
        if (cachePages) {
            for (uint32_t i = 0U; i < pageCount; ++i) {
                const int32_t page = blockTableGm_.GetValue(tableBase + i);
                pageIds[i] = page;

                if (i > 0U) {
                    const int64_t delta = static_cast<int64_t>(page) -
                        static_cast<int64_t>(pageIds[i - 1U]);
                    if (i == 1U) {
                        pageDelta = delta;
                    }
                    positiveAffine = positiveAffine && delta > 0 && delta == pageDelta;
                }

                allPagesValid = allPagesValid && page >= 0 &&
                    static_cast<uint64_t>(page) < tiling_.numPhysicalPages;
                if (i > 0U && static_cast<int64_t>(page) ==
                    static_cast<int64_t>(pageIds[i - 1U]) + 1) {
                    hasAdjacentRun = true;
                }
            }
        }
        hasAdjacentRun = hasAdjacentRun && pagesContiguous_;
        // Skip clear only when legal full pages overwrite every allocated slot byte.
        const bool fullOverwrite = allPagesValid && fullSlotCoverage;
        if (!fullOverwrite) {
            ZeroKeyL1Slot(slot);
        }

        // Preserve every original per-page NZ address in one positive-affine multi-ND copy.
        // Contiguous delta-one pages keep merged loading; strided delta-one pages use multi-ND.
        // Tail, partial, invalid and non-affine macros retain the original loader.
        if (fullOverwrite && positiveAffine &&
            (pageDelta > 1 || (pageDelta == 1 && !pagesContiguous_)) &&
            affineSourceBounds_ && static_cast<uint64_t>(pageDelta) < affineMaxPageDelta_) {
            Nd2NzParams nd2nz;
            nd2nz.ndNum = pageCount;
            nd2nz.nValue = tiling_.summariesPerPage;
            nd2nz.dValue = QkInputDim(tiling_);
            nd2nz.srcNdMatrixStride = static_cast<uint64_t>(pageDelta) * tiling_.cacheStridePage;
            nd2nz.srcDValue = QkInputDim(tiling_);
            nd2nz.dstNzC0Stride = kRowsAlign;
            nd2nz.dstNzNStride = 1U;
            nd2nz.dstNzMatrixStride = tiling_.summariesPerPage * kPagedInputC0;
            const uint64_t sourceOffset =
                static_cast<uint64_t>(static_cast<uint32_t>(pageIds[0])) * tiling_.cacheStridePage;
            DataCopy(kL1_[slot], pagedKeySummaryGm_[sourceOffset], nd2nz);
            return 1U;
        }

        if (fullOverwrite && !hasAdjacentRun) {
            Nd2NzParams nd2nz;
            nd2nz.ndNum = 1U;
            nd2nz.nValue = tiling_.summariesPerPage;
            nd2nz.dValue = QkInputDim(tiling_);
            nd2nz.srcNdMatrixStride = 0U;
            nd2nz.srcDValue = QkInputDim(tiling_);
            nd2nz.dstNzC0Stride = kRowsAlign;
            nd2nz.dstNzNStride = 1U;
            nd2nz.dstNzMatrixStride = 0U;
            const uint32_t destinationPageStride = tiling_.summariesPerPage * kPagedInputC0;
            const uint64_t cacheStridePage = tiling_.cacheStridePage;
            uint32_t i = 0U;
            for (; i + 4U <= pageCount; i += 4U) {
                const uint64_t sourceOffset0 =
                    static_cast<uint64_t>(static_cast<uint32_t>(pageIds[i])) * cacheStridePage;
                const uint64_t sourceOffset1 =
                    static_cast<uint64_t>(static_cast<uint32_t>(pageIds[i + 1U])) * cacheStridePage;
                const uint64_t sourceOffset2 =
                    static_cast<uint64_t>(static_cast<uint32_t>(pageIds[i + 2U])) * cacheStridePage;
                const uint64_t sourceOffset3 =
                    static_cast<uint64_t>(static_cast<uint32_t>(pageIds[i + 3U])) * cacheStridePage;
                DataCopy(kL1_[slot][i * destinationPageStride],
                    pagedKeySummaryGm_[sourceOffset0], nd2nz);
                DataCopy(kL1_[slot][(i + 1U) * destinationPageStride],
                    pagedKeySummaryGm_[sourceOffset1], nd2nz);
                DataCopy(kL1_[slot][(i + 2U) * destinationPageStride],
                    pagedKeySummaryGm_[sourceOffset2], nd2nz);
                DataCopy(kL1_[slot][(i + 3U) * destinationPageStride],
                    pagedKeySummaryGm_[sourceOffset3], nd2nz);
            }
            for (; i < pageCount; ++i) {
                const uint64_t sourceOffset =
                    static_cast<uint64_t>(static_cast<uint32_t>(pageIds[i])) * cacheStridePage;
                DataCopy(kL1_[slot][i * destinationPageStride],
                    pagedKeySummaryGm_[sourceOffset], nd2nz);
            }
            return pageCount;
        }

        uint32_t runCount = 0U;
        for (uint32_t localPage = 0U; localPage < pageCount;) {
            const int32_t page = cachePages ? pageIds[localPage] :
                blockTableGm_.GetValue(tableBase + localPage);
            const bool zeroFill = page < 0 || static_cast<uint64_t>(page) >= tiling_.numPhysicalPages;
            uint32_t length = 1U;
            if (pagesContiguous_ || zeroFill) {
                while (localPage + length < pageCount) {
                    const int32_t nextPage = cachePages ? pageIds[localPage + length] :
                        blockTableGm_.GetValue(tableBase + localPage + length);
                    const bool nextZeroFill = nextPage < 0 ||
                        static_cast<uint64_t>(nextPage) >= tiling_.numPhysicalPages;
                    if (nextZeroFill != zeroFill ||
                        (!zeroFill && static_cast<int64_t>(nextPage) !=
                            static_cast<int64_t>(page) + static_cast<int64_t>(length))) {
                        break;
                    }
                    ++length;
                }
            }
            if (!zeroFill) {
                const uint32_t logicalRow = localPage * tiling_.summariesPerPage;
                const uint32_t rowsRemaining = kCount - logicalRow;
                const uint32_t runRows = MinValue(length * tiling_.summariesPerPage,
                                                   rowsRemaining);
                CopyContiguousRunToL1(slot, logicalRow, runRows, page, kRowsAlign);
            }
            ++runCount;
            localPage += length;
        }
        return runCount;
    }

    __aicore__ inline uint32_t ScanPartialMacroPageRuns(uint32_t request, uint32_t kStart,
                                                     uint32_t kCount, uint32_t slot,
                                                     uint32_t kRowsAlign)
    {
        const uint32_t firstPage = kStart / tiling_.summariesPerPage;
        const uint32_t firstOffset = kStart % tiling_.summariesPerPage;
        const uint32_t pageCount = (firstOffset + kCount + tiling_.summariesPerPage - 1U) /
            tiling_.summariesPerPage;
        if (pageCount == 0U || pageCount > kPagedMaxPagesPerMacro ||
            firstPage > tiling_.pageCols || pageCount > tiling_.pageCols - firstPage) {
            return 0U;
        }
        // 初始化后无效页和未用 K 尾部保持为零，有效物理页段再按需覆盖。
        ZeroKeyL1Slot(slot);

        uint32_t runCount = 0U;
        const uint64_t tableBase = static_cast<uint64_t>(request) *
            tiling_.blockTableStrideRequest + firstPage;
        for (uint32_t localPage = 0U; localPage < pageCount;) {
            const int32_t page = blockTableGm_.GetValue(tableBase + localPage);
            const bool zeroFill = page < 0 || static_cast<uint64_t>(page) >= tiling_.numPhysicalPages;
            uint32_t length = 1U;
            if (pagesContiguous_ || zeroFill) {
                while (localPage + length < pageCount) {
                    const int32_t nextPage = blockTableGm_.GetValue(tableBase + localPage + length);
                    const bool nextZeroFill = nextPage < 0 ||
                        static_cast<uint64_t>(nextPage) >= tiling_.numPhysicalPages;
                    if (nextZeroFill != zeroFill ||
                        (!zeroFill && static_cast<int64_t>(nextPage) !=
                            static_cast<int64_t>(page) + static_cast<int64_t>(length))) {
                        break;
                    }
                    ++length;
                }
            }
            if (!zeroFill) {
                const uint32_t pageRow = localPage * tiling_.summariesPerPage;
                const uint32_t logicalRow = pageRow > firstOffset ? pageRow - firstOffset : 0U;
                const uint32_t sourceInPageRow = localPage == 0U ? firstOffset : 0U;
                const uint32_t rowsRemaining = kCount - logicalRow;
                const uint32_t runRows = MinValue(length * tiling_.summariesPerPage - sourceInPageRow,
                                                   rowsRemaining);
                CopyContiguousRunToL1(slot, logicalRow, runRows, page, kRowsAlign, sourceInPageRow);
            }
            ++runCount;
            localPage += length;
        }
        return runCount;
    }

    uint64_t affineMaxPageDelta_ = 0U;
    bool affineSourceBounds_ = false;
    bool pagesContiguous_ = false;
    GlobalTensor<QkInputType> pagedKeySummaryGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> cuSeqLensGm_;
};
} // 命名空间 IndexerRegionScorePaged

#endif
