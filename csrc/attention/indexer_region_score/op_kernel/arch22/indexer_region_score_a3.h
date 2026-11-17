// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#ifndef INDEXER_REGION_SCORE_A3_H
#define INDEXER_REGION_SCORE_A3_H

#include "kernel_operator.h"
#include "a3_tiling_data.h"

namespace IndexerRegionScoreA3 {
using namespace AscendC;

// A query bundle is owned by one Cube/AIV pair. Only a bounded H-by-128 dot tile
// crosses GM. Scores retain the source's quartet reduction order; no floating
// atomics or cross-core floating reductions are used.
class A3IndexerRegionScore {
public:
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR weights, GM_ADDR key,
        GM_ADDR lengths, GM_ADDR table, GM_ADDR cu, GM_ADDR output, GM_ADDR ids,
        GM_ADDR userWorkspace, const PagedTilingData &data, TPipe *pipe)
    {
        data_ = data;
        pipe_ = pipe;
        query_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(query));
        key_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(key));
        weightsFloat_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weights));
        weightsBits_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(weights));
        weightsHalf_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(weights));
        lengths_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(lengths));
        table_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(table));
        cu_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cu));
        ids_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ids));
        scores_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
        if (data_.batchTokens != 0) {
            scores_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace + data_.scorePoolOffset));
        }
        // Metadata belongs to this invocation only; every Init reloads it.
        singleRequestCached_ = BatchedVector() && data_.numRequests == 1U;
        singleRegionPower2_ = false;
        singleRegionShift_ = 0;
        if (singleRequestCached_) {
            singleBegin_ = cu_.GetValue(0);
            singleEnd_ = cu_.GetValue(1);
            singleLength_ = lengths_.GetValue(0);
            const uint32_t region = data_.regionSize;
            singleRegionPower2_ = region != 0U && (region & (region - 1U)) == 0U;
            if (singleRegionPower2_) {
                for (uint32_t value = region; value > 1U; value >>= 1U) ++singleRegionShift_;
            }
        }
        core_ = GetBlockIdx();
        if ASCEND_IS_AIV {
            core_ /= 2;
            sub_ = GetSubBlockIdx();
        }
        bridge_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            userWorkspace + data_.reservedStride0 + core_ * kBridgeBytesPerCore));
        if ASCEND_IS_AIC {
            InitCube();
        } else {
            InitVector();
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            CubeRows();
        } else {
            VectorRows();
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll<false>();
        if ASCEND_IS_AIV {
            SelectRows();
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr uint16_t kReady = 0;
    static constexpr uint16_t kFree = 1;
    static constexpr uint32_t kSelectionWindow = 4096;
    // One H4 IDs-only UB arena; byte aliases follow the frozen round7 contract.
    static constexpr uint32_t kArenaBytes = 153600;
    static constexpr uint32_t kBatchRows = 4;
    static constexpr uint32_t kBatchColumns = 640;
    static constexpr uint32_t kBatchInputElements = 2560;
    static constexpr uint32_t kBatchPackedElements = 5120;
    static constexpr uint32_t kBatchPackedRowWords = 1280;
    static constexpr uint32_t kBatchRowBytes = 2560;
    static constexpr uint32_t kBatchOutputRowBytes = 2048;
    static constexpr uint32_t kBatchInputOffset = 0;
    static constexpr uint32_t kBatchPackedAOffset = 10240;
    static constexpr uint32_t kBatchPackedBOffset = 30720;
    static constexpr uint32_t kBatchIndexOffset = 51200;
    static constexpr uint32_t kBatchOrderedOffset = 61440;
    static constexpr uint32_t kBatchOutputOffset = 71680;
    static constexpr uint32_t kBatchLogicalOffset = 79872;
    static constexpr uint32_t kBatchWideOffset = 82432;
    static constexpr uint32_t kBatchFloatIdsOffset = 84992;
    static constexpr uint32_t kBatchGreaterOffset = 87552;
    static constexpr uint32_t kBatchEqualOffset = 87808;
    static constexpr uint32_t kBatchCutoffOffset = 88064;
    // Scoring lifetime aliases; batch selection starts after global SyncAll.
    static constexpr uint32_t kArenaDotsOffset = 0, kArenaDotsElements = 8192;
    static constexpr uint32_t kArenaAccOffset = 32768, kArenaAccElements = 2048;
    static constexpr uint32_t kArenaPartialOffset = 40960, kArenaPartialElements = 64;
    static constexpr uint32_t kArenaWeightOffset = 41216, kArenaWeightElements = 8192;
    static constexpr uint32_t kArenaWindowOffset = 73984, kArenaWindowElements = 4096;
    static constexpr uint32_t kArenaHeapScoreOffset = 90368, kArenaHeapElements = 512;
    static constexpr uint32_t kArenaHeapIdsOffset = 92416;


    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) const { return a < b ? a : b; }
    __aicore__ inline uint32_t Align16(uint32_t n) const { return (n + 15U) / 16U * 16U; }

    __aicore__ inline bool Request(uint32_t token, uint32_t &request, uint32_t &visible) const
    {
        visible = 0;
        for (uint32_t n = 0; n < data_.numRequests; ++n) {
            const int32_t begin = singleRequestCached_ ? singleBegin_ : cu_.GetValue(n);
            const int32_t end = singleRequestCached_ ? singleEnd_ : cu_.GetValue(n + 1);
            if (begin < 0 || end < begin || static_cast<uint32_t>(end) > data_.totalTokens) continue;
            if (token >= static_cast<uint32_t>(begin) && token < static_cast<uint32_t>(end)) {
                request = n;
                const int32_t length = singleRequestCached_ ? singleLength_ : lengths_.GetValue(n);
                const int64_t position = static_cast<int64_t>(length) - (end - begin) + token - begin;
                if (position > 0) {
                    const uint64_t quotient = singleRegionPower2_ ?
                        (static_cast<uint64_t>(position) >> singleRegionShift_) :
                        static_cast<uint64_t>(position / data_.regionSize);
                    visible = Min(static_cast<uint32_t>(quotient), data_.seqK);
                }
                return true;
            }
        }
        return false;
    }

    __aicore__ inline uint64_t ScoreBase(uint32_t group, uint32_t token) const
    {
        const uint32_t row = data_.batchTokens == 0 ? token : token - data_.batchBegin;
        return static_cast<uint64_t>(group) * data_.outputStrideGroup +
            static_cast<uint64_t>(row) * data_.outputStrideToken;
    }

    __aicore__ inline float Weight(uint32_t group, uint32_t token, uint32_t head) const
    {
        const uint64_t offset = static_cast<uint64_t>(token) * data_.weightStrideToken +
            static_cast<uint64_t>(group) * data_.weightStrideGroup + head;
        float value;
        if (data_.weightDtype == 0) {
            value = weightsFloat_.GetValue(offset);
        } else if (data_.weightDtype == 1) {
            union { uint32_t bits; float number; } converted;
            converted.bits = static_cast<uint32_t>(weightsBits_.GetValue(offset)) << 16U;
            value = converted.number;
        } else {
            value = static_cast<float>(weightsHalf_.GetValue(offset));
        }
        return data_.headScale == 1.0F ? value : value * data_.headScale;
    }

    __aicore__ inline void InitCube()
    {
        pipe_->InitBuffer(qL1Buffer_, kHeadCapacity * kDimension * sizeof(bfloat16_t));
        pipe_->InitBuffer(kL1Buffer_, kRegionTile * kDimension * sizeof(bfloat16_t));
        pipe_->InitBuffer(qL0Buffer_, kHeadCapacity * kDimension * sizeof(bfloat16_t));
        pipe_->InitBuffer(kL0Buffer_, kRegionTile * kDimension * sizeof(bfloat16_t));
        pipe_->InitBuffer(cL0Buffer_, kHeadCapacity * kRegionTile * sizeof(float));
        qL1_ = qL1Buffer_.Get<bfloat16_t>();
        kL1_ = kL1Buffer_.Get<bfloat16_t>();
        qL0_ = qL0Buffer_.Get<bfloat16_t>();
        kL0_ = kL0Buffer_.Get<bfloat16_t>();
        cL0_ = cL0Buffer_.Get<float>();
        SetFlag<HardEvent::M_MTE2>(0);
        SetFlag<HardEvent::FIX_M>(0);
    }

    __aicore__ inline bool BatchedVector() const
    {
        return data_.batchTokens != 0 && data_.headsPerGroup == 4;
    }

    __aicore__ inline void CacheBundleWeights(uint32_t group, uint32_t token, uint32_t queries)
    {
        for (uint32_t q = 0; q < queries; ++q) {
            for (uint32_t head = 0; head < 4; ++head) {
                Duplicate(weight_[(q * 4 + head) * kVectorColumns], Weight(group, token + q, head), kVectorColumns);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceBundle(uint32_t queries)
    {
        BinaryRepeatParams repeats{1, 1, 1, 8, 32, 32};
        Mul(acc_, dots_, weight_, uint64_t(kVectorColumns), uint8_t(queries), repeats);
        PipeBarrier<PIPE_V>();
        for (uint32_t head = 1; head < 4; ++head) {
            MulAddDst(acc_, dots_[head * kVectorColumns], weight_[head * kVectorColumns],
                uint64_t(kVectorColumns), uint8_t(queries), repeats);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void InitVector()
    {
        if (BatchedVector()) {
            pipe_->InitBuffer(batchArenaBuffer_, kArenaBytes);
            batchArena_ = batchArenaBuffer_.Get<uint8_t>();
            batchArena_.SetSize(kArenaBytes);
            dots_ = BatchView<float>(kArenaDotsOffset, kArenaDotsElements);
            acc_ = BatchView<float>(kArenaAccOffset, kArenaAccElements);
            partial_ = BatchView<float>(kArenaPartialOffset, kArenaPartialElements);
            weight_ = BatchView<float>(kArenaWeightOffset, kArenaWeightElements);
            window_ = BatchView<float>(kArenaWindowOffset, kArenaWindowElements);
            heapScores_ = BatchView<float>(kArenaHeapScoreOffset, kArenaHeapElements);
            heapIds_ = BatchView<int32_t>(kArenaHeapIdsOffset, kArenaHeapElements);
            return;
        }
        pipe_->InitBuffer(dotBuffer_, kHeadCapacity * kVectorColumns * sizeof(float));
        pipe_->InitBuffer(accBuffer_, (BatchedVector() ? 32U : 1U) * kVectorColumns * sizeof(float));
        pipe_->InitBuffer(partBuffer_, kVectorColumns * sizeof(float));
        pipe_->InitBuffer(weightBuffer_, (BatchedVector() ? 128U : 1U) * kVectorColumns * sizeof(float));
        pipe_->InitBuffer(windowBuffer_, kSelectionWindow * sizeof(float));
        pipe_->InitBuffer(heapScoreBuffer_, kMaxTopK * sizeof(float));
        pipe_->InitBuffer(heapIdBuffer_, kMaxTopK * sizeof(int32_t));
        dots_ = dotBuffer_.Get<float>();
        acc_ = accBuffer_.Get<float>();
        partial_ = partBuffer_.Get<float>();
        weight_ = weightBuffer_.Get<float>();
        window_ = windowBuffer_.Get<float>();
        heapScores_ = heapScoreBuffer_.Get<float>();
        heapIds_ = heapIdBuffer_.Get<int32_t>();
    }

    __aicore__ inline bool ContiguousKey(uint32_t request, uint32_t visible)
    {
        if (data_.summariesPerPage != 1 || data_.cacheStrideRegion != kDimension ||
            data_.cacheStridePage != kDimension) return false;
        if (cachedRequest_ != request) {
            cachedRequest_ = request;
            cachedPrefix_ = 0;
            cachedLinear_ = true;
            cachedFirstPage_ = 0;
        }
        if (!cachedLinear_) return false;
        while (cachedPrefix_ < visible) {
            const int32_t page = table_.GetValue(static_cast<uint64_t>(request) *
                data_.blockTableStrideRequest + cachedPrefix_);
            if (cachedPrefix_ == 0) cachedFirstPage_ = page;
            if (page < 0 || static_cast<uint32_t>(page) >= data_.numPhysicalPages ||
                static_cast<int64_t>(page) != static_cast<int64_t>(cachedFirstPage_) + cachedPrefix_) {
                cachedLinear_ = false;
                return false;
            }
            ++cachedPrefix_;
        }
        return true;
    }

    __aicore__ inline void LoadCubeTile(uint32_t group, uint32_t token,
        uint32_t request, uint32_t regionStart, uint32_t count, uint32_t queries, bool linear, bool loadQuery)
    {
        WaitFlag<HardEvent::M_MTE2>(0);
        const uint32_t m = Align16(data_.headsPerGroup * queries);
        if (loadQuery) {
            InitConstValueParams<uint16_t> clearQ(1, static_cast<uint16_t>(m * kDimension / 16), 0, 0);
            InitConstValue(qL1_.ReinterpretCast<uint16_t>(), clearQ);
        }
        InitConstValueParams<uint16_t> clearK(1, static_cast<uint16_t>(kRegionTile * kDimension / 16), 0, 0);
        InitConstValue(kL1_.ReinterpretCast<uint16_t>(), clearK);
        PipeBarrier<PIPE_MTE2>();
        if (loadQuery) {
            Nd2NzParams qCopy{};
            qCopy.ndNum = 1;
            qCopy.nValue = data_.headsPerGroup;
            qCopy.dValue = kDimension;
            qCopy.srcDValue = kDimension;
            qCopy.dstNzC0Stride = m;
            qCopy.dstNzNStride = 1;
            for (uint32_t q = 0; q < queries; ++q) {
                DataCopy(qL1_[q * data_.headsPerGroup * 16], query_[
                    static_cast<uint64_t>(token + q) * data_.qStrideToken +
                    static_cast<uint64_t>(group) * data_.qStrideGroup], qCopy);
            }
        }
        // Key rows are paged. Adjacent physical rows share one ND-to-NZ copy.
        Nd2NzParams keyCopy{};
        keyCopy.ndNum = 1;
        keyCopy.nValue = 1;
        keyCopy.dValue = kDimension;
        keyCopy.srcDValue = kDimension;
        keyCopy.dstNzC0Stride = kRegionTile;
        keyCopy.dstNzNStride = 1;
        if (linear) {
            keyCopy.nValue = count;
            DataCopy(kL1_, key_[(static_cast<uint64_t>(cachedFirstPage_) + regionStart) * kDimension], keyCopy);
        }
        for (uint32_t r = linear ? count : 0; r < count;) {
            const uint32_t logical = regionStart + r;
            const int32_t page = table_.GetValue(static_cast<uint64_t>(request) *
                data_.blockTableStrideRequest + logical / data_.summariesPerPage);
            if (page < 0 || static_cast<uint32_t>(page) >= data_.numPhysicalPages) {
                ++r;
                continue;
            }
            const uint64_t offset = static_cast<uint64_t>(page) * data_.cacheStridePage +
                (logical % data_.summariesPerPage) * data_.cacheStrideRegion;
            uint32_t run = 1;
            // Coalesce only rows whose physical offsets are exactly adjacent.
            // Invalid/strided/shuffled pages retain their zero/small-copy path.
            if (data_.cacheStrideRegion == kDimension &&
                data_.cacheStridePage == data_.summariesPerPage * kDimension) {
                while (r + run < count) {
                    const uint32_t next = logical + run;
                    const int32_t nextPage = table_.GetValue(static_cast<uint64_t>(request) *
                        data_.blockTableStrideRequest + next / data_.summariesPerPage);
                    if (nextPage < 0 || static_cast<uint32_t>(nextPage) >= data_.numPhysicalPages) break;
                    const uint64_t nextOffset = static_cast<uint64_t>(nextPage) * data_.cacheStridePage +
                        (next % data_.summariesPerPage) * data_.cacheStrideRegion;
                    if (nextOffset != offset + static_cast<uint64_t>(run) * kDimension) break;
                    ++run;
                }
            }
            keyCopy.nValue = run;
            DataCopy(kL1_[r * 16], key_[offset], keyCopy);
            r += run;
        }
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
        if (loadQuery) {
            LoadData2DParams loadQ{};
            loadQ.repeatTimes = kDimension / 16;
            loadQ.srcStride = m / 16;
            loadQ.ifTranspose = false;
            for (uint32_t block = 0; block < m / 16; ++block) {
                LoadData(qL0_[block * kDimension * 16], qL1_[block * 256], loadQ);
            }
        }
        LoadData2DParams loadK{};
        loadK.repeatTimes = kRegionTile * kDimension / 256;
        loadK.srcStride = 1;
        loadK.ifTranspose = false;
        LoadData(kL0_, kL1_, loadK);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
    }

    // Fill the otherwise padded M=16 rows for H=4. A bundle is split at
    // request boundaries; each segment shares its request's paged key.
    __aicore__ inline uint32_t BundleWidth() const
    {
        return data_.headsPerGroup == 4 ? (data_.batchTokens != 0 ? 32U : 4U) : 1U;
    }

    __aicore__ inline uint32_t Segment(uint32_t token, uint32_t limit,
        uint32_t request, uint32_t &maxVisible) const
    {
        uint32_t queries = 1;
        while (queries < limit) {
            uint32_t nextRequest = 0, nextVisible = 0;
            if (!Request(token + queries, nextRequest, nextVisible) || nextRequest != request) break;
            if (nextVisible > maxVisible) maxVisible = nextVisible;
            ++queries;
        }
        return queries;
    }

    __aicore__ inline void CubeRows()
    {
        const uint32_t width = BundleWidth();
        const uint32_t bundles = (data_.batchCount + width - 1) / width;
        const uint32_t rows = data_.groups * bundles;
        for (uint32_t row = core_; row < rows; row += data_.usedCubeCores) {
            const uint32_t group = row / bundles;
            const uint32_t first = row % bundles * width;
            const uint32_t count = Min(width, data_.batchCount - first);
            for (uint32_t q = 0; q < count;) {
                const uint32_t token = data_.batchBegin + first + q;
                uint32_t request = 0, visible = 0;
                if (!Request(token, request, visible)) { ++q; continue; }
                const uint32_t queries = Segment(token, count - q, request, visible);
                // IDs-only selects every visible region in this bundle.
                // Match producer/consumer skipping; keep final flag credits intact.
                if (data_.batchTokens != 0 && visible <= (data_.selectionConfig & 0xFFFFU)) {
                    q += queries;
                    continue;
                }
                const uint32_t m = Align16(data_.headsPerGroup * queries);
                const bool linear = ContiguousKey(request, visible);
                for (uint32_t r = 0; r < visible; r += kRegionTile) {
                    LoadCubeTile(group, token, request, r, Min(kRegionTile, visible - r), queries, linear, !BatchedVector() || r == 0);
                    WaitFlag<HardEvent::FIX_M>(0);
                    MmadParams mm{};
                    mm.m = m;
                    mm.n = kRegionTile;
                    mm.k = kDimension;
                    mm.cmatrixInitVal = true;
                    Mmad(cL0_, qL0_, kL0_, mm);
                    SetFlag<HardEvent::M_MTE2>(0);
                    SetFlag<HardEvent::M_FIX>(0);
                    WaitFlag<HardEvent::M_FIX>(0);
                    CrossCoreWaitFlag(kFree);
                    FixpipeParamsV220 fix{};
                    fix.mSize = m;
                    fix.nSize = kRegionTile;
                    fix.srcStride = m;
                    fix.dstStride = kRegionTile;
                    fix.quantPre = QuantMode_t::NoQuant;
                    fix.reluEn = true;
                    Fixpipe<float, float, CFG_ROW_MAJOR>(bridge_, cL0_, fix);
                    CrossCoreSetFlag<2, PIPE_FIX>(kReady);
                    SetFlag<HardEvent::FIX_M>(0);
                }
                q += queries;
            }
        }
        CrossCoreWaitFlag(kFree);
        WaitFlag<HardEvent::M_MTE2>(0);
        WaitFlag<HardEvent::FIX_M>(0);
    }

    __aicore__ inline void Reduce(uint32_t group, uint32_t token, uint32_t dotHead)
    {
        for (uint32_t h = 0; h < data_.headsPerGroup; h += 4) {
            auto partial = h == 0 ? acc_ : partial_;
            Muls(partial, dots_[(dotHead + h) * kVectorColumns], Weight(group, token, h), kVectorColumns);
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 1; j < 4; ++j) {
                Duplicate(weight_, Weight(group, token, h + j), kVectorColumns);
                PipeBarrier<PIPE_V>();
                MulAddDst(partial, dots_[(dotHead + h + j) * kVectorColumns], weight_, kVectorColumns);
                PipeBarrier<PIPE_V>();
            }
            if (h != 0) {
                Add(acc_, acc_, partial_, kVectorColumns);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void VectorRows()
    {
        CrossCoreSetFlag<2, PIPE_MTE2>(kFree);
        const uint32_t width = BundleWidth();
        const uint32_t bundles = (data_.batchCount + width - 1) / width;
        const uint32_t rows = data_.groups * bundles;
        for (uint32_t row = core_; row < rows; row += data_.usedCubeCores) {
            const uint32_t group = row / bundles;
            const uint32_t first = row % bundles * width;
            const uint32_t count = Min(width, data_.batchCount - first);
            for (uint32_t q = 0; q < count;) {
                const uint32_t token = data_.batchBegin + first + q;
                uint32_t request = 0, visible = 0;
                if (!Request(token, request, visible)) { ++q; continue; }
                const uint32_t firstVisible = visible;
                const uint32_t queries = Segment(token, count - q, request, visible);
                // IDs-only selects every visible region in this bundle.
                // Match producer/consumer skipping; keep final flag credits intact.
                if (data_.batchTokens != 0 && visible <= (data_.selectionConfig & 0xFFFFU)) {
                    q += queries;
                    continue;
                }
                if (BatchedVector()) CacheBundleWeights(group, token, queries);
                for (uint32_t r = 0; r < visible; r += kRegionTile) {
                    CrossCoreWaitFlag(kReady);
                    DataCopyParams copy{};
                    copy.blockCount = data_.headsPerGroup * queries;
                    copy.blockLen = kVectorColumns * sizeof(float) / 32;
                    copy.srcStride = kVectorColumns * sizeof(float) / 32;
                    DataCopy(dots_, bridge_[sub_ * kVectorColumns], copy);
                    SetFlag<HardEvent::MTE2_V>(0);
                    WaitFlag<HardEvent::MTE2_V>(0);
                    CrossCoreSetFlag<2, PIPE_MTE2>(kFree);
                    if (BatchedVector()) ReduceBundle(queries);
                    const uint32_t batchStart = r + sub_ * kVectorColumns;
                    if (BatchedVector() && batchStart < firstVisible &&
                        firstVisible - batchStart >= kVectorColumns && data_.outputStrideToken <= UINT32_MAX / 4U) {
                        SetFlag<HardEvent::V_MTE3>(0);
                        WaitFlag<HardEvent::V_MTE3>(0);
                        DataCopyExtParams store{static_cast<uint16_t>(queries), kVectorColumns * 4U, 0,
                            static_cast<uint32_t>(data_.outputStrideToken * 4U - kVectorColumns * 4U), 0};
                        DataCopyPad(scores_[ScoreBase(group, token) + batchStart], acc_, store);
                        SetFlag<HardEvent::MTE3_V>(0);
                        WaitFlag<HardEvent::MTE3_V>(0);
                    } else {
                    for (uint32_t item = 0; item < queries; ++item) {
                        uint32_t itemRequest = 0, itemVisible = 0;
                        (void)Request(token + item, itemRequest, itemVisible);
                        if (!BatchedVector()) Reduce(group, token + item, item * data_.headsPerGroup);
                        const uint32_t start = r + sub_ * kVectorColumns;
                        if (start < itemVisible) {
                            const uint32_t length = Min(kVectorColumns, itemVisible - start);
                            SetFlag<HardEvent::V_MTE3>(0);
                            WaitFlag<HardEvent::V_MTE3>(0);
                            DataCopyExtParams store{1, length * 4U, 0, 0, 0};
                            DataCopyPad(scores_[ScoreBase(group, token + item) + start], acc_[BatchedVector() ? item * kVectorColumns : 0], store);
                            SetFlag<HardEvent::MTE3_V>(0);
                            WaitFlag<HardEvent::MTE3_V>(0);
                        }
                    }
                    }
                    SetFlag<HardEvent::V_MTE2>(0);
                    WaitFlag<HardEvent::V_MTE2>(0);
                }
                q += queries;
            }
        }
    }

    // Better means a higher score or the smaller logical ID at exact ties.
    __aicore__ inline bool Better(float a, int32_t ai, float b, int32_t bi) const
    {
        return a > b || (a == b && ai < bi);
    }

    __aicore__ inline bool HeapBetter(float a, int32_t ai, float b, int32_t bi) const
    {
        return reverseHeap_ ? Better(b, bi, a, ai) : Better(a, ai, b, bi);
    }

    __aicore__ inline void Swap(uint32_t a, uint32_t b)
    {
        const float score = heapScores_.GetValue(a);
        const int32_t id = heapIds_.GetValue(a);
        heapScores_.SetValue(a, heapScores_.GetValue(b));
        heapIds_.SetValue(a, heapIds_.GetValue(b));
        heapScores_.SetValue(b, score);
        heapIds_.SetValue(b, id);
    }

    __aicore__ inline void Push(float score, int32_t id, uint32_t &size)
    {
        uint32_t at = size++;
        heapScores_.SetValue(at, score);
        heapIds_.SetValue(at, id);
        while (at != 0) {
            const uint32_t parent = (at - 1) / 2;
            if (!HeapBetter(heapScores_.GetValue(parent), heapIds_.GetValue(parent), score, id)) break;
            Swap(parent, at);
            at = parent;
        }
    }

    __aicore__ inline void ReplaceRoot(float score, int32_t id, uint32_t size)
    {
        heapScores_.SetValue(0, score);
        heapIds_.SetValue(0, id);
        uint32_t at = 0;
        while (at * 2 + 1 < size) {
            uint32_t child = at * 2 + 1;
            if (child + 1 < size && HeapBetter(heapScores_.GetValue(child), heapIds_.GetValue(child),
                heapScores_.GetValue(child + 1), heapIds_.GetValue(child + 1))) ++child;
            if (!HeapBetter(heapScores_.GetValue(at), heapIds_.GetValue(at),
                heapScores_.GetValue(child), heapIds_.GetValue(child))) break;
            Swap(at, child);
            at = child;
        }
    }

    __aicore__ inline void SiftDownIds(uint32_t at, uint32_t size)
    {
        const int32_t id = heapIds_.GetValue(at);
        while (at * 2 + 1 < size) {
            uint32_t child = at * 2 + 1;
            if (child + 1 < size && heapIds_.GetValue(child + 1) > heapIds_.GetValue(child)) ++child;
            const int32_t childId = heapIds_.GetValue(child);
            if (id >= childId) break;
            heapIds_.SetValue(at, childId);
            at = child;
        }
        heapIds_.SetValue(at, id);
    }

    __aicore__ inline void SortIds(uint32_t size)
    {
        // Sort only integer IDs. A max heap yields ascending output with
        // bounded O(K log K) work, including large equal-score tie sets.
        for (uint32_t root = size / 2; root != 0; --root) SiftDownIds(root - 1, size);
        for (uint32_t end = size; end > 1; --end) {
            const int32_t largest = heapIds_.GetValue(0);
            heapIds_.SetValue(0, heapIds_.GetValue(end - 1));
            heapIds_.SetValue(end - 1, largest);
            SiftDownIds(0, end - 1);
        }
    }

    __aicore__ inline void OrderSelectedIds(uint32_t size, uint32_t visible, bool complement)
    {
        if (visible > kSelectionWindow) {
            SortIds(size);
            return;
        }
        // Scoring has completed and Process synchronized all cores. Reuse
        // the first 16 KiB of dots_ for integer membership flags.
        auto flags = dots_.ReinterpretCast<int32_t>();
        const uint32_t aligned = (visible + 7U) / 8U * 8U;
        Duplicate(flags, int32_t(0), aligned);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        for (uint32_t i = 0; i < size; ++i) {
            flags.SetValue(static_cast<uint32_t>(heapIds_.GetValue(i)), 1);
        }
        uint32_t out = 0;
        for (uint32_t id = 0; id < visible; ++id) {
            if ((flags.GetValue(id) != 0) != complement) heapIds_.SetValue(out++, static_cast<int32_t>(id));
        }
    }

    __aicore__ inline uint32_t SelectByThreshold(uint32_t group, uint32_t token,
        uint32_t visible, uint32_t k)
    {
        // Selection runs after the global scoring barrier. Reuse 24 KiB of
        // the 32 KiB dot buffer; no additional UB or GM allocation is needed.
        const uint32_t aligned = (visible + 63U) / 64U * 64U;
        auto packed = dots_[0];
        auto scratch = dots_[2048];
        auto ordered = dots_[4096];
        auto index = dots_.ReinterpretCast<uint32_t>()[5120];
        DataCopyExtParams copy{1, visible * 4U, 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(window_, scores_[ScoreBase(group, token)], copy, pad);
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
        union { uint32_t bits; float value; } negativeInfinity;
        negativeInfinity.bits = 0xFF800000U;
        for (uint32_t i = visible; i < aligned; ++i) window_.SetValue(i, negativeInfinity.value);
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        Duplicate(index, uint32_t(0), aligned);
        PipeBarrier<PIPE_V>();
        Sort<float, true>(packed, window_, index, scratch, aligned / 32U);
        PipeBarrier<PIPE_V>();
        Extract(ordered, index, packed, aligned / 32U);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        const float threshold = ordered.GetValue(k - 1);
        // Strict boundary proves membership even when selected scores contain ties.
        // Exact cutoff ties retain the original stable logical-ID tie path.
        if (data_.batchTokens != 0 && data_.headsPerGroup == 4 && visible > k && threshold > ordered.GetValue(k)) {
            auto directIds = index.ReinterpretCast<int32_t>();
            auto directMask = dots_.ReinterpretCast<uint8_t>()[0];
            CreateVecIndex(directIds, int32_t(0), aligned);
            CompareScalar(directMask, window_, threshold, CMPMODE::GE, aligned);
            PipeBarrier<PIPE_V>();
            uint64_t directCount = 0;
            GatherMask(heapIds_, directIds, directMask.ReinterpretCast<uint32_t>(),
                true, visible, {1, 1, 8, 1}, directCount);
            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);
            return static_cast<uint32_t>(directCount);
        }
        // Find the first tie by binary search; the hardware sort's tie order
        // is irrelevant because only the threshold value is used.
        uint32_t lo = 0, hi = k;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (ordered.GetValue(mid) > threshold) lo = mid + 1;
            else hi = mid;
        }
        const uint32_t ties = k - lo;
        auto logicalIds = index.ReinterpretCast<int32_t>();
        auto tieIds = scratch.ReinterpretCast<int32_t>();
        auto greater = dots_.ReinterpretCast<uint8_t>()[0];
        auto equal = dots_.ReinterpretCast<uint8_t>()[256];
        auto cutoffMask = dots_.ReinterpretCast<uint8_t>()[512];
        // packed/scratch/ordered storage is dead after reading the threshold.
        CreateVecIndex(logicalIds, int32_t(0), aligned);
        CreateVecIndex(ordered, 0.0F, aligned);
        CompareScalar(greater, window_, threshold, CMPMODE::GT, aligned);
        CompareScalar(equal, window_, threshold, CMPMODE::EQ, aligned);
        PipeBarrier<PIPE_V>();
        uint64_t tieCount = 0;
        GatherMask(tieIds, logicalIds, equal.ReinterpretCast<uint32_t>(),
            true, visible, {1, 1, 8, 1}, tieCount);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        const int32_t cutoff = tieIds.GetValue(ties - 1);
        CompareScalar(cutoffMask, ordered, static_cast<float>(cutoff), CMPMODE::LE, aligned);
        PipeBarrier<PIPE_V>();
        auto equalWords = equal.ReinterpretCast<uint16_t>();
        auto greaterWords = greater.ReinterpretCast<uint16_t>();
        And(equalWords, equalWords, cutoffMask.ReinterpretCast<uint16_t>(), aligned / 16U);
        PipeBarrier<PIPE_V>();
        Or(greaterWords, greaterWords, equalWords, aligned / 16U);
        PipeBarrier<PIPE_V>();
        uint64_t selected = 0;
        GatherMask(heapIds_, logicalIds, greater.ReinterpretCast<uint32_t>(),
            true, visible, {1, 1, 8, 1}, selected);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        return static_cast<uint32_t>(selected);
    }

    __aicore__ inline void SelectRows()
    {
        const uint32_t k = data_.selectionConfig & 0xFFFFU;
        const uint32_t rows = data_.groups * data_.batchCount;
        const uint32_t vectorCore = GetBlockIdx();
        const uint32_t stride = data_.usedCubeCores * 2U;
        for (uint32_t row = vectorCore; row < rows;) {
            if (BatchedVector() && k == 512U && TrySelectTwelveRows(row)) {
                row += kLargeRows * stride;
                continue;
            }
            if (BatchedVector() && k == 512U && TrySelectFourRows(row)) {
                row += 4U * stride;
                continue;
            }
            const uint32_t group = row / data_.batchCount;
            const uint32_t token = data_.batchBegin + row % data_.batchCount;
            uint32_t request = 0, visible = 0;
            (void)Request(token, request, visible);
            uint32_t size = 0;
            if (visible <= k) {
                const uint32_t alignedVisible = (visible + 7U) / 8U * 8U;
                if (alignedVisible != 0) CreateVecIndex(heapIds_, int32_t(0), alignedVisible);
                const uint32_t alignedK = (k + 7U) / 8U * 8U;
                if (alignedK > alignedVisible)
                    Duplicate(heapIds_[alignedVisible], int32_t(-1), alignedK - alignedVisible);
                SetFlag<HardEvent::V_S>(0);
                WaitFlag<HardEvent::V_S>(0);
                for (uint32_t i = visible; i < Min(alignedVisible, k); ++i) heapIds_.SetValue(i, -1);
                size = k;
            } else if (visible <= 1024) {
                size = SelectByThreshold(group, token, visible, k);
            } else {
                // A heap of the rejected IDs is smaller when K is near R.
                // Comparator reversal preserves the original score/ID total order.
                const bool complement = visible <= kSelectionWindow && visible - k < k;
                const uint32_t limit = complement ? visible - k : k;
                reverseHeap_ = complement;
                const uint64_t base = ScoreBase(group, token);
                for (uint32_t begin = 0; begin < visible; begin += kSelectionWindow) {
                    const uint32_t count = Min(kSelectionWindow, visible - begin);
                    DataCopyExtParams copy{1, count * 4U, 0, 0, 0};
                    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
                    DataCopyPad(window_, scores_[base + begin], copy, pad);
                    SetFlag<HardEvent::MTE2_S>(0);
                    WaitFlag<HardEvent::MTE2_S>(0);
                    for (uint32_t i = 0; i < count; ++i) {
                        const float score = window_.GetValue(i);
                        const int32_t id = static_cast<int32_t>(begin + i);
                        if (size < limit) {
                            Push(score, id, size);
                        } else if (HeapBetter(score, id, heapScores_.GetValue(0), heapIds_.GetValue(0))) {
                            ReplaceRoot(score, id, size);
                        }
                    }
                }
                OrderSelectedIds(size, visible, complement);
                if (complement) size = k;
            }
            for (uint32_t i = size; i < k; ++i) heapIds_.SetValue(i, -1);
            SetFlag<HardEvent::S_MTE3>(0);
            WaitFlag<HardEvent::S_MTE3>(0);
            const uint64_t out = (static_cast<uint64_t>(group) * data_.totalTokens + token) * k;
            DataCopyExtParams copy{1, k * 4U, 0, 0, 0};
            DataCopyPad(ids_[out], heapIds_, copy);
            SetFlag<HardEvent::MTE3_S>(0);
            WaitFlag<HardEvent::MTE3_S>(0);
            row += stride;
        }
    }

    template <typename T>
    __aicore__ inline LocalTensor<T> BatchView(uint32_t bytes, uint32_t count)
    {
        auto tensor = batchArena_.ReinterpretCast<T>()[bytes / sizeof(T)];
        tensor.SetSize(count);
        return tensor;
    }

    __aicore__ inline void SortFourRows()
    {
        auto input = BatchView<float>(kBatchInputOffset, kBatchInputElements);
        auto a = BatchView<float>(kBatchPackedAOffset, kBatchPackedElements);
        auto b = BatchView<float>(kBatchPackedBOffset, kBatchPackedElements);
        auto index = BatchView<uint32_t>(kBatchIndexOffset, kBatchInputElements);
        Duplicate(index, uint32_t(0), kBatchInputElements);
        PipeBarrier<PIPE_V>();
        Sort32(a, input, index, 80);
        PipeBarrier<PIPE_V>();
        uint16_t length32[4] = {32, 32, 32, 32};
        uint32_t sorted[4] = {0, 0, 0, 0};
        MrgSort<float, false>(b, MrgSortSrcList<float>(a[0], a[64], a[128], a[192]),
            length32, sorted, 15, 20);
        PipeBarrier<PIPE_V>();
        uint16_t length128[4] = {128, 128, 128, 128};
        for (uint32_t j = 0; j < kBatchRows; ++j) {
            const uint32_t p = kBatchPackedRowWords * j;
            MrgSort<float, false>(a[p], MrgSortSrcList<float>(b[p], b[p + 256], b[p + 512], b[p + 768]),
                length128, sorted, 15, 1);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < kBatchRows; ++j) {
            const uint32_t p = kBatchPackedRowWords * j + 1024U;
            DataCopy(a[p], b[p], 256);
        }
        PipeBarrier<PIPE_V>();
        uint16_t lengthFinal[4] = {512, 128, 0, 0};
        for (uint32_t j = 0; j < kBatchRows; ++j) {
            const uint32_t p = kBatchPackedRowWords * j;
            MrgSort<float, false>(b[p], MrgSortSrcList<float>(a[p], a[p + 1024], a[p], a[p]),
                lengthFinal, sorted, 3, 1);
        }
        PipeBarrier<PIPE_V>();
        // A ablation: threshold reads packed score fields directly; no Extract.
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
    }

    __aicore__ inline bool SelectSortedBatchRow(uint32_t j, uint32_t visible)
    {
        auto input = BatchView<float>(kBatchInputOffset + j * kBatchRowBytes, kBatchColumns);
        auto ordered = BatchView<float>(kBatchPackedBOffset + j * kBatchPackedRowWords * sizeof(float), kBatchPackedRowWords);
        auto output = BatchView<int32_t>(kBatchOutputOffset + j * kBatchOutputRowBytes, kMaxTopK);
        auto logical = BatchView<int32_t>(kBatchLogicalOffset, kBatchColumns);
        auto wide = BatchView<int32_t>(kBatchWideOffset, kBatchColumns);
        auto floatIds = BatchView<float>(kBatchFloatIdsOffset, kBatchColumns);
        auto greater = BatchView<uint8_t>(kBatchGreaterOffset, 256);
        auto equal = BatchView<uint8_t>(kBatchEqualOffset, 256);
        auto cutoffMask = BatchView<uint8_t>(kBatchCutoffOffset, 256);
        const float threshold = ordered.GetValue(2U * 511U);
        const float next = ordered.GetValue(2U * 512U);
        CreateVecIndex(logical, int32_t(0), 640);
        uint64_t count = 0;
        if (threshold > next) {
            CompareScalar(greater, input, threshold, CMPMODE::GE, 640);
        } else {
            uint32_t lo = 0, hi = 512;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) / 2;
                if (ordered.GetValue(2U * mid) > threshold) lo = mid + 1;
                else hi = mid;
            }
            const uint32_t ties = 512U - lo;
            CompareScalar(greater, input, threshold, CMPMODE::GT, 640);
            CompareScalar(equal, input, threshold, CMPMODE::EQ, 640);
            PipeBarrier<PIPE_V>();
            uint64_t tieCount = 0;
            GatherMask(wide, logical, equal.ReinterpretCast<uint32_t>(), true, visible, {1, 1, 8, 1}, tieCount);
            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);
            if (ties == 0U || tieCount < ties) return false;
            const int32_t cutoff = wide.GetValue(ties - 1U);
            CreateVecIndex(floatIds, 0.0F, 640);
            CompareScalar(cutoffMask, floatIds, static_cast<float>(cutoff), CMPMODE::LE, 640);
            PipeBarrier<PIPE_V>();
            And(equal.ReinterpretCast<uint16_t>(), equal.ReinterpretCast<uint16_t>(),
                cutoffMask.ReinterpretCast<uint16_t>(), 40);
            PipeBarrier<PIPE_V>();
            Or(greater.ReinterpretCast<uint16_t>(), greater.ReinterpretCast<uint16_t>(),
                equal.ReinterpretCast<uint16_t>(), 40);
        }
        PipeBarrier<PIPE_V>();
        GatherMask(wide, logical, greater.ReinterpretCast<uint32_t>(), true, visible, {1, 1, 8, 1}, count);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        if (count != 512U) return false;
        DataCopy(output, wide, 512);
        PipeBarrier<PIPE_V>();
        return true;
    }

    __aicore__ inline bool TrySelectFourRows(uint32_t row)
    {
        const uint32_t stride = data_.usedCubeCores * 2U;
        const uint32_t rows = data_.groups * data_.batchCount;
        if (row + 3U * stride >= rows) return false;
        const uint32_t group = row / data_.batchCount;
        uint32_t tokens[kBatchRows], visible[kBatchRows], firstRequest = 0;
        for (uint32_t j = 0; j < kBatchRows; ++j) {
            const uint32_t current = row + j * stride;
            if (current / data_.batchCount != group) return false;
            tokens[j] = data_.batchBegin + current % data_.batchCount;
            uint32_t request = 0;
            if (!Request(tokens[j], request, visible[j]) || visible[j] < 513U || visible[j] > 639U) return false;
            if (j == 0U) firstRequest = request;
            else if (request != firstRequest) return false;
        }
        auto input = BatchView<float>(kBatchInputOffset, kBatchInputElements);
        union { uint32_t bits; float value; } negativeInfinity;
        negativeInfinity.bits = 0xFF800000U;
        Duplicate(input, negativeInfinity.value, kBatchInputElements);
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        for (uint32_t j = 0; j < kBatchRows; ++j) {
            DataCopyExtParams copy{1, visible[j] * 4U, 0, 0, 0};
            DataCopyPadExtParams<float> pad{true, 0,
                static_cast<uint8_t>((8U - visible[j] % 8U) % 8U), negativeInfinity.value};
            DataCopyPad(input[j * kBatchColumns], scores_[ScoreBase(group, tokens[j])], copy, pad);
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        SortFourRows();
        // No GM output until every row passes; failed batches may safely
        // discard all arena views and retry the original per-row selector.
        for (uint32_t j = 0; j < kBatchRows; ++j) {
            if (!SelectSortedBatchRow(j, visible[j])) return false;
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        for (uint32_t j = 0; j < kBatchRows; ++j) {
            auto output = BatchView<int32_t>(kBatchOutputOffset + j * kBatchOutputRowBytes, kMaxTopK);
            const uint64_t out = (static_cast<uint64_t>(group) * data_.totalTokens + tokens[j]) * 512U;
            DataCopyExtParams copy{1, 2048, 0, 0, 0};
            DataCopyPad(ids_[out], output, copy);
        }
        SetFlag<HardEvent::MTE3_S>(0);
        WaitFlag<HardEvent::MTE3_S>(0);
        return true;
    }

    TBuf<TPosition::VECCALC> batchArenaBuffer_;
    LocalTensor<uint8_t> batchArena_;
    // A12 aliases: input and packed B remain live through membership.
    static constexpr uint32_t kLargeRows = 12, kLargeColumns = 640;
    static constexpr uint32_t kLargeInputElements = 7680, kLargePackedElements = 15360;
    static constexpr uint32_t kLargePackedRowWords = 1280, kLargeRowBytes = 2560;
    static constexpr uint32_t kLargeOutputRowBytes = 2048;
    static constexpr uint32_t kLargeInputOffset = 0, kLargePackedAOffset = 30720;
    static constexpr uint32_t kLargePackedBOffset = 92160, kLargeOutputOffset = 30720;
    static constexpr uint32_t kLargeLogicalOffset = 55296, kLargeWideOffset = 57856;
    static constexpr uint32_t kLargeFloatIdsOffset = 60416, kLargeGreaterOffset = 62976;
    static constexpr uint32_t kLargeEqualOffset = 63232, kLargeCutoffOffset = 63488;

    __aicore__ inline void SortTwelveRows()
    {
        auto input = BatchView<float>(kLargeInputOffset, kLargeInputElements);
        auto a = BatchView<float>(kLargePackedAOffset, kLargePackedElements);
        auto b = BatchView<float>(kLargePackedBOffset, kLargePackedElements);
        auto index = BatchView<uint32_t>(kLargePackedBOffset, kLargeInputElements);
        Duplicate(index, uint32_t(0), kLargeInputElements);
        PipeBarrier<PIPE_V>();
        Sort32(a, input, index, 240);
        PipeBarrier<PIPE_V>(); // Index storage is dead before B merge output.
        uint16_t length32[4] = {32, 32, 32, 32};
        uint32_t sorted[4] = {0, 0, 0, 0};
        MrgSort<float, false>(b, MrgSortSrcList<float>(a[0], a[64], a[128], a[192]),
            length32, sorted, 15, 60);
        PipeBarrier<PIPE_V>();
        uint16_t length128[4] = {128, 128, 128, 128};
        for (uint32_t j = 0; j < kLargeRows; ++j) {
            const uint32_t p = kLargePackedRowWords * j;
            MrgSort<float, false>(a[p], MrgSortSrcList<float>(b[p], b[p + 256], b[p + 512], b[p + 768]),
                length128, sorted, 15, 1);
        }
        PipeBarrier<PIPE_V>();
        DataCopyParams tail{};
        tail.blockCount = kLargeRows;
        tail.blockLen = 32;
        tail.srcStride = 128;
        tail.dstStride = 128;
        DataCopy(a[1024], b[1024], tail);
        PipeBarrier<PIPE_V>();
        uint16_t lengthFinal[4] = {512, 128, 0, 0};
        for (uint32_t j = 0; j < kLargeRows; ++j) {
            const uint32_t p = kLargePackedRowWords * j;
            MrgSort<float, false>(b[p], MrgSortSrcList<float>(a[p], a[p + 1024], a[p], a[p]),
                lengthFinal, sorted, 3, 1);
        }
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        // Packed A is now dead and may hold output IDs and membership scratch.
    }

    __aicore__ inline bool SelectSortedLargeRow(uint32_t j, uint32_t visible)
    {
        auto input = BatchView<float>(kLargeInputOffset + j * kLargeRowBytes, kLargeColumns);
        auto ordered = BatchView<float>(kLargePackedBOffset + j * kLargePackedRowWords * sizeof(float), kLargePackedRowWords);
        auto output = BatchView<int32_t>(kLargeOutputOffset + j * kLargeOutputRowBytes, kMaxTopK);
        auto logical = BatchView<int32_t>(kLargeLogicalOffset, kLargeColumns);
        auto wide = BatchView<int32_t>(kLargeWideOffset, kLargeColumns);
        auto floatIds = BatchView<float>(kLargeFloatIdsOffset, kLargeColumns);
        auto greater = BatchView<uint8_t>(kLargeGreaterOffset, 256);
        auto equal = BatchView<uint8_t>(kLargeEqualOffset, 256);
        auto cutoffMask = BatchView<uint8_t>(kLargeCutoffOffset, 256);
        const float threshold = ordered.GetValue(2U * 511U);
        const float next = ordered.GetValue(2U * 512U);
        // logical IDs are built once per A12 batch and remain read-only.
        uint64_t count = 0;
        if (threshold > next) {
            CompareScalar(greater, input, threshold, CMPMODE::GE, 640);
        } else {
            uint32_t lo = 0, hi = 512;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) / 2;
                if (ordered.GetValue(2U * mid) > threshold) lo = mid + 1;
                else hi = mid;
            }
            const uint32_t ties = 512U - lo;
            CompareScalar(greater, input, threshold, CMPMODE::GT, 640);
            CompareScalar(equal, input, threshold, CMPMODE::EQ, 640);
            PipeBarrier<PIPE_V>();
            uint64_t tieCount = 0;
            GatherMask(wide, logical, equal.ReinterpretCast<uint32_t>(), true, visible, {1, 1, 8, 1}, tieCount);
            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);
            if (ties == 0U || tieCount < ties) return false;
            const int32_t cutoff = wide.GetValue(ties - 1U);
            CreateVecIndex(floatIds, 0.0F, 640);
            CompareScalar(cutoffMask, floatIds, static_cast<float>(cutoff), CMPMODE::LE, 640);
            PipeBarrier<PIPE_V>();
            And(equal.ReinterpretCast<uint16_t>(), equal.ReinterpretCast<uint16_t>(),
                cutoffMask.ReinterpretCast<uint16_t>(), 40);
            PipeBarrier<PIPE_V>();
            Or(greater.ReinterpretCast<uint16_t>(), greater.ReinterpretCast<uint16_t>(),
                equal.ReinterpretCast<uint16_t>(), 40);
        }
        PipeBarrier<PIPE_V>();
        GatherMask(wide, logical, greater.ReinterpretCast<uint32_t>(), true, visible, {1, 1, 8, 1}, count);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        if (count != 512U) return false;
        DataCopy(output, wide, 512);
        PipeBarrier<PIPE_V>();
        return true;
    }


    __aicore__ inline bool TrySelectTwelveRows(uint32_t row)
    {
        const uint32_t stride = data_.usedCubeCores * 2U;
        const uint32_t rows = data_.groups * data_.batchCount;
        if (row >= rows || rows - 1U - row < (kLargeRows - 1U) * stride) return false;
        const uint32_t group = row / data_.batchCount;
        uint32_t tokens[kLargeRows], visible[kLargeRows], firstRequest = 0;
        for (uint32_t j = 0; j < kLargeRows; ++j) {
            const uint32_t current = row + j * stride;
            if (current / data_.batchCount != group) return false;
            tokens[j] = data_.batchBegin + current % data_.batchCount;
            uint32_t request = 0;
            if (!Request(tokens[j], request, visible[j]) || visible[j] < 513U || visible[j] > 639U) return false;
            if (j == 0U) firstRequest = request;
            else if (request != firstRequest) return false;
        }
        auto input = BatchView<float>(kLargeInputOffset, kLargeInputElements);
        union { uint32_t bits; float value; } negativeInfinity;
        negativeInfinity.bits = 0xFF800000U;
        Duplicate(input, negativeInfinity.value, kLargeInputElements);
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        for (uint32_t j = 0; j < kLargeRows; ++j) {
            DataCopyExtParams copy{1, visible[j] * 4U, 0, 0, 0};
            DataCopyPadExtParams<float> pad{true, 0,
                static_cast<uint8_t>((8U - visible[j] % 8U) % 8U), negativeInfinity.value};
            DataCopyPad(input[j * kLargeColumns], scores_[ScoreBase(group, tokens[j])], copy, pad);
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        SortTwelveRows();
        auto logical = BatchView<int32_t>(kLargeLogicalOffset, kLargeColumns);
        CreateVecIndex(logical, int32_t(0), kLargeColumns);
        PipeBarrier<PIPE_V>();
        // No GM output until every row passes; failed batches may safely
        // discard all arena views and retry the original per-row selector.
        for (uint32_t j = 0; j < kLargeRows; ++j) {
            if (!SelectSortedLargeRow(j, visible[j])) return false;
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        auto output = BatchView<int32_t>(kLargeOutputOffset, kLargeRows * kMaxTopK);
        if (stride - 1U <= UINT32_MAX / kLargeOutputRowBytes) {
            const uint64_t out = (static_cast<uint64_t>(group) * data_.totalTokens + tokens[0]) * kMaxTopK;
            DataCopyExtParams copy{static_cast<uint16_t>(kLargeRows), kLargeOutputRowBytes, 0,
                (stride - 1U) * kLargeOutputRowBytes, 0};
            DataCopyPad(ids_[out], output, copy);
        } else {
            for (uint32_t j = 0; j < kLargeRows; ++j) {
                const uint64_t out = (static_cast<uint64_t>(group) * data_.totalTokens + tokens[j]) * kMaxTopK;
                DataCopyExtParams copy{1, kLargeOutputRowBytes, 0, 0, 0};
                DataCopyPad(ids_[out], output[j * kMaxTopK], copy);
            }
        }
        SetFlag<HardEvent::MTE3_S>(0);
        WaitFlag<HardEvent::MTE3_S>(0);
        return true;
    }


    PagedTilingData data_{};
    TPipe *pipe_ = nullptr;
    uint32_t core_ = 0, sub_ = 0;
    bool singleRequestCached_ = false, singleRegionPower2_ = false;
    uint32_t singleRegionShift_ = 0;
    int32_t singleBegin_ = 0, singleEnd_ = 0, singleLength_ = 0;
    bool reverseHeap_ = false;
    uint32_t cachedRequest_ = 0xFFFFFFFFU, cachedPrefix_ = 0;
    int32_t cachedFirstPage_ = 0;
    bool cachedLinear_ = true;
    GlobalTensor<bfloat16_t> query_, key_;
    GlobalTensor<float> weightsFloat_, scores_, bridge_;
    GlobalTensor<uint16_t> weightsBits_;
    GlobalTensor<half> weightsHalf_;
    GlobalTensor<int32_t> lengths_, table_, cu_, ids_;
    TBuf<TPosition::A1> qL1Buffer_;
    TBuf<TPosition::B1> kL1Buffer_;
    TBuf<TPosition::A2> qL0Buffer_;
    TBuf<TPosition::B2> kL0Buffer_;
    TBuf<TPosition::CO1> cL0Buffer_;
    TBuf<TPosition::VECCALC> dotBuffer_, accBuffer_, partBuffer_, weightBuffer_;
    TBuf<TPosition::VECCALC> windowBuffer_, heapScoreBuffer_, heapIdBuffer_;
    LocalTensor<bfloat16_t> qL1_, kL1_, qL0_, kL0_;
    LocalTensor<float> cL0_, dots_, acc_, partial_, weight_, window_, heapScores_;
    LocalTensor<int32_t> heapIds_;
};
} // namespace IndexerRegionScoreA3
#endif
