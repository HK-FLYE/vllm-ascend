# Paged indexer region score

This operator adapts `experimental/attention/indexer_region_score` from
ops-transformer for the measured model cases on Ascend950. It accepts native
FP16, BF16, or FP8 E4M3FN query and key summaries, with FP32 accumulation.
Query and key summaries must have the same dtype; weights can be FP16, BF16,
or FP32. `D=256`; query groups `G` can be 1 or a positive multiple of 4,
and heads per group `H` can be any multiple of 4 from 4 through 128.
Key summaries are shared by all query groups.

```python
from vllm_ascend.ops.indexer_region_score import indexer_region_score

scores, region_ids = indexer_region_score(
    query,                    # [T, G, H, 256]
    weights,                  # [T, G, H]
    key_summary,             # [P, S, 1, 256]
    actual_kv_lengths,        # int32 [N], includes the current query tokens
    block_table,              # int32 [N, C], logical page -> physical page
    cu_seq_lens,              # int32 [N + 1], packed query offsets
    max_region_len=8192,
    top_k=512,
    region_size=8,
    cache_mode=1,
    mask_mode=3,
    sort_mode=1,
)
```

`scores` is FP32 `[G, T, R_out]`, with `R_out=max_region_len` by default;
a preallocated output may have larger capacity. `region_ids` is INT32
`[G, T, top_k]` and contains logical region IDs in ascending order. IDs are
selected by descending score, with the smaller logical ID winning ties.
Unused slots contain `-1`. These IDs do not include physical page mapping,
the current region, or downstream packing.
`top_k` is in `[1, 512]`. Selection assumes finite FP32 scores.

The API accepts noncontiguous inputs and output views. ACLNN converts inputs
as needed and copies results back into the caller's output views. Aligned
paged cache views with contiguous page contents, gaps between pages, and
nonzero layer offsets retain zero-copy access; other key layouts are
materialized. Optional `out` and `selected_region_ids` buffers are reused.
Output views must not overlap internally, with inputs, or with each other.

To omit the public score tensor, use the separate IDs-only API:

```python
from vllm_ascend.ops.indexer_region_score import indexer_region_score_topk

region_ids = indexer_region_score_topk(
    query, weights, key_summary, actual_kv_lengths, block_table, cu_seq_lens,
    max_region_len=8192, top_k=512,
    # selected_region_ids=preallocated_ids,  # optional [G,T,top_k] INT32 buffer
)
```

Its scoring, causal visibility, tie breaking, ID ordering and padding match the
complete-score API. Omitting `out` from `indexer_region_score` still allocates
and returns scores; it does not select IDs-only execution.

IDs-only execution uses an internal, fixed **256 MiB total ACLNN workspace
limit**, including the CANN library prefix and allocation reserve. It computes
into a `[G,batch_T,roundup8(R)]` FP32 pool, selects IDs, and reuses the pool for
the next token batch. Request positions remain global across batches. If even
one token batch cannot fit, the call fails. Input storage and final IDs are
outside this workspace limit; it is not a whole-model memory limit.

Following the original bf16 TopK-only implementation, Query, weights, request
metadata and the optional IDs buffer must be contiguous. Key supports aligned,
page-inner-contiguous views, including inter-page gaps and layer offsets. The
IDs buffer's storage base and view offset must be 32-byte aligned. Unsupported
layouts are rejected without materializing full input copies. These stricter
requirements apply to IDs-only execution; the complete-score interface keeps
its noncontiguous input/output support.

At the native layer, passing `output=nullptr` to
`aclnnVllmAscendIndexerRegionScoreGetWorkspaceSize` selects the same bounded
path. `aclnnVllmAscendIndexerRegionScoreTopKGetWorkspaceSize` and
`aclnnVllmAscendIndexerRegionScoreTopK` provide convenience entry points.
The internal GE score slot has shape `[0]` when disabled. No public workspace
budget parameter is added. The migrated scope remains D=256, K<=512 and
cache/mask/sort modes 1/3/1.

For query token `j` within request `n`, the visible historical region count is
`clamp((actual_kv_lengths[n] - query_length[n] + j) // region_size,
0, max_region_len)`. For each visible region, the score is
`sum_h(weights[t, g, h] * head_scale * relu(dot(query[t, g, h], key[n, r])))`
for each query group `g`.
Only visible scores are defined. Masked columns and extra output capacity
must not be read as candidates.

The mode parameters remain public, with defaults `cache_mode=1`,
`mask_mode=3`, and `sort_mode=1`. Only this combination is currently
supported; other values are rejected. Continuous keys, D=128, other
masks, and score-ordered output are omitted. The
Python interface exposes the operator for models that already produce paged
key summaries; existing quantized lightning-indexer calls use a different
data contract.

Build the native operator with the repository's CANN build, then rebuild the
Torch extension to register `_C_ascend.npu_indexer_region_score`:

```bash
bash csrc/build.sh --pkg --soc=ascend950 --ops=indexer_region_score -j8 -O3
```

vLLM initializes the native extension during startup. Standalone callers can
initialize it with `vllm_ascend.utils.enable_custom_op()` before the first
operator call, after sourcing the installed OPP environment.

The scoring stage retains the native Cube/Vector pipeline and paged key
reuse across query groups. H=4 retains its compiled specialization; other
head counts use a common reduction with the same FP32 arithmetic order as
the source operator. The selection stage uses the source FP32 radix threshold primitive
with a 16384-region window and Ascend C `Sort` for logical IDs and retained
score keys. Short histories generate IDs directly. This keeps the large
history path efficient while removing unused layouts, proxy dimensions and sorting
policies. The operator builds with the CANN APIs already used by this
repository.

The source performance records cover 26 model cases, all with G=1, H=4,
D=256, `top_k=512`, and `region_size=8`. Those records use score-ordered output
and are historical reference data. A performance comparison for this migration
must measure both implementations with logical-ID ordering, the same inputs,
and the complete score-and-selection kernel sequence.

See [validation results](VALIDATION.md) for the compiled configurations,
precision coverage and matched model-case measurements.
