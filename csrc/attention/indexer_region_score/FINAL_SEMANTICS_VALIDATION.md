# Final layout and workspace semantics

Validated on 2026-11-12 against source commit
`a44c2d02dd8626a656162228fff63956a07e4ec7` on
`personal_indexer_region_score_A5_bf16`, under
`ops-transformer-personal/experimental/attention/indexer_region_score`.

## Current contract

The single Python/Torch operator family is preserved: omit `out` or pass
`None` to return IDs; provide `out` to return `(out, ids)`. Caller-supplied
score and IDs buffers are reused.

Both modes now normalize noncontiguous and misaligned inputs in the outer
ACLNN layer. Compatible aligned page-inner-contiguous Key views retain
direct reads, including gaps between pages and layer offsets. Other layouts
use Contiguous and, when needed, byte-preserving aligned TensorMove copies.
Noncontiguous or unaligned outputs use temporary aligned storage followed
by ViewCopy into the caller's view. Uninitialized output buffers are not
read to prepare those temporary outputs.

Score output is FP32 `[G,T,R_out]`, with `R_out >= max_region_len`, in eager,
compiled and functionalized execution. Graph paths copy the R-wide score
prefix into the supplied view; they accept wider capacity and noncontiguous
outputs. Only visible scores are defined; masked columns and extra capacity
remain unspecified. IDs are INT32 `[G,T,top_k]` and may also be strided or
have an unaligned logical offset.

The fixed **256 MiB limit applies to the IRS algorithm allocation**, including
its intrinsic library prefix, auxiliary storage, score pool and alignment
reserve. Outer input materialization and output writeback storage are
additional. ACLNN GetWorkspaceSize returns the full executor requirement,
which may exceed 256 MiB. Inputs and final outputs also remain outside the
algorithm allocation; this is not a whole-model memory limit.

IDs-only still reuses one bounded batch score pool and does not allocate a
full `[G,T,R]` score output. Inputs whose minimum algorithm batch cannot fit
are rejected. The migrated D256, G/H, dtype, K<=512 and cache/mask/sort=1/3/1
scope and scoring/selection mathematics are preserved.

## Validation

Environment: Ascend950PR, CANN 9.2.0, PyTorch 2.10.0 and torch_npu
2.10.0.post6. The native package and complete repository Torch extension
compiled successfully.

- **351 device tests passed** on physical device 2. This includes the previous
  coverage, automatic normalization in IDs-only, wide/strided score and IDs
  buffers, all three input dtypes, variable G/H, odd K and true multi-batch
  execution.
- Large backing regressions cover Q and Key with both aligned and unaligned
  `offset > numel`, in both output modes. Aligned offset views of weights,
  lengths, block tables and cumulative sequence lengths are also covered.
- Real `aot_eager` fullgraph and `torch.func.functionalize` tests cover both
  modes, strided inputs/outputs, wider score capacity, output aliases and
  rejection of malformed dtype, shape or device.
- CPU wrapper tests: **11 passed, with 18 subtests**. Meta and symbolic
  FakeTensor checks passed for the four fixed-return dispatcher overloads.
- Additional probes on physical device 3 passed all nine Q/K and weight
  dtype combinations over multiple selection windows, zero regions,
  offset views, guarded output writes, and actual NPUGraph replay with
  Key mutation and with a forced multi-batch input.
- TorchAir is not installed in this environment; no TorchAir validation
  is claimed.

An initial new score-output fixture inferred INT64 from an integer sentinel;
it was corrected to explicit FP32. A separate offset probe exposed an SDK
descriptor edge: Contiguous may return a shared-storage alias with reduced
StorageShape while retaining the logical offset. Final view construction
now preserves the original caller backing capacity when storage is shared,
and applies the offset once. Alignment materialization separately validates
capacity and clips TensorMove to the exact logical dense span. The final
351-test result uses this corrected native package.

## Actual total executor workspace

These direct native probes create no dense score output. Their incompatible
interleaved Key layout requires outer materialization. GetWorkspaceSize
returns the full allocation, and launch/IDs match an exact tie oracle.

| IDs-only case | Total executor bytes | Outer logical Key copy bytes | Dense score bytes avoided |
| --- | ---: | ---: | ---: |
| BF16 G1/T2049/H4/R32769/K512 | 281084416 | 16781312 | 268574724 |
| FP8 G1/T16384/H4/R131072/K512 | 285214208 | 33554432 | 8589934592 |

Both totals exceed **268435456 bytes**, as permitted by the final contract.
The second case avoids an 8 GiB dense score output. Final IDs use 4196352
and 33554432 bytes respectively, outside the returned workspace. Minimum
algorithm-batch rejection remains verified in both probes.

## Code size and standard paths

| Production component | Lines |
| --- | ---: |
| Kernel and shared tiling POD | 2505 |
| Host, ACLNN, Torch adapter and operator CMake | 782 |
| Python API | 84 |
| Existing registration, Meta, Functionalize and build additions | 128 |
| Total | 3499 |

Counts include comments and blank lines; tests, docs, validation data and
external shared libraries are excluded. Existing registration/build changes
are counted against the pre-migration base, consistently with earlier counts.

All nine packaged scoring/selection kernel binaries are byte-identical to
the preceding validated package. A new complete `op_summary`/`task_time`
audit of three standard model cases found no additional conversion or copy
tasks for their aligned layouts; IDs and visible full scores matched the
archived source implementation. This is not a new full 26-case performance
qualification, and no host/Python latency improvement is claimed.

## Evidence

The corrected private package is
`/tmp/irs-package-final-semantics2/vendors/irs_vllm_migration_transformer`.
Build, validation scripts and logs are under `/tmp/irs-final-semantics`:
`precision-fixed.log`, `extra-probe-fixed.json`, `workspace-outer-fixed.json`,
`workspace-large-fixed.json` and `model-pipeline-audit.json` record the final
results. Source and binary hashes are in the
[provenance record](validation/final_semantics_provenance.json).

The preceding [single-interface report](UNIFIED_TORCH_VALIDATION.md) and
[TopK-only report](TOPK_ONLY_VALIDATION.md) remain historical records. Their
stricter layout, output-capacity and total-executor-budget restrictions no
longer describe the current API. The previously documented device-6 stall
also occurred with the unchanged baseline and is not passing evidence here.
