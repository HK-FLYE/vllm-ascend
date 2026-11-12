# TopK-only validation

This is the historical validation of commit `69ebd50ad`, before the single
Torch interface was synchronized. Its separate Python/ACLNN TopK convenience
entry points have since been removed. Current behavior and validation are in
[unified Torch validation](UNIFIED_TORCH_VALIDATION.md).

Validated on 2026-11-12 with Ascend950PR, CANN 9.2.0, PyTorch 2.10.0 and
torch_npu 2.10.0.post6. The complete repository Torch extension and all nine
native Q/K and weight dtype combinations were rebuilt. Validation used a
private OPP installation; the source worktrees and global OPP were not modified.

## Source contract

The authoritative source is ops-transformer-personal's
`experimental/attention/indexer_region_score`, on branch
`personal_indexer_region_score_A5_bf16`, HEAD `4f11a7a479` plus the user's
synchronized, uncommitted changes. All 33 production files were compared
against the original repository's registered TopK-only worktree
`/tmp/irs_topk_only_20261111/worktree` at `d87170aae3` and are identical.
That reference follows `edd7196c64` and `b9d851be26`, with `4f11a7a479`
in its ancestry. Source hashes are retained in the validation record.

The complete-score Python API still returns `(scores, ids)` and allocates
scores when `out` is omitted. The separate `indexer_region_score_topk` API
returns only IDs. At ACLNN level, the existing entry selects the bounded
path when `output=nullptr`; the TopK convenience entry calls the same path.
The internal GE output0 is empty FP32 `[0]`, selected by `return_score=false`.

The fixed total ACLNN workspace limit is 256 MiB, including the platform
library prefix and allocation reserve. Global request and causal coordinates
are preserved while a batch-local score pool is reused. Inputs and final
IDs are outside this workspace limit. TopK-only rejects unsupported layouts
instead of creating full input copies. The complete-score interface retains
noncontiguous input and output support.

The migrated scope remains paged storage, D=256, cache/mask/sort modes 1/3/1,
K=1..512, G=1 or a positive multiple of four, and H=4..128 in steps of four.
This does not extend the migration to the source's other modes or D=128.

## Code size

Counts include comments, blank lines, both Python APIs, native interfaces,
CMake and additions to existing Torch registration files. Tests, documentation
and external shared CANN libraries are excluded, consistently with the preceding
migration count.

| Component | Lines |
| --- | ---: |
| Kernel and shared tiling POD | 2505 |
| Host, ACLNN, Torch adapter and operator CMake | 790 |
| Python APIs | 111 |
| Existing Torch registration, Meta and CMake additions | 92 |
| Total | 3498 |

The extra mode adds 33 total lines over the previous 3465-line implementation.
Repeated radix byte stages, page scanning and Host plan comparisons were
consolidated. The H=4 arithmetic specialization and optimized page-loading
paths are retained. Ten thousand randomized Host shard plans matched the
preceding implementation exactly.

## Correctness and graph execution

- All **249 device/integration tests passed** against the final package on
  physical device 2, including the preceding 235 tests.
- CPU wrapper tests: **9 passed, with 15 subtests**.
- The Host workspace planner passed **375 accepted-plan checks**, together
  with zero dimension, overflow, region limit and minimum-batch rejection guards.
- Nine actual multi-batch combinations cover FP16/BF16/FP8, G=1/H=4 and
  G=4/H=12, K=512 and K=37, empty requests, a request crossing the batch
  boundary, causal padding and a three-batch tail. Dense scores exceed 256 MiB.
- Extra probes passed all nine Q/K and weight dtype combinations with
  R=33001 and K=37, spanning more than two selection windows. They also
  covered zero regions, contiguous Query views with offset=1 in all three
  dtypes, metadata offsets, and repeated guarded odd-K output writes.
- Invalid Key layouts and unaligned IDs are rejected by TopK-only while
  the complete-score path continues accepting those views with correct IDs.
- Default and `.out` dispatch, output aliases, Meta and symbolic FakeTensor
  dimensions passed. Real NPUGraph capture/replay passed with Key mutation
  and with a forced multi-batch G4/T1901/R9000/K37 input.
- TorchAir is not installed in this environment; no TorchAir validation is claimed.

## Native workspace without score allocation

These probes create inputs, final IDs and the returned workspace, without
creating a dense score tensor or its descriptor. Both the original nullable
score entry and the TopK convenience entry were launched and checked against
an exact tie-ID oracle.

| Case | Dense score bytes avoided | Actual ACLNN workspace bytes | Final IDs bytes |
| --- | ---: | ---: | ---: |
| BF16 G1/T2049/H4/R32769/K512, page stride 3 | 268574724 | 264302592 | 4196352 |
| FP8 G1/T16384/H4/R131072/K512, page stride 3 | 8589934592 | 251659264 | 33554432 |

Both entries report identical workspace sizes and produce identical correct
IDs. The long prefill avoids an 8 GiB dense score output. A G4/R16777216
input whose minimum one-token score pool cannot fit is rejected before
launch, returning zero workspace size and a null executor.

## Matched model-case performance

All 26 source model-case shapes were measured on physical device 4 using
identical regenerated binary-fraction inputs, logical-ID ordering, the same
warmup and interleaved three-way measurements. Each case has ten samples
per implementation, for **780 measured samples**. Visible scores of the two
complete-score paths match bitwise; all three paths' IDs match exactly.

| Comparison | Geometric-mean speedup |
| --- | ---: |
| Migrated complete-score / archived source complete-score | 1.109x |
| Migrated IDs-only / archived source complete-score | 1.129x |

Both migrated paths are faster than the archived source complete-score path
in all 26 cases. IDs-only takes 0.9825x the migrated complete-score time in
geometric mean; individual ratios range from 0.9118 to 1.0137. These are
kernel durations, excluding host API and Python allocation time. This does
not measure speedup over a freshly built source TopK-only kernel.

Complete `op_summary` and `task_time` audits found no additional conversion,
TensorMove, ViewCopy or device-copy tasks in the standard model cases. Every
accepted session started idle and its monitor detected no foreign process.
Device-5 attempts interrupted by other jobs were rejected, and the final
26-case series was collected on device 4. The per-case medians and all raw
samples are in [performance.csv](validation/topk_only_performance.csv) and
[samples.csv](validation/topk_only_performance_samples.csv).

## Reproduction and evidence

The standard tests are in the existing CPU and single-card operator test files.
The standalone Host planner test can be run from the repository root:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  csrc/attention/indexer_region_score/tests/bounded_workspace_test.cpp \
  -o /tmp/irs_bounded_workspace_test
/tmp/irs_bounded_workspace_test
```

Final package: `/tmp/irs-package-noscore3/vendors/irs_vllm_migration_transformer`.
Build, device, workspace, graph and profiler logs are under `/tmp/irs-noscore`.
The [machine-readable validation record](validation/topk_only_provenance.json)
stores artifact hashes and evidence paths. The repository formatting and
manual-stage checks passed, including the new untracked files. A final
independent local code review found no substantive issues.

Physical device 6 stalled on the small R9000/K512 case with both the modified
operator and the unchanged previous API5 package. The cause has not been
localized beyond a device-dependent execution wait. Those runs are excluded
from passing evidence; their logs are retained. No device reset was performed.
