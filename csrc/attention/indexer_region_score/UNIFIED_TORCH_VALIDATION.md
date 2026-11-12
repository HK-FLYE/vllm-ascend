# Unified Torch interface validation

This records the interface revision at source commit `6f4169fcdc`. Its strict
IDs-only layouts, total-executor workspace cap and functionalized score-output
restrictions were superseded by source commit `a44c2d02dd`; see
[final semantics validation](FINAL_SEMANTICS_VALIDATION.md) for current behavior.

Validated on 2026-11-12 against source commit
`6f4169fcdc2b24081b39710dabefaa0c689d5d8e` on
`personal_indexer_region_score_A5_bf16`, under
`ops-transformer-personal/experimental/attention/indexer_region_score`.

## Interface contract

The public Python API is `indexer_region_score`:

| Call | Result |
| --- | --- |
| Omit `out`, or pass `out=None` | IDs tensor |
| Supply `out` | `(out, ids)` |
| Supply `selected_region_ids` | Reuse the supplied IDs in either mode |

The independent `indexer_region_score_topk` Python/Torch entry and ACLNN
TopK convenience aliases have been removed, matching the source change.
The one dispatcher name `_C_ascend.npu_indexer_region_score` has fixed-return
overloads: default IDs, `.score` functional score/IDs, `.ids_out` mutating
IDs and `.out` mutating score/IDs. Mutators return no aliases; the public
wrapper returns the caller's buffers after dispatch.

A shared C++ Functionalize implementation lowers mutators to the functional
overloads and buffer copies. As in the source, functionalized score `out`
must be contiguous with exact `[G,T,R]` shape. Eager score output still
accepts larger capacity and noncontiguous views. IDs-only keeps the existing
contiguity/alignment requirements and fixed 256 MiB workspace limit.

The migrated paged D256, G/H, dtype, K<=512 and cache/mask/sort=1/3/1 scope
is preserved. The existing native ACLNN main entry still selects bounded
IDs-only execution through `output=nullptr`.

## Validation

Environment: Ascend950PR, CANN 9.2.0, PyTorch 2.10.0 and torch_npu
2.10.0.post6. Native package and the complete repository Torch extension
both compiled successfully.

- **258 device tests passed** on physical device 7. This retains the prior
  249-case coverage and adds dispatcher return/mutation checks, three
  `aot_eager` fullgraph cases, three actual `torch.func.functionalize` cases
  and two functionalization rejection cases.
- CPU wrapper tests: **9 passed, with 12 subtests**.
- Meta and symbolic FakeTensor checks cover all four overloads, dynamic
  token/group dimensions, output shapes and void mutation returns.
- Additional probes passed on physical device 1: all nine Q/K and weight
  dtype pairs across multiple selection windows, Q offset=1 in every
  input dtype, zero regions, guarded odd-K output buffers and layout
  rejection with eager full-score compatibility.
- Real NPUGraph capture/replay passed with Key mutation and with a true
  multi-batch G4/T1901/R9000/K37 input, using the single public API.
- A direct native null-score run for FP8 G1/T16384/H4/R131072/K512 used
  **251659264 bytes** of ACLNN workspace and **33554432 bytes** of final
  IDs, without creating the **8589934592-byte** dense score tensor.
  Minimum-batch rejection also passed.
- The native shared library exports only the main GetWorkspaceSize/run
  pair for this operator; the Torch dispatcher exposes only the unified
  name and its four overloads.

An initial AOT test failure came from the standalone test loader not
registering its dynamically loaded wrapper in `sys.modules`. The fixture
was corrected and the final full suite passed. The test environment does
not have vLLM installed; tests load the actual repository wrapper and the
compiled repository extension directly. TorchAir is not installed and
was not validated.

## Kernel and code size

The interface consolidation changes no scoring, selection, tiling or
workspace planner code. Eight relevant source files and all **nine packaged
kernel binaries** are byte-identical to the preceding validated package.
The earlier 26-case kernel performance data remains a record of this same
kernel implementation; no new Python or host-latency speedup is claimed.

| Production component | Lines |
| --- | ---: |
| Kernel and shared tiling POD | 2505 |
| Host, ACLNN, Torch adapter and operator CMake | 775 |
| Python API | 75 |
| Existing registration, Meta, Functionalize and build additions | 141 |
| Total | 3496 |

Counts include comments and blank lines, excluding tests, docs, validation
data and external shared libraries. Registration/build additions are
counted against the pre-migration base, consistently with previous counts.

## Evidence

The private package is
`/tmp/irs-package-unified/vendors/irs_vllm_migration_transformer`.
Build and validation scripts/logs are under `/tmp/irs-unified`; the final
Torch build log is `torch-build-final.log`, and device results are in
`precision-final.log` and `extra-probe-final.json`.
The [provenance record](validation/unified_interface_provenance.json)
contains source, binary and script hashes, code counts and evidence paths.

The previous [TopK-only report](TOPK_ONLY_VALIDATION.md) remains historical
evidence for workspace, matched model-case measurements and the device-6
stall also reproduced in the unchanged previous baseline. Those device-6
runs are not treated as passing evidence for this interface update.
