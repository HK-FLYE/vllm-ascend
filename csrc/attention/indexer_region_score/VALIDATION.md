# Migration validation

The 2026-11-12 IDs-only extension, current 3498-line code count, bounded
workspace measurements and final regression results are recorded in
[TopK-only validation](TOPK_ONLY_VALIDATION.md). The results below describe
the preceding complete-score migration as validated on 2026-11-10.

Validated on 2026-11-10 with Ascend950PR, CANN 9.2.0, PyTorch 2.10.0 and
torch_npu 2.10.0.post6. Device tests used the compiled repository Torch
extension and an independently installed custom OPP package.

## Production code size

Counts include blank lines, comments, CMake, the Torch adapter, Python API and
additions to existing registration/build files. Tests and documentation are
excluded.

| Component | Lines |
| --- | ---: |
| Kernel and shared tiling POD | 2687 |
| Host, ACLNN, Torch adapter and operator CMake | 674 |
| Python API | 63 |
| Existing Torch registration, Meta and CMake additions | 41 |
| Total | 3465 |

The source operator's kernel, Host and Torch implementation occupied about
11294 lines. The migrated implementation specializes paged storage,
D=256, causal historical regions and logical-ID-ordered output. G can be 1
or a positive multiple of 4; H can be any multiple of 4 from 4 through 128.
The public Key argument is `key_summary`. The `cache_mode`, `mask_mode`, and
`sort_mode` parameters are retained; defaults and currently supported values
are 1, 3, and 1 respectively. Other mode combinations are rejected.
Noncontiguous inputs and output views are supported through CANN conversion
and copy-back APIs, while compatible paged cache views retain zero-copy access.

## Correctness and integration

- All nine Q/K and weight dtype combinations compiled with the repository's
  CANN build. The generated sources matched the working-tree source hashes.
- The complete repository Torch extension compiled, loaded and dispatched the
  new Meta implementation successfully. Output aliases and a dynamic
  FakeTensor token dimension were also checked.
- Python wrapper unit tests: 6 passed, with 12 additional subtests.
- Unchanged build-cache prerequisite tests previously passed all 112 cases.
- Device precision tests: all 235 cases passed on physical device 0.
  Visible FP32 scores and selected IDs were compared exactly against an independent CPU oracle
  using exactly representable binary-fraction inputs. This includes the
  previous 155 cases and 80 additional API/layout cases.
- Non-32-byte-aligned preallocated ID views also passed 35 device checks:
  G=4, T=9, K=1/2/3/4/7/8/17, each repeated five times.
- An independent C++ check confirmed identical shard plans for 280 G=1/H=4
  configurations and valid tile capacities for every supported H.
- The complete `bash format.sh ci` check passed. A separate check included
  every new untracked source and documentation file.

Device coverage includes FP16/BF16/FP8 E4M3FN, all three weight dtypes,
all 32 supported head counts, G=1/4/8, shared keys with independent group
queries and weights, full query tiles at H=12/20/68/96/128,
negative weights, invalid physical pages, arbitrary page tails, poisoned
inter-page gaps, nonzero cache layer offsets, empty requests, zero history,
preallocated output aliases and guards, small K, K=512, equal scores, and
multiple 16384-region carry merges.

API/layout coverage includes the `key_summary` keyword, default and explicit
mode parameters, invalid modes through both Python and native dispatch,
strided Query/weights/request metadata, transposed and interleaved keys,
unaligned Query/Key storage offsets, single-page noncanonical strides,
noncontiguous score/ID output views with untouched-storage guards, combined
input/output views, and default zero-element score outputs in all Q/K dtypes.
CANN's canonical TensorV2 workspace description with an empty stride is
handled as dense storage while retaining offset and span validation.

```bash
python -m pytest -q --confcutdir=tests/ut/ops \
  tests/ut/ops/test_indexer_region_score.py
python -m pytest -q \
  --confcutdir=tests/e2e/nightly/single_node/ops/singlecard_ops \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_indexer_region_score.py
```

Device tests require the custom OPP environment and the rebuilt Torch
extension to be loaded. The integration environment did not contain vLLM;
tests loaded the compiled native extension and the real Python wrapper
directly, without unrelated model imports.

## Performance comparison

The comparison uses the source model-case shapes and sequence metadata,
regenerated deterministic inputs, the same device and logical-ID ordering
for both operators. Measurements include the complete scoring and selection
kernel. Historical score-ordered results are not used as acceptance data.

All 26 source model cases passed bitwise visible-score and exact selected-ID
comparison, and all 26 improved in performance. The geometric-mean speedup is
**1.138x**, with individual speedups of **1.040x to 1.289x**.

Measurements used physical device 5 and CANN 9.2.0. Each implementation has
10 samples per case, collected in five ABBA blocks, for 520 samples. Every
measured target followed the same 256 MiB FP16 ArgMax warmup. Profiler task
duration excludes host API time and the warmup kernel. Each session started
idle; monitoring detected no foreign process during the application.

Full `op_summary` and `task_time` audits found no additional conversion,
TensorMove, ViewCopy, or memory-copy tasks for the 26 standard model cases
or the two additional FP8 prefill cases. Restoring noncontiguous support
does not add device-copy work to these standard inputs.

Q/K values are integers in `[-8,8]/8`; weights are integers in `[1,8]/8`.
Physical pages form contiguous valid-prefix runs, modulo the recorded pool
size, with unused page capacity set to `-1`. The original model shapes and
sequence lengths are preserved. These results cover matched regenerated
inputs with that distribution.

| Additional FP8 prefill case | Source median, us | Migrated median, us | Speedup |
| --- | ---: | ---: | ---: |
| S01, T=1024, R=128 | 63.9925 | 20.1665 | 3.173x |
| S06, T=256, R=16384 | 74.3040 | 62.5470 | 1.188x |

These two additional cases passed the same strict precision checks and are
reported separately from the 26-case main statistic.

Before this API/layout compatibility update, three representative G=1/H=4
cases compared the G/H extension with the earlier fixed-G/H migration,
using separate processes to isolate the two packages
with the same operator name. Each version has ten samples per case. The
new-to-previous time ratios are 0.9671 for BF16 S01, 1.0135 for BF16 S06,
and 1.0205 for FP8 R9000; their geometric mean is 1.0001.

Three additional variable-G/H cases were measured separately against the
source, with ten samples per implementation and exact score/ID agreement:

| G/H case, R=2048 | Source median, us | Migrated median, us | Time change |
| --- | ---: | ---: | ---: |
| G=4, H=12, T=16, FP16 | 28.7840 | 31.0670 | +7.93% |
| G=4, H=128, T=8, BF16 | 104.9235 | 101.7375 | -3.04% |
| G=8, H=64, T=8, FP8 | 75.7140 | 72.9070 | -3.71% |

The first case is slower than the source's group-fused implementation.
The compact implementation processes groups sequentially while reusing
resident keys; the measured 26 original model cases all use G=1/H=4.
The main performance statistic therefore does not establish a speedup for
every supported G/H configuration.

A separate layout-cost experiment used the same migrated package and the
BF16 S01 shape, with identical logical values. Both calls use paged storage
and mode values 1/3/1. The strided variant interleaves all six inputs and
both outputs with stride 2 and a nonzero storage offset.

| Storage layout | Score kernel, us | Sum of device tasks, us | Device task span, us |
| --- | ---: | ---: | ---: |
| Contiguous | 17.6135 | 17.6135 | 17.7500 |
| Strided views | 15.9780 | 59.6590 | 60.7500 |

The strided variant includes Slice, StridedSlice, and ViewCopy work. These
figures include conversion and copy-back costs, excluding host API time and
warmup; they are not end-to-end serving latency or a guarantee for other
layouts. The layout experiment is excluded from source-comparison statistics.

- [All 26 case medians](validation/performance.csv)
- [All 520 main samples](validation/performance_samples.csv)
- [Additional FP8 prefill medians](validation/fp8_prefill.csv)
- [Additional FP8 prefill samples](validation/fp8_prefill_samples.csv)
- [Historical H=4 comparison before the API update](validation/previous_migration.csv)
- [Additional variable-G/H medians](validation/variable_groups_heads.csv)
- [All 60 variable-G/H samples](validation/variable_groups_heads_samples.csv)
- [Device pipeline audits](validation/pipeline_audit.json)
- [Contiguous versus strided layout costs](validation/layout_cost.csv)
- [All 20 layout-cost samples](validation/layout_cost_samples.csv)
- [Measurement identity and source/package hashes](validation/provenance.json)

The baseline is the source operator's November 9 `outline_fix/package_920`
archive, measured alongside the migrated package with ID ordering enabled.
Detailed call manifests, profiler outputs and the measured script snapshots
are retained locally under `/tmp/irs-api/perf/`.

Independent local review found no required code changes.
