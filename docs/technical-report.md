# Descale MVC CPU and Memory Optimization Technical Report

**Date:** 2026-08-05

**Scope:** Linux Release CPU backend, fixed-kernel execution, and the original `descale` plugin comparison

**Status:** Final evidence review for the current release artifact

## 1. Executive Summary

The current `dsmvc` CPU implementation is faster because it reduces both
executor overhead and intermediate data movement. The most important result is
not a higher DRAM bandwidth number. On the tested Ryzen 9 5950X, measured
bandwidth stays close to 44-46 GB/s while the candidate transfers fewer bytes,
incurs fewer DRAM fills, and spends fewer cycles waiting on the memory system.

Against the `before-destination-forward` baseline, the v4 Release candidate
shows the following R32 throughput ranges across b1/b3/b5/b7:

| Format | Candidate / baseline FPS at R32 | Best v4 case |
|---|---:|---:|
| Float32 | 1.240x - 1.287x | b7: 1.287x |
| GRAY16 | 1.314x - 1.395x | b7: 1.395x |
| YUV420P10 | 1.187x - 1.294x | b5: 1.294x |

The memory profile explains these gains. At R32, Float32 b5 falls from
55.02 to 43.27 MB/frame and Float32 b7 from 55.01 to 43.21 MB/frame. GRAY16
b5 falls from 47.73 to 34.47 MB/frame and GRAY16 b7 from 49.08 to 34.75
MB/frame. The corresponding CPI and DRAM-fill reductions are large, while
measured GB/s remains effectively saturated.

The original `descale` comparison is a separate layer of evidence. On the
same supplied Digimon source and release VapourSynth environment, the current
plugin is 13.13x faster for the R1 `getfnative` scan and 6.45x faster at R32.
The narrower `getfnative_v2` scan reaches 3.10x at R1 and 2.56x at R32; the
`selectkernel` scan reaches 6.02x and 2.46x respectively.

All 12 v4 format/kernel pixel checks have equal SHA-256 output, zero different
samples, and zero maximum error. The low-request result is intentionally
reported as mixed: the main gains are at R8/R32, while some R1 cells are near
measurement noise or regress. The report does not claim a universal R1 gain.

## 2. Scope, Baselines, and Provenance

### 2.1 Version identity

The repository is currently at `HEAD=c33a585`. The main implementation change
for the integer/data-movement pipeline is commit `f776a76`; the current branch
also contains the follow-up standalone planner/dependency cleanup in
`293cc0a`, benchmark cleanup in `d971659`, and the README update in
`c33a585`.

The CPU optimization history used in this report is:

| Commit | Role |
|---|---|
| `4917f7b` | CPU executor refactor, inverse-only plans, and packed-plan caching |
| `3a69c60` | AxisPlan validation, cache ownership, and sealing safety |
| `4682cbc` | AVX2 paired-column specialization for Spline64/b7 |
| `f776a76` | Integer pipeline, destination-ordered/streamed paths, fused writeback, and memory-phase limiting |
| `293cc0a` | Remove the external GetNative-VF build dependency and keep the planner in this repository |

The v4 fixed-kernel A/B uses these exact binaries:

| Artifact | SHA-256 |
|---|---|
| `before-destination-forward` baseline | `ba4d31a428d7840853773706ec2e5d08a2c1dd5e576b8ee7b34474d79f6f07df` |
| v4 candidate | `48af462cd5cc4c0bea65b0aaddaa5b085ae1fee7fd3293ee95a92d0d09cb364d` |

The v4 candidate is also the current checked-out Release plugin at
`out/linux-release-generic-20260805/final/dsmvc.so`. The baseline above is a
pre-v4 `dsmvc` build, not the original `descale` plugin. The original plugin is
compared separately in Section 9.

### 2.2 Evidence hierarchy

The report uses measured artifacts as the authority for numeric claims:

1. v4 Release blank-pipeline throughput and pixel checks.
2. v4 perf/PCM memory profile.
3. The CPU optimization decision report and its stage-level A/B results.
4. The release benchmark JSON and generated release report for the original
   `descale` comparison.
5. Git history and Codex history for intent, rejected hypotheses, and decision
   provenance only.

The local Codex records were inspected in
`/home/owen/.codex/history.jsonl` and the related session records. The mounted
Codex records under
`/run/media/owen/1A16B65916B6361B/Documents and Settings/lsy39/.codex` were
also checked, including earlier Windows/GetNative-VF CPU profiling and AVX2
recurrence discussions. Those records document the decision to optimize for
general workloads, avoid CCD-specific tuning, use streamed RHS generation,
fuse integer conversion, and test R1/R8/R32 across b1/b3/b5/b7. They do not
serve as current Linux benchmark evidence because their binaries, platform,
and workload definitions differ.

## 3. Benchmark Methodology

### 3.1 Fixed-kernel A/B

The primary v4 comparison uses an in-memory VapourSynth `std.BlankClip` at
1920x1080, with a 1080 -> 810 descale geometry. It removes decoder and source
I/O cost and measures the plugin execution path and VapourSynth frame plumbing.

| Parameter | Value |
|---|---|
| CPU | AMD Ryzen 9 5950X, 16 cores / 32 logical CPUs |
| OS | Linux 7.0.0-28-generic, glibc 2.43 |
| Memory | 30.3 GiB |
| VapourSynth | Core R78, API R4.2/R3.6 |
| Formats | Float32, GRAY16, YUV420P10 |
| Kernels | b1, b3, b5, b7 |
| Requests | R1, R8, R32 |
| Frames | 1,200 measured frames |
| Warmup | 64 frames |
| Repeated runs | 3 independent VSPipe processes |
| Memory concurrency | `auto` |

The build is a generic x86-64 Release build. AVX2/FMA is isolated to
`src/cpu_executor_avx2.cpp`; the build does not use LTO, PGO, `-march=native`,
or fast-math flags. This keeps the measured binary representative of a
portable release configuration rather than a machine-specific tuned build.

The v4 throughput artifact contains 36 throughput cells and 12 pixel checks.
The memory artifact contains 192 profile cells: 144 perf samples and 48 PCM
samples. Perf and PCM groups were collected in separate VSPipe processes;
profile groups below 99% running were rejected.

### 3.2 End-to-end original-plugin comparison

The release benchmark uses the supplied Digimon source, the LSMASH source
filter, the same candidate graph, and threads 1/8/16/32. It evaluates three
workloads:

| Case | Candidate space | What it measures |
|---|---:|---|
| `getfnative` | 30,800 | Broad height and scaler-family search |
| `getfnative_v2` | 3,200 | Vertical-only height search |
| `selectkernel` | 101 | Fixed-height kernel-parameter search |

The graph includes planning, `FrameEval`, descale, reconstruction, `Expr`,
border cropping, `PlaneStats`, frame delivery, and VSPipe overhead. It is a
realistic end-to-end workload, not a pure kernel measurement.

### 3.3 Statistics and correctness

FPS ratios are candidate FPS divided by baseline FPS. CPU milliseconds/frame,
CPI, L2 fills, DRAM fills, GB/s, and MB/frame are taken from the corresponding
artifact tables. Pixel checks compare output hashes and report different
samples, maximum error, and mean error.

The CPU-stage decision gate required a 3% target improvement, at least three
pooled normalized MADs, and no statistically significant regression above 1%
in the other guarded regimes. This gate was applied to CPU scheduling and
kernel-specialization experiments; the final v4 blank-pipeline run is a
three-run release matrix and should not be confused with the seven-run CPU
decision gate.

## 4. CPU Optimization Timeline

### 4.1 Planner and executor ownership

The executor was separated from the older generalized planner path and made
inverse-only. A packed CPU plan stores vector-friendly weights and band factors,
and built-in plans can be prepared once and reused through an owning cache.
This removes repeated packing and makes plan lifetime explicit.

The safety stage then strengthened `AxisPlan::valid()`, rejected malformed CSR
offsets and non-finite coefficients, made borrowed-plan packing invocation-local,
and added acquire/release sealing for prepared plans. These changes are
primarily correctness and concurrency work, but they prevent unsafe cache reuse
from invalidating performance measurements.

### 4.2 AVX2 paired columns

The accepted CPU specialization dispatches half-bandwidth 7 to a paired-column
AVX2 kernel. The pair reuses coefficient broadcasts for two independent YMM
accumulators while preserving the arithmetic order of each column.

The decision report measured the accepted b7 pair against the generic b7 path:

| Case | R1 | R8 | R32 |
|---|---:|---:|---:|
| Spline64/b7 FPS ratio | 1.071x | 1.046x | 0.988x |
| CPU seconds/frame change | -6.66% | -4.18% | +0.02% |

The R1 and R8 gains cleared the gate. The R32 loss was within three pooled
MADs, so the specialization was retained. The comparable b5 paired-column
experiment peaked at only +2.68% at R1 and did not clear the 3% gate.

### 4.3 Rejected scheduling directions

Tile-before-slice and pool-disabled tiling variants caused significant low-
request regressions and were rejected. Disabling the existing opportunistic
WorkerPool improved saturated R32 b5/b7 throughput by about 3.4-3.6%, but lost
about 72% at R1 and 12.6-13.9% at R8. The existing pool policy was therefore
retained.

Rolling recurrence was skipped because the required accepted b5 comparator did
not exist. No CCD-specific affinity or topology tuning was accepted; the
optimization target remained general throughput across request counts and
formats.

## 5. CPU Executor Optimizations in the Final Build

The final executor combines the following mechanisms:

1. **Owning packed-plan reuse.** `PackedCpuPlan` stores padded dimensions,
   vector weights, band factors, inverse diagonals, and source-oriented CSR
   metadata. Prepared plans are retained by `shared_ptr`; borrowed plans are
   copied into invocation-local ownership.
2. **AVX2 fast paths.** Float and integer paths use a separate AVX2/FMA
   translation unit. b7 uses paired columns; scalar tails remain available for
   unaligned or incomplete widths.
3. **Bounded opportunistic parallelism.** The existing WorkerPool is kept after
   A/B testing because it protects R1/R8 behavior even though it is not optimal
   for every saturated R32 cell.
4. **Strict numerical guards.** Axis plans, strides, conversion scales, output
   ranges, and finite coefficient values are validated before execution.

These optimizations reduce executor overhead and improve arithmetic throughput,
but they do not alone explain the largest v4 R32 gains. The dominant additional
change is the reduction of intermediate memory traffic described next.

## 6. Memory Traffic and Data-Movement Optimization

### 6.1 Old dataflow pressure

The wide-kernel path performs a horizontal inverse, a vertical RHS accumulation,
banded solves, and integer conversion. A naive implementation repeatedly
clears and updates a full Float32 intermediate, then reads that intermediate
again for the backward solve and final integer writeback. At high request counts,
the intermediate is large enough that extra passes become DRAM traffic rather
than useful arithmetic.

### 6.2 Source-oriented RHS metadata

`PackedCpuPlan` now includes `source_offsets`, `source_destinations`, and
`source_weights`. These arrays invert the vertical CSR view so the executor can
walk each source row once, reuse its horizontal result, and update only the
vertical destinations reached by that source row. This makes source-row reuse
explicit and avoids repeatedly rediscovering the same transpose relationships.

### 6.3 Destination-ordered streamed RHS

For overlapping 2D frames, the final path generates the vertical RHS in
destination-row order. It gathers the horizontal source rows needed by one
destination row, performs the vertical forward operation immediately, and
keeps only the active working set. This removes the need to materialize and
revisit a broad intermediate for every frame.

The source-oriented and destination-ordered paths are complementary. The
buffered path is useful when one frame can exploit internal parallelism; the
streamed path is selected when multiple frames overlap and reducing the
working set has higher value than another independent full-frame pass.

### 6.4 Horizontal row/block cache

Horizontal results are retained in a bounded cache of source-row blocks. The
packed plan records the number of source blocks needed by the vertical stencil,
and the executor uses age-based replacement. The AVX2 implementation processes
8-row blocks and handles the final partial block without widening the public
API. This reduces repeated horizontal work and keeps the streamed path's live
data compact.

### 6.5 Fused backward solve and integer writeback

The backward solve now has integer-output variants. Each solved Float32 value is
scaled, clamped, rounded, and written to `uint8` or `uint16` output while the
backward pass is already visiting it. The final Float32 RHS is therefore not
read a second time by a separate conversion pass.

### 6.6 Memory-phase concurrency limiter

The plugin uses a shared limiter for the memory-heavy phase. Its default limit
is half the logical CPU count, clamped to the supported range, and applies only
to fused-integer wide adaptive 2D cases whose core request count exceeds that
limit. Float32 processing is not constrained by the default limiter. Setting
`DSMVC_MEMORY_CONCURRENCY` explicitly applies the experimental override to all
adaptive 2D formats and kernel widths; `0` disables the limiter.

This is a concurrency limit, not CPU affinity or CCD binding. Its purpose is to
avoid letting too many independent working sets evict one another at R32.

## 7. Incremental v4 A/B Results

### 7.1 Aggregate throughput ranges

The final v4 matrix shows a consistent pattern: low-request cells are mixed,
R8 is generally positive, and R32 gains are broadest.

| Format | R1 ratio range | R8 ratio range | R32 ratio range |
|---|---:|---:|---:|
| Float32 | 0.956x - 1.025x | 1.099x - 1.202x | 1.240x - 1.287x |
| GRAY16 | 0.975x - 0.999x | 1.038x - 1.116x | 1.314x - 1.395x |
| YUV420P10 | 0.984x - 1.039x | 0.995x - 1.138x | 1.187x - 1.294x |

Representative wide-kernel cells make the CPU-time reduction visible:

| Format / kernel | Baseline -> candidate FPS | FPS ratio | CPU ms/frame baseline -> candidate |
|---|---:|---:|---:|
| Float32 b5, R32 | 822.16 -> 1056.95 | 1.286x | 36.423 -> 18.709 |
| Float32 b7, R32 | 818.05 -> 1052.59 | 1.287x | 36.574 -> 18.032 |
| GRAY16 b5, R32 | 940.17 -> 1304.90 | 1.388x | 31.735 -> 13.551 |
| GRAY16 b7, R32 | 907.91 -> 1266.82 | 1.395x | 32.441 -> 13.548 |
| YUV420P10 b5, R32 | 650.86 -> 841.97 | 1.294x | 45.644 -> 21.408 |
| YUV420P10 b7, R32 | 647.66 -> 817.51 | 1.262x | 45.590 -> 21.428 |

### 7.2 R1 stability interpretation

The v4 b7 R1 long run measured Float32 at 0.9886x, while its same-binary
self-control measured 1.0113x. The full three-run v4 matrix measured Float32
b7 at 0.9972x. These observations support a near-unity/noise interpretation
for that b7 R1 result, not a deterministic b7 regression. Other R1 cells do
show real tradeoffs, including Float32 b1 at 0.9561x, so the final claim is
limited to strong R8/R32 improvement and mixed R1 behavior.

## 8. CPU, Cache, and DRAM Attribution

The v4 memory profile provides the following representative R32 changes:

| Format / kernel | CPI baseline -> candidate | DRAM fills/frame baseline -> candidate | GB/s baseline -> candidate | MB/frame baseline -> candidate |
|---|---:|---:|---:|---:|
| Float32 b5 | 2.164 -> 0.987 | 60,010 -> 19,725 | 45.24 -> 45.86 | 55.02 -> 43.27 |
| Float32 b7 | 1.734 -> 0.758 | 55,458 -> 20,851 | 45.25 -> 45.69 | 55.01 -> 43.21 |
| GRAY16 b5 | 1.823 -> 0.676 | 60,204 -> 8,940 | 45.30 -> 45.66 | 47.73 -> 34.47 |
| GRAY16 b7 | 1.493 -> 0.537 | 58,462 -> 10,331 | 45.32 -> 44.69 | 49.08 -> 34.75 |

The candidate can complete more work at approximately the same measured
bandwidth because each frame requires less traffic. Lower L2 miss ratios and
fewer DRAM fills are consistent with the destination streaming, row caching,
fused writeback, and memory-phase limiting described in Section 6.

The accepted b7 CPU profile independently measured a 22.34% reduction in
vertical retired instructions/frame and a 20.74% reduction in estimated
PMCx024 events/frame at R32. At R1, vertical PMCx024 events/frame fell 33.76%.
The sampled R32 vertical CPI increased from 2.160 to 3.106 in that targeted
profile even though throughput was neutral, so sampled cycles were not used as
the specialization acceptance signal.

## 9. Comparison with the Original `descale` Plugin

The original-plugin release artifact compares the current Release plugin with
`libdescale.so` using the same supplied Digimon source and candidate graphs.
The results are:

| Workload | Candidates | R1 old -> current | R1 speedup | R32 old -> current | R32 speedup |
|---|---:|---:|---:|---:|---:|
| `getfnative` | 30,800 | 13.569 -> 178.114/s | 13.13x | 55.581 -> 358.623/s | 6.45x |
| `getfnative_v2` | 3,200 | 109.425 -> 339.658/s | 3.10x | 159.571 -> 408.623/s | 2.56x |
| `selectkernel` | 101 | 18.700 -> 112.489/s | 6.02x | 51.218 -> 126.175/s | 2.46x |

The same release error sweep found unchanged best candidates and heights for
all three workloads. The largest reported output/reconstruction absolute error
was `2.38419e-07` for `getfnative`; the other two cases reported zero.

There is an important measurement boundary. In
`benchmark-results/release-benchmark-20260805/release-benchmark.json`, old
performance and paired error data were freshly measured, while current Release
performance was reused from the preceding current-only refresh. Therefore the
ratios are valid recorded release results, but this artifact is not a fully
synchronous same-session rerun of both plugins.

The blank fixed-kernel results corroborate the same direction without decoder
cost. For example, at R32/R32T32 the current plugin reaches 1.44x for
Bilinear, 1.48x for Lanczos3, and 1.48x for Spline64 against the original
plugin in the release blank-clip table.

## 10. Correctness, Compatibility, and Risk

### 10.1 Pixel correctness

The v4 Release matrix covers Float32, GRAY16, and YUV420P10 with b1/b3/b5/b7.
All 12 cells have:

- equal SHA-256 output;
- zero different samples;
- zero maximum error; and
- zero mean error.

The integer fused paths are therefore not accepted solely on throughput. They
also pass the same output comparison as the buffered baseline path.

### 10.2 Engine and sanitizer checks

The CPU decision artifact records fresh Release engine tests as passed, with a
maximum b5/b7 scalar-versus-AVX2 error of `7.15256e-7`, no non-finite outputs,
and unchanged guards. ASan/UBSan tests also passed with zero sanitizer errors.
ThreadSanitizer had passed at the earlier safety stage, but was not rerun after
the arithmetic-only b7 commit. The installed VapourSynth integration baseline
was incompatible with the test harness signature, so that run stopped before
pixel checks; this is an environment/baseline compatibility limitation, not a
failed v4 pixel comparison.

### 10.3 Compatibility contract

The optimization commits keep the public VapourSynth filter API and format
behavior intact. The CPU backend continues to expose scalar fallback behavior,
AVX2 dispatch when available, and explicit integer range conversion. The
standalone planner follow-up removes the build-time GetNative-VF dependency;
it is a packaging and ownership improvement, not a reason to reinterpret the
v4 performance ratios.

## 11. Evidence Quality, Limitations, and Next Steps

### 11.1 Limitations

- BlankClip results exclude decoder and source I/O. They measure plugin and
  frame-plumbing behavior, not total playback throughput.
- The v4 PMU profile uses one run per cell. It is strong attribution evidence,
  but not a replacement for a repeated throughput distribution.
- The memory profile artifact records `perf_event_paranoid=0` during collection.
  The current system value after profiling is `4`; these states must not be
  conflated when reproducing the profile.
- The original-plugin release comparison reuses current performance, as noted
  above.
- The tested machine is a Zen 3 DDR4 system. The absolute memory ceiling and
  the best `DSMVC_MEMORY_CONCURRENCY` value may differ on DDR5, another CCD
  topology, or a different ISA.
- No accepted experiment establishes that CCD affinity, manual prefetch, or a
  rolling recurrence improves the general plugin. Those directions remain
  unsupported by the current evidence.

### 11.2 Recommended follow-up

1. Rerun the original-plugin and current-plugin E2E measurements in the same
   session with identical run counts and process isolation.
2. Repeat the v4 blank/profile matrix on at least one DDR5 platform and one
   non-5950X AVX2 CPU to validate the default memory-phase limit.
3. Keep R1 and R8 as explicit release gates. A future memory optimization
   should retain the broad R32 benefit without turning the current mixed R1
   behavior into a deterministic regression.
4. Preserve the raw profile and benchmark artifacts when changing the limiter
   or the streamed-cache size; those parameters are hardware-sensitive.

## Appendix A. Evidence Index

| Evidence | Location |
|---|---|
| v4 throughput, 36 cells and 12 pixel checks | [`destination-forward-v4-release-throughput`](../benchmark-results/destination-forward-v4-release-throughput-20260805/report.md) |
| v4 perf/PCM memory profile | [`destination-forward-v4-memory-profile`](../benchmark-results/destination-forward-v4-memory-profile-20260805/report.md) |
| v4 R1 stability and self-control | [`v4-r1-stability`](../benchmark-results/destination-forward-v4-r1-stability-20260805/report.md), [`v4-self-control`](../benchmark-results/destination-forward-v4-self-control-b7-r1-20260805/report.md) |
| CPU specialization decisions and targeted profiles | [`cpu-optimization decision report`](../benchmark-results/cpu-optimization-20260804/decision-report.md) |
| Original `descale` release comparison | [`release-benchmark.json`](../benchmark-results/release-benchmark-20260805/release-benchmark.json), [`release-benchmark.md`](release-benchmark.md) |
| CPU executor implementation | [`cpu_executor.cpp`](../src/cpu_executor.cpp), [`cpu_executor_avx2.cpp`](../src/cpu_executor_avx2.cpp), [`cpu_packed.hpp`](../src/cpu_packed.hpp) |
| Plugin dispatch and memory limiter | [`vs_plugin.cpp`](../src/vs_plugin.cpp) |
| Public executor and conversion declarations | [`engine.hpp`](../include/dsmvc/engine.hpp) |
| Source/decision history | Git commits listed in Section 2; local and mounted Codex history described in Section 2.2 |
