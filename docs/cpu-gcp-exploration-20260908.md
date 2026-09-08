# CPU exploration on GCP (2026-09-08)

This change retains ordered AVX-512 RHS specialization, wider column groups, and a bounded F32 two-pass route after independent C4/C4A experiments. All percentages below are reductions in elapsed time against the frozen working tree at the start of this task. That baseline already included the earlier CPU and GPU optimizations.

## Retained behavior

- AVX-512 B5/B7 horizontal kernels specialize ascending RHS spans 6/7 and 9/10. All other spans retain the existing accumulation path.
- AVX-512 F32 columns with half-bandwidth 5, 7, or 11 share coefficients across four 16-lane vectors. Incomplete 64-column groups keep the existing 16-column and AVX2/scalar tail handling.
- When the selected CPU path is AVX-512, both F32 axes have one of those three half-bandwidths, output width is at least 64, and input height is at least 16, 2D execution uses separate horizontal/vertical passes and a reusable, 64-byte aligned intermediate. Existing horizontal ISA selection is preserved: Lanczos6 horizontal execution remains AVX2.
- NEON B11 F32 columns use four independent 4-lane accumulators, preserving ascending RHS and descending triangular-solve order, with the existing remainder path.

No public option or build flag was added. F64 and integer dispatch retain their previous paths. The current README describes the updated routing; `docs/release-description.md` remains the historical v0.1.2 description.

## Measured results

| Workload | Lanczos3 | Spline64 | Lanczos6 |
| --- | ---: | ---: | ---: |
| C4 F32 2D engine, active height 951.5 | 28.46% | 36.28% | 28.98% |
| C4 fixed-size VapourSynth Descale | 20.60% | 18.36% | 16.87% |
| C4 GetNative-style candidate graph | 4.07% | 5.91% | 6.36% |
| C4A fixed-size VapourSynth Descale | control | control | 18.76% |
| C4A GetNative-style candidate graph | control | control | 4.38% |

C4 used a Xeon Platinum 8581C (Emerald Rapids), GCC 13.3, and `c4-highcpu-2`. C4A used Neoverse-V2, GCC 15.2, and `c4a-highcpu-2`. Both were Spot VMs restored from the 2026-09-08 verified snapshots. Release builds used the same compiler options within each pair. Native CMake selected AVX2/AVX-512 on C4 and NEON on C4A.

Each cell is the median of three alternating AB/BA/AB pairs. C4 final graph cells used 256 fixed-size frames or 96 sampled heights; C4A used 128 fixed-size frames or 32 sampled heights. The complete candidate graph included Descale, reconstruction, the thresholded difference expression, a five-pixel crop, and PlaneStats. Input was predecoded; video decoding was outside this experiment. These are independent within-host comparisons, not a cross-host throughput comparison.

The C4 candidate-graph paired ranges were Lanczos3 +0.13% to +6.97%, Spline64 -0.81% to +7.02%, and Lanczos6 +6.22% to +7.56%. The short scan measurements have variation; the medians do not establish a minimum improvement for every run.

The isolated AVX-512 RHS screen reduced row time by 8.25% / 4.93%. NEON B11 grouping reduced column time by 40.32% and engine 2D time by 19.60%. These component measurements use separate candidates and must not be added to the final 2D or graph gains.

## Expression-stage opportunity

An independent API4 probe implemented only `abs(a-b) > 0.015 ? abs(a-b) : 0`. On the C4A candidate graph, replacing R75 `std.Expr` with this NEON probe reduced total time by 65.52%, 63.16%, 56.05% for Lanczos3, Spline64, and Lanczos6. The same optimized dsmvc binary was used on both sides. All sampled pixels and PlaneStats matched.

This prototype remains in the experiment artifacts; it is not part of dsmvc or a general Expr replacement. The R75 scalar ARM and x86 JIT paths differed on NaN comparison semantics; the probe was validated against each architecture separately, including threshold neighbors, signed zero, infinities, NaNs, and SIMD tails.

## Correctness and limits

- Linux Release CTest passed 6/6 on both architectures; ASan/UBSan passed 5/5 C++ suites on each. The macOS Release build with Metal enabled passed 13/13; no local Metal performance result was used for admission.
- The x86 route candidate passed 84 complete-buffer comparisons across three kernels, four active heights, and F32/F64/u8/u16 paths. NEON passed 28 corresponding comparisons. Final x86 geometry probes covered 18 combinations from 257x145 through a 3840x2160 tiled plane. All tested complete outputs were byte-identical to their baselines.
- Guard-page tests cover B11 destination sizes 12/13/56, column counts around 16/32/64 and the 1024-column tile threshold, odd strides, padding, changed input on reuse, and wide-band integer paths. A separate baseline/final guard oracle matched 8,603,776 output bytes exactly.
- Final x86 parallel checks submitted 32 requests with four VapourSynth threads, for three kernels and three repetitions per binary. All 576 tested frames and their statistics matched the baseline.
- Timed workloads were pinned to CPU 0. Final C4 admission required CPU 1 to be at least 99% idle during the timed region; outer process telemetry was also retained. Earlier whole-process gates and all invalid observations remain archived. Sixteen of eighteen final geometry cells obtained a valid paired run. Tail/Lanczos6 and downscale/Spline64 remained inconclusive for performance; their numeric checks passed.
- The 4K input was a tiled version of the same source plane, and the small input was a crop. These test layout/size behavior, not additional native-resolution videos. The source-frame SHA-256 was `d50727fbfe6a6cb47eee915d5e92372292c41f7280d4d80d8db37312817e36bd`.
- Performance evidence applies to the tested C4/C4A configurations. Windows, AMD, other x86 CPUs, other ARM CPUs, and full video decoding were not benchmarked in this task.

Full records, rejected candidates, source manifests, binaries, compiler commands, PMU data, and cleanup evidence are under `artifacts/gcp-perf-explore-20260908/`. The detailed report is `artifacts/gcp-perf-explore-20260908/report.md`.
