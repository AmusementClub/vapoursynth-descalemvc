# Descale MVC Release Benchmark

## Executive Summary

This package compares the current Release build against the original descale plugin on the same Digimon source, VapourSynth runtime, decoder, geometry, and thread configurations.
 The current plugin was built with generic x86-64 Release code and an AVX2/FMA-only executor TU.

| Workload | Result |
|---|---:|
| E2E getfnative candidates | 361.447 candidates/s at R32T32 |
| Fixed kernel coverage | 40 algorithm/thread/implementation cases, 4,000 frames each |
| BlankClip kernel coverage | 80 implementation/thread/kernel cases, 8,000 frames each |
| Error coverage | 34,101 candidates across three recipes |

At R1T1, the current Release is substantially faster on the complete candidate scans: `getfnative` 13.569 -> 178.796 candidates/s (13.18x), `getfnative_v2` 3.11x, and `selectkernel` 6.07x.
At R32T32, the gains narrow to 2.63x-6.50x because the workload reaches this machine's shared memory data-movement ceiling. The fixed-kernel R8-R32 results show the same convergence: available memory stays high, so the bottleneck is local memory bandwidth and/or cache/DRAM access and queueing latency rather than capacity.
A DDR5 platform is therefore expected to improve the high-thread results by raising the memory-system ceiling, especially when channel configuration and timings are favorable. The gain should be treated as an upper-bound improvement opportunity, not a guaranteed linear speedup, because the graph still contains planner, synchronization, and frame-movement overhead.

## Test System and Run Configuration

| Item | Configuration |
|---|---|
| CPU | `AMD Ryzen 9 5950X 16-Core Processor (16C/32T)` |
| OS | `Ubuntu26.04`, `Linux 7.0.0-28-generic x86_64`, `glibc 2.43` |
| Memory | `32 GiB` DDR4@3600Mhz |
| VapourSynth | `Core R78; API R4.2; API R3.6` |
| Input | 1920x1080; 1080p HEVC-10bit MKV |
| Source filter | `lsmas` |
| Source decoder options | decoder `hevc_cuvid`, prefer_hw `0`, RAP verification `0` |
| Descale geometry | base `1778x1000`, native target `1440x810` |
| Thread sweep | `R1T1`, `R8T8`, `R16T16`, `R32T32`; each cell uses `core.num_threads=N` and `--requests N` |
| Performance repetition | One fresh VSPipe process per implementation/case/thread cell; wall time includes decode, graph setup, filtering, PlaneStats, and shutdown |
| BlankClip throughput | In-memory `std.BlankClip`, 1920x1080 GRAYS, fixed 810p geometry, 8,000 frames per cell; no decoder or source filter |

The reference scripts are the three supplied `.vpy` files. They use `core.lsmas.LWLibavSource` and `muf.getnative`; the release benchmark uses the explicitly supplied Digimon MKV and the source filter shown above, then expands the same descale/reconstruction/statistics graph so old and current plugin namespaces can be selected independently.

## E2E Case Definitions

The measured graph starts with `source -> ShufflePlanes(plane=0, GRAY) -> resize.Point(format=GRAYS)`. For each candidate it calls the old namespace (`core.descale`) or current namespace (`core.dsmvc`, `backend=cpu`) through `Debilinear`, `Debicubic`, `Delanczos`, `Despline16`, or `Despline36`, reconstructs with the matching `core.resize.*` kernel, then applies `std.Expr`, a 5-pixel border crop, and `PlaneStats`. The output is statistics, not an encoded video stream.

| Case | Scenario | Reference call shape | Measured candidate space |
|---|---|---|---|
| `getfnative` | Full non-vertical GetNative candidate scan | `muf.getnative(src, rescaler, src_heights=arange(700, 980, 0.1), base_height=1000)` | frame 12493; 11 scalers x 2,800 heights = 30,800 candidates |
| `getfnative_v2` | Vertical-only GetNative candidate scan | `muf.getnative(src, rescaler, src_heights=arange(840, 880, 0.1), base_height=1000, vertical_only=True)` | frame 358; 8 scalers x 400 heights = 3,200 candidates |
| `selectkernel` | Kernel-parameter selection at a fixed height | `muf.getnative(src, src_heights=719.8, base_height=1000, ex_thr=0.012, rescalers=...)` | frame 1111; bilinear + 10x10 Bicubic b/c grid = 101 candidates |

`getfnative` is the broad normal search: it scans both height and scaler family on a non-vertical geometry. `getfnative_v2` is the narrower vertical-only search. `selectkernel` holds height at 719.8 and scans kernel parameters, so it isolates kernel-selection cost from height search.

## Build and Provenance

- Build: `Release`, CMake platform defaults, with AVX2/FMA isolated to `cpu_executor_avx2.cpp` when the target is x86_64
- Link: version-script export of `VapourSynthPluginInit2`, RELRO, NOW, and pthread
- No LTO, PGO, native CPU tuning, or fast-math flags

## E2E Thread Scaling

![E2E thread scaling](e2e-scaling.svg)

The chart reports candidates per second for the complete candidate graph. It includes planner/cache work, FrameEval, reconstruction, Expr, PlaneStats, frame delivery, and VSPipe process overhead.

| Case | R1T1 old -> new | R8T8 old -> new | R16T16 old -> new | R32T32 old -> new |
|---|---:|---:|---:|---:|
| `getfnative` | 13.569 -> 178.796 (13.18x) | 55.093 -> 400.023 (7.26x) | 57.472 -> 378.583 (6.59x) | 55.581 -> 361.447 (6.50x) |
| `getfnative_v2` | 109.425 -> 340.156 (3.11x) | 164.063 -> 451.760 (2.75x) | 163.007 -> 439.617 (2.70x) | 159.571 -> 419.635 (2.63x) |
| `selectkernel` | 18.700 -> 113.573 (6.07x) | 49.605 -> 135.703 (2.74x) | 51.893 -> 132.263 (2.55x) | 51.218 -> 121.554 (2.37x) |

## E2E Error Comparison

The error sweep evaluates every candidate in each recipe on the same training frame. Metrics compare old and current reconstructed output against the source after the benchmark's 5-pixel border crop.

| Case | Candidates | Best old | Best current | Changed | Max output abs | Max reconstruction abs |
|---|---:|---|---|---|---:|---:|
| `getfnative` | 30,800 | bilinear@979.2 (0.00053924) | bilinear@979.2 (0.00053924) | False | 2.38419e-07 | 2.38419e-07 |
| `getfnative_v2` | 3,200 | bicubic_b1.0_c0.0@876.7 (0.000476052) | bicubic_b1.0_c0.0@876.7 (0.000476052) | False | 0 | 0 |
| `selectkernel` | 101 | bicubic_b0.0_c0.0@719.8 (0.00122793) | bicubic_b0.0_c0.0@719.8 (0.00122793) | False | 0 | 0 |

### Consolidated per-algorithm minima

Each row groups all parameter variants of one algorithm family and keeps the best old/current candidate within that family. For example, the 100 Bicubic variants in `selectkernel` become one row. `Delta MAE` and `Delta height` are `current - old`; a negative MAE is an improvement.

| Case | Algorithm family | Candidates | Old best (candidate; height / MAE) | Current best (candidate; height / MAE) | Delta MAE | Delta height | Candidate changed | Height changed |
|---|---|---:|---|---|---:|---:|---|---|
| `getfnative` | `bicubic` | 14,000 | `bicubic_b0.0_c0.5@979.3`; 979.3 / 0.000544565 | `bicubic_b0.0_c0.5@979.3`; 979.3 / 0.000544565 | +0 | +0.0 | False | False |
| `getfnative` | `bilinear` | 2,800 | `bilinear@979.2`; 979.2 / 0.00053924 | `bilinear@979.2`; 979.2 / 0.00053924 | +0 | +0.0 | False | False |
| `getfnative` | `lanczos2` | 2,800 | `lanczos2@979.2`; 979.2 / 0.00054352 | `lanczos2@979.2`; 979.2 / 0.00054352 | +0 | +0.0 | False | False |
| `getfnative` | `lanczos3` | 2,800 | `lanczos3@979.2`; 979.2 / 0.000584545 | `lanczos3@979.2`; 979.2 / 0.000584545 | +0 | +0.0 | False | False |
| `getfnative` | `lanczos4` | 2,800 | `lanczos4@979.3`; 979.3 / 0.000617304 | `lanczos4@979.3`; 979.3 / 0.000617304 | +0 | +0.0 | False | False |
| `getfnative` | `spline16` | 2,800 | `spline16@979.2`; 979.2 / 0.000559219 | `spline16@979.2`; 979.2 / 0.000559219 | +0 | +0.0 | False | False |
| `getfnative` | `spline36` | 2,800 | `spline36@979.2`; 979.2 / 0.000579655 | `spline36@979.2`; 979.2 / 0.000579655 | +0 | +0.0 | False | False |
| `getfnative_v2` | `bicubic` | 2,000 | `bicubic_b1.0_c0.0@876.7`; 876.7 / 0.000476052 | `bicubic_b1.0_c0.0@876.7`; 876.7 / 0.000476052 | +0 | +0.0 | False | False |
| `getfnative_v2` | `bilinear` | 400 | `bilinear@877.5`; 877.5 / 0.000483636 | `bilinear@877.5`; 877.5 / 0.000483636 | +0 | +0.0 | False | False |
| `getfnative_v2` | `spline16` | 400 | `spline16@877.4`; 877.4 / 0.000477543 | `spline16@877.4`; 877.4 / 0.000477543 | +0 | +0.0 | False | False |
| `getfnative_v2` | `spline36` | 400 | `spline36@876.8`; 876.8 / 0.000476866 | `spline36@876.8`; 876.8 / 0.000476866 | +0 | +0.0 | False | False |
| `selectkernel` | `bicubic` | 100 | `bicubic_b0.0_c0.0@719.8`; 719.8 / 0.00122793 | `bicubic_b0.0_c0.0@719.8`; 719.8 / 0.00122793 | +0 | +0.0 | False | False |
| `selectkernel` | `bilinear` | 1 | `bilinear@719.8`; 719.8 / 0.00123996 | `bilinear@719.8`; 719.8 / 0.00123996 | +0 | +0.0 | False | False |

## Fixed Kernel Throughput

Each cell is `old FPS -> current Release FPS (speedup)` for 4,000 source frames at fixed 810p geometry. This isolates the fixed-kernel path while retaining source decoding and VSPipe frame delivery.

![Fixed kernel thread scaling](fixed-kernel-scaling.svg)


## BlankClip Throughput

Each cell is `old FPS -> current Release FPS (speedup)` for 8,000 frames from an in-memory 1920x1080 GRAYS `std.BlankClip` at fixed 810p geometry. There is no decoder, source filter, or input-video content; this isolates the fixed-kernel execution path and VapourSynth frame plumbing.

![BlankClip fixed kernel thread scaling](blank-fixed-kernel-scaling.svg)

| Kernel | R1T1 | R8T8 | R16T16 | R32T32 |
|---|---:|---:|---:|---:|
| `bilinear` | 905.429 -> 1987.081 (2.19x) | 864.647 -> 1212.995 (1.40x) | 767.385 -> 1205.674 (1.57x) | 717.039 -> 1054.317 (1.47x) |
| `bicubic (0, 0.5)` | 523.091 -> 1405.558 (2.69x) | 794.589 -> 1210.783 (1.52x) | 766.976 -> 1194.648 (1.56x) | 715.681 -> 1043.564 (1.46x) |
| `lanczos2` | 524.669 -> 1406.171 (2.68x) | 799.986 -> 1208.500 (1.51x) | 771.968 -> 1184.898 (1.53x) | 715.580 -> 1043.497 (1.46x) |
| `lanczos3` | 256.213 -> 846.429 (3.30x) | 785.503 -> 1137.909 (1.45x) | 770.586 -> 1174.760 (1.52x) | 702.963 -> 1044.179 (1.49x) |
| `lanczos4` | 196.204 -> 700.510 (3.57x) | 764.390 -> 967.554 (1.27x) | 766.360 -> 1144.399 (1.49x) | 696.529 -> 1035.326 (1.49x) |
| `lanczos5` | 156.652 -> 522.378 (3.33x) | 727.444 -> 870.733 (1.20x) | 761.272 -> 1103.489 (1.45x) | 693.835 -> 1015.063 (1.46x) |
| `lanczos6` | 129.671 -> 434.968 (3.35x) | 678.308 -> 774.256 (1.14x) | 759.645 -> 1057.559 (1.39x) | 692.513 -> 982.050 (1.42x) |
| `spline16` | 523.460 -> 1377.237 (2.63x) | 782.897 -> 1215.002 (1.55x) | 771.402 -> 1197.272 (1.55x) | 717.221 -> 1040.051 (1.45x) |
| `spline36` | 256.828 -> 832.300 (3.24x) | 789.201 -> 1162.324 (1.47x) | 770.546 -> 1179.764 (1.53x) | 705.717 -> 1041.521 (1.48x) |
| `spline64` | 195.770 -> 708.200 (3.62x) | 756.119 -> 1003.110 (1.33x) | 768.937 -> 1136.727 (1.48x) | 699.173 -> 1038.158 (1.48x) |
