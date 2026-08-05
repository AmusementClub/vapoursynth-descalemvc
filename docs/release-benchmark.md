# Descale MVC Release Benchmark

## Executive Summary

This package compares the current Release build against the original descale plugin on the same Digimon source, VapourSynth runtime, decoder, geometry, and thread configurations.
 The current plugin was built with generic x86-64 Release code and an AVX2/FMA-only executor TU.

| Workload | Result |
|---|---:|
| E2E getfnative candidates | 358.623 candidates/s at R32T32 |
| Fixed kernel coverage | 40 algorithm/thread/implementation cases, 4,000 frames each |
| BlankClip kernel coverage | 80 implementation/thread/kernel cases, 8,000 frames each |
| Error coverage | 34,101 candidates across three recipes |

At R1T1, the current Release is substantially faster on the complete candidate scans: `getfnative` 13.569 -> 178.114 candidates/s (13.13x), `getfnative_v2` 3.10x, and `selectkernel` 6.02x.
At R32T32, the gains narrow to 2.56x-6.45x because the workload reaches this machine's shared memory data-movement ceiling. The fixed-kernel R8-R32 results show the same convergence: available memory stays high, so the bottleneck is local memory bandwidth and/or cache/DRAM access and queueing latency rather than capacity.
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
- Link: version-script export of `VapourSynthPluginInit`, RELRO, NOW, and pthread
- No LTO, PGO, native CPU tuning, or fast-math flags

## E2E Thread Scaling

![E2E thread scaling](e2e-scaling.svg)

The chart reports candidates per second for the complete candidate graph. It includes planner/cache work, FrameEval, reconstruction, Expr, PlaneStats, frame delivery, and VSPipe process overhead.

| Case | R1T1 old -> new | R8T8 old -> new | R16T16 old -> new | R32T32 old -> new |
|---|---:|---:|---:|---:|
| `getfnative` | 13.569 -> 178.114 (13.13x) | 55.093 -> 391.178 (7.10x) | 57.472 -> 373.662 (6.50x) | 55.581 -> 358.623 (6.45x) |
| `getfnative_v2` | 109.425 -> 339.658 (3.10x) | 164.063 -> 442.975 (2.70x) | 163.007 -> 427.313 (2.62x) | 159.571 -> 408.623 (2.56x) |
| `selectkernel` | 18.700 -> 112.489 (6.02x) | 49.605 -> 130.049 (2.62x) | 51.893 -> 129.004 (2.49x) | 51.218 -> 126.175 (2.46x) |

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
| `bilinear` | 905.429 -> 1118.063 (1.23x) | 864.647 -> 1089.204 (1.26x) | 767.385 -> 1059.942 (1.38x) | 717.039 -> 1029.197 (1.44x) |
| `bicubic (0, 0.5)` | 523.091 -> 920.558 (1.76x) | 794.589 -> 1146.916 (1.44x) | 766.976 -> 1092.015 (1.42x) | 715.681 -> 1032.368 (1.44x) |
| `lanczos2` | 524.669 -> 916.095 (1.75x) | 799.986 -> 1139.471 (1.42x) | 771.968 -> 1091.078 (1.41x) | 715.580 -> 1028.773 (1.44x) |
| `lanczos3` | 256.213 -> 642.088 (2.51x) | 785.503 -> 1068.997 (1.36x) | 770.586 -> 1105.402 (1.43x) | 702.963 -> 1040.409 (1.48x) |
| `lanczos4` | 196.204 -> 552.692 (2.82x) | 764.390 -> 973.509 (1.27x) | 766.360 -> 1114.636 (1.45x) | 696.529 -> 1033.407 (1.48x) |
| `lanczos5` | 156.652 -> 428.908 (2.74x) | 727.444 -> 859.585 (1.18x) | 761.272 -> 1082.981 (1.42x) | 693.835 -> 1014.916 (1.46x) |
| `lanczos6` | 129.671 -> 382.282 (2.95x) | 678.308 -> 767.966 (1.13x) | 759.645 -> 1041.396 (1.37x) | 692.513 -> 977.961 (1.41x) |
| `spline16` | 523.460 -> 916.940 (1.75x) | 782.897 -> 1125.032 (1.44x) | 771.402 -> 1092.939 (1.42x) | 717.221 -> 1025.248 (1.43x) |
| `spline36` | 256.828 -> 654.828 (2.55x) | 789.201 -> 1101.847 (1.40x) | 770.546 -> 1112.049 (1.44x) | 705.717 -> 1037.778 (1.47x) |
| `spline64` | 195.770 -> 554.729 (2.83x) | 756.119 -> 992.142 (1.31x) | 768.937 -> 1092.615 (1.42x) | 699.173 -> 1032.725 (1.48x) |
