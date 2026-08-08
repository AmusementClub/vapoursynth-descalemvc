# Descale MVC Release Benchmark

## Executive Summary

This package compares the current Release build against the original descale plugin on the same Digimon source, VapourSynth runtime, decoder, geometry, and thread configurations.
All performance and error results below were freshly measured. The current plugin was built for Apple arm64 with the isolated NEON/FMA executor and generic `-O3 -flto=full` Release flags. The opt-in build also contains the experimental Metal backend measured below.

| Workload | Result |
|---|---:|
| E2E getfnative candidates | 492.604 new CPU candidates/s at R32T32; full Metal sweep unavailable |
| Fixed kernel coverage | 40 algorithm/thread/implementation cases, 4,000 frames each |
| BlankClip kernel coverage | 80 implementation/thread/kernel cases, 8,000 frames each |
| CPU error coverage | 34,101 candidates across three recipes; Metal full-sweep error unavailable |
| Metal coverage | 17 explicit CPU/Metal cells at full resources |

At R1T1, the current Release is substantially faster on the complete candidate scans: `getfnative` 19.686 -> 58.822 candidates/s (2.99x), `getfnative_v2` 1.42x, and `selectkernel` 2.55x.
At R32T32, the gains narrow to 1.47x-2.34x because the workload reaches this machine's shared memory data-movement ceiling. The fixed-kernel R8-R32 results show the same convergence: available memory stays high, so the bottleneck is local memory bandwidth and/or cache/DRAM access and queueing latency rather than capacity.
A DDR5 platform is therefore expected to improve the high-thread results by raising the memory-system ceiling, especially when channel configuration and timings are favorable. The gain should be treated as an upper-bound improvement opportunity, not a guaranteed linear speedup, because the graph still contains planner, synchronization, and frame-movement overhead.

## Test System and Run Configuration

| Item | Configuration |
|---|---|
| CPU | `Apple M4 Max (16C/16T)` |
| OS | `Darwin 27.0.0 arm64`, ` ` |
| Memory | `128.0 GiB` physical memory at report generation |
| VapourSynth | `Core R78; API R4.2; API R3.6` |
| Input | `[LoliHouse] DIGIMON BEATBREAK - 40 [WebRip 1080p HEVC-10bit AAC SRTx2].mkv`, 1920x1080; supplied Digimon 1080p HEVC-10bit MKV |
| Source filter | `ffms2` |
| Source decoder options | decoder `default`, prefer_hw `0`, RAP verification `-1` |
| Descale geometry | base `1778x1000`, native target `1440x810` |
| Thread sweep | `R1T1`, `R8T8`, `R16T16`, `R32T32`; each cell uses `core.num_threads=N` and `--requests N` |
| Performance repetition | One fresh VSPipe process per implementation/case/thread cell; wall time includes decode, graph setup, filtering, PlaneStats, and shutdown |
| BlankClip throughput | In-memory `std.BlankClip`, 1920x1080 GRAYS, fixed 810p CPU geometry, 8,000 frames per cell; Metal supplement uses 1692x952 |
| Error sweep | R32T32 recipe, `4` worker processes, `1` worker thread per process |

The reference scripts are the three supplied `.vpy` files. They use `core.lsmas.LWLibavSource` and `muf.getnative`; the release benchmark uses the explicitly supplied Digimon MKV and the source filter shown above, then expands the same descale/reconstruction/statistics graph so old and current plugin namespaces can be selected independently. The target experimental Metal backend is restricted to the measured `1920x1080 -> 1692x952` fixed recipe, so it cannot execute the complete arbitrary-height E2E candidate graphs.

## E2E Case Definitions

The measured graph starts with `source -> ShufflePlanes(plane=0, GRAY) -> resize.Point(format=GRAYS)`. For each candidate it calls the old namespace (`core.descale`) or current namespace (`core.dsmvc`, `backend=cpu`) through `Debilinear`, `Debicubic`, `Delanczos`, `Despline16`, or `Despline36`, reconstructs with the matching `core.resize.*` kernel, then applies `std.Expr`, a 5-pixel border crop, and `PlaneStats`. The output is statistics, not an encoded video stream.

| Case | Scenario | Reference call shape | Measured candidate space |
|---|---|---|---|
| `getfnative` | Full non-vertical GetNative candidate scan | `muf.getnative(src, rescaler, src_heights=arange(700, 980, 0.1), base_height=1000)` | frame 12493; 11 scalers x 2,800 heights = 30,800 candidates |
| `getfnative_v2` | Vertical-only GetNative candidate scan | `muf.getnative(src, rescaler, src_heights=arange(840, 880, 0.1), base_height=1000, vertical_only=True)` | frame 358; 8 scalers x 400 heights = 3,200 candidates |
| `selectkernel` | Kernel-parameter selection at a fixed height | `muf.getnative(src, src_heights=719.8, base_height=1000, ex_thr=0.012, rescalers=...)` | frame 1111; bilinear + 10x10 Bicubic b/c grid = 101 candidates |

`getfnative` is the broad normal search: it scans both height and scaler family on a non-vertical geometry. `getfnative_v2` is the narrower vertical-only search. `selectkernel` holds height at 719.8 and scans kernel parameters, so it isolates kernel-selection cost from height search.

## Build and Provenance

- Source SHA-256: `864d552f8e2ead057ebd2c202c7580442a5f22c8acecd08167eb8a07110d1bf4`
- Source filter: `ffms2`
- Build: native Apple `arm64` Release; NEON/FMA is isolated to `cpu_executor_neon.cpp`
- Optimization: generic `-O3 -flto=full`; no model-specific `-mcpu`, PGO, or fast-math flags
- Link: Full LTO with the API4-only `VapourSynthPluginInit2` export
- Metal: experimental Apple arm64 opt-in build; CPU-only builds do not contain the backend

## E2E Thread Scaling

![E2E thread scaling; full Metal sweep unavailable](../release-benchmark-arm64-svg/e2e-scaling.svg)

The chart and CPU columns report candidates per second for the complete candidate graph. They include planner/cache work, FrameEval, reconstruction, Expr, PlaneStats, frame delivery, and VSPipe process overhead. The old and existing CPU values below are frozen. `newmetal` is `n/a` because explicit Metal rejects the arbitrary candidate geometries in this full sweep; the smoke rerun reached the backend's geometry validation before a complete candidate graph could run.

| Case | R1T1 old -> newcpu -> newmetal (CPUx / Metalx) | R8T8 old -> newcpu -> newmetal (CPUx / Metalx) | R16T16 old -> newcpu -> newmetal (CPUx / Metalx) | R32T32 old -> newcpu -> newmetal (CPUx / Metalx) |
|---|---:|---:|---:|---:|
| `getfnative` | 19.686 -> 58.822 -> n/a (2.99x / n/a) | 141.459 -> 412.927 -> n/a (2.92x / n/a) | 205.361 -> 565.377 -> n/a (2.75x / n/a) | 210.479 -> 492.604 -> n/a (2.34x / n/a) |
| `getfnative_v2` | 44.420 -> 63.126 -> n/a (1.42x / n/a) | 320.716 -> 450.845 -> n/a (1.41x / n/a) | 456.624 -> 637.019 -> n/a (1.40x / n/a) | 424.790 -> 624.565 -> n/a (1.47x / n/a) |
| `selectkernel` | 21.240 -> 54.102 -> n/a (2.55x / n/a) | 117.257 -> 223.028 -> n/a (1.90x / n/a) | 161.850 -> 260.534 -> n/a (1.61x / n/a) | 153.153 -> 263.074 -> n/a (1.72x / n/a) |

## E2E Error Comparison

The CPU error sweep evaluates every candidate in each recipe on the same training frame. Metrics compare old and existing CPU reconstructed output against the source after the benchmark's 5-pixel border crop. The separate Metal level is shown below; it is unavailable for the full arbitrary-height recipes in this worktree.

### CPU error

| Case | Candidates | Best old | Best current | Changed | Max output abs | Max reconstruction abs |
|---|---:|---|---|---|---:|---:|
| `getfnative` | 30,800 | bilinear@979.2 (0.00053924) | bilinear@979.2 (0.00053924) | False | 0.0309417 | 8.16584e-06 |
| `getfnative_v2` | 3,200 | bicubic_b1.0_c0.0@876.7 (0.000476052) | bicubic_b1.0_c0.0@876.7 (0.000476052) | False | 2.98023e-06 | 2.38419e-07 |
| `selectkernel` | 101 | bicubic_b0.0_c0.0@719.8 (0.00122794) | bicubic_b0.0_c0.0@719.8 (0.00122793) | False | 1.43051e-06 | 4.76837e-07 |

### Metal error

Explicit `backend=metal` rejects `getfnative` candidates outside the fixed `1920x1080 -> 1692x952` recipe, while `getfnative_v2` and `selectkernel` do not contain that supported geometry. Therefore no full Metal error level can be reported without changing the target backend implementation.

| Case | Candidates | Best old | Best newmetal | Changed | Max output abs | Max reconstruction abs |
|---|---:|---|---|---|---:|---:|
| `getfnative` | 30,800 | n/a | n/a | n/a | n/a | n/a |
| `getfnative_v2` | 3,200 | n/a | n/a | n/a | n/a | n/a |
| `selectkernel` | 101 | n/a | n/a | n/a | n/a | n/a |

### Consolidated CPU per-algorithm minima

Each row groups all parameter variants of one algorithm family and keeps the best old/current candidate within that family. For example, the 100 Bicubic variants in `selectkernel` become one row. `Delta MAE` and `Delta height` are `current - old`; a negative MAE is an improvement.

| Case | Algorithm family | Candidates | Old best (candidate; height / MAE) | Current best (candidate; height / MAE) | Delta MAE | Delta height | Candidate changed | Height changed |
|---|---|---:|---|---|---:|---:|---|---|
| `getfnative` | `bicubic` | 14,000 | `bicubic_b0.0_c0.5@979.3`; 979.3 / 0.000544565 | `bicubic_b0.0_c0.5@979.3`; 979.3 / 0.000544565 | -2.38789e-11 | +0.0 | False | False |
| `getfnative` | `bilinear` | 2,800 | `bilinear@979.2`; 979.2 / 0.00053924 | `bilinear@979.2`; 979.2 / 0.00053924 | +0 | +0.0 | False | False |
| `getfnative` | `lanczos2` | 2,800 | `lanczos2@979.2`; 979.2 / 0.00054352 | `lanczos2@979.2`; 979.2 / 0.00054352 | +2.08621e-11 | +0.0 | False | False |
| `getfnative` | `lanczos3` | 2,800 | `lanczos3@979.2`; 979.2 / 0.000584545 | `lanczos3@979.2`; 979.2 / 0.000584545 | +5.22201e-11 | +0.0 | False | False |
| `getfnative` | `lanczos4` | 2,800 | `lanczos4@979.3`; 979.3 / 0.000617304 | `lanczos4@979.3`; 979.3 / 0.000617304 | +1.12194e-11 | +0.0 | False | False |
| `getfnative` | `spline16` | 2,800 | `spline16@979.2`; 979.2 / 0.000559219 | `spline16@979.2`; 979.2 / 0.000559219 | +3.13889e-11 | +0.0 | False | False |
| `getfnative` | `spline36` | 2,800 | `spline36@979.2`; 979.2 / 0.000579655 | `spline36@979.2`; 979.2 / 0.000579655 | +6.82371e-11 | +0.0 | False | False |
| `getfnative_v2` | `bicubic` | 2,000 | `bicubic_b1.0_c0.0@876.7`; 876.7 / 0.000476052 | `bicubic_b1.0_c0.0@876.7`; 876.7 / 0.000476052 | +3.35503e-12 | +0.0 | False | False |
| `getfnative_v2` | `bilinear` | 400 | `bilinear@877.5`; 877.5 / 0.000483636 | `bilinear@877.5`; 877.5 / 0.000483636 | +0 | +0.0 | False | False |
| `getfnative_v2` | `spline16` | 400 | `spline16@877.4`; 877.4 / 0.000477543 | `spline16@877.4`; 877.4 / 0.000477543 | -4.29881e-12 | +0.0 | False | False |
| `getfnative_v2` | `spline36` | 400 | `spline36@876.8`; 876.8 / 0.000476866 | `spline36@876.8`; 876.8 / 0.000476866 | +2.9356e-12 | +0.0 | False | False |
| `selectkernel` | `bicubic` | 100 | `bicubic_b0.0_c0.0@719.8`; 719.8 / 0.00122794 | `bicubic_b0.0_c0.0@719.8`; 719.8 / 0.00122793 | -5.87782e-09 | +0.0 | False | False |
| `selectkernel` | `bilinear` | 1 | `bilinear@719.8`; 719.8 / 0.00123996 | `bilinear@719.8`; 719.8 / 0.00123996 | +0 | +0.0 | False | False |

## Fixed Kernel Throughput

Each cell is `old FPS -> new CPU FPS -> new Metal FPS (CPU multiplier / Metal multiplier)`. The old and existing CPU values remain the original fixed 810p measurements. The rerun Metal values use the target backend's only supported fixed recipe, `1920x1080 -> 1692x952`, with the same source and thread requests; `n/a` means that kernel is not supported by experimental Metal.

[Open the full fixed-kernel scaling chart](../release-benchmark-arm64-svg/fixed-kernel-scaling.svg)

| Kernel | R1T1 old -> newcpu -> newmetal (CPUx / Metalx) | R8T8 old -> newcpu -> newmetal (CPUx / Metalx) | R16T16 old -> newcpu -> newmetal (CPUx / Metalx) | R32T32 old -> newcpu -> newmetal (CPUx / Metalx) |
|---|---:|---:|---:|---:|
| `bilinear` | 158.496 -> 963.109 -> 861.637 (6.08x / 5.44x) | 866.063 -> 1107.740 -> 1032.706 (1.28x / 1.19x) | 796.648 -> 1029.778 -> 956.591 (1.29x / 1.20x) | 826.897 -> 1051.358 -> 992.604 (1.27x / 1.20x) |
| `bicubic (0, 0.5)` | 62.520 -> 930.238 -> 837.276 (14.88x / 13.39x) | 440.183 -> 1046.563 -> 1041.149 (2.38x / 2.37x) | 513.271 -> 1002.715 -> 986.844 (1.95x / 1.92x) | 522.842 -> 1041.293 -> 988.708 (1.99x / 1.89x) |
| `lanczos2` | 63.393 -> 934.913 -> n/a (14.75x / n/a) | 434.147 -> 1035.687 -> n/a (2.39x / n/a) | 518.879 -> 1019.059 -> n/a (1.96x / n/a) | 535.333 -> 1032.159 -> n/a (1.93x / n/a) |
| `lanczos3` | 47.340 -> 634.932 -> 543.630 (13.41x / 11.48x) | 310.286 -> 866.993 -> 849.056 (2.79x / 2.74x) | 381.292 -> 857.290 -> 813.506 (2.25x / 2.13x) | 392.117 -> 887.585 -> 845.422 (2.26x / 2.16x) |
| `lanczos4` | 39.036 -> 566.974 -> n/a (14.52x / n/a) | 260.827 -> 795.033 -> n/a (3.05x / n/a) | 325.738 -> 804.930 -> n/a (2.47x / n/a) | 330.851 -> 809.096 -> n/a (2.45x / n/a) |
| `lanczos5` | 31.400 -> 393.203 -> n/a (12.52x / n/a) | 217.551 -> 675.718 -> n/a (3.11x / n/a) | 275.435 -> 665.788 -> n/a (2.42x / n/a) | 284.424 -> 680.164 -> n/a (2.39x / n/a) |
| `lanczos6` | 33.638 -> 346.550 -> n/a (10.30x / n/a) | 231.585 -> 609.767 -> n/a (2.63x / n/a) | 280.403 -> 622.174 -> n/a (2.22x / n/a) | 280.636 -> 626.820 -> n/a (2.23x / n/a) |
| `spline16` | 64.375 -> 934.795 -> n/a (14.52x / n/a) | 456.179 -> 1026.481 -> n/a (2.25x / n/a) | 546.771 -> 1021.833 -> n/a (1.87x / n/a) | 540.066 -> 1061.465 -> n/a (1.97x / n/a) |
| `spline36` | 47.163 -> 639.231 -> 544.192 (13.55x / 11.54x) | 317.770 -> 850.103 -> 841.202 (2.68x / 2.65x) | 389.112 -> 872.888 -> 830.674 (2.24x / 2.13x) | 402.177 -> 894.668 -> 851.578 (2.22x / 2.12x) |
| `spline64` | 39.025 -> 569.521 -> 482.677 (14.59x / 12.37x) | 259.705 -> 747.721 -> 767.099 (2.88x / 2.95x) | 331.288 -> 811.067 -> 790.177 (2.45x / 2.39x) | 338.321 -> 823.717 -> 783.048 (2.43x / 2.31x) |

## Metal Throughput

Metal is an Apple arm64 opt-in backend. The release-facing measurement uses one full-resource configuration (`R32T32`, the maximum request setting in this benchmark suite). Each cell compares a separate CPU process with a separate explicit Metal process on the same fixed graph and reports `CPU FPS -> Metal FPS (Metal/CPU)`.

Each cell uses 512 frames, 5 measured pairs, and 1 warmup pair.

[Open the full-resource CPU/Metal chart](../release-benchmark-arm64-svg/metal-scaling.svg)

The chart uses one panel per format/kernel and paired bars for CPU and explicit Metal. The table reports all supported cells at the same full-resource setting.

| Format / kernel | Full resources (R32T32) |
|---|---:|
| `GRAYS Float32 / bilinear` | 2170.348 -> 1533.996 (0.707x) |
| `GRAYS Float32 / bicubic` | 1988.178 -> 1316.443 (0.662x) |
| `GRAYS Float32 / spline36` | 1425.543 -> 1220.700 (0.856x) |
| `GRAYS Float32 / lanczos3` | 1457.477 -> 1249.519 (0.857x) |
| `GRAYS Float32 / spline64` | 1287.952 -> 1116.259 (0.867x) |
| `YUV420P8 / bilinear` | 1886.754 -> 1512.784 (0.802x) |
| `YUV420P8 / spline16` | 1359.587 -> 1157.750 (0.852x) |
| `YUV420P8 / bicubic` | 1328.764 -> 1158.222 (0.872x) |
| `YUV420P8 / spline36` | 687.365 -> 1020.699 (1.485x) |
| `YUV420P8 / lanczos3` | 677.872 -> 1005.488 (1.483x) |
| `YUV420P8 / spline64` | 569.829 -> 851.608 (1.494x) |
| `YUV420P10 / bilinear` | 1677.578 -> 1339.596 (0.799x) |
| `YUV420P10 / spline16` | 1235.987 -> 1082.021 (0.875x) |
| `YUV420P10 / bicubic` | 1228.207 -> 1066.563 (0.868x) |
| `YUV420P10 / spline36` | 664.058 -> 942.351 (1.419x) |
| `YUV420P10 / lanczos3` | 664.322 -> 940.883 (1.416x) |
| `YUV420P10 / spline64` | 562.680 -> 800.943 (1.423x) |

Correctness, limited/full range propagation, 1-LSB integer tolerance, narrow/wide batches, cancellation reuse, illegal-route rejection, and Metal-off loading are enforced by the API4 evaluator and plugin integration tests. Metal callbacks complete before returning their frame; output `n` still depends only on input `n`.

## BlankClip Throughput

Each cell is `old FPS -> current Release FPS (speedup)` for 8,000 frames from an in-memory 1920x1080 GRAYS `std.BlankClip` at fixed 810p geometry. There is no decoder, source filter, or input-video content; this isolates the fixed-kernel execution path and VapourSynth frame plumbing.

[Open the blank fixed-kernel scaling chart](../release-benchmark-arm64-svg/blank-fixed-kernel-scaling.svg)

| Kernel | R1T1 | R8T8 | R16T16 | R32T32 |
|---|---:|---:|---:|---:|
| `bilinear` | 166.448 -> 1680.667 (10.10x) | 1256.098 -> 4052.502 (3.23x) | 1855.165 -> 4554.530 (2.46x) | 1862.772 -> 4426.212 (2.38x) |
| `bicubic (0, 0.5)` | 73.912 -> 1543.385 (20.88x) | 506.981 -> 3647.177 (7.19x) | 815.629 -> 4273.838 (5.24x) | 813.587 -> 4092.429 (5.03x) |
| `lanczos2` | 69.261 -> 1542.779 (22.27x) | 473.797 -> 3630.894 (7.66x) | 843.436 -> 4265.832 (5.06x) | 829.630 -> 4199.040 (5.06x) |
| `lanczos3` | 46.542 -> 806.745 (17.33x) | 339.724 -> 1850.526 (5.45x) | 542.648 -> 2272.907 (4.19x) | 542.877 -> 2208.141 (4.07x) |
| `lanczos4` | 38.702 -> 681.405 (17.61x) | 282.410 -> 1532.518 (5.43x) | 441.595 -> 1864.336 (4.22x) | 433.803 -> 1856.372 (4.28x) |
| `lanczos5` | 30.282 -> 364.948 (12.05x) | 232.607 -> 986.474 (4.24x) | 354.159 -> 1139.764 (3.22x) | 352.566 -> 1173.512 (3.33x) |
| `lanczos6` | 30.671 -> 310.141 (10.11x) | 234.797 -> 818.328 (3.49x) | 341.343 -> 985.355 (2.89x) | 344.123 -> 978.756 (2.84x) |
| `spline16` | 66.678 -> 1551.732 (23.27x) | 544.752 -> 3750.744 (6.89x) | 767.984 -> 4224.073 (5.50x) | 836.479 -> 4192.133 (5.01x) |
| `spline36` | 46.716 -> 804.720 (17.23x) | 359.990 -> 2000.195 (5.56x) | 539.218 -> 2219.197 (4.12x) | 542.075 -> 2291.478 (4.23x) |
| `spline64` | 38.285 -> 696.879 (18.20x) | 293.693 -> 1624.460 (5.53x) | 432.330 -> 1847.430 (4.27x) | 438.089 -> 1847.940 (4.22x) |

## Interpretation

The Release build removes the previous build-condition confounder: current and original plugins are now compared with the current plugin compiled using the recommended optimized parameters. Fixed-kernel results expose the long-running executor cost, while E2E candidate scanning also includes graph construction and per-candidate statistics. Their scaling curves therefore answer different performance questions.

The full fixed-kernel table shows where the current executor wins or loses by algorithm and thread count. The error table shows whether those throughput differences change candidate selection or reconstructed output. Together they are the release-facing performance and compatibility record.

## Retained Artifacts

This repository snapshot retains the benchmark result reports and SVG charts only.
Per-cell JSON and CSV files, error shards, command logs, and generated subreports
are intentionally omitted from Git.
