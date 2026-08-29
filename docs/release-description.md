# dsmvc v0.1.1

This release adds JET v12-compatible `blur` kernel stretching and substantially
improves Vulkan throughput, especially for GetNative-style dynamic filter
graphs. The default `blur=1.0` path preserves the existing kernel behavior.

## Highlights

- Adds `blur` to every fixed-kernel filter, generic `Descale`, the native API,
  and the Python wrappers.
- Uses JET v12 stretching semantics: `ceil(base_support * blur)` support and
  `kernel(distance / blur)` weights, with the original Lanczos taps window.
- Adds optimized Vulkan fixed-band execution through H11, including
  blur-driven H9 plans.
- Extends ARM64 NEON wide-band specialization through H9 for blur-expanded
  plans.
- Caches successful Vulkan availability checks once per process, avoiding
  repeated instance creation and device enumeration in dynamic filter graphs.

## Vulkan Performance

Fixed-band figures are five alternating R8T8 samples of a 4,000-frame
BlankClip workload, comparing generic and optimized Vulkan builds from the
same blur-capable source snapshot. The largest measured gain is **1.96x**.

| Band | Representative kernel | Generic Vulkan | v0.1.1 Vulkan | Gain |
|---|---|---:|---:|---:|
| H1 | Bilinear | 398.169 fps | 437.292 fps | +9.8% |
| H3 | Bicubic | 435.561 fps | 615.293 fps | +41.3% |
| H5 | Lanczos3 | 352.860 fps | 613.897 fps | +74.0% |
| H7 | Spline64 | 297.074 fps | 581.746 fps | +95.8% |
| H9 | Lanczos5 | 306.813 fps | 487.834 fps | +59.0% |
| H11 | Lanczos6 | 269.071 fps | 428.799 fps | +59.4% |
| H9 with blur | Spline64 `blur=1.25` | 305.726 fps | 486.045 fps | +59.0% |

Active GPU utilization averaged 97.26% at a 2812 MHz median SM clock; the
blurred H9 run measured 97.57% at 2820 MHz.

For the complete 30,800-candidate GetNative graph, every final thread level is
the median of three fresh VSPipe processes after one warmup and an idle GPU
gate.

| Threads | Three v0.1.1 samples (candidates/s) | Median | Pre-cache Vulkan | Gain |
|---|---|---:|---:|---:|
| R1T1 | 124.103 / 123.195 / 124.007 | **124.007** | 47.117 | **2.632x** |
| R8T8 | 357.684 / 357.701 / 357.537 | **357.684** | 133.789 | **2.674x** |
| R16T16 | 354.490 / 354.225 / 354.461 | **354.461** | 135.136 | **2.623x** |
| R32T32 | 353.604 / 353.068 / 352.886 | **353.068** | 134.730 | **2.621x** |

The construction bottleneck was repeated Vulkan instance/device probing. A
2,000-filter creation test improves from 63.645 s to 0.386 s (**164.9x**).
The separately paired R32T32 baseline was 134.865 candidates/s, making the
final result **2.618x** faster under the exact A/B comparison. R32 averaged
98.41% active GPU utilization at a 2820 MHz median SM clock. The
availability-cache change leaves fixed-plan H5/H9/H11 throughput within
1.009x-1.024x of the pre-fix candidate.

## Compatibility

- Existing calls retain `blur=1.0` behavior; blur is applied entirely during
  plan construction with no extra frame or memory round trip.
- Healthy geometries agree with the Float64 reference to normal Float32
  rounding; integer output remains within 1 code value of the same-precision
  CPU reference.
- Validation passed ctest 35/35 and full VapourSynth CUDA/Vulkan integration
  on the RTX 5080 host.

CUDA and Vulkan remain explicit backends. See the [README](../README.md) and
[full benchmark report](release-benchmark.md) for API details, absolute
performance tables, and methodology.
