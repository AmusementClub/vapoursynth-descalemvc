# dsmvc v0.1.3

This release speeds up selected CPU and GPU Float32 descale paths while
preserving the existing API, precision policy, and automatic backend selection.

## Highlights

- Reduces horizontal RHS work for common AVX2, AVX-512, and NEON kernels,
  with aligned private workspaces and ordered arithmetic.
- Shares coefficients across wider AVX-512 and NEON column groups. Eligible
  AVX-512 Float32 2D operations use separate horizontal and vertical passes;
  Lanczos6 horizontal work retains its AVX2 implementation.
- Enables the validated fixed-span GPU optimization in selected fused
  Float32 routes: CUDA half-bandwidths 5/7, Vulkan 5/7/11, and Metal 5/7/11.
  Other routes retain their dynamic loops. This release adds no tuning switches.
- Consolidates the Metal build option, propagates the macOS deployment target
  to the Metal compiler, and keeps Vulkan policy assertions active in Release.
- Adds buffer-boundary and GPU RHS coverage, including the permitted Vulkan
  fused or separate multiply/add rounding models, and serializes Metal tests that share
  the device, and packages macOS arm64 with a verified 36-entry Metal library.

## CPU vs v0.1.2

All ratios below are **v0.1.2 time / v0.1.3 candidate time** on the same host
and requested ISA. Larger is faster; 1.000x is unchanged. These are fresh
comparisons against the latest published release, not a product of older gains.

The width workload preserves the v0.1.2 release description's 256 rows,
1920 → 1692 geometry, four kernels, Release build, and GCC 13.3. Both hosts
have two vCPUs: Intel Xeon Platinum 8581C and AMD EPYC 9B45. Buffers use the
same controlled alignment and are filled before timing. Five A/B or B/A pairs
use three calls per sample, followed by longer calibrated samples.

The main table uses five longer pairs, approximately 1.0–1.3 seconds or more
per sample. Each accepted group requires CPU1 to be at least 99% idle in
every timed sample. Environment-invalid groups are retained and are not used
for these ratios.

| Filter | Intel AVX2 | Intel AVX-512 | AMD AVX2 | AMD AVX-512 |
|---|---:|---:|---:|---:|
| Debilinear | 1.012x | 1.013x | 0.995x | 0.999x |
| Debicubic | 1.004x | 1.008x | 0.986x | 0.993x |
| Delanczos (`taps=3`) | 1.400x | 1.296x | 1.575x | 1.350x |
| Despline64 | 1.424x | 1.406x | 1.645x | 1.493x |

Debilinear and Debicubic are effectively flat in this workload; AMD Debicubic
has a small measured slowdown (0.986x on AVX2). Lanczos3 and Spline64 improve
on both tested CPUs and both requested ISA paths.

A supplemental predecoded Float32 2D case, 1920×1080 → 1692×952, measures
the full CPU inverse with automatic ISA selection:

| Filter | Intel auto | AMD auto |
|---|---:|---:|
| Delanczos (`taps=3`) | 2.187x | 1.753x |
| Despline64 | 2.169x | 1.740x |
| Delanczos (`taps=6`) | 1.955x | 1.451x |

All measured cross-version outputs are byte-identical. The legacy height
fixture (951 → 952 with automatic precision) selects **F64** for every tested
kernel; its results do not measure the new F32 column route. The supplemental
2D case explicitly checks that both axes use F32. These are engine timings,
not video decode or complete GetNative scan timings.

The [full benchmark report](https://github.com/AmusementClub/vapoursynth-descalemvc/blob/codex/release-v0.1.3/docs/release-benchmark-v0.1.3.md)
includes the original five-by-three results, absolute times, every paired
ratio, environment failures, and reproduction instructions.

## GPU vs v0.1.2 (RTX 5080)

Fresh comparisons use the actual Linux release archives on the same RTX 5080,
NVIDIA driver 595.84, Ryzen 9 5950X, and VapourSynth R78. The v0.1.3 package
is built from `d0ce31fb46e3c8e90cd8fc1307316554d119ba49`. Ratios are the median
of five alternating A/B or B/A pairs: **candidate FPS / v0.1.2 FPS**.
Larger is faster. Every planned case and all five pairs are retained.

Both versions load the same predecoded Float32 input before timing. Fixed
geometry is 1920x1080 to 1692x952, with active height 951.5, symmetric padding,
128 untimed warmup frames and 2,048 measured requests. The candidate graph
scans 512 heights across 700.0-979.9, including reconstruction, error threshold,
crop and PlaneStats. It is a bounded GetNative-style workload. Decode and the
complete historical 30,800-candidate scan are outside this comparison.

### Default RHS selection, one request and one thread

| Kernel | CUDA fixed | Vulkan fixed | CUDA candidate graph | Vulkan candidate graph |
|---|---:|---:|---:|---:|
| Bilinear | 1.001x | 1.002x | 0.989x | 0.991x |
| Bicubic | 1.007x | 1.000x | 0.991x | 0.992x |
| Lanczos3 | 1.002x | 1.002x | 1.008x | 1.026x |
| Spline64 | 1.000x | 0.999x | 1.005x | 1.056x |
| Lanczos6 | 1.000x | 0.998x | 0.994x | 1.039x |
| Lanczos9 | 0.999x | 1.000x | 0.998x | 0.993x |

### Default RHS selection, 32 requests and 32 threads

| Kernel | CUDA candidate graph | Vulkan candidate graph |
|---|---:|---:|
| Lanczos3 | 1.000x | 1.081x |
| Spline64 | 0.996x | 1.145x |
| Lanczos6 | 0.991x | 1.110x |

### Forced fused RHS, fixed geometry, one request and one thread

These cells set the existing RHS policy override to fused execution. Their
ratios describe that specific route; they do not establish the same gain under
automatic RHS selection.

| Kernel | CUDA | Vulkan |
|---|---:|---:|
| Bilinear | 0.999x | 1.001x |
| Bicubic | 1.001x | 1.000x |
| Lanczos3 | 1.103x | 1.166x |
| Spline64 | 1.066x | 1.223x |
| Lanczos6 | 0.998x | 1.231x |
| Lanczos9 | 1.001x | 0.998x |

Default fixed geometry is nearly unchanged across both backends. CUDA default
scans remain close to v0.1.2; Vulkan gains depend on kernel and concurrency.
Small measured slowdowns are retained in the tables. Vulkan Bilinear scan
pairs span 0.892-1.102x around its 0.991x median, so that cell does not establish
a stable change. The full report includes every pair range.

All 672 paired sampled frames match in logical pixel SHA-256 and PlaneStats.
An additional 96 paired cases cover six kernels, CUDA/Vulkan, automatic/fused
RHS, and normal, tail, downscale and 4K dimensions. Every cross-version output
hash matches; repeated outputs are exact and all CPU-scalar comparisons meet
`3e-6`. These samples do not establish that every timed frame was hashed.
The hardware results cover Linux on this RTX 5080; Metal, Windows GPU runtime,
and other GPU architectures retain their separate evidence boundaries.

The [full GPU report](https://github.com/AmusementClub/vapoursynth-descalemvc/blob/codex/release-v0.1.3/docs/release-benchmark-gpu-v0.1.3.md)
provides all paired ranges, absolute FPS, whole-process wall ratios, workload
scripts and output hashes in its linked machine-readable data.

## Compatibility and packages

- Existing `opt=0/1/2/3`, Python enum values, precision settings, and backend
  selection retain their meaning. Explicit AVX-512 still requires the supported ISA.
- The Float32 scalar/SIMD and mixed CPU/GPU tolerance contracts are unchanged.
  Fixed routes remain deterministic; concurrency-dependent mixed routing retains
  the existing numerical tolerance.
- Packages: Linux x64 CPU/CUDA/Vulkan, Windows x64 CPU/CUDA/Vulkan, and macOS
  arm64 CPU/Metal (deployment target macOS 13.3). macOS 13.3 is the target,
  not a claim of testing on that OS version.
- CUDA packages contain native SM75/86/89/120 and PTX75/120 targets. Hardware
  coverage does not imply execution on every compiled GPU architecture.
- `SHA256SUMS` covers all three package archives. The macOS archive also
  includes its source/build manifest and per-file checksums.

The earlier GPU and incremental CPU reports retain their own baselines and
workload limits; their percentages are not cumulative v0.1.2-to-v0.1.3 results.
