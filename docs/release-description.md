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
