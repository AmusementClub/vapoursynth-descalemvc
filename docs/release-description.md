# dsmvc v0.1.4

This release speeds up the CUDA and Vulkan backends and adds two optional ways
to move frames to the GPU without staging copies. The API, precision policy,
CPU and Metal paths, and automatic backend selection are unchanged.

## Highlights

- Solves Float32 plans with half bandwidth 1, 3, 5 or 7 in 2 to 32 chunks per
  row or column on CUDA and Vulkan, instead of one thread walking each vector
  end to end. GPU time for a 1080p Float32 frame on CUDA drops from 0.53 ms to
  0.09 ms. Other plans and Float64 keep the sequential solve.
- Submits the planes of a YUV or RGB frame to Vulkan in one command buffer,
  recorded step by step so the planes' dispatches overlap on the GPU.
- Removes the Vulkan transpose-only passes: the right-hand side reads either
  layout, and a GPU frame's plane is read where it lies.
- Copies large planes between frame memory and staging with non-temporal
  stores on x86-64, and stops clearing Vulkan staging before overwriting it.
- CUDA no longer spins a CPU core while it waits for a stream, and copies
  16-bit and Float32 2D frames straight from frame memory by default.
- **GPU frames (VapourSynth R81):** a `std.GPUUpload()` clip is processed on
  the core's own Vulkan device, or by CUDA through exported memory, and stays
  on the GPU. This needs a build against the API 4.3 headers; see
  "Compatibility and packages".
- **Direct transfers on vapoursynth-classic (R57.A14 and later):** opt-in with
  `DSMVC_CUDA_DIRECT_IO=1` or `DSMVC_VULKAN_DIRECT_IO=1`. Frame pool blocks are
  registered or imported once and rows are copied straight between them and
  the device.
- Fixes found on the way: a deadlock between filters that share one source
  under concurrent requests (CUDA input cache), a crash or hang at process
  exit when NVIDIA's background shader compile was still running, and a crash
  at exit under Vulkan synchronization validation.

[`docs/gpu-performance.md`](https://github.com/AmusementClub/vapoursynth-descalemvc/blob/v0.1.4/docs/gpu-performance.md)
records the profiles, every change with its measurement, what was tried and
dropped, and all switches.
[`docs/frame-io.md`](https://github.com/AmusementClub/vapoursynth-descalemvc/blob/v0.1.4/docs/frame-io.md)
describes GPU frames and direct transfers.

## GPU vs v0.1.3 (RTX 5080)

Both plugins were built from source with the same options on the same host:
RTX 5080, NVIDIA driver 615.71.09, Ryzen 9 5950X, VapourSynth R78. v0.1.3 is
the tagged commit; the candidate is this release's source. Ratios are
**candidate FPS / v0.1.3 FPS**, each the median of five alternating A/B or B/A
runs. Larger is faster.

The workload is a fixed 1920x1080 to 1280x720 descale of fresh host frames
(3840x2160 to 2560x1440 for the 4K row), 32 requests in flight on 32 core
threads, 64 untimed warmup frames and 700 to 4000 measured frames. `gray` is
Float32; `rgb` is Float32 RGB. It measures the plugin with a synthetic source;
decode and GetNative-style candidate scans are outside this comparison. The
GPU also drives the desktop of the test machine.

| Workload | CUDA v0.1.3 | CUDA v0.1.4 | CUDA ratio | Vulkan v0.1.3 | Vulkan v0.1.4 | Vulkan ratio |
|---|---:|---:|---:|---:|---:|---:|
| Debilinear, gray | 713 | 840 | 1.18x | 524 | 861 | 1.64x |
| Debicubic, gray | 715 | 841 | 1.18x | 610 | 852 | 1.40x |
| Delanczos, gray | 723 | 838 | 1.16x | 591 | 858 | 1.45x |
| Despline36, gray | 722 | 839 | 1.16x | 593 | 830 | 1.40x |
| Despline64, gray | 721 | 839 | 1.16x | 589 | 856 | 1.45x |
| Debicubic, YUV420 8-bit | 1716 | 2017 | 1.18x | 745 | 2244 | 3.01x |
| Debicubic, YUV420 16-bit | 913 | 1058 | 1.16x | 603 | 1075 | 1.78x |
| Debicubic, RGB | 229 | 268 | 1.17x | 193 | 262 | 1.36x |
| Despline64, YUV420 8-bit | 1818 | 1947 | 1.07x | 625 | 2108 | 3.37x |
| Debicubic, gray, 4K | 186 | 201 | 1.08x | 145 | 196 | 1.35x |

Float32 host frames are bound by host copies and PCIe on both backends, which
is why all kernels land near the same figure. Vulkan gains most on 8-bit YUV,
where v0.1.3 was bound by its single queue.

**Outputs.** CPU output is byte-identical to v0.1.3. Of 40 GPU digests (five
kernels, four formats, both backends, 32 frames each), 34 are identical to
v0.1.3. The six that differ are Float32 Lanczos3 and Spline64 cases, and
`DSMVC_GPU_SOLVE_PARTITION=0` reproduces the v0.1.3 digest for each: the
chunked solve is exact algebra, and only Float32 rounding of the state carried
between chunks can differ. The largest difference from the sequential solve on
the full-size routes tested is 4.8e-7, against the 2e-5 tolerance of the
integration tests.

### GPU frames (VapourSynth R81)

GPU frames are new, so there is no v0.1.3 figure. Debicubic 1080p to 720p on
a GPU resident clip, fps:

| Format | Vulkan | CUDA |
|---|---:|---:|
| Float32 gray | 7648 | 7977 |
| YUV420 8-bit | 5278 | - |
| YUV420 16-bit | 4993 | - |
| Float32 RGB | 3489 | - |

A chain of three filters on a GPU clip measured 430 fps on Vulkan against 226
fps on host frames, with about half the PCIe traffic. Output is bit-identical
to the host path on every tested route. CUDA figures for YUV and RGB GPU
frames were not remeasured for this release; they are bound by the core's
plane export (see the performance page).

### Direct transfers (vapoursynth-classic R57.A15)

Vulkan with direct transfers on, Debicubic 1080p to 720p, measured 2590 fps on
8-bit YUV420 (2244 staged on R78 above) and 395 fps on Float32 RGB (262
staged). CUDA's new default transfer already copies 16-bit and Float32 frames
from frame memory on any core; direct transfers add to it for 8-bit and
one-axis routes. Output digests are identical with direct transfers on and off for every route
tested.

## Compatibility and packages

- Existing arguments, `opt` values, Python enum values, precision settings and
  backend selection keep their meaning. No script needs to change.
- The Float32 scalar/SIMD and mixed CPU/GPU tolerance contracts are unchanged.
- The plugin still declares API 4.0 and loads on API 4 cores, including
  vapoursynth-classic. GPU frames are compiled in only when the plugin is
  built against headers that include `VSVulkan4.h` (API 4.3) with Vulkan
  enabled, and are enabled at run time on R81 only (`DSMVC_GPU_FRAMES=force`
  allows another core). Packages built against the R57 SDK do not take GPU
  clips.
- A Vulkan slot now holds the workspace of all planes of a frame: about three
  times the previous figure for RGB (245 MB per slot at 4K).
- New switches, all optional: `DSMVC_GPU_SOLVE_PARTITION=0`,
  `DSMVC_VULKAN_FRAME_SUBMIT=0`, `DSMVC_VULKAN_FOLD_TRANSPOSE=0`,
  `DSMVC_HOST_COPY_STREAM_MIN`, `DSMVC_CUDA_HOST_TRANSFER`,
  `DSMVC_CUDA_DIRECT_IO`, `DSMVC_VULKAN_DIRECT_IO`, `DSMVC_GPU_FRAMES`. Each
  restores the previous behavior or enables an opt-in path; the linked pages
  list them all.
- Packages: Linux x64 CPU/CUDA/Vulkan, Linux arm64 CPU/NEON, Windows x64
  CPU/CUDA/Vulkan, and macOS arm64 CPU/Metal, as in v0.1.3.

Evidence boundaries: all GPU measurements and GPU-frame, direct-transfer and
exit tests ran on Linux with one RTX 5080. The Windows handle paths for GPU
frames and direct transfers are written but were not exercised. AMD and Intel
Vulkan drivers and multi-GPU hosts are untested; the test suite also passes
on Mesa lavapipe under the Vulkan validation layer.
