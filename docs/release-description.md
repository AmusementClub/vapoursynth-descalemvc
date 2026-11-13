# dsmvc v0.1.4

This release makes the CUDA and Vulkan backends faster and cheaper on the
CPU, and adds two ways to move frames without staging copies: GPU frames on
VapourSynth R81 and direct transfers on vapoursynth-classic. The filter API,
precision policy, CPU and Metal paths, and automatic backend selection are
unchanged, so no script needs to change.

## Highlights

- **Faster GPU backends on ordinary host frames.** Against v0.1.3, CUDA is 7
  to 17% faster with about half the CPU time per frame, and Vulkan is 1.3x to
  3.4x faster.
- **GPU frames (VapourSynth R81).** A `std.GPUUpload()` clip is processed on
  the core's own Vulkan device, or by CUDA through exported memory, and stays
  on the GPU. dsmvc's own GPU work runs at 2300 to 9900 fps at 1080p.
- **Direct transfers on vapoursynth-classic (R57.A14 and later).** Used
  automatically: rows are copied straight between the core's frame memory and
  the device, without staging copies, at 1.4x to 3.5x the v0.1.3 frame rate.
  `DSMVC_CUDA_DIRECT_IO=0` or `DSMVC_VULKAN_DIRECT_IO=0` turns them off.
- **Fixes.** A deadlock between CUDA filters that share one source under
  concurrent requests, a crash or hang at process exit while NVIDIA's
  background shader compile was still running, and a crash at exit under
  Vulkan synchronization validation.

## What changed inside

- Float32 solves with half bandwidth 1, 3, 5 or 7 run in 2 to 32 chunks per
  row or column on both backends, instead of one thread walking each vector
  end to end. GPU time for a 1080p Float32 frame on CUDA drops from 0.53 ms
  to 0.09 ms. Other plans and Float64 keep the sequential solve.
- Vulkan submits all planes of a YUV or RGB frame in one command buffer, with
  the planes' dispatches overlapping on the GPU, and no longer runs
  transpose-only passes.
- CUDA sleeps instead of spinning a CPU core while it waits, and copies
  16-bit and Float32 frames straight from frame memory by default.
- Large planes are copied between frame memory and staging with non-temporal
  stores on x86-64.

[`docs/gpu-performance.md`](https://github.com/AmusementClub/vapoursynth-descalemvc/blob/v0.1.4/docs/gpu-performance.md)
has every change with its own measurement, what bounds each route now, the
fixes, and all switches.
[`docs/frame-io.md`](https://github.com/AmusementClub/vapoursynth-descalemvc/blob/v0.1.4/docs/frame-io.md)
describes GPU frames and direct transfers.

## Performance (RTX 5080)

Both plugins were built from source with the same options and measured on one
host: RTX 5080 (driver 615.71.09, also driving the desktop), Ryzen 9 5950X,
Linux. The workload is Debicubic 1920x1080 to 1280x720 on fresh frames from a
synthetic source (4K: 3840x2160 to 2560x1440), 16 requests in flight, default
settings. Figures are fps, the median of five alternating runs. Decoding and
GetNative-style candidate scans are outside this comparison.

### Host frames (VapourSynth R80)

| Format | CUDA v0.1.3 | CUDA v0.1.4 | | Vulkan v0.1.3 | Vulkan v0.1.4 | |
|---|---:|---:|---:|---:|---:|---:|
| Float32 gray | 725 | 841 | 1.16x | 591 | 858 | 1.45x |
| Float32 RGB | 229 | 269 | 1.17x | 195 | 261 | 1.34x |
| YUV420 8-bit | 1722 | 2009 | 1.17x | 746 | 2243 | 3.01x |
| YUV420 16-bit | 913 | 1058 | 1.16x | 603 | 1086 | 1.80x |
| Float32 gray, 4K | 185 | 201 | 1.08x | 145 | 191 | 1.32x |
| YUV420 8-bit, Despline64 | 1822 | 1946 | 1.07x | 625 | 2110 | 3.37x |

CPU time per frame, with both builds paced to the same frame rate, falls by
45 to 57% on CUDA and by 23 to 34% on Vulkan. The CPU backend is unchanged
(1130 against 1134 fps on Float32 gray).

Float32 host frames are now bound by host copies and PCIe on both backends,
not by the GPU. Vulkan gains most on 8-bit YUV, where v0.1.3 was bound by its
single queue.

### GPU frames (VapourSynth R81)

GPU frames are new in this release, so there is no v0.1.3 figure. On a
GPU-resident clip (uploaded once, nothing downloaded):

| Filter, format | CUDA | Vulkan |
|---|---:|---:|
| Debilinear, Float32 gray | 8042 | 9914 |
| Debicubic, Float32 gray | 7907 | 7616 |
| Despline64, Float32 gray | 7961 | 4555 |
| Debicubic, YUV420 8-bit | 2798 | 5282 |
| Debicubic, YUV420 16-bit | 2780 | 4929 |
| Debicubic, Float32 RGB | 2714 | 3454 |
| Debicubic, Float32 gray, 4K | 2317 | 2906 |

Between `std.GPUUpload` and `std.GPUDownload`, the core's transfers set the
pace: one Debicubic on Float32 gray runs at about 1075 fps on either backend,
and a chain of three Debicubic runs at 1461 fps on CUDA and 1747 fps on
Vulkan. 8-bit YUV420 runs at 2259 fps on CUDA and 3539 fps on Vulkan. Output
is bit-identical to the host path on every tested route.

### vapoursynth-classic (R57.A15)

"Staged" is v0.1.4 with direct transfers turned off.

| Format | CUDA v0.1.3 | CUDA staged | CUDA v0.1.4 | Vulkan v0.1.3 | Vulkan staged | Vulkan v0.1.4 |
|---|---:|---:|---:|---:|---:|---:|
| Float32 gray | 699 | 829 | 1406 | 592 | 818 | 1220 |
| Float32 RGB | 229 | 265 | 437 | 192 | 257 | 396 |
| YUV420 8-bit | 1761 | 2075 | 2446 | 744 | 2184 | 2586 |

Output digests are identical with direct transfers on and off for every route
tested.

### Outputs

CPU output is byte-identical to v0.1.3. Of 40 GPU digests (five kernels, four
formats, both backends, 32 frames each), 34 are identical to v0.1.3. The six
that differ are Float32 Lanczos3 and Spline64 cases, and
`DSMVC_GPU_SOLVE_PARTITION=0` reproduces the v0.1.3 digest for each. The
chunked solve is exact algebra; only the Float32 rounding of the state carried
between chunks can differ. The largest difference from the sequential solve
on the full-size routes tested is 4.8e-7, against the 2e-5 tolerance of the
integration tests.

## Compatibility and packages

- Existing arguments, `opt` values, Python enum values, precision settings
  and backend selection keep their meaning. The Float32 scalar/SIMD and mixed
  CPU/GPU tolerance contracts are unchanged.
- The plugin still declares API 4.0 and loads on API 4 cores, including
  vapoursynth-classic. The Linux x64 and Windows x64 packages are built
  against the API 4.3 headers of R81 and include GPU frames, which are
  enabled at run time on R81 only (`DSMVC_GPU_FRAMES=force` allows another
  core). On any other core the same binary behaves as before.
- On vapoursynth-classic R57.A14 and later, the GPU backends keep frame
  memory registered with the device: page-locked for CUDA, imported for
  Vulkan, up to half of the core's frame cache
  (`DSMVC_CUDA_DIRECT_IO_MAX_MB`, `DSMVC_VULKAN_DIRECT_IO_MAX_MB`). If a block
  cannot be registered, that frame is staged as before.
- A Vulkan slot now holds the workspace of all planes of a frame: about three
  times the previous figure for RGB (245 MB per slot at 4K).
- On Linux with an NVIDIA driver, a Vulkan process may wait at exit for the
  driver's shader compile to finish (about 4 s, and only when the driver's
  shader cache holds an interrupted compile). `DSMVC_VULKAN_TEARDOWN_WAIT=0`
  disables the wait.
- Every change has a switch that restores the previous behavior
  (`DSMVC_GPU_SOLVE_PARTITION=0`, `DSMVC_VULKAN_FRAME_SUBMIT=0`,
  `DSMVC_VULKAN_FOLD_TRANSPOSE=0`, `DSMVC_HOST_COPY_STREAM_MIN=0`,
  `DSMVC_CUDA_WAIT=spin`, `DSMVC_CUDA_HOST_TRANSFER=staging-decoupled`,
  `DSMVC_CUDA_DIRECT_IO=0`, `DSMVC_VULKAN_DIRECT_IO=0`); the
  linked pages list them all.
- Packages: Linux x64 CPU/CUDA/Vulkan, Linux arm64 CPU/NEON, Windows x64
  CPU/CUDA/Vulkan, and macOS arm64 CPU/Metal, as in v0.1.3.

**Evidence boundaries.** All GPU measurements and the GPU-frame,
direct-transfer and exit tests ran on Linux with one RTX 5080. The Windows
handle paths for GPU frames and direct transfers are written but were not
exercised; direct transfers are on by default there too, and
`DSMVC_CUDA_DIRECT_IO=0` / `DSMVC_VULKAN_DIRECT_IO=0` restore staging. AMD
and Intel Vulkan drivers and multi-GPU hosts are untested; the test suite
also passes on Mesa lavapipe under the Vulkan validation layer.
