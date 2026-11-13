# Frame I/O: classic direct transfers and GPU frames

dsmvc's CUDA and Vulkan backends can avoid copying frames through staging
buffers in two independent ways. Both are additive: nothing changes for a script
that does not use them, and every unsupported case either keeps the old path or
fails with a message that names the remedy.

This page describes what was built and what was measured.

## 1. Direct transfers on vapoursynth-classic

[vapoursynth-classic](https://github.com/AmusementClub/vapoursynth-classic)
R57.A14 and later expose their frame pool blocks (`getVSFrameMemoryAPI`, the
`VSFrameMemory.h` extension). dsmvc registers (CUDA, `cudaHostRegister`) or
imports (Vulkan, `VK_EXT_external_memory_host`) a block on its **second** use and
then copies rows straight between the block and the device. PCIe transfers still
happen; the host-side staging copies do not.

**Opt-in.** Set `DSMVC_CUDA_DIRECT_IO=1` and/or `DSMVC_VULKAN_DIRECT_IO=1`.

| Variable | Meaning |
|---|---|
| `DSMVC_CUDA_DIRECT_IO`, `DSMVC_VULKAN_DIRECT_IO` | `1` enables. Unset or `0`: unchanged behavior. |
| `DSMVC_CUDA_DIRECT_IO_MAX_MB`, `DSMVC_VULKAN_DIRECT_IO_MAX_MB` | Cap on bytes registered per core. Default: half of the core's frame cache. |
| `DSMVC_CUDA_DIRECT_IO_STATS`, `DSMVC_VULKAN_DIRECT_IO_STATS` | `1` prints registrations, hits, misses, cap refusals and hook releases when a core's last filter is freed. |
| `DSMVC_CUDA_DIRECT_IO_SIDES`, `DSMVC_VULKAN_DIRECT_IO_SIDES` | Bit mask, `1` source, `2` destination (diagnostic). |

An explicit `DSMVC_CUDA_HOST_TRANSFER` keeps its meaning and disables block
registration, so the two mechanisms never overlap. Input and output are checked
independently; a side that is not in a registered block is staged as before.
Any failure to register or import (alignment, no coherent memory type,
allocation limits, cap reached) falls back to staging; results never depend on
registration.

Since the later CUDA transfer default (`DSMVC_CUDA_HOST_TRANSFER` unset means
`auto`, see [gpu-performance.md](gpu-performance.md)), 16-bit and Float32 2D
frames are copied by the driver straight from frame memory on any core. The
driver still transfers registered blocks directly, but the executor no longer
looks them up, so the statistics show no hits or misses for those frames; the
block lookup now matters for 8-bit frames and the rows-only and columns-only
routes. The table below was measured before that default, against staging.

Defaults that were measured, not assumed: an 8-bit CUDA source stays staged
(a direct 8-bit upload was ~13% slower than the packed pinned upload on the test
machine), everything else uses both sides.

### Measured (RTX 5080, Ryzen 9 5950X, driver 615.71.09, classic R57.A15)

1080p to 720p, 32 requests in flight, 32 core threads, fresh frames per request,
median of 3-5 alternating runs. The GPU also drives the desktop, so treat small
differences as noise.

| Route | CUDA staged | CUDA direct | Vulkan staged | Vulkan direct |
|---|---:|---:|---:|---:|
| Float32 gray 2D | 688 fps | 1375 (2.0x) | 582 | 876 (+51%) |
| Float32 RGB 2D | 228 | 434 (1.9x) | 191 | 282 (+47%) |
| Float32 gray 4K to 1440p | 185 | 325 (+76%) | 145 | 260 (+80%) |
| 16-bit YUV420 2D | 904 | 1288 (+42%) | 624 | 639 (flat) |
| 8-bit YUV420 2D | 1722 | 1935 (+12%) | 753 | 748 (flat) |
| Float32 rows only | 1315 | 1309 (flat) | 609 | 990 (+63%) |

The existing per-call `DSMVC_CUDA_HOST_TRANSFER=registered` mode measured 444 fps
on the gray case. Output digests are identical with direct transfers on and off
for every route (2D, one axis, 8/16-bit, Float64, pool churn).

## 2. GPU frames (VapourSynth R81 and later)

With a GPU resident clip (`clip.std.GPUUpload()`), dsmvc runs on the GPU without
moving the image to the host:

```python
gpu = src.std.GPUUpload()
a = core.dsmvc.Debicubic(gpu, 1280, 720, backend="vulkan")
b = core.dsmvc.Delanczos(a, 960, 540, backend="cuda")   # same physical GPU
result = b.std.GPUDownload()
```

| Request on a GPU clip | Behavior |
|---|---|
| `backend="auto"` or `"vulkan"` | Kernels run on the core's own Vulkan device; output stays on the GPU. |
| `backend="cuda"` | CUDA reads and writes the same buffers through exported memory. The CUDA device must be the core's GPU (matched by UUID), else an error. |
| `backend="cpu"` (or `metal`) | A deliberate boundary: `std.GPUDownload` once, then the requested host path. |
| Core other than R81, or `DSMVC_GPU_FRAMES=0` | The clip is downloaded with a logged warning and processed on the host. The Vulkan table is never touched. `DSMVC_GPU_FRAMES=force` allows an untested core. |
| Unsupported format | An error that names the remedy; never a silent CPU solve. |

Supported on GPU frames: Float32 clips (2D, one axis), 8-16 bit integer clips when
both width and height change, Float64 plans (`f64mode=2` and the automatic
retention), with the same results, `_Range`/`_ColorRange` handling and Float64
nonfinite failures as the host path. Not supported yet: one-axis integer GPU
clips (they need a conversion subgraph) and CUDA/Vulkan on a different GPU than
the core's.

### Compatibility

The plugin is compiled against the 4.3 headers (for the table layout) but still
declares API 4.0, so one binary loads on VapourSynth R78, vapoursynth-classic,
and R80+. The residency-aware signatures (`vnode:all`) are registered only when
the host reports API 4.3. `VSVULKANAPI` has no version of its own, so GPU frames
are enabled for tested cores only; the tested core is **R81** (header sha256
`56f0cb5a...` for `VSVulkan4.h`, `d079c0e1...` for `VapourSynth4.h`).

The plugin builds the GPU path when `VSVulkan4.h` is next to the VapourSynth
headers and Vulkan is enabled. A CUDA-only build registers the plain signatures
and does not take GPU clips (CUDA on GPU frames needs the Vulkan table of the
same build). macOS (CPU/Metal) builds never see it.

### Measured

Three chained filters, 1080p Float32, GPU clip vs host frames (RTX 5080):

| | Host frames | GPU frames |
|---|---:|---:|
| Vulkan: throughput | 226 fps | 430 fps (1.9x) |
| Vulkan: PCIe per frame (to GPU / from GPU) | 18.9 / 19.3 MB | 9.9 / 3.4 MB |
| CUDA: throughput | 273 fps | 353 fps (1.3x) |
| CUDA: PCIe per frame (to GPU / from GPU) | 18.3 / 15.4 MB | 10.4 / 4.2 MB |

The resident figures are one upload and one download for the whole chain. Output
is bit-identical to the host path on every tested route (2D/rows/columns, gray,
RGB, YUV444/420, 8/10/16-bit, odd byte counts, Float64, mixed
Vulkan-CUDA-Vulkan chains, fan-out under 16 concurrent requests with a 128 MB
frame cache, out-of-order requests, import cache limited to one entry).

### How it works

- **Vulkan:** the executor adopts the core's `VkDevice` and queue (every submit
  is bracketed by the core's queue lock) and compiles the same SPIR-V on it. Input
  planes are copied pitch-aware into the existing workspace after a device-side
  wait on the plane's ready pair; results are copied into the planes of a
  `newGPUVideoFrame`. A runtime lives exactly as long as the last filter using it,
  and its packed plans are purged with it, so it never outlives the core's device.
- **CUDA:** an input frame is taken over with `getExportableFrameFilter`, its
  producers are waited for on the host (`waitGPUFrame`), and each allocation is
  imported once (`cudaImportExternalMemory`, cached by the core's stable
  `memoryId`, bounded by `DSMVC_CUDA_IMPORT_LIMIT`, default 64). Where the core
  exports its timeline semaphores the producers are waited for on the CUDA
  stream instead (see [gpu-performance.md](gpu-performance.md)). The output is
  written by CUDA and the stream is synchronized before the frame is returned.
  File descriptors are consumed or closed on every path. The node requests its
  input `rpStrictSpatial`, so a frame this filter alone consumes is handed over in
  place; when something else holds the frame (fan-out, a cached source) the core
  copies it once per frame on the GPU and shares that copy.

## Verification and what was not done

The original integration suite (`vs_integration.py`, baseline descale plugin)
passes on R78, R80 and R81; the default paths are within noise of the original
commit (ratios 0.989-1.005 over six cases). Not done: Windows, AMD/Intel, multiple
GPUs, low-VRAM behavior, device loss, `compute-sanitizer`, and benchmarks of real decoded content or
GetNative-style candidate scans.

## Known limits and open items

- Direct transfers were run on Linux only. The Windows handle paths are written
  but not exercised.
- GPU frames: executions are synchronous; there are no producer pairs and no
  cross-API semaphores yet, so a chain does not overlap work across filters.
  Memory that dsmvc allocates on the core's device is not yet charged to the
  core's VRAM budget (`reserveGPUMemory`).
- Transformed-input content caching is disabled on the GPU path.
- One unexplained segfault was seen once, in the staged Vulkan path (direct
  transfers off), while running the route matrix on the RTX 5080. It did not
  reproduce in more than 450 further runs of the same sequences, including a
  150-run stress with core dumps enabled; no core was captured, so the cause is
  unknown. It was most likely the exit crash in NVIDIA's shader compiler
  later found and fixed (see "Crash or hang at exit" in
  [gpu-performance.md](gpu-performance.md)).
- Only an RTX 5080 was available; AMD/Intel Vulkan and multi-GPU hosts are
  untested.

## Tests

| Test | What it covers |
|---|---|
| `dsmvc_host_blocks_tests` | Block registry: second-use policy, cap, bounds, stale addresses, hooks, drops, concurrency (no GPU needed). |
| `benchmarks/classic_direct_io_matrix.sh <bench> <plugin> <cuda\|vulkan> [frames]` | Staged vs direct digests and statistics for 10 routes on a classic core. Build the host with `benchmarks/classic_frame_bench.cpp`. |
| `tests/vs_gpu_frames.py <plugin>` (ctest `dsmvc_vs_gpu_frames` with `-DDSMVC_VS_GPU_PYTHON=<R81 python>`) | GPU frames against host frames on both backends, chains, fan-out, concurrency, integer, Float64, descriptors. |
