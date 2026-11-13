# GPU performance

What the v0.1.4 GPU work changed in the CUDA and Vulkan backends, and what it
measures. Every table was measured on 2026-11-13 with one harness.

**Setup.** RTX 5080 (driver 615.71.09, also driving the desktop), Ryzen 9
5950X, Linux. `benchmarks/classic_frame_bench.cpp` is a native VapourSynth
host whose source makes a fresh frame per request. Debicubic 1920x1080 to
1280x720 unless a row says otherwise (4K: 3840x2160 to 2560x1440), 16 requests
in flight, default settings. Builds run alternately; figures are fps, median
of five runs (three for classic and for single switches).

## Results

### Host frames (VapourSynth R80)

| Route | v0.1.3 | v0.1.4 | fps | CPU per frame¹ |
|---|---:|---:|---:|---:|
| CUDA, Float32 gray | 725 | 841 | +16% | -48% |
| CUDA, Float32 RGB | 229 | 269 | +17% | -45% |
| CUDA, 8-bit YUV420 | 1722 | 2009 | +17% | -57% |
| CUDA, 16-bit YUV420 | 913 | 1058 | +16% | -49% |
| CUDA, Float32 gray, 4K | 185 | 201 | +8% | |
| CUDA, 8-bit YUV420, Despline64 | 1822 | 1946 | +7% | |
| Vulkan, Float32 gray | 591 | 858 | +45% | -34% |
| Vulkan, Float32 RGB | 195 | 261 | +34% | -23% |
| Vulkan, 8-bit YUV420 | 746 | 2243 | 3.0x | -26% |
| Vulkan, 16-bit YUV420 | 603 | 1086 | +80% | -34% |
| Vulkan, Float32 gray, 4K | 145 | 191 | +32% | |
| Vulkan, 8-bit YUV420, Despline64 | 625 | 2110 | 3.4x | |
| CPU backend, Float32 gray (control) | 1130 | 1134 | 0% | |
| CPU backend, 8-bit YUV420 (control) | 1765 | 1799 | +2% | |

¹ Process CPU time per frame with both builds paced to the same frame rate
(a Python script on R80, `BlankClip` source, ten seconds of frames, median of
three). Unpaced comparisons mislead: the faster build saturates host memory
bandwidth and every copy takes more cycles.

### vapoursynth-classic (R57.A15)

"Direct" is the opt-in direct transfer of [frame-io.md](frame-io.md)
(`DSMVC_CUDA_DIRECT_IO=1`, `DSMVC_VULKAN_DIRECT_IO=1`).

| Route | v0.1.3 | v0.1.4 | v0.1.4 direct | Direct against v0.1.3 |
|---|---:|---:|---:|---:|
| CUDA, Float32 gray | 699 | 829 | 1406 | 2.0x |
| CUDA, Float32 RGB | 229 | 265 | 437 | 1.9x |
| CUDA, 8-bit YUV420 | 1761 | 2075 | 2446 | 1.4x |
| Vulkan, Float32 gray | 592 | 818 | 1220 | 2.1x |
| Vulkan, Float32 RGB | 192 | 257 | 396 | 2.1x |
| Vulkan, 8-bit YUV420 | 744 | 2184 | 2586 | 3.5x |

### GPU frames (VapourSynth R81)

v0.1.3 has no GPU frames, so "before" is the first GPU-frame build, from
before the changes on this page.

**GPU-resident:** the source is uploaded once and looped on the GPU and
nothing is downloaded, so this is the throughput of dsmvc's own GPU work.

| Filter, format | CUDA before | CUDA now | | Vulkan before | Vulkan now | |
|---|---:|---:|---:|---:|---:|---:|
| Debilinear, gray | 841 | 8042 | 9.6x | 812 | 9914 | 12.2x |
| Debicubic, gray | 1495 | 7907 | 5.3x | 1635 | 7616 | 4.7x |
| Delanczos, gray | 1240 | 7763 | 6.3x | 1519 | 6023 | 4.0x |
| Despline36, gray | 1237 | 7802 | 6.3x | 1525 | 6023 | 3.9x |
| Despline64, gray | 1128 | 7961 | 7.1x | 1345 | 4555 | 3.4x |
| Debicubic, gray, width only | 1984 | 7970 | 4.0x | 3161 | 7996 | 2.5x |
| Debicubic, 8-bit YUV420 | 1447 | 2798 | 1.9x | 897 | 5282 | 5.9x |
| Debicubic, 16-bit YUV420 | 1424 | 2780 | 2.0x | 886 | 4929 | 5.6x |
| Debicubic, RGB | 1150 | 2714 | 2.4x | 558 | 3454 | 6.2x |
| Debicubic, gray, 4K | 627 | 2317 | 3.7x | 644 | 2906 | 4.5x |

**Between `std.GPUUpload` and `std.GPUDownload`:**

| Clip | CUDA before | CUDA now | | Vulkan before | Vulkan now | |
|---|---:|---:|---:|---:|---:|---:|
| Float32 gray | 1076 | 1073 | 0% | 1085 | 1081 | 0% |
| Float32 RGB | 330 | 329 | 0% | 336 | 335 | 0% |
| 8-bit YUV420 | 1420 | 2259 | +59% | 862 | 3539 | 4.1x |
| 16-bit YUV420 | 1376 | 1353 | -2% | 852 | 1468 | +72% |
| Three Debicubic in a row, gray² | 472 | 1461 | 3.1x | 520 | 1747 | 3.4x |

² To 1600x900, 1280x720 and 960x540, from a Python script on R81.

A single filter on Float32 frames does not gain, because the core's upload
and download bound it. The chain of three now runs as fast as one filter.

### What each change contributes

v0.1.4 with one switch set to its old behavior, against the default.

| Change | Route | Old | Now | |
|---|---|---:|---:|---:|
| Partitioned solve | CUDA, resident, gray | 4429 | 7917 | +79% |
| | CUDA, host, 8-bit YUV420 | 1726 | 2003 | +16% |
| | CUDA, host, gray | 790 | 837 | +6% |
| | Vulkan, resident, gray | 1775 | 7621 | 4.3x |
| | Vulkan, host, 8-bit YUV420 | 1221 | 2228 | +82% |
| | Vulkan, host, gray | 833 | 858 | +3% |
| Vulkan without transpose passes | resident, gray | 5410 | 7608 | +41% |
| | resident, gray, 4K | 1698 | 2915 | +72% |
| | resident, gray, width only | 3178 | 7977 | 2.5x |
| One Vulkan submission per frame | resident, 8-bit YUV420 | 3079 | 5274 | +71% |
| | resident, RGB | 2554 | 3488 | +37% |
| | host, 8-bit YUV420 | 1752 | 2254 | +29% |
| CUDA waits for GPU frames on the stream | resident, gray | 5137 | 7956 | +55% |
| | resident, 8-bit YUV420 | 2727 | 2806 | +3% |
| Streaming host copies | Vulkan, host, gray | 694 | 848 | +22% |
| | Vulkan, host, RGB | 213 | 261 | +23% |
| | CUDA, host, 8-bit YUV420 | 1919 | 2009 | +5% |
| CUDA driver copy for 16-bit and Float32 | host, gray | 877 | 841 | -4% |
| | host, 16-bit YUV420 | 1023 | 1060 | +4% |
| CUDA blocking waits | host, gray | 839 | 840 | 0% |
| | host, 8-bit YUV420 | 2069 | 2013 | -3% |

The last two are there for CPU time, not frame rate: the spin took 46% of the
process's CPU on 8-bit YUV, and the driver copy used 38% less CPU than dsmvc's
staging on Float32 gray (profiles taken during the work).

## What changed

**Partitioned solve (CUDA and Vulkan).** The old solve walked each row or
column end to end in one thread: a 1080p frame offered 720 to 1920 threads to
84 SMs, and they waited on memory latency. `src/solve_partition.hpp` splits
each vector into 2 to 32 chunks of at least 32 samples. Each chunk is solved
from a zero carry-in, one thread per vector carries the true state from chunk
to chunk with small transfer matrices (computed once per plan, in double
precision), and each chunk is solved again from its true carry-in. GPU time
per 1080p Float32 gray frame on CUDA: 0.53 ms to 0.09 ms (nsys).

- It is exact algebra; only Float32 rounding of the carried state can differ.
  Output is bit-identical on 12 of 14 full-size routes tested, and the largest
  difference is 4.8e-7 (a near-identity 1080 to 1070 plan) against the 2e-5
  tolerance of the integration tests.
- Used for Float32 plans with half bandwidth 1, 3, 5 or 7 whose vectors hold
  two chunks. Other plans, Float64 and CUDA's row-major horizontal layout keep
  the sequential kernels.
- A Vulkan device must allow 32 x chunks invocations per workgroup and
  4 KiB x half_bandwidth of shared memory, else it uses the sequential
  shaders.

**Vulkan without transpose passes.** The right-hand side reads either layout,
so a Float32 source is read as uploaded and the vertical pass reads the
horizontal result where it lies. Two of the six dispatches of a 2D descale
only transposed; both are gone (an integer source keeps the first, which
converts it). Width-only descales now use the partitioned solve too.

**One Vulkan submission per frame.** The planes of a YUV or RGB frame are
recorded into one command buffer, step by step (upload, right-hand side,
solve, ..., download) with each step covering all planes and one barrier per
step. The planes' dispatches then overlap on the GPU, which is most of the
gain; recording plane after plane gained only 4 to 7%. Float64 plans, one-axis
routes and frames that do not fit one slot run per plane as before. A slot's
workspace now holds all planes: about 245 MB per slot for 4K RGB, against 82.

**CUDA waits.** A slot sleeps on a `cudaEventBlockingSync` event instead of
spinning in `cudaStreamSynchronize`. On GPU frames, the input's producer is
waited for on the CUDA stream through the core's exported timeline semaphore
(`cudaWaitExternalSemaphoresAsync`) instead of a host wait, which cost a third
of the time; the host wait remains the fallback.

**CUDA host transfers.** By default, 16-bit and Float32 2D frames are copied
by the driver straight from frame memory, and 8-bit frames go through dsmvc's
pinned staging.

**Streaming host copies.** `src/host_copy.*` copies planes of 1 MiB or more
with non-temporal stores on x86-64. glibc uses ordinary stores below 3/4 of
the shared cache (24 MB on the 5950X), which reads every destination line
first: a third of a copy's traffic on a host bound by memory bandwidth. Vulkan
on host frames spent 93% of its CPU in `memmove` with the GPU 52% busy.

**Smaller.** GPU-frame planes with tight rows are read and written in place.
Vulkan staging is no longer cleared before it is overwritten (14% of the
process's CPU). `AxisPlan::valid()` remembers a successful check instead of
rescanning every coefficient per frame and plane (2.2%).

## Environment switches

The defaults are the measured configuration; each switch restores the
previous behavior.

| Variable | Default | Meaning |
|---|---|---|
| `DSMVC_GPU_SOLVE_PARTITION` | on | `0` uses the sequential solve kernels. |
| `DSMVC_GPU_SOLVE_MIN_CHUNK` | `32` | Shortest chunk, 8 to 4096. Shorter chunks give more parallelism and more rounding difference. |
| `DSMVC_HOST_COPY_STREAM_MIN` | 1 MiB | Smallest plane, in bytes, copied with non-temporal stores. `0` never streams. |
| `DSMVC_CUDA_WAIT` | blocking | `spin` restores the spinning wait. |
| `DSMVC_CUDA_HOST_TRANSFER` | `auto` | `staging-decoupled` is the previous default; `staging`, `pageable` and `registered` keep their meaning. |
| `DSMVC_CUDA_GPU_FRAME_WAIT` | stream | `host` waits with `waitGPUFrame`. |
| `DSMVC_VULKAN_FOLD_TRANSPOSE` | on | `0` restores the transpose passes and the sequential width-only solve. |
| `DSMVC_VULKAN_FRAME_SUBMIT` | on | `0` submits once per plane. |
| `DSMVC_VULKAN_TEARDOWN_WAIT` | on | `0` disables the wait for the driver's compile threads (see below). |
| `DSMVC_VULKAN_TEARDOWN_TRACE` | off | `1` reports each such wait. |

## Where the time goes now

- **Host frames: host copies and PCIe.** The GPU is 52 to 58% busy on Float32
  frames. This is why Float32 RGB and 4K gain least, and why classic direct
  transfers still double the frame rate.
- **CUDA on GPU frames: the core's `exportGPUPlane`.** One call takes 43 us,
  calls do not overlap, and every plane of every frame needs two. That is
  about 16000 exports per second: 8000 fps for one plane at any size up to
  1080p, 2800 for three. Removing it needs the core to report a plane's
  `memoryId` and offset without a handle.
- **Vulkan: about 35 us per submission that does not overlap** with other
  submissions (`vkQueueSubmit` itself is 7 us). A 1080p gray frame takes
  130 us of the queue, so 95 us is work, and the work is throughput: three
  times the rows or columns take 2.5 to 2.6 times as long.
- **The solve is 64% of a frame's GPU time on CUDA.** It reads every value
  four times and writes it twice, so it runs at memory throughput.
- **Kernel launches do not matter.** Six launches cost 22 us per frame, so
  CUDA graphs were not worth adding.

## Fixes found on the way

**CUDA deadlock with shared inputs.** Rows-only, columns-only and 8-bit 2D
executions took a slot before joining the shared transposed input, so a
consumer could wait for its producer while holding the slot the producer
needed. Several filters on one source with concurrent requests hung for good.
Every route now waits for its producer before taking a slot.

**Crash or hang at exit in NVIDIA's shader compiler (Vulkan, host frames).**

- *Cause.* NVIDIA compiles shaders on its own threads after
  `vkCreateComputePipelines` returns. A process that ends during a compile
  leaves an incomplete entry in the driver's shader cache. Every later process
  then recompiles that pipeline about 3 s after it starts, for several
  seconds, and crashes or hangs if it exits meanwhile, which leaves the entry
  incomplete again. Failures therefore came in bursts. The partitioned-solve
  shader made the window wide (about 6 s for Despline36).
- *Fix.* When the last dsmvc filter on the owned Vulkan device is freed,
  dsmvc waits until the driver's threads have been idle for 100 ms, at most
  20 s; with nothing running the check costs about 2 ms. The shader's two
  passes are also one rolled loop now, which shortens the recompile to about
  4 s.
- *Result.* With a poisoned cache (`__GL_SHADER_DISK_CACHE_PATH` at a
  directory where a Despline36 run was killed 1.5 s in), failed exits went
  from 25 of 30 to 0 of 30 on R80, and from 18 of 20 to 0 of 20 on classic.
  Each run also completes the cache entry, so the next process starts
  normally.
- *Limits.* Hosts that exit without freeing their filters are not covered: an
  exit handler is too late. GPU frames use the core's device and get no wait
  (0 of 20 failed there). Tested on Linux with driver 615.71.09 only; other
  platforms do not wait.
- *Did not help:* creating every pipeline at start-up,
  `vkGetPipelineCacheData`, empty dispatches,
  `VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT`, disabling the disk cache,
  `__GL_NO_DSO_FINALIZER=1`, never destroying the runtime. A wait capped at
  3 s made it worse (16 of 30): giving up mid-compile is the worst moment to
  tear down.

**Crash at exit under Vulkan synchronization validation.** The runtime called
`vkDeviceWaitIdle` in an exit handler, after the validation layer's static
state was gone. The device is now waited for when the last filter is freed:
48 of 48 clean exits, against 0 of 12.

## Verification

- 63 ctest tests pass, including the partitioned solve against the CPU
  backend at full size, and seven filters sharing one source under 24
  concurrent requests (Float32, 16-bit, 8-bit). The previous build hangs in
  that test on CUDA.
- Output digests of 140 host routes (5 kernels, 4 formats, 2D, odd sizes, one
  axis, both backends) are equal with the host-copy and Vulkan-pass changes on
  and off, except rows-only Despline64 by rounding (3.6e-7 from the CPU
  backend either way). One submission per frame is bit-identical on and off.
- Staged and direct digests are equal on all classic routes, both backends.
- Vulkan under the Khronos validation layer with synchronization validation:
  no errors on host, direct-transfer and GPU-frame routes.
- The benchmark on this page ran about 720 processes without a failed or hung
  run.

## Not done

- **A queue per Vulkan slot.** Implemented and removed: after the partitioned
  solve, eight queues measured +9% on 8-bit frames, +1% on 16-bit and -5% on
  Float32, with 3 to 11% more CPU per frame.
- **CUDA: one wait per frame, U and V in one launch, no transpose passes.**
  CUDA still processes planes one call at a time and keeps its two transposes
  (12% of its GPU time). On GPU frames the core's exports are the bound.
- **Folding the right-hand side into the solve** would save one read in
  seven.
- **Staging in cache-sized chunks on Vulkan.** The driver does this for CUDA;
  Vulkan still packs whole frames.
- **Float64 kernels** are unchanged.
- **Other GPUs.** Everything here was measured on one RTX 5080. The Vulkan
  limits check was not exercised on a device that needs it.
