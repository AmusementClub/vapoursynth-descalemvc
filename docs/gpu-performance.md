# GPU performance: partitioned solve, waits and transfers

This page records a round of GPU work (November 2026) driven by profiling, and
what it measured. Machine: RTX 5080 (driver 615.71.09), Ryzen 9 5950X,
VapourSynth R78 for host frames and R81 for GPU frames. The GPU also drives the
desktop; figures are medians of three alternating runs against the previous
build.

## What was slow

nsys on a 1080p to 720p Debicubic (Float32 gray, CUDA):

- **GPU time 0.53 ms per frame**, almost all in the two LDL^T solves (0.34 ms
  horizontal, 0.16 ms vertical). The solve walked each row or column from end
  to end in one thread, so a frame offered 720 to 1920 threads to 84 SMs and
  waited on memory latency.
- **Host frames were bound by host memory, not by the GPU.** Packing an 8 MB
  frame into pinned memory took about 8.6 ms of wall time per frame with 16 in
  flight; perf put 51% of the process's CPU in `memmove`.
- **CUDA spun while waiting.** No wait mode was set, so with one context on a
  32-thread CPU `cudaStreamSynchronize` spins: 46% of the process's CPU on
  8-bit YUV.
- **Vulkan serialised its slots.** One queue, and every compute barrier waits
  for all earlier work on it: fps stopped growing at two slots (8-bit YUV 733
  to 743 fps from 2 to 8 slots, against CUDA's 1138 to 1955 from 2 to 8
  streams). The partitioned solve shortens each serialised piece of work,
  which recovered most of this (see "Not done" for the queue experiment).
- **Smaller costs:** the Vulkan backend cleared the whole staging buffer before
  overwriting it (14% of the process's CPU), and `AxisPlan::valid()` rescanned
  every coefficient on every frame and plane (2.2%).

## What changed

### Partitioned solve (CUDA and Vulkan)

`src/solve_partition.hpp` splits each vector into 2 to 32 chunks of at least
32 samples. Each chunk is solved from a zero carry-in, one thread per vector
then carries the true state from chunk to chunk with small transfer matrices
computed once per plan (in double precision), and each chunk is solved again
from its true carry-in; the same for the backward substitution. It is exact
algebra: only Float32 rounding of the carried state can differ from the
sequential solve, and because descale plans forget their carry-in within a
few dozen samples the output is bit-identical on 12 of 14 full-size routes
tested (largest difference 4.8e-7 on a near-identity 1080 to 1070 plan, against
the 2e-5 tolerance of the integration tests). Kernels:
`dsmvc_cuda_solve_chunked_w{1,3,5,7}` and `src/vulkan/shaders/solve_chunked.comp`
(one SPIR-V module per half bandwidth and chunk count, 20 in all, with the
workgroup size fixed in each; a plan's pipeline is created when the plan is
packed, which happens when a filter is created).

Used for Float32 plans with half bandwidth 1, 3, 5 or 7 whose vectors hold two
chunks; other plans, Float64 and the row-major horizontal layout keep the
sequential kernels. A Vulkan device must allow 32 x chunks invocations per
workgroup and 4 KiB x half_bandwidth of shared memory, else the sequential
shaders are used.

| Variable | Meaning |
|---|---|
| `DSMVC_GPU_SOLVE_PARTITION=0` | Use the sequential solve kernels. |
| `DSMVC_GPU_SOLVE_MIN_CHUNK` | Shortest chunk (default 32, 8 to 4096). Shorter chunks give more parallelism and more rounding difference. |

GPU time per 1080p Float32 gray frame (CUDA, nsys): 0.53 ms to 0.09 ms. The
standalone benchmark (`benchmarks/cuda_solve_bench.cu`) has per-plan figures.

### CUDA waits

Each execution slot waits on a `cudaEventBlockingSync` event, so a thread
sleeps instead of spinning. `DSMVC_CUDA_WAIT=spin` restores the old wait.

### CUDA host transfers

`DSMVC_CUDA_HOST_TRANSFER` unset now means `auto`: 16-bit and Float32 frames are
copied by the driver straight from frame memory (the existing `pageable` route;
the driver stages pageable memory in chunks that stay in cache), 8-bit frames
through dsmvc's pinned staging. Measured after the solver change, 1080p 2D:

| | staging | driver (pageable) |
|---|---:|---:|
| Float32 gray | 789 fps | 933 fps, 38% less CPU |
| Float32 RGB | 259 | 304, 47% less CPU |
| Float32 4K gray | 210 | 229, 37% less CPU |
| 16-bit YUV420 | 1057 | 1122, 41% less CPU |
| 8-bit YUV420 | 2113 | 1710 (slower) |

`staging`, `staging-decoupled` (the previous default), `pageable` and
`registered` keep their meaning. Rows-only and columns-only routes always stage.

### Smaller changes

- Vulkan staging: only the bytes between the packed image and the word-aligned
  copy size are zeroed.
- `AxisPlan::valid()` remembers a successful check per object (copies and moves
  start unchecked).

### Fixes found on the way

- **CUDA deadlock with shared inputs.** Rows-only, columns-only and 8-bit 2D
  executions took a slot before joining the shared transposed input, so a
  consumer could wait for its producer while holding the slot the producer
  needed. Several filters on one source with concurrent requests hung for good
  (the previous build hangs in `tests/vs_gpu_concurrency.py`). Every route now
  joins the shared input and waits for its producer before taking a slot, as
  the Vulkan backend already did.
- **Crash or hang at exit in NVIDIA's shader compiler (Vulkan, host
  frames).** Processes sometimes segfaulted or never exited after their last
  frame, on a driver thread in `libnvidia-gpucomp` (`+0x2ab6dd8`, the hang
  spinning next to it) under `libnvidia-glvkspirv` `_nv002nvvm`. Mechanism, as
  measured with driver 615.71.09:
  - NVIDIA compiles shaders on its own threads after
    `vkCreateComputePipelines` returns. When the process ends during a
    compile (a kill, a crash), the driver's shader cache keeps an incomplete
    entry for that pipeline.
  - From then on, every process that creates the pipeline compiles it again
    about 3 s after it starts, for several seconds, and stores the result
    only if the compile finishes. A process that exits during that compile
    crashes or hangs: the exit handlers that run before dsmvc's (static
    objects created later than dsmvc's runtime, likely in the compiler
    itself) destroy state the compiler still uses. The entry stays
    incomplete, so the next process is in the same state. This is why the
    failures came in bursts per filter that ended only when one run outlived
    the compile.
  - The partitioned-solve shader made the window wide: the recompile of the
    Despline36 pipeline took about 6 s. Other shaders compile in a fraction
    of that, which is why the failures were seen with the partition only.

  Reproduction: point `__GL_SHADER_DISK_CACHE_PATH` at an empty directory,
  kill a 2000-frame Despline36 run 1.5 s in, then run it again from a copy of
  that directory each time. Exits failed in 25 of 30 runs on R80 and 18 of 20
  on vapoursynth-classic (each executable has its own cache entry).

  Fix: when the last dsmvc filter on the owned Vulkan device is freed
  (hosts free their filters before they exit, so this is before any exit
  handler), dsmvc waits until the driver's threads have been idle for 100 ms,
  at most 20 s. The driver's threads are those created while dsmvc set up the
  device; a check with nothing running costs about 2 ms. The same wait runs
  again before the device is destroyed, for hosts that exit without freeing
  their filters; that second wait runs among the exit handlers and is not
  shown to help (on R80 the crash happened while it slept), so hosts that
  exit without freeing their filters are not covered. In the reproduction:
  0 of 30 failures on R80 (25 of 30 without the wait), 0 of 20 on classic
  (18 of 20), and every run completed the cache entry, so the next process
  starts normally (exit waits of about 4.2 s, in that state only). A Python
  script that ends without freeing its clip (the clip is freed while the
  interpreter shuts down) waited the same way and completed the entry in 15
  of 15 runs; vspipe did not reach the slow recompile in the reproduction. The
  partitioned shader's two passes are also one rolled loop instead of two
  inlined copies, which shortened that recompile from about 6 s to about 4 s
  at the same frame rate. `DSMVC_VULKAN_TEARDOWN_WAIT=0` disables the wait,
  `DSMVC_VULKAN_TEARDOWN_TRACE=1` reports each wait.

  What did not help, measured: creating every pipeline at start-up (more to
  compile), `vkGetPipelineCacheData`, empty dispatches, pipeline executable
  statistics, `VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT`, disabling the
  disk cache, `__GL_NO_DSO_FINALIZER=1`, never destroying the runtime, and
  freeing the runtime with its last filter (295 ms per filter create and free).
  A wait at exit with a 3 s limit made it worse (16 of 30, mostly hangs):
  giving up in the middle of the compile is the worst moment to tear down. A
  wait at exit alone is not enough on R80 either; the failure happened while
  dsmvc's exit-time wait was still sleeping.

## Measured

Host frames, R78, 1080p to 720p Debicubic (4K: 3840x2160 to 2560x1440), 2000
frames, 16 requests in flight; CPU is process CPU time per frame (it includes
producing the source frames).

| Route | Before | After | fps | CPU per frame |
|---|---:|---:|---:|---:|
| CUDA Float32 gray | 792 fps | 940 fps | +19% | -46% |
| CUDA Float32 RGB | 261 | 304 | +17% | -50% |
| CUDA Float32 4K gray | 209 | 229 | +10% | -44% |
| CUDA 8-bit YUV420 | 1830 | 2107 | +15% | -19% |
| CUDA 16-bit YUV420 | 1000 | 1126 | +13% | -45% |
| Vulkan Float32 gray | 630 | 758 | +20% | see below |
| Vulkan Float32 RGB | 208 | 247 | +19% | see below |
| Vulkan Float32 4K gray | 161 | 205 | +27% | see below |
| Vulkan 8-bit YUV420 | 743 | 1640 | +121% | see below |
| Vulkan 16-bit YUV420 | 615 | 947 | +54% | see below |
| CPU backend (control) | 1338 / 1926 | 1334 / 1986 | 0 / +3% | 0 |

GPU frames, R81 (`std.GPUUpload`, dsmvc, `std.GPUDownload`):

| Chain | Before | After |
|---|---:|---:|
| One Debicubic, CUDA | 1456 fps | 1463 fps |
| One Debicubic, Vulkan | 1529 | 1475 (-3.5%) |
| Three Debicubic (1600x900, 1280x720, 960x540), CUDA | 472 | 1332 (2.8x) |
| Three Debicubic, Vulkan | 521 | 1677 (3.2x) |

A single filter on GPU frames is limited by the core's upload and download, so
it does not gain (Vulkan loses a little: the partitioned solve needs a separate
right-hand-side dispatch); chains gain because each filter's GPU time shrinks.

**CPU per frame at higher fps.** Unpaced, Vulkan used more CPU per frame after
the change (8-bit YUV 1.52 to 2.17 ms). perf shows dsmvc's own samples fell (the
driver's spin in its fence wait and `AxisPlan::valid()` are gone) while
`memmove`/`memset` per frame tripled: at twice the frame rate the process
saturates host memory bandwidth and every copy stalls longer. Paced to the same
frame rate for both builds:

| Route, paced to | Before | After | CPU per frame |
|---|---:|---:|---:|
| CUDA 8-bit YUV420, 1500 fps | 2.39 ms | 1.25 ms | -48% |
| CUDA Float32 gray, 700 fps | 4.35 | 1.83 | -58% |
| Vulkan 8-bit YUV420, 600 fps | 1.15 | 0.99 | -14% |
| Vulkan 16-bit YUV420, 500 fps | 1.90 | 1.48 | -22% |
| Vulkan Float32 gray, 500 fps | 2.42 | 1.85 | -24% |
| Vulkan Float32 RGB, 180 fps | 6.97 | 4.65 | -33% |
| GPU frames, one Debicubic, CUDA, 1200 fps | 3.70 | 2.91 | -21% |
| GPU frames, one Debicubic, Vulkan, 1200 fps | 2.29 | 2.77 | +21% |
| GPU frames, three Debicubic, CUDA, 400 fps | 4.58 | 2.66 | -42% |
| GPU frames, three Debicubic, Vulkan, 450 fps | 2.23 | 1.94 | -13% |

The one increase is not dsmvc's: perf attributes all of the extra time to
`memmove` inside the core's `std.GPUUpload` and `std.GPUDownload`
(`gpuUploadGetFrame`, `gpuDownloadGetFrame`); dsmvc's own samples fell.

## Second round: host copies and Vulkan passes

A review of the round above with perf and `nvidia-smi dmon` (same machine).

**What was slow**

- **Vulkan on host frames spent 93% of its CPU in `memmove`** with the GPU
  52% busy. glibc writes with ordinary stores below 3/4 of the shared cache
  (24 MB on this CPU), so packing an 8.3 MB Float32 plane first read every
  destination line from memory: a third of a copy's traffic on a host that is
  bound by memory bandwidth.
- **Vulkan on GPU frames was bound by the GPU** (96% busy). Two of the six
  dispatches of a 2D descale only transposed, and the plane was copied into
  the workspace first.
- **Rows-only (width-only) descales used the sequential solve** on Vulkan:
  3160 fps on GPU frames against 12000 for columns-only.

**What changed**

- `src/host_copy.*`: both backends copy planes of 1 MiB or more with
  non-temporal stores on x86-64 (`DSMVC_HOST_COPY_STREAM_MIN`, bytes; 0 never
  streams). CUDA's default route for 16-bit and Float32 frames is the
  driver's own copy and does not change.
- Vulkan 2D: the right-hand side reads either layout, so a Float32 source is
  read row-major as uploaded and the vertical pass reads the horizontal
  result where it lies; neither transpose pass remains (an integer source
  keeps the first one, which converts it). For a row-major source the
  right-hand side runs the samples of a row along the subgroup (`rhs.comp`,
  push constant 18) when the plan's half bandwidth is above 1. A GPU frame's
  plane with tight rows is bound and read in place instead of copied.
- Vulkan rows-only: partitioned solve on vector-major data, then one
  transpose. Its output can differ from the sequential solve by rounding, as
  the 2D path's already could (Despline64 and Lanczos4: 3.6e-7 from the CPU
  backend either way).
- `DSMVC_VULKAN_FOLD_TRANSPOSE=0` restores the previous Vulkan passes.

**Measured** (1080p to 720p Debicubic unless named, median of 3, before =
the two switches above set to the old behavior):

| Route | Before | After | |
|---|---:|---:|---:|
| R80 host frames, Vulkan, Float32 gray | 692 fps | 849 | +23% |
| R80 host frames, Vulkan, 8-bit YUV420 | 1638 | 1755 | +7% |
| R80 host frames, Vulkan, 16-bit YUV420 | 899 | 994 | +11% |
| R80 host frames, Vulkan, Float32 RGB | 227 | 261 | +15% |
| R80 host frames, Vulkan, 4K gray | 182 | 195 | +7% |
| R80 host frames, CUDA, Float32 gray | 840 | 834 | -1% |
| R80 host frames, CUDA, 8-bit YUV420 | 1935 | 2006 | +4% |
| R80 host frames, CUDA, 16-bit YUV420 | 1051 | 1052 | 0 |
| classic, Vulkan, staged | 682 | 811 | +19% |
| classic, Vulkan, direct | 1202 | 1216 | +1% |
| R81 GPU frames with upload and download, Vulkan | 1082 | 1080 | 0 |
| R81 resident, Vulkan, Debilinear | 6331 | 9885 | +56% |
| R81 resident, Vulkan, Debicubic | 5372 | 7596 | +41% |
| R81 resident, Vulkan, Delanczos | 4577 | 6016 | +31% |
| R81 resident, Vulkan, Despline16 | 5399 | 7587 | +41% |
| R81 resident, Vulkan, Despline36 | 4614 | 6016 | +30% |
| R81 resident, Vulkan, Despline64 | 3774 | 4538 | +20% |
| R81 resident, Vulkan, width only | 3156 | 7996 | +153% |
| R81 resident, Vulkan, RGB | 1803 | 2549 | +41% |
| R81 resident, Vulkan, 8-bit YUV420 | 2837 | 3053 | +8% |
| R81 resident, Vulkan, 4K | 1712 | 2919 | +71% |

"Resident" is `classic_frame_bench --gpu-resident`: the source is uploaded
once and looped on the GPU and nothing is downloaded.

Checked (again after the third round): 63 ctest tests (new:
`dsmvc_host_copy_tests`), `vs_gpu_frames.py` on R81 with both waits, the classic direct I/O digest matrix on both backends, and output digests
of 140 host routes (5 kernels, 4 formats, 2D, odd sizes, one axis, both
backends) equal with the new paths on and off, except rows-only Despline64 as
noted. `DSMVC_VULKAN_VALIDATION=1` reports nothing on the host routes.

## Third round: where the time goes now

Measured with nsys, timers around single calls, and frames from 64x64 up
(to separate fixed costs from work).

**CUDA on GPU frames**

- `waitGPUFrame` (a host wait with one submission round trip in the core)
  cost a third of the time. The input's producer pair is now waited for on
  the CUDA stream: the core's timeline semaphore is exported once per filter,
  imported (`cudaImportExternalSemaphore`) and queued with
  `cudaWaitExternalSemaphoresAsync`. `DSMVC_CUDA_GPU_FRAME_WAIT=host` keeps
  the host wait, which is also the fallback when the core cannot export
  semaphores or the import fails.
- Float32 planes with tight rows are read and written in place (no
  device-to-device copies).

| GPU-resident, CUDA | Host wait | Now |
|---|---:|---:|
| Debicubic 1080p to 720p | 5112 fps | 7977 (+56%) |
| Despline64 | 5073 | 7924 |
| Width only | 5289 | 7949 |
| 4K to 1440p | 2122 | 2320 |
| 8-bit YUV420 | 2765 | 2815 |
| RGB | 2634 | 2744 |
| CUDA then CUDA chain | 2732 | 4056 (+48%) |
| Vulkan then CUDA chain | 3698 | 4866 (+32%) |

What bounds it now: `exportGPUPlane`. One call takes 43 us alone and the
calls do not overlap (471 us per frame of waiting with 16 requests in flight);
every plane of every frame needs two, because the core hands out an
allocation's identity only together with a new file descriptor. That is about
16000 exports per second: 8000 fps for one plane at any size up to 1080p (a
64x64 frame runs at 8100 fps too), 2800 for three. Removing it needs the core
to report a plane's `memoryId` and offset without a handle.

**CUDA graphs: not worth it.** Six kernel launches cost 22 us per frame
(3.7 us each, uncontended), against 86 us of exports on GPU frames and
transfers on host frames. Launch calls only bound frames far smaller than
video (320x180 host frames: 16700 fps, 60 us of serialized API time each).

**Vulkan**

- A submission has about 35 us of cost that does not overlap with other
  submissions (64x64 frames: 28400 per second with 16 in flight; 64 us of
  latency alone). A 1080p gray frame takes 130 us of the queue, so 95 us is
  work.
- The work is throughput, not latency: three times the rows or columns take
  2.5 to 2.6 times as long. Solving the planes of a frame in one dispatch
  would therefore gain little; one submission per frame saves the 35 us twice
  for YUV: about 70 of 329 us on GPU frames, 70 of 570 us on host frames.
- `vkQueueSubmit` itself is 7 us.

**The solve** is 64% of the GPU time of a frame on CUDA (two of 35 to 45 us
at 1080p). It reads every value four times and writes it twice, so it runs at
memory throughput; folding the right-hand side in would save one read in
seven.

**Host frames** are unchanged by this round: the GPU is 52 to 58% busy on
Float32 frames and the rest is host copies and PCIe.

**Left**

- CUDA Float32 host frames: dsmvc's pinned staging with streaming copies
  measured 874 fps against 829 for the driver's copy. Not changed: one
  measurement, and the driver's route used 38% less CPU before.
- CUDA still has its two transpose passes (12% of its GPU time); they only
  matter above 1080p on GPU frames, where exports are not the bound.

### One Vulkan submission per frame

A YUV or RGB frame whose planes are 2D Float32 or integer work is now recorded
into one command buffer and submitted once, on host frames and GPU frames
(`Executor::inverse_2d_frame`; Float64 plans, one-axis routes, single planes
and the other backends run per plane as before, and so does a frame that does
not fit one slot). `DSMVC_VULKAN_FRAME_SUBMIT=0` restores a submission per
plane.

Recording one plane after another in the shared command buffer gained only 4
to 7%: a barrier orders everything after it behind everything before it, so
the planes still ran one at a time. The frame is therefore recorded step by
step (upload, right-hand side, solve, ... download), each step for all planes,
followed by one barrier command that covers the planes' ranges. The planes'
dispatches then overlap on the GPU, which is where most of the gain comes
from; the saved submissions are the smaller part.

Median of five runs, fps, 1080p to 720p unless noted. "Before" is the build
of the sections above; the new build with the switch off measures the same
(within 1%) on every row.

| route | before | after | change |
|---|---:|---:|---:|
| GPU frames, Debicubic gray | 7648 | 7648 | +0% |
| GPU frames, Debicubic yuv420p8 | 3053 | 5278 | +73% |
| GPU frames, Debicubic yuv420p16 | 2959 | 4993 | +69% |
| GPU frames, Debicubic RGB | 2581 | 3489 | +35% |
| GPU frames, Despline64 yuv420p8 | 2031 | 3540 | +74% |
| GPU frames, Debilinear RGB | 3293 | 4467 | +36% |
| GPU frames, Debicubic RGB, 4K to 1440p | 988 | 981 | -1% |
| R81 host frames via upload and download, yuv420p8 | 2945 | 3537 | +20% |
| R80 host, Debicubic gray | 855 | 858 | +0% |
| R80 host, Debicubic yuv420p8 | 1752 | 2242 | +28% |
| R80 host, Debicubic yuv420p16 | 993 | 1083 | +9% |
| R80 host, Debicubic RGB | 261 | 261 | +0% |
| R80 host, Despline64 yuv420p8 | 1440 | 2103 | +46% |
| classic direct I/O, Debicubic yuv420p8 | 1825 | 2590 | +42% |
| classic direct I/O, Debicubic RGB | 392 | 395 | +1% |

Float32 RGB host frames and 4K GPU frames do not move: the first is bound by
host copies and PCIe, the second by GPU throughput, and neither by
submissions.

Costs and notes:

- A slot's workspace holds all planes of a frame: about 245 MB per slot for
  4K RGB, against 82 MB.
- A frame joins its shared inputs in address order and publishes them after
  its one submission. Plane order would deadlock two clips that hold the
  planes of one source in different orders (`ShufflePlanes`).
- Outputs are bit-identical to the previous build with the switch on and off:
  5 kernels x 4 formats x 2 geometries on R80, the classic direct I/O matrix
  (10 routes), and `vs_gpu_frames.py`. No validation or synchronization
  messages on host, direct I/O and GPU-frame routes.
- Fixed on the way: with synchronization validation on, every run crashed or
  hung at exit (also in the previous build). The owned runtime is destroyed
  in an exit handler and called `vkDeviceWaitIdle` there, after the
  validation layer's own static state was gone. The device is now waited for
  when the last filter is freed, and the exit-time wait only runs if
  something was submitted since: 48 of 48 clean exits, against 0 of 12. This
  is not the driver compile crash above (that one needs no layer).

## Verification

- All 62 ctest tests pass (rerun after the exit-crash fix and the shader
  change), including `dsmvc_vs_gpu_concurrency` (new): the
  partitioned solve against the CPU backend at full size on both backends, and
  seven filters sharing one source under 24 concurrent requests, compared with
  sequential runs, for Float32, 16-bit and 8-bit. Against the previous build
  the test hangs on CUDA (the deadlock above).
- Full-size outputs of both backends against the previous build: 14 routes
  (bands 1 to 7, odd sizes, rows, columns, RGB, YUV 8/16-bit, 4K), bit-exact
  except the two noted above.
- The classic direct I/O digest matrix: staged and direct digests equal on all
  routes, both backends.
- Vulkan under `VK_LAYER_KHRONOS_validation` with synchronization validation
  (fan-out, 8-bit YUV, gray, RGB): no errors, no hazards.
- The fan-out stress also passes with 1 and 2 CUDA streams and 1 and 2 Vulkan
  slots; the full test suite passed three times in a row, and the new test ten
  times in a row outside ctest.

## Not done

- **A queue per Vulkan execution slot.** Implemented (per-queue timeline
  semaphores, cross-queue waits for shared inputs; clean under synchronization
  validation) and removed again. Before the partitioned solve, Vulkan's single
  queue was the bottleneck; after it, eight queues measured +9% fps on 8-bit
  frames, +1% on 16-bit and -5% on Float32, with 3-11% more CPU per frame at
  equal fps, and it made the exit crash above reproducible. Not worth it.

- **One wait per frame on CUDA, and U+V in one launch.** Vulkan now submits a
  frame once (above). CUDA still processes planes one call at a time: on GPU
  frames it is bound by the core's plane exports, not by launches or waits.
- **Folding the right-hand side into the solve.** (The Vulkan transposes are
  gone, see the second round; CUDA keeps its passes.)
- **Staging written by dsmvc in cache-sized chunks.** For CUDA the driver's
  pageable route does this; the Vulkan backend still packs whole frames, and
  has no driver equivalent short of importing host memory (classic direct I/O).
- Float64 kernels are unchanged. Only one GPU (RTX 5080) was available; the
  Vulkan limits check was not exercised on a device that needs it.
- Exit-time compile wait, limits:
  - GPU frames (R81) use the core's Vulkan device and get no wait: dsmvc
    cannot tell the driver's threads from the core's there. The reproduction
    above did not fail on that path (0 of 20, no recompile ran).
  - A host that keeps another thread busy inside the driver at exit could
    make the wait run to its 20 s limit.
  - Tested on Linux with driver 615.71.09 only; other platforms do not wait.
  - Debugging a stuck process: the benchmark host
    (`benchmarks/classic_frame_bench.cpp`) prints the faulting stack and
    allows a debugger to attach with `BENCH_CRASH_TRACE=1`, and keeps the
    process alive with `--exit-sleep S`.
