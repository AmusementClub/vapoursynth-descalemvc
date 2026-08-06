# Release benchmark

This directory contains the reproducible old/current Release comparison used
to produce [`docs/release-benchmark.md`](../docs/release-benchmark.md).

The pipeline is split into four Python runners:

- `release_benchmark.py` consolidates the E2E, fixed-kernel, BlankClip, and
  error results.
- `e2e_benchmark.py` measures the three real-video candidate scans.
- `fixed_kernel_benchmark.py` measures fixed kernels on the supplied video.
- `blank_fixed_kernel_benchmark.py` measures the same kernels from an
  in-memory `std.BlankClip`.

`cpu_api_regression.py` is a focused ABI regression runner for comparing a
preserved API3 plugin with the API4-only build on `backend=cpu`. It runs each
ABI in a separate VSPipe process, alternates their order, gives every measured
process its own warm-up, and flags a cell when median VSPipe throughput drops
past the selected threshold. For example:

```sh
python3 benchmarks/cpu_api_regression.py \
  --api3-plugin /path/to/api3/dsmvc.so \
  --api4-plugin /path/to/api4/dsmvc.so \
  --vspipe /path/to/vspipe \
  --output profile-results/cpu-api4-regression \
  --frames 5000 --warmup-frames 128 --runs 3 \
  --threads 1 8 16 32 \
  --kernels bilinear bicubic_b0_c0_5 spline64
```

It writes raw JSON/CSV samples, a summary CSV and Markdown report, and the
exact warm-up and measurement commands. VSPipe-internal FPS is the primary
metric; the default regression threshold is 3%.

Each runner invokes its matching VapourSynth graph:

- `vspipe_e2e.vpy`
- `vspipe_fixed_kernel.vpy`
- `vspipe_blank_fixed_kernel.vpy`

The graphs use the supplied source and plugin paths passed as VSPipe
arguments. They do not write into the VapourSynth installation. The release
runner uses separate VSPipe processes for each implementation and thread
configuration, then writes machine-readable results and SVG charts under
`benchmark-results/`.

Performance runners execute an untimed throwaway warm-up before each measured
cell. This warms persistent driver/module/page caches and GPU clocks, while the
measured fresh VSPipe process still pays for its own VapourSynth and CUDA
context initialization. Use sufficiently long measured runs and compare both
the external wall time and the VSPipe-internal time recorded in JSON; do not use
the warm-up sample itself as a performance result.

The E2E graph bounds VapourSynth's frame cache with
`--performance-cache-mb` (512 MiB by default). A full GetNative scan therefore
runs as one graph by default, avoiding repeated decoder, VapourSynth, and CUDA
startup. `--performance-batch-size N` splits the candidate interval into fresh
VSPipe processes only as a low-memory fallback; `0` means one complete bounded
graph. `--cuda-plan-cache-mb` controls the CUDA packed-plan LRU (16 MiB by
default), which is intentionally small because height-scan plans are normally
used once. Candidate ranges are half-open and each scaler group is preserved
when a range crosses a scaler boundary.

On the reference 16 GiB RTX 5080, the complete 30,800-candidate GetNative scan
peaked at about 2.7 GiB host RSS and 504 MiB of CUDA process memory with the
bounded single graph and adaptive slots. The same scan with
`DSMVC_CUDA_STREAMS=4` peaked at about 2.6 GiB RSS and 406 MiB of CUDA process
memory. Full-scan timings varied by roughly 3% across runs, so these data do not
establish a stable throughput advantage for either slot policy. The earlier
multi-GiB failure was VapourSynth frame-cache retention (about 12.5 GiB without
the bound), not 30,800 resident CUDA plans. If host memory is still constrained,
use a positive performance batch size; expect repeated process/context startup
and measure the aggregate wall time rather than one batch's FPS.

For the complete Linux/macOS release procedure, see
[`mac_release_benchmark_guide.md`](mac_release_benchmark_guide.md). The
published report and its charts are kept in [`docs/`](../docs/).
