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

Each runner invokes its matching VapourSynth graph:

- `vspipe_e2e.vpy`
- `vspipe_fixed_kernel.vpy`
- `vspipe_blank_fixed_kernel.vpy`

The graphs use the supplied source and plugin paths passed as VSPipe
arguments. They do not write into the VapourSynth installation. The release
runner uses separate VSPipe processes for each implementation and thread
configuration, then writes machine-readable results and SVG charts under
`benchmark-results/`.

For the complete Linux/macOS release procedure, see
[`mac_release_benchmark_guide.md`](mac_release_benchmark_guide.md). The
published report and its charts are kept in [`docs/`](../docs/).
