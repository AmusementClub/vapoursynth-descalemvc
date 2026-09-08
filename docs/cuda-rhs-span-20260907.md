# CUDA F32 RHS span experiment (2026-09-07)

> Status update (2026-09-08): the admitted routes are now enabled in normal builds and the experiment switches have been removed. CUDA wide/generic expansion was rejected during broader testing; Vulkan and Metal use bounded dedicated routes. See [default-promotion validation](gpu-default-promotion-20260908.md). Results below describe the original experiment.


`DSMVC_CUDA_RHS_SPAN_EXPERIMENT` enables ordered fixed-count RHS accumulation
inside the fused CUDA F32 inverse kernels. It defaults to **OFF**. The experiment
preserves the sparse entry order, uses one `fmaf` accumulator, and specializes
H5 counts 6/7, H7 counts 9/10, and H11 counts 13/14. Other counts retain the
dynamic loop. Plan construction and the public ABI are unchanged.

The initial experiment also changed standalone RHS kernels. That version raised
their register count from 38 to 48 and regressed several cases, including a
39.49% increase in bilinear vertical RHS time. The retained experiment changes
only fused inverse kernels. Standalone RHS and F64 kernels remain unchanged.

## Measured scope

RTX 5080, driver 595.84, CUDA 13.3.73, GCC 15.2, VapourSynth R78; native SM120
plus compute_120 PTX for this local test only. Release architecture defaults
remain unchanged. All comparisons use the same predecoded 1920x1080 GRAYS frame
and three alternating A/B pairs. Both versions retain the previous CPU changes.

The isolated GPU axis benchmark uses 64 threads and the default horizontal
column-major entry point, with plans and input already resident. It excludes
transfers, transposes, plan construction, and readback. The independent vertical
input has 1920 vectors; it is not the intermediate image of a complete 2D run.

| Kernel | Horizontal fused time reduction | Vertical fused time reduction | Candidate FPS change |
|---|---:|---:|---:|
| Lanczos3 | 12.66% | 13.07% | +1.20% |
| Spline64 | 8.59% | 8.73% | +0.75% |
| Lanczos6 | 1.81% | 1.95% | -0.25% |

Candidate throughput uses the actual CUDA plugin at R1T1, `f64mode=1`, default
adaptive split RHS, and 32 distinct heights, including reconstruction, Expr,
cropping, and PlaneStats. Video decoding is excluded. Steady BlankClip throughput
is nearly unchanged because reused plans normally use the unchanged split path.
The much larger row-major tiled-kernel gains are from an alternative layout and
must not be presented as default plugin throughput gains.

## Validation and decision

- Baseline CTest: 8/8; final candidate: 9/9.
- New direct CUDA tests: 42 route/band combinations, sparse counts 0..17,
  noncontiguous indices, vector tails, and output guards; exact ordered-FMA
  reference agreement. The same tests also pass against the baseline library.
- Compute Sanitizer memcheck: zero errors.
- Plugin output: 24 configurations and 396 frame SHA256/PlaneStats comparisons
  agree across adaptive, forced fused, and forced split modes.
- With the option OFF, all 28 SM120 kernel SASS bodies match the baseline.
  With it ON, only three fused F32 inverse kernels differ; the other 25 match.

Keep this as an opt-in experiment. The isolated kernel improvement is real for
these configurations, but a general pipeline speedup has not been established.
Other kernels, precisions, architectures, devices, and concurrency settings are
not admitted by this limited campaign.

Full evidence is in `artifacts/cuda-rhs-span-20260907/`: `report.md`, the CSV
tables, `decision.json`, `integrity.json`, and the archived `results/` tree.
