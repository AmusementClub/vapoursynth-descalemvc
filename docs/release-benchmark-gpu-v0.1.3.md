# GPU package benchmark: v0.1.2 vs v0.1.3 candidate

The published v0.1.2 and pre-merge v0.1.3 candidate Linux CPU/CUDA/Vulkan archives were tested on the same local RTX 5080 through SSH. The candidate is built from `d0ce31fb46e3c8e90cd8fc1307316554d119ba49`. This is a release-package comparison, not a product of previous incremental gains.

Primary ratio is candidate FPS / baseline FPS for each AB/BA pair, summarized by the median of all five pairs. VSPipe prints FPS with more significant digits than elapsed seconds; raw FPS reduces display quantization. Whole-process wall ratios are retained separately and include graph startup and warmup. No case is selected or discarded based on gain. The [R78 VSPipe source](https://github.com/vapoursynth/vapoursynth/blob/R78/src/vspipe/vspipe.cpp#L540-L560) starts the output timer after script evaluation, which contains source preparation and this harness's warmup.

## Results in context

- Default fixed-geometry throughput is nearly unchanged across both backends and all six kernels: medians range from -0.19% to +0.70%.
- Vulkan default candidate graphs improve by +2.59%, +5.57%, +3.90% at R1T1, and +8.13%, +14.51%, +10.97% at R32T32 for Lanczos3, Spline64 and Lanczos6 respectively.
- CUDA default candidate graphs remain close to the baseline. Its R32T32 medians are -0.03%, -0.42%, -0.86%, with all three five-pair ranges crossing 1.0x.
- Forced fused RHS improves CUDA Lanczos3 and Spline64 by +10.27%, +6.64%, and Vulkan Lanczos3, Spline64 and Lanczos6 by +16.55%, +22.33%, +23.10%. Those gains apply to the stated route and geometry.
- Small measured slowdowns are retained, including CUDA Bilinear/Bicubic candidate graphs and Vulkan Bicubic/Lanczos9 candidate graphs. Vulkan Bilinear candidate pairs span 0.8922-1.1021x around a 0.9912x median, so that cell does not establish a stable change. These are measured package results, not an attribution of every difference to one kernel edit.
- The campaign contains 420 timed processes and 583,680 measured frame requests. The longest process is 23.574 seconds. Both packages contain native SM75/86/89/120 and PTX75/120 code; hardware execution here covers the RTX 5080 only.

## Environment and workload

- Local host: AMD Ryzen 9 5950X, NVIDIA RTX 5080, driver 595.84, Linux 7.0.0-31, VapourSynth R78. CUDA/Vulkan use the same installed driver and runtime for both packages. Vulkan explicitly uses the NVIDIA ICD.
- Source: one predecoded 1920×1080 Float32 frame, SHA-256 `d50727fbfe6a6cb47eee915d5e92372292c41f7280d4d80d8db37312817e36bd`, loaded once outside timing. No decoding or input filling occurs inside the measured segment.
- Fixed hot path: 1920×1080 → 1692×952, active height 951.5, proportional width, centered fractional offsets, symmetric padding, forced Float32. A single filter processes 128 untimed warmup frames and 2,048 distinct measured frame requests from the same cached source.
- Candidate graph: 512 unique heights selected across 700.0–979.9 at integer tenths, including 951.5. Each candidate creates its descale plan, reconstructs to 1920×1080, applies the 0.015 error threshold, crops five pixels on each edge, and computes PlaneStats. It warms a separate 810p geometry for 64 frames. This preserves the GetNative-style operations but is a bounded subset, not the complete historical 30,800-candidate/11-scaler scan.
- R1T1 is pinned to CPU0. R32T32 uses the host's 32 logical CPUs. Both variants use identical frame counts, height lists, cache limit (512 MiB), warmup, affinity, and request/thread count. Automatic RHS selection is unchanged except in the separately labelled forced-fused cells.
- Every measured process records the mapped plugin path and SHA-256, exact command, timestamps, GPU telemetry, and external wall time. The GPU must be at most 5% busy with no other compute process before starting; no other compute process may remain after exit. Each process has a 29-second timeout.
- The initial isolated VSPipe configuration could not discover its embedded Python runtime and produced no performance samples. That failed setup is retained in `attempt0`; the active campaign uses the host's existing runtime configuration while still checking plugin identity and rejecting unexpected autoload.

## Package identities

| Package | Archive SHA-256 | Plugin SHA-256 |
|---|---|---|
| v0.1.2 | `78bc910d6b63fe1bd280ade6b74a48283d4a69c15d3ae31f7ac3d29b0f5bc640` | `e640e75e686f6c0214197b79f17a553793d59ca5a051e2d611e40af3cc6191ac` |
| v0.1.3-draft | `abc20ae2706cf9a7971ae4f39931f999c003dcefe8f4418b39b83b6f9ff9398e` | `b3f6618438f054ba423e1655215cfed7494de52602a348f91ef0219f78214b19` |

## Default RHS, fixed hot path, R1T1

| Kernel | CUDA speedup | CUDA pair range | Vulkan speedup | Vulkan pair range |
|---|---:|---:|---:|---:|
| bilinear | 1.0008x (+0.08%) | 0.9959–1.0017x | 1.0016x (+0.16%) | 0.9999–1.0041x |
| bicubic | 1.0070x (+0.70%) | 0.9996–1.0097x | 1.0001x (+0.01%) | 0.9939–1.0078x |
| lanczos3 | 1.0017x (+0.17%) | 0.9942–1.0077x | 1.0019x (+0.19%) | 0.9989–1.0029x |
| spline64 | 1.0004x (+0.04%) | 0.9928–1.0061x | 0.9990x (-0.10%) | 0.9962–1.0063x |
| lanczos6 | 1.0000x (-0.00%) | 0.9938–1.0054x | 0.9981x (-0.19%) | 0.9949–1.0004x |
| lanczos9 | 0.9990x (-0.10%) | 0.9966–1.0035x | 1.0005x (+0.05%) | 1.0001–1.0012x |

## Default RHS, candidate graph, R1T1

| Kernel | CUDA speedup | CUDA pair range | Vulkan speedup | Vulkan pair range |
|---|---:|---:|---:|---:|
| bilinear | 0.9889x (-1.11%) | 0.9859–0.9929x | 0.9912x (-0.88%) | 0.8922–1.1021x |
| bicubic | 0.9908x (-0.92%) | 0.9892–0.9930x | 0.9917x (-0.83%) | 0.9886–0.9954x |
| lanczos3 | 1.0083x (+0.83%) | 1.0012–1.0106x | 1.0259x (+2.59%) | 1.0243–1.0296x |
| spline64 | 1.0051x (+0.51%) | 1.0020–1.0074x | 1.0557x (+5.57%) | 1.0533–1.1180x |
| lanczos6 | 0.9939x (-0.61%) | 0.9935–0.9955x | 1.0390x (+3.90%) | 1.0377–1.0392x |
| lanczos9 | 0.9976x (-0.24%) | 0.9901–0.9992x | 0.9933x (-0.67%) | 0.9862–0.9940x |

## Default RHS, candidate graph, R32T32

| Kernel | CUDA speedup | CUDA pair range | Vulkan speedup | Vulkan pair range |
|---|---:|---:|---:|---:|
| lanczos3 | 0.9997x (-0.03%) | 0.9971–1.0066x | 1.0813x (+8.13%) | 1.0772–1.0858x |
| spline64 | 0.9958x (-0.42%) | 0.9841–1.0060x | 1.1451x (+14.51%) | 1.1426–1.1477x |
| lanczos6 | 0.9914x (-0.86%) | 0.9830–1.0079x | 1.1097x (+10.97%) | 1.1085–1.1102x |

## Forced fused RHS, fixed hot path, R1T1

| Kernel | CUDA speedup | CUDA pair range | Vulkan speedup | Vulkan pair range |
|---|---:|---:|---:|---:|
| bilinear | 0.9985x (-0.15%) | 0.9952–1.0045x | 1.0015x (+0.15%) | 0.9971–1.0049x |
| bicubic | 1.0009x (+0.09%) | 1.0001–1.0192x | 1.0000x (+0.00%) | 0.9970–1.0024x |
| lanczos3 | 1.1027x (+10.27%) | 1.0981–1.1051x | 1.1655x (+16.55%) | 1.1630–1.1658x |
| spline64 | 1.0664x (+6.64%) | 1.0634–1.0679x | 1.2233x (+22.33%) | 1.2221–1.2254x |
| lanczos6 | 0.9977x (-0.23%) | 0.9969–0.9985x | 1.2310x (+23.10%) | 1.2275–1.2315x |
| lanczos9 | 1.0011x (+0.11%) | 1.0001–1.0048x | 0.9977x (-0.23%) | 0.9878–1.0005x |

## Absolute rates and whole-process time

Absolute FPS is the median of individual samples. The speedup above is the median of paired ratios, which can differ from the ratio of these medians.

| Case | Baseline FPS | Candidate FPS | Whole-process wall speedup | Longest process, seconds |
|---|---:|---:|---:|---:|
| cuda-bilinear-hot-r1-automatic | 480.07 | 480.61 | 1.0014x | 4.998 |
| cuda-bilinear-scan-r1-automatic | 185.54 | 183.43 | 0.9944x | 3.484 |
| cuda-bilinear-hot-r1-fused | 481.19 | 481.21 | 1.0006x | 4.999 |
| cuda-bicubic-hot-r1-automatic | 665.15 | 670.99 | 1.0083x | 3.733 |
| cuda-bicubic-scan-r1-automatic | 159.35 | 157.95 | 0.9930x | 3.864 |
| cuda-bicubic-hot-r1-fused | 443.17 | 443.53 | 1.0032x | 5.450 |
| cuda-lanczos3-hot-r1-automatic | 577.29 | 578.12 | 1.0056x | 4.234 |
| cuda-lanczos3-scan-r1-automatic | 125.45 | 126.56 | 1.0104x | 4.741 |
| cuda-lanczos3-hot-r1-fused | 382.21 | 419.88 | 1.0964x | 6.172 |
| cuda-spline64-hot-r1-automatic | 544.79 | 545.47 | 0.9996x | 4.478 |
| cuda-spline64-scan-r1-automatic | 122.46 | 122.99 | 1.0077x | 4.903 |
| cuda-spline64-hot-r1-fused | 337.87 | 360.32 | 1.0636x | 6.935 |
| cuda-lanczos6-hot-r1-automatic | 313.45 | 313.15 | 1.0017x | 7.405 |
| cuda-lanczos6-scan-r1-automatic | 77.06 | 76.61 | 0.9967x | 7.465 |
| cuda-lanczos6-hot-r1-fused | 210.40 | 209.86 | 0.9985x | 10.829 |
| cuda-lanczos9-hot-r1-automatic | 262.67 | 262.47 | 1.0012x | 8.752 |
| cuda-lanczos9-scan-r1-automatic | 59.42 | 59.30 | 0.9984x | 9.572 |
| cuda-lanczos9-hot-r1-fused | 174.09 | 174.52 | 1.0042x | 12.973 |
| cuda-lanczos3-scan-r32-automatic | 416.85 | 416.31 | 1.0083x | 1.963 |
| cuda-spline64-scan-r32-automatic | 410.80 | 409.22 | 1.0088x | 2.004 |
| cuda-lanczos6-scan-r32-automatic | 389.87 | 386.51 | 1.0025x | 2.203 |
| vulkan-bilinear-hot-r1-automatic | 408.45 | 409.26 | 1.0028x | 5.906 |
| vulkan-bilinear-scan-r1-automatic | 166.38 | 163.90 | 0.9949x | 3.897 |
| vulkan-bilinear-hot-r1-fused | 408.77 | 408.98 | 1.0031x | 5.898 |
| vulkan-bicubic-hot-r1-automatic | 591.55 | 589.98 | 1.0005x | 4.245 |
| vulkan-bicubic-scan-r1-automatic | 136.42 | 135.23 | 0.9884x | 4.546 |
| vulkan-bicubic-hot-r1-fused | 310.82 | 310.47 | 1.0018x | 7.591 |
| vulkan-lanczos3-hot-r1-automatic | 550.03 | 550.35 | 1.0012x | 4.539 |
| vulkan-lanczos3-scan-r1-automatic | 103.37 | 106.13 | 1.0238x | 5.730 |
| vulkan-lanczos3-hot-r1-fused | 248.35 | 289.27 | 1.1543x | 9.333 |
| vulkan-spline64-hot-r1-automatic | 493.79 | 492.69 | 0.9994x | 5.025 |
| vulkan-spline64-scan-r1-automatic | 96.16 | 101.42 | 1.0490x | 6.109 |
| vulkan-spline64-hot-r1-fused | 204.91 | 250.66 | 1.2108x | 11.182 |
| vulkan-lanczos6-hot-r1-automatic | 342.70 | 341.35 | 0.9978x | 6.931 |
| vulkan-lanczos6-scan-r1-automatic | 66.79 | 69.37 | 1.0349x | 8.549 |
| vulkan-lanczos6-hot-r1-fused | 147.88 | 182.04 | 1.2197x | 15.278 |
| vulkan-lanczos9-hot-r1-automatic | 206.06 | 206.15 | 1.0007x | 11.133 |
| vulkan-lanczos9-scan-r1-automatic | 44.91 | 44.60 | 0.9937x | 12.580 |
| vulkan-lanczos9-hot-r1-fused | 94.87 | 94.64 | 0.9983x | 23.574 |
| vulkan-lanczos3-scan-r32-automatic | 297.57 | 321.00 | 1.0652x | 2.553 |
| vulkan-spline64-scan-r32-automatic | 238.15 | 272.71 | 1.1045x | 3.010 |
| vulkan-lanczos6-scan-r32-automatic | 171.45 | 190.10 | 1.0836x | 3.917 |

## Output validation and limits

Each timed cell has separate output proofs: four fixed-geometry frames or 32 distributed candidate frames, preserving requested concurrency. All 672 paired proof frames match in logical pixel SHA-256 and PlaneStats. These are sampled output proofs, not a claim that every timed frame was hashed.

An additional matrix covers 96 paired backend/policy/kernel/shape cases: normal, tail, downscale, and 4K; six kernels; CUDA/Vulkan; automatic/fused RHS. Both versions are checked against CPU scalar at 3e-6 and their two repeated output frames must be exact. Baseline/candidate output hashes match for every matrix case. The maximum observed scalar-reference error is 1.311302185e-06. The 4K input tiles the original frame; this is not a separate decoded 4K video.

All numerical proofs, raw performance samples, environment readings, package binaries, graph/runner scripts, and setup failures are retained in the verified evidence archive. The results do not establish Metal, Windows GPU, other NVIDIA architectures, video-decode throughput, every kernel/shape, or stable gains outside the measured request/thread configurations.

## Reproduction material

The [machine-readable data](release-benchmark-gpu-v0.1.3.json) includes all 420 timed samples, paired output hashes, the complete graph and proof runner, package identities, and the verified evidence archive hash. Write its `graph_source` to `graph.vpy` beside the input frame, load the chosen package by its full plugin path, and use the following command shapes with the same environment, affinity and preflight checks described above. Run each cell in alternating A/B order for five pairs.

```sh
# Fixed geometry; set backend to cuda or vulkan and the desired kernel.
taskset -c 0 vspipe -a plugin=/path/to/dsmvc.so -a backend=cuda \
  -a kernel=lanczos3 -a scenario=hot -a threads=1 -a frames=2048 \
  -r 1 -s 128 -e 2175 graph.vpy --

# Candidate graph; for R32T32 use threads=32, -r 32 and CPUs 0-31.
taskset -c 0 vspipe -a plugin=/path/to/dsmvc.so -a backend=cuda \
  -a kernel=lanczos3 -a scenario=scan -a threads=1 -a frames=512 \
  -r 1 graph.vpy --
```

Use `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and the NVIDIA Vulkan ICD. Leave both `DSMVC_CUDA_SPLIT_RHS` and `DSMVC_VULKAN_SPLIT_RHS` unset for automatic selection; set both to `0` for forced fused cells. The raw data records the exact binary and policy for each sample. The source frame is identified by its hash and is not redistributed in the repository.

This historical benchmark retains the exact candidate-package hashes above. The tagged release is rebuilt from the merged main-branch source, which retains these runtime kernels and adds Linux arm64 CPU packaging. The timings describe the recorded candidate binaries; they are not a second timing campaign on the rebuilt release archives.
