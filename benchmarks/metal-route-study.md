# Apple Silicon CPU/Metal route study

> Evidence provenance: the measurements and artifact hashes in this document
> were produced by the preserved API3 control before the API4 migration. They
> define the experimental routing boundary, but do not by themselves establish
> API4 correctness or throughput. Use `validate_api4_apple_arm.sh` for current
> migration evidence.

## Scope

This experiment asks whether Apple Silicon unified memory makes Metal useful
alongside the existing ARM64 NEON executor for the fixed 1920x1080 to
1692x952 recipe. It is built only when `DSMVC_BUILD_METAL_EXPERIMENTS=ON`.
The normal build has no Metal executor. In the opt-in build, explicit
`backend="metal"` enters the measured coordinator for supported fixed GRAYS or
YUV420P8/P10 recipes. `auto` is eligible only for the fixed wide-kernel YUV420
recipe at 256 or more frames, on a default Metal device that reports unified
memory and an Apple M-series name, and is activated only after 16 callbacks are
observed concurrently. Short clips, narrow kernels, GRAYS, and low-concurrency
`auto` requests stay on CPU; explicit Metal remains available for measurement
in supported cases and can be slower.

The Metal library is compiled with strict Float32 options:

```text
-fno-fast-math -ffp-contract=off
```

Axis plans, pipeline state, shared buffers, and plan uploads are persistent and
outside the timed region. Timed Metal routes still include the explicit host
copies required by this standalone model. The integer benchmark additionally
models P8/P10 limited-range conversion, all three YUV420 planes, and the row
strides observed from VapourSynth R78 at the fixed geometry.

## Float32 result

The candidate on the measured Apple M4 Max is a batch of 16 independent frames
split between 12 concurrent NEON frames and four Metal frames. The standalone
benchmark rotates route order for each sample. Speedup confidence intervals
bootstrap the paired per-cycle ratios with a fixed seed.

| Case | Ratio of medians | Paired median | Bootstrap 95% lower |
| --- | ---: | ---: | ---: |
| B1 bilinear | 1.049x | 1.053x | 1.039x |
| B3 bicubic | 1.109x | 1.105x | 1.092x |
| B5 Lanczos3 | 1.221x | 1.222x | 1.213x |
| B7 Spline64 | 1.195x | 1.186x | 1.179x |

Maximum absolute error across these routes is below `2.9e-6`. The experimental
gate is correctness plus a paired-bootstrap lower bound of at least `1.05x`
against the batch-16 NEON route for the selected kernel.

B1 is not a stable candidate. An earlier run reached a `1.053x` lower bound,
but a clean Xcode 27 rebuild reached only `1.039x`. B1 is therefore rejected
for adaptive Metal allocation. B3, B5, and B7 clear the gate in the fresh run;
the wider B5/B7 cases retain the most useful margin.

A 13 NEON plus three Metal split also improved all four medians, but 12 plus
four is the stronger wide-kernel allocation for this Float32 fixture.

## Integer YUV420 result

The source-faithful integer run uses fused limited-range normalization in the
horizontal Metal pass, Float32 recurrence, round-to-nearest-even integer
conversion, one buffered NEON frame, and streamed remaining NEON frames. The
horizontal chroma request matches `prepare_filter_requests()` exactly:

```text
0.25 - 0.25 * destination_width / source_width
    + src_left * chroma_source / source_width
```

R78 reported destination byte strides of `1696/864/864` for P8 and
`3392/1696/1696` for P10 at 1692x952. The benchmark records those exact values
instead of assuming 64-byte row padding.

The selected batch-16 policy uses 12 NEON plus four Metal frames for B1/B3 and
nine NEON plus seven Metal frames for B5/B7, with CPU concurrency 16 and eight
respectively and a Metal threadgroup width of 128. The 31-sample paired result
is:

| Case | Ratio of medians | Paired median | Bootstrap 95% lower |
| --- | ---: | ---: | ---: |
| P8 B1 bilinear | 1.243x | 1.233x | 1.151x |
| P8 Spline16 | 1.221x | 1.252x | 1.244x |
| P8 B3 bicubic | 1.254x | 1.256x | 1.248x |
| P8 Spline36 | 1.523x | 1.518x | 1.493x |
| P8 B5 Lanczos3 | 1.519x | 1.536x | 1.508x |
| P8 B7 Spline64 | 1.484x | 1.487x | 1.483x |
| P10 B1 bilinear | 1.079x | 1.079x | 1.070x |
| P10 Spline16 | 1.154x | 1.156x | 1.150x |
| P10 B3 bicubic | 1.158x | 1.154x | 1.142x |
| P10 Spline36 | 1.471x | 1.466x | 1.463x |
| P10 B5 Lanczos3 | 1.477x | 1.477x | 1.471x |
| P10 B7 Spline64 | 1.354x | 1.354x | 1.351x |

All 12 cases clear the `1.05x` lower-bound gate and differ from NEON by at most
one output code value. A 32/64/128/256 threadgroup sweep selected 128; 256
regressed. Moving the wide split from 8/8 to 9/7 raises its minimum robust lower
bound while retaining large margins in the P8 B5 cases. Moving the narrow split
to 13/3 hurt B3, so narrow remains 12/4.

## Plugin-level result

The opt-in plugin path uses a load-aware coordinator. Full groups contain 16
independent VapourSynth frame callbacks. Narrow plans assign 12 frames to CPU
and four to Metal; wide plans assign nine to CPU and seven to Metal. CPU slots
are admitted immediately, so a full group no longer requires all 16 callbacks
to block inside the coordinator. A partial GPU group waits 500 microseconds;
if CPU work from the same in-flight window is still active, the deadline is
extended until that work drains and then receives one more 500-microsecond
fill window. Otherwise it falls back to CPU immediately. GPU submission and
shared staging buffers remain serialized.

The R78 integration test used spatially patterned YUV420P8 and YUV420P10 input,
alternating limited/full `_Range`, and every measured kernel. Frame properties
proved real GPU participation: `_DSMVCMetalBatch` was four for narrow plans and
seven for wide plans. Both range modes reached the GPU in every case, and all
three output planes differed from the CPU route by at most one code value.

The same test passed a 7-frame CPU-only tail, 17- and 32-frame full-plus-tail
groups, cancellation followed by a fresh request, invalid-route rejection,
retained output-frame access after graph release, and both high- and
low-concurrency `backend="auto"` cases. High-concurrency 256-frame wide auto
graphs reached `_DSMVCMetalBatch=7`; prefetch-4 graphs produced zero GPU
frames. A separate VSPipe 17-frame P10/Spline64 proof exported seven
`_DSMVCMetalBatch=7` frames and a final CPU tail.

End-to-end VSPipe wall time includes process startup, graph construction, Metal
pipeline/buffer initialization, frame delivery, tail flush, and shutdown. Five
order-rotated samples per route produced these current-candidate results. The
wide rows below use a separate 21-sample control/candidate A/B with bootstrap
over paired ratios; the other rows use the five-sample CPU/candidate runner.

| Workload | Requests | Result |
| --- | ---: | ---: |
| 96 frames, P8/P10 Spline36/Lanczos3/Spline64, explicit Metal | 16/32 | ratio of medians `0.942x` to `1.013x`; the short-clip route is not a speed gate |
| 256 frames, P8/P10 wide `auto` | 4 | ratio of medians `0.995x` to `1.004x`; the admission probe stays effectively CPU-only |
| 256 frames, P8/P10 wide `auto` | 16/32 | ratio of medians `1.100x` to `1.235x`; paired-sample minima `1.073x` to `1.223x` |
| 512 frames, P8/P10 wide bulk-copy candidate versus row-copy control | 16/32 | ratio of medians `1.005x` to `1.039x`; bootstrap 95% lower `0.998x` to `1.034x` |
| 512 frames, P8/P10 narrow explicit Metal | 1/4/16/32 | current-candidate CPU comparison remains below `1x` across the measured cases |

The 21-sample wide A/B is the current bulk-copy candidate against the
pre-bulk-copy control. All six P10 cases clear a bootstrap lower bound of
`1.0228x` or higher. Five of six P8 cases clear `1x`; P8 Spline64/R16 has a
ratio-of-medians of `1.0053x` but a bootstrap lower bound of `0.9984x`, so it is
reported as a median improvement without a stable lower-bound claim. The
bulk-copy change is therefore useful evidence for P10 and most P8 wide cases,
not a universal P8 guarantee.

The current-candidate 256-frame auto run covers P8/P10 Spline36, Lanczos3, and
Spline64 at R4/R16/R32. R16/R32 paired minima are at least `1.073x` and
`1.114x` for P10, and at least `1.112x` and `1.203x` for P8. The R4 rows stay
near one, as intended by the observed-callback admission gate. A current
candidate Spline64 length sweep gives weakest paired minima of `1.0489x` at
192 frames and `1.0883x` at 256 frames across P8/P10 and R16/R32.

The bulk-staging candidate used for the A/B artifacts above has SHA-256
`dbc4749b8e210a1663e073ecdf2d1b71235dd71821068b3ccc6e01cb33e11377`.
The current wide A/B artifacts are
`/tmp/dsmvc-metal-bulkcopy-ab-p8-512-21-final-20260806.json` (SHA-256
`20eb3fc4ec4d2127598f6ad33e363f946f4cb80a13093b8dc123faf115217370`) and
`/tmp/dsmvc-metal-bulkcopy-ab-p10-512-21-final-20260806.json` (SHA-256
`16d695f261c9aa760be16c0c9b93b89220efa1fd01ec10fd30bc5e96467a728c`).
The current-candidate short-wide artifact is
`/tmp/dsmvc-plugin-metal-admission-final-short-wide-new-20260806.json` (SHA-256
`67bfcd0b1a0f26d62852a0aa5933623de94f33de8bf5494da9bbd8c39e306521`).
The current-candidate auto artifact is
`/tmp/dsmvc-plugin-metal-admission-final-auto-256-new-20260806.json` (SHA-256
`59493841896272aa404750df3657bceda90ccad401039db984acca75f86fa188`).
The current-candidate narrow 512-frame artifact is
`/tmp/dsmvc-plugin-metal-admission-final-narrow-512-new-20260806.json`
(SHA-256
`be659c2e725af914a95dd3553a6c3b9766d576b722d3fc003fa3fb652ad8d2c5`).
The current-candidate Spline64 length artifacts are
`/tmp/dsmvc-plugin-metal-admission-final-length-192-new-20260806.json` (SHA-256
`e9807fadea42e39f65aea649a1115b897ec985ad91196a0028d3cf2e87b00e3b`) and
`/tmp/dsmvc-plugin-admission-final-length-256-new-20260806.json` (SHA-256
`13b5f54998f103d3bf77c0736f7638ad953d3771e6b67c0eca78fdfd27071d78`).

### GRAYS B1 plugin route

Explicit `backend="metal"` now accepts B1 bilinear for the fixed GRAYS recipe.
The R78 integration test routes four of each 16 callbacks to Metal and compares
spatially patterned output against CPU with maximum absolute error
`9.5367431640625e-7`. It also verifies that GRAYS B1 `auto` emits no Metal frame
properties and therefore retains its CPU route.

An initial candidate also made GRAYS B1 automatic on a detected Apple M-series
device. Alternating 512-frame BlankClip A/B runs rejected that policy:

| Route | Samples | R16 ratio of medians | R16 95% lower | R32 ratio of medians | R32 95% lower |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 `auto` Metal candidate | 7 | `0.7375x` | `0.6970x` | `0.7311x` | `0.7041x` |
| B1 explicit Metal | 5 | `0.7293x` | `0.6999x` | `0.7270x` | `0.7151x` |

The explicit result confirms that the regression comes from entering the Metal
route, rather than the `auto` predicate. Standalone 12/4, 13/3, 14/2, and 15/1
CPU/Metal split scans reached `1.1004x`, `1.1565x`, `1.0761x`, and `1.0203x`,
respectively, but none can bridge the roughly 27% plugin-level loss. Further
split tuning was therefore stopped.

The rejected auto candidate was
`6bce57d7c3e65d44b7b8bbdab5679977292c519936ae35071b7db3a3aeab03e0`
against CPU control
`513d4bdfbc32614082cebed7b60b3ea13094aca058f7f33d60de89a028ce70f7`.
Its auto A/B artifact is
`/private/tmp/dsmvc-grays-b1-auto-ab-512-7-20260806.json` (SHA-256
`ef6c12fca52c67830a28cf4d30a98c07426bec7e8c05358fffdcbffcde96aa67`),
and its explicit A/B artifact is
`/private/tmp/dsmvc-grays-b1-explicit-ab-512-5-20260806.json` (SHA-256
`6626f51e64dcee708ca064d4d2cc1cce3fe05344c653532d21f8b28798dba72a`).
The final plugin retains explicit B1 Metal but restores B1 `auto` to CPU; its
SHA-256 is
`6e952591e8c30123d08eef10bab9e999f707393c595bfa81982397629cb46fa1`.
The final CPU-fallback A/B medians are `1.0087x` at R16 and `1.0033x` at R32 in
`/private/tmp/dsmvc-grays-b1-final-auto-fallback-512-5-20260806.json`
(SHA-256
`bf31b92c0cdf51d6c6a9f35a6c1677d5d51dfa7befa0d8e6cdcb8b058842bd74`).

A Debug UBSan-only build passed all four CTests, including the complete
VapourSynth Metal integration (`107.22 s`). A combined ASan/UBSan build passed
the engine, coordinator, and staging tests, but the Python integration process
aborted before plugin execution with `Interceptors are not working`, even when
the Xcode ASan runtime was supplied through `DYLD_INSERT_LIBRARIES`. Complete
Metal executor ASan coverage is therefore still unverified; the UBSan result
must not be presented as equivalent memory-safety coverage.

## Route decisions

Accepted for further measurement:

- Float32 independent-frame heterogeneous execution at batch 16, using 12 NEON
  frames and four Metal frames for B3/B5/B7 on the measured M4 Max.
- Fixed GRAYS B1 explicit Metal access for correctness and profiling. It is not
  a performance default and is deliberately excluded from `auto`.
- Integer YUV420 independent-frame execution at batch 16, using 12/4 for narrow
  kernels and 9/7 for wide kernels at threadgroup width 128.
- Pure Metal integer YUV batches for B5/B7 clear the standalone speed gate, but
  heterogeneous execution retains the stronger margin.
- One command buffer containing all integer axis and conversion passes, with
  persistent shared buffers and immutable plan buffers.
- Equal-stride bulk staging for both P8 and P10, with a row-wise fallback for
  mismatched strides. P8 Spline64/R16 remains a documented statistical boundary
  rather than a reason to add a kernel-specific copy policy.
- Auto routing for the opt-in build only when the fixed wide recipe has at
  least 256 frames, core threads are at least 16, and the coordinator observes
  16 active callbacks; low-concurrency admission remains CPU-only.

Rejected as production choices by current evidence:

- Float32 per-frame Metal-to-Metal execution. At batch one it is roughly four
  to five times slower than NEON; batching narrows but does not reverse the gap.
- Automatic GRAYS Float32 B1 dispatch. Standalone split measurements did not
  survive the real plugin schedule: explicit and automatic Metal both delivered
  only about `0.73x` CPU throughput. Integer YUV420 B1 is a separate measured
  route and does clear its standalone gate.
- NEON-to-Metal and Metal-to-NEON axis splitting. Synchronization and the
  remaining host copy outweigh useful overlap.
- Direct wrapping of ordinary R78 frame planes as a zero-copy contract. Tested
  pointers were 64-byte aligned but `page + 64`; direct
  `newBufferWithBytesNoCopy` failed for P10 luma, while aligning down would
  expose memory outside the plane allocation.
- Broad automatic Metal dispatch without eligibility gates. Short clips and
  narrow kernels regress, and the plugin cannot infer an external request
  window from core thread count alone; the observed-concurrency activation
  threshold is required.

## Trace interpretation

Both checked trace sets use 200 measured batches after seven warmups, or 3200
output frames. The Metal trace contains one measured command-buffer submission
per batch. Command-buffer IDs, rather than labels, join submissions to GPU
intervals.

The trace analyzer reported a Metal allocation peak of 354,140,160 bytes
(`337.73 MiB`), consistent with the persistent batch-16 Float32 working set.
Instruments perturbs the GPU path heavily: the measured command-buffer GPU
interval union is about 18.1 ms per batch in the trace, so trace duration is
used for attribution only. Standalone `routes.json` samples are the performance
authority.

The P10 B7 YUV trace covers 1800 NEON and 1400 Metal frames. All 200 command
buffers join to GPU intervals. Instruments coalesces the nine Y/U/V horizontal,
vertical, and conversion encoders into one row containing all nine labels. The
allocation peak is 398,704,640 bytes (`380.23 MiB`) versus 398,135,920 bytes
requested by the benchmark for persistent data and plan buffers. The traced GPU
interval union is about 14.26 ms per batch or 2.04 ms per Metal frame; this is
again perturbed trace attribution, not the standalone throughput result.

## Plugin phase attribution

The first plugin-level phase trace was captured on 2026-08-06 from the rebuilt
signpost candidate, whose SHA-256 is
`91105f146774785735ec9052fdfdec875ca70078e41d7b958b8b566da8ca5bac`. The
pre-signpost control was preserved separately at
`/private/tmp/dsmvc-plugin-metal-presignpost-control-20260806/dsmvc.so` with
SHA-256
`dbc4749b8e210a1663e073ecdf2d1b71235dd71821068b3ccc6e01cb33e11377`.

The workload was the fixed P10/Spline64 two-axis recipe, 4096 output frames,
explicit Metal, and 16 requested VapourSynth threads. The trace used the native
VSPipe executable at
`/Users/owen/vapoursynth/venv/lib/python3.14/site-packages/vapoursynth/vspipe`;
the `venv/bin/vspipe` entry point is a Python wrapper that launches a child and
must not be passed directly to `xctrace --launch` when a single target process is
required. `Logging` plus `Points of Interest` was configured with
`benchmarks/xctrace_plugin_signposts.json`, and Metal System Trace was recorded
as a separate equivalent run.

The signpost export contains two Instruments interval views. The analyzer
deduplicates their 7168 source rows to 3584 plugin intervals, excludes the first
16 of 256 batches, and verifies 240 measured batches (3840 frames): 2160 CPU
frames routed nine per batch and 1680 Metal frames routed seven per batch.

| Phase or measurement | Count | Median | Mean | Share/qualification |
| --- | ---: | ---: | ---: | --- |
| CPU frame (`DSMVCPluginCpuFrame`) | 2160 | 11.106 ms | 11.058 ms | concurrent work; summed durations are not wall time |
| Metal batch | 240 | 16.033 ms | 16.051 ms | reference interval |
| Host upload | 240 | 0.824 ms | 0.822 ms | 5.12% of summed batch phase time |
| Encode | 240 | 0.0237 ms | 0.0247 ms | 0.15% |
| CPU-visible wait | 240 | 14.718 ms | 14.730 ms | 91.77%; includes commit and `waitUntilCompleted` |
| Host download | 240 | 0.469 ms | 0.472 ms | 2.94% |

Metal System Trace found one command-buffer submission and one coalesced
encoder-list row per measured batch. Its per-command GPU interval union had a
14.389 ms median (14.404 ms mean), so the separate-run host-wait/GPU median
ratio was `1.0229x` and the difference was 0.329 ms. This is distribution-level
comparison, not per-command timestamp correlation. The traced allocation peak
was 175,046,656 bytes.

The staging metadata was constant across all measured batches: 21 `memcpy`
calls and 43,545,600 uploaded bytes for the seven Metal frames, followed by 21
download calls and 33,906,320 downloaded bytes. The combined diagnostic was 42
calls and 77,451,920 bytes per batch. Signpost overlap showed nine CPU frames
routed per batch, with an observed peak of eight simultaneously active CPU
frame intervals during a Metal batch; the Metal batch metadata consistently
reported seven GPU frames.

This trace does not support implementing double-buffered staging or command
submission as the next change: the dominant wait interval is almost entirely
GPU residency, while host copies account for about 8.1% of the batch. The next
measurement should target Metal kernel memory/instruction attribution or a
different workload envelope before changing synchronization. These timings are
attribution evidence only and do not replace alternating end-to-end throughput
or correctness gates.

Requested data working sets are 19,332,608 bytes (`18.437 MiB`) per P8 frame
and 24,849,664 bytes (`23.698 MiB`) per P10 frame. B7 uploads 541,296 bytes of
unique plan buffers. The current plugin diagnostic reports 42 staging copy
calls for a seven-frame wide batch, rather than the 28,448 row copies used by
the control path. It reports 38,832,360 bytes for a P8 batch and 77,451,920
bytes for P10. Those values include row padding except after the final row of
each plane; compared with logical plane bytes, the extra span is 20,904 bytes
per P8 frame (`0.378%`) and 11,408 bytes per P10 frame (`0.103%`). A direct CTest
also covers the different-stride fallback and verifies that it copies only
logical row bytes without touching destination padding.

The JSON also reports source-level logical Metal buffer accesses. For B7 these
previously totaled about `1.115 GiB` per P8 Metal frame and `1.140 GiB` per P10
Metal frame, including repeated recurrence and plan reads. The fixed B5/B7
kernels now retain their forward and backward recurrence windows in explicit
per-thread registers, removing those repeated result-buffer reads while
preserving the original descending accumulation order. These are still
program-level accesses before cache, coalescing, and compiler effects; they are
not measured DRAM traffic and must not be compared directly with physical
bandwidth.

The explicit-register candidate has SHA-256
`d03e0fa6e3de6d0aaf7c45d2067a517901382a8026eac41495cae6b39fcd78fb`.
Against the `91105f146774785735ec9052fdfdec875ca70078e41d7b958b8b566da8ca5bac`
control, the 512-frame plugin A/B covers P8/P10 Spline36, Lanczos3, and
Spline64 at R16/R32 with 21 order-rotated samples. Its ratio of medians is
`1.032x` to `1.073x`; paired-bootstrap 95% lower bounds are `1.031x` to
`1.067x`. The artifact is
`/private/tmp/dsmvc-plugin-register-explicit-final-512-21-20260806.json`
(SHA-256
`3c3b2882905467be6fbd847531c99a7439f0d735f01ce31aa9d8226000a8e999`).
After the kernel change, a 31-sample standalone split scan preferred 8/8 over
9/7 for P8 but not P10. The matching plugin A/B rejected that P8 split: all six
P8 cases regressed by `1.9%` to `3.1%`, while the unchanged P10 9/7 cases stayed
near one. The plugin therefore retains 9/7 for both formats; standalone route
balance is not used as a substitute for the VSPipe scheduling result. The split
screen is
`/private/tmp/dsmvc-plugin-register-explicit-p8-split8-screen-512-5-20260806.json`.

The default CPU Counters configuration emits four-lane CPU Bottlenecks sample
weight arrays. They are useful for comparing attribution inside the measured
`MeasuredLoop` signpost, but they are not retired instructions, cycles, cache
misses, or another exact hardware event count. Do not relabel or sum them as
instruction counts.

## Reproduction

Build and run only the standalone comparison:

```bash
cmake -S . -B /tmp/dsmvc-metal-routes-build -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DDSMVC_VAPOURSYNTH_SDK=/path/to/vapoursynth-sdk \
    -DDSMVC_VS_PYTHON=/path/to/vapoursynth/python \
    -DDSMVC_BUILD_METAL_EXPERIMENTS=ON \
    -DBUILD_TESTING=ON
cmake --build /tmp/dsmvc-metal-routes-build \
    --target dsmvc_metal_route_benchmark dsmvc_metal_yuv_benchmark \
             dsmvc_engine_tests --parallel
/tmp/dsmvc-metal-routes-build/dsmvc_metal_route_benchmark \
    --samples 31 --warmups 7 --threads-per-threadgroup 32 \
    --batch-size 16 --heterogeneous-cpu-frames 12 --assert \
    --json-out /tmp/dsmvc-metal-routes.json
python3 benchmarks/analyze_metal_routes.py \
    /tmp/dsmvc-metal-routes.json \
    --json-out /tmp/dsmvc-metal-routes-analysis.json \
    --assert-route neon+metal --assert-case b7-spline64

/tmp/dsmvc-metal-routes-build/dsmvc_metal_yuv_benchmark \
    --samples 31 --warmups 7 --threads-per-threadgroup 128 \
    --batch-size 16 --narrow-cpu-frames 12 --wide-cpu-frames 9 \
    --narrow-cpu-concurrency 16 --wide-cpu-concurrency 8 --assert \
    --json-out /tmp/dsmvc-metal-yuv-routes.json
python3 benchmarks/analyze_metal_routes.py \
    /tmp/dsmvc-metal-yuv-routes.json --baseline-route neon \
    --json-out /tmp/dsmvc-metal-yuv-routes-analysis.json \
    --assert-route neon+metal --assert-case yuv420p10-b7-spline64

ctest --test-dir /tmp/dsmvc-metal-routes-build \
    -R 'dsmvc_(vs_metal_integration|metal_yuv_staging_tests)' \
    --output-on-failure
python3 benchmarks/metal_plugin_benchmark.py \
    --plugin /tmp/dsmvc-metal-routes-build/dsmvc.so \
    --vspipe /path/to/vspipe \
    --formats p8 p10 \
    --kernels bilinear spline16 bicubic spline36 lanczos3 spline64 \
    --requests 1 4 16 32 --frames 512 --samples 5 --warmups 1 \
    --json-out /tmp/dsmvc-metal-plugin.json

python3 benchmarks/metal_plugin_ab_benchmark.py \
    --control-plugin /tmp/dsmvc-metal-control.so \
    --candidate-plugin /tmp/dsmvc-metal-routes-build/dsmvc.so \
    --vspipe /path/to/vspipe --formats p8 p10 \
    --kernels spline36 lanczos3 spline64 --requests 16 32 \
    --frames 512 --samples 21 --warmups 1 \
    --json-out /tmp/dsmvc-metal-plugin-ab.json
```

For CPU Counters, Metal System Trace, XML exports, measured-loop filtering, and
normalization over the exact frame count:

```bash
benchmarks/profile_metal_routes.sh \
    --vapoursynth-sdk /path/to/vapoursynth-sdk
benchmarks/profile_metal_yuv.sh \
    --vapoursynth-sdk /path/to/vapoursynth-sdk
```

Each run has a unique output directory. The driver records source identity,
host and Xcode metadata, effective Ninja commands, engine test output, raw
standalone samples, raw traces, exported XML tables, and normalized JSON.

For plugin phase attribution, use the native VSPipe binary and the dedicated
analyzer after exporting `OSSignpostIntervals` from a Logging/Points-of-Interest
trace and the four Metal System Trace tables:

```bash
python3 benchmarks/analyze_metal_plugin_profile.py \
    --plugin /tmp/dsmvc-plugin-metal-admission-20260806/dsmvc.so \
    --signpost-intervals /tmp/plugin-phases-intervals.xml \
    --submissions /tmp/metal-submissions.xml \
    --encoders /tmp/metal-encoders.xml \
    --gpu-intervals /tmp/metal-gpu-intervals.xml \
    --allocated /tmp/metal-allocated.xml \
    --expected-output-frames 4096 --warmup-batches 16 \
    --batch-size 16 --cpu-frames-per-batch 9 \
    --metal-frames-per-batch 7 --format p10 --kernel spline64 \
    --requests 16 --json-out /tmp/dsmvc-metal-plugin-profile.json
```

## Production boundary

The coordinator, correctness, frame-lifetime, and dynamic-admission gates are
complete for the fixed opt-in recipe. The normal build remains CPU-only. In an
opt-in Metal build, `auto` is deliberately limited to the measured wide
YUV420P8/P10 recipe, 256 or more frames, core threads at least 16, an observed
16-callback activation threshold, and a detected unified-memory Apple M-series
Metal device; all other cases use CPU. Fixed GRAYS B1 is exposed only through
an explicit Metal request because its automatic plugin route failed the
throughput gate. The final plugin binary passed fresh correctness and
paired-sample gates for the accepted envelope, and the default non-Metal build
still passes its engine test and CPU smoke. This is experimental Apple Silicon
routing evidence, not a claim that the whole plugin is globally optimized, nor
a portability claim for non-Apple hosts or unmeasured geometries/formats.
