# v0.1.3 prerelease CPU benchmark

All ratios below are **v0.1.2 time / v0.1.3 candidate time** on the same host
and requested ISA. Larger is faster; 1.000x is unchanged. These are fresh
comparisons against the latest published release, not a product of older gains.

The width workload preserves the v0.1.2 release description's 256 rows,
1920 → 1692 geometry, four kernels, Release build, and GCC 13.3. Both hosts
have two vCPUs: Intel Xeon Platinum 8581C and AMD EPYC 9B45. Buffers use the
same controlled alignment and are filled before timing. Five A/B or B/A pairs
use three calls per sample, followed by longer calibrated samples.

The main table uses five longer pairs, approximately 1.0–1.3 seconds or more
per sample. Each accepted group requires CPU1 to be at least 99% idle in
every timed sample. Environment-invalid groups are retained and are not used
for these ratios.

| Filter | Intel AVX2 | Intel AVX-512 | AMD AVX2 | AMD AVX-512 |
|---|---:|---:|---:|---:|
| Debilinear | 1.012x | 1.013x | 0.995x | 0.999x |
| Debicubic | 1.004x | 1.008x | 0.986x | 0.993x |
| Delanczos (`taps=3`) | 1.400x | 1.296x | 1.575x | 1.350x |
| Despline64 | 1.424x | 1.406x | 1.645x | 1.493x |

Debilinear and Debicubic are effectively flat in this workload; AMD Debicubic
has a small measured slowdown (0.986x on AVX2). Lanczos3 and Spline64 improve
on both tested CPUs and both requested ISA paths.

A supplemental predecoded Float32 2D case, 1920×1080 → 1692×952, measures
the full CPU inverse with automatic ISA selection:

| Filter | Intel auto | AMD auto |
|---|---:|---:|
| Delanczos (`taps=3`) | 2.187x | 1.753x |
| Despline64 | 2.169x | 1.740x |
| Delanczos (`taps=6`) | 1.955x | 1.451x |

All measured cross-version outputs are byte-identical. The legacy height
fixture (951 → 952 with automatic precision) selects **F64** for every tested
kernel; its results do not measure the new F32 column route. The supplemental
2D case explicitly checks that both axes use F32. These are engine timings,
not video decode or complete GetNative scan timings.

The [full benchmark report](https://github.com/AmusementClub/vapoursynth-descalemvc/blob/codex/release-v0.1.3/docs/release-benchmark-v0.1.3.md)
includes the original five-by-three results, absolute times, every paired
ratio, environment failures, and reproduction instructions.

## Legacy five pairs × three calls

This table retains the short sampling count from the v0.1.2 description. Each side runs in an independent process with one warm call, and pair order alternates AB/BA. The very short timing windows are not sufficient for the CPU1 environment gate, so the stable table above is the performance conclusion. Input/output pointers are controlled to offset 16 modulo 64 on both sides; historical allocator placement was not controlled.

| Filter | Intel AVX2 | Intel AVX-512 | AMD AVX2 | AMD AVX-512 |
|---|---:|---:|---:|---:|
| Debilinear | 1.011x | 1.007x | 0.996x | 0.994x |
| Debicubic | 1.017x | 0.976x | 0.989x | 0.994x |
| Delanczos (`taps=3`) | 1.395x | 1.272x | 1.542x | 1.337x |
| Despline64 | 1.417x | 1.353x | 1.559x | 1.456x |

## Legacy height geometry, longer samples

All of these requests select F64, including when `avx512` is requested. The Intel medians are approximately flat. AMD has wider scatter, including a 0.969x Spline64 AVX2 median; this data does not establish broad non-regression of F64 performance. Numerical equality passed.

| Filter | Intel AVX2 | Intel AVX-512 | AMD AVX2 | AMD AVX-512 |
|---|---:|---:|---:|---:|
| Debilinear | 0.999x | 1.000x | 1.025x | 0.998x |
| Debicubic | 0.999x | 1.007x | 1.041x | 0.998x |
| Delanczos (`taps=3`) | 0.998x | 1.000x | 1.010x | 1.020x |
| Despline64 | 0.994x | 0.998x | 0.969x | 1.006x |

## Accepted absolute times and pair ranges

Absolute times below are medians of individual time-per-call samples. Speedup is the median of the five paired ratios, not a ratio of these two medians.

| Host | Case | v0.1.2 ms/call | Candidate ms/call | Paired speedup | Five-pair range | Min CPU1 idle |
|---|---|---:|---:|---:|---:|---:|
| intel | width-bilinear-avx2-to-avx2 | 0.277695 | 0.274287 | 1.011768x | 1.006763–1.017700x | 100.000% |
| intel | width-bilinear-avx512-to-avx512 | 0.277866 | 0.274332 | 1.012873x | 1.011564–1.014798x | 100.000% |
| intel | width-bicubic-avx2-to-avx2 | 0.504528 | 0.502885 | 1.004146x | 1.000953–1.006497x | 99.213% |
| intel | width-bicubic-avx512-to-avx512 | 0.505106 | 0.501208 | 1.008027x | 1.002046–1.008704x | 100.000% |
| intel | width-lanczos3-avx2-to-avx2 | 0.776942 | 0.556113 | 1.399601x | 1.391229–1.400774x | 99.010% |
| intel | width-lanczos3-avx512-to-avx512 | 0.590323 | 0.455564 | 1.295886x | 1.294610–1.299926x | 100.000% |
| intel | width-spline64-avx2-to-avx2 | 0.998250 | 0.700830 | 1.424262x | 1.420241–1.427138x | 100.000% |
| intel | width-spline64-avx512-to-avx512 | 0.769662 | 0.548366 | 1.405901x | 1.399368–1.406487x | 100.000% |
| intel | height-bilinear-avx2-to-avx2 | 6.864051 | 6.848301 | 0.999077x | 0.997215–1.014401x | 99.010% |
| intel | height-bilinear-avx512-to-avx512 | 6.861269 | 6.834377 | 1.000242x | 0.994468–1.005584x | 99.020% |
| intel | height-bicubic-avx2-to-avx2 | 7.742320 | 7.755861 | 0.998904x | 0.993515–0.999118x | 100.000% |
| intel | height-bicubic-avx512-to-avx512 | 7.799039 | 7.738785 | 1.006510x | 0.997971–1.009411x | 100.000% |
| intel | height-lanczos3-avx2-to-avx2 | 8.418804 | 8.436903 | 0.997572x | 0.995903–0.999514x | 99.000% |
| intel | height-lanczos3-avx512-to-avx512 | 8.423107 | 8.426954 | 0.999537x | 0.999210–1.003446x | 100.000% |
| intel | height-spline64-avx2-to-avx2 | 9.089238 | 9.142459 | 0.994178x | 0.990644–1.003504x | 100.000% |
| intel | height-spline64-avx512-to-avx512 | 9.117453 | 9.146855 | 0.997930x | 0.992791–1.011046x | 99.010% |
| intel | 2d-lanczos3 | 5.440039 | 2.484788 | 2.187333x | 2.183889–2.200359x | 100.000% |
| intel | 2d-spline64 | 6.538545 | 3.011039 | 2.168633x | 2.167850–2.174726x | 100.000% |
| intel | 2d-lanczos6 | 9.333553 | 4.776154 | 1.954712x | 1.953559–1.959491x | 100.000% |
| amd | width-bilinear-avx2-to-avx2 | 0.242484 | 0.243217 | 0.995252x | 0.993352–0.999249x | 100.000% |
| amd | width-bilinear-avx512-to-avx512 | 0.245208 | 0.245249 | 0.999306x | 0.994638–1.003932x | 99.010% |
| amd | width-bicubic-avx2-to-avx2 | 0.426791 | 0.433024 | 0.985610x | 0.982441–0.989557x | 100.000% |
| amd | width-bicubic-avx512-to-avx512 | 0.429473 | 0.432983 | 0.993096x | 0.988330–0.994939x | 99.010% |
| amd | width-lanczos3-avx2-to-avx2 | 0.733557 | 0.466001 | 1.574940x | 1.573365–1.579351x | 100.000% |
| amd | width-lanczos3-avx512-to-avx512 | 0.493372 | 0.367368 | 1.349824x | 1.327193–1.360327x | 99.020% |
| amd | width-spline64-avx2-to-avx2 | 0.929437 | 0.563549 | 1.644643x | 1.606235–1.651904x | 100.000% |
| amd | width-spline64-avx512-to-avx512 | 0.678410 | 0.454701 | 1.493001x | 1.484846–1.506445x | 100.000% |
| amd | height-bilinear-avx2-to-avx2 | 5.729850 | 5.550731 | 1.025431x | 0.975140–1.056289x | 100.000% |
| amd | height-bilinear-avx512-to-avx512 | 9.430519 | 9.695184 | 0.998386x | 0.924534–1.008173x | 100.000% |
| amd | height-bicubic-avx2-to-avx2 | 6.071332 | 5.992891 | 1.040926x | 0.976647–1.084067x | 100.000% |
| amd | height-bicubic-avx512-to-avx512 | 6.044777 | 6.030931 | 0.997927x | 0.956951–1.009729x | 99.074% |
| amd | height-lanczos3-avx2-to-avx2 | 6.886642 | 6.784401 | 1.010092x | 0.947325–1.071344x | 100.000% |
| amd | height-lanczos3-avx512-to-avx512 | 6.764019 | 6.764659 | 1.020464x | 0.944034–1.066787x | 100.000% |
| amd | height-spline64-avx2-to-avx2 | 12.183972 | 12.411140 | 0.968643x | 0.929552–1.015705x | 100.000% |
| amd | height-spline64-avx512-to-avx512 | 7.571489 | 7.522674 | 1.006489x | 0.905124–1.026307x | 100.000% |
| amd | 2d-lanczos3 | 5.120177 | 2.977114 | 1.752619x | 1.678353–1.829392x | 99.507% |
| amd | 2d-spline64 | 6.806168 | 3.851677 | 1.740132x | 1.675432–1.798112x | 99.561% |
| amd | 2d-lanczos6 | 9.364673 | 6.397487 | 1.451453x | 1.428489–1.473226x | 100.000% |

## Method and provenance

- Baseline: published `v0.1.2`, commit `0e9f5b444c8766702a11b5148b35659634226162`. Candidate runtime sources are the release branch; documentation, the workflow, and proof-file deduplication were updated after the benchmark freeze. The compiled CPU sources and CMake configuration are unchanged after that freeze.
- Both versions were freshly built from independent frozen sources on each host with GCC 13.3, Release, native CPU SIMD enabled, and GPU backends disabled. Both versions passed all six CPU/API4 CTests on both hosts.
- Dedicated GCP Spot VMs: `c4-highcpu-2` (Xeon 8581C, northamerica-northeast1-a) and `c4d-highcpu-2` (EPYC 9B45, us-central1-a). No PMU numbers are used in this release comparison. CPU0 affinity is inherited by all benchmark threads. CPU1 idle excludes iowait. Do not compare absolute throughput across these hosts as a controlled CPU-platform study.
- The width input is the deterministic historical pattern `(((i * 17 + 13) % 257) - 128) / 128`, stride 1920, 256 rows, active length 1691.5555555555557 and centered fractional shift. The height fixture has 951 source rows, 952 output rows, 1692 columns, stride 1920, active length 951.5 and shift 0.25. Border mode is mirror.
- Initial stable samples calibrate the faster side to approximately one second; an invalid case is retried as a whole, at most three groups. Cases with no valid group were rerun with a 1.3-second target, synchronization before each group, and deduplication of already-verified proof files. Selection is the first complete valid group, never the best speedup. Every group was below 30 seconds.
- The supplemental 2D input is one predecoded 1920×1080 Float32 frame, SHA-256 `d50727fbfe6a6cb47eee915d5e92372292c41f7280d4d80d8db37312817e36bd`. Output is 1692×952, active height 951.5 and proportional active width, mirror border, aligned 64-byte buffers and output stride 1696. The full `CpuExecutor::inverse_2d` call is timed with fixed iterations and five alternating pairs. Input loading, plan construction, proof writes, and scalar-oracle checks are outside the timer.
- Every measured output is checked across versions with SHA-256. Width/height proofs contain logical rows; the supplemental driver hashes its full initialized output allocation. The supplemental scalar-oracle tolerance is 2e-5, while the release-to-release comparison is exact. No output fill occurs inside the timer.
- The complete frozen sources, compile logs/commands, binaries, inputs, output proofs, and raw logs are retained in the local prerelease evidence archive. The public [data file](release-benchmark-v0.1.3.json) includes all summary groups, rejected environments, selected output hashes, source fingerprints, and binary identities. It does not substitute older incremental GPU or CPU measurements for a new release comparison.

## Reproduce the release-description workload

```sh
export CC=gcc CXX=g++
export DSMVC_VAPOURSYNTH_INCLUDE_DIR=/path/to/vapoursynth/include
python3 benchmarks/release_cpu_version_ab.py \
  --baseline v0.1.2 --output artifacts/release-v013-ab \
  --samples 5 --iterations 3 --stable-seconds 1.3 --jobs 2
```

Run on an otherwise idle Linux x86 host with AVX2, two visible CPUs, and the baseline tag available. The runner freezes both sources, builds/tests both, records capabilities and hashes, and retains numerical proofs. An AVX-512 host also compares the explicit AVX-512 route. The compatibility wrappers accept `DSMVC_BASELINE_REF`; the v0.1.1 wrapper remains available for historical comparisons.

The supplemental driver and paired orchestration are preserved with the evidence as `matrix_timed_driver.cpp` and `followup.py`; their compiled libraries are exactly those used in the release-description workload. This supplemental experiment does not replace the maintained public runner above.
