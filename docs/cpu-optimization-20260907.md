# CPU optimization and ARM PMU, 2026-09-07

The AVX-512 horizontal solver now aligns its two private thread-local workspaces to 64 bytes and specializes the B7 forward/backward substitutions, reusing the existing B5 fixed-band implementation. Each workspace reserves 15 extra floats so its 16-float blocks stay within a cache line. Public input/output alignment and the descending FMA order are unchanged. The existing CPU path selection remains in place.

The SIMD buffer contract test adds 35 guard-page cases covering the smallest legal B7 system, workspace growth and shrinkage, row-block boundaries, padded strides, and output padding. AxisPlan validation and native 16×16 transpose experiments were rejected after end-to-end screening and are absent from the final change.

## Scoped C4 gate

Baseline: `0e9f5b444c8766702a11b5148b35659634226162`. Candidate uses the same GCC 13.3 Release build on a GCP C4 Spot VM, Xeon Platinum 8581C / Emerald Rapids. Workloads use CPU0, one VS thread and one request, the historical DIGIMON episode 40 source, FFMS2, and the repository VSPipe recipes. Each case has six alternating paired runs without profiling overhead. Process wall time includes decode, graph construction, execution and output disposal.

| Workload | Outputs | Paired wall-time reduction | Bootstrap 95% interval |
|---|---:|---:|---:|
| Lanczos4 video | 3,200 | 3.683% | 1.660% to 4.836% |
| Spline64 video | 3,200 | 3.206% | 2.802% to 3.814% |
| GetFnative | 3,190 | 2.391% | 0.554% to 3.368% |
| GetFnative v2 | 3,200 | 0.612% | -7.013% to 5.499% |

Both B7 targets meet the established point-estimate threshold of at least 3%, with positive confidence intervals. The full 13-case set has no statistically significant regression exceeding 1%. This does not establish 1% noninferiority for every protected case; v2 and SelectKernel remain noisy. Eight other video controls use 320 consecutive frames, and SelectKernel uses 101 candidates. GetFnative selects 290 uniformly spaced, endpoint-preserving heights for each of its 11 scalers from the original full range.

Short controls, v2 and SelectKernel were repeated with pipe-EOF timing to remove the original file-output wait polling granularity. Their complete case groups replace the initial measurements; all successful samples and superseded groups are retained. The primary dataset has 156 accepted runs. Paired median speedup is converted to wall reduction; intervals use 50,000 paired bootstrap resamples from the existing benchmark analysis helper.

PMU measurements support the intended mechanism: per-call split loads decrease by about 93% for Lanczos3 rows and 95% for Spline64 rows; Spline64 instructions per call decrease by about 33%. These phase counters are diagnostic, while acceptance uses the complete plugin workflow above.

Validation: all 15,451 workflow outputs match the baseline exactly in logical pixel bytes and PlaneStats floating-point values; x86 Release CTest 6/6, ASan+UBSan C++ tests 5/5, and macOS native CTest 5/5 pass. The measured source and final local source match. No claim extends to all AVX-512 hosts, concurrent request configurations, full-video scene coverage, or unmodified F64/AVX2/ARM paths.

## ARM findings

A separate C4A Spot VM, Neoverse V2 and GCC 15.2, completed 121 phase configurations, 13 bounded real workflows, and Arm Top-down on 19 phase configurations plus three real workflows. Native baseline CTest passes 6/6. No ARM implementation change is included.

For GetFnative, Expr and PlaneStats account for 71.88% and 6.40% of cycles samples; dsmvc accounts for 13.01%. For v2 the corresponding shares are 80.94%, 7.13% and 6.02%. In the tested VapourSynth R75 source, Expr's compiler construction is x86-only and ARM evaluates this expression through the per-pixel interpreter. An equivalent vectorized error-expression implementation is therefore the highest-priority pipeline experiment, with exact pixel/statistic validation required.

Wide-kernel video scans shift work into dsmvc: its share reaches 50.76% for Lanczos6. The Lanczos3 horizontal phase spends 87.15% of samples in the generic horizontal solver. Fixed-band NEON substitution and independent row-group scheduling merit separate paired experiments. CPU Frontend Bound is distinct from the software AxisPlan construction stage, and sampling fractions are not exact component wall-time shares.

Raw results, source/binary manifests, rejected attempts, all confidence intervals and cleanup proofs are preserved under `artifacts/cpu-opt-arm-pmu-20260907/`, with `report.md` and `arm-report.md` as entry points. Both Spot VMs and their auto-delete disks were removed after local archive hash verification; snapshots remain intact.
