# Shared CPU SIMD optimization, 2026-09-07

Wide-band Float32 solvers repeatedly loaded recurrence state from transposed output rows. They now use aligned contiguous float workspaces and a shared fixed-index recurrence window. The NEON, generic/fixed AVX2 and AVX-512 implementations preserve the original coefficient/FMA order and public buffer contracts. B5/B7/B9/B11 use the shared window where the existing ISA dispatch permits it; AVX-512 horizontal dispatch remains B5/B7. Common AVX-512 tap counts are unrolled in their original order.

This exploration follows the user's instruction to remove fixed percentage performance thresholds. Retention is based on repeatable workload-specific benefit, numerical compatibility and implementation value. It is not a blanket no-regression claim.

## Incremental measurements

The baseline includes the preceding AVX-512 alignment/B7 change on top of `0e9f5b4`. Its source archive SHA256 is `ee2e44e6856c2bc232b830f1a582c022923032562db3d1f4aab3b60a56ea169b`.

At active height 951.5, paired wall-time reductions are:

| Host/path | Lanczos3 rows | Spline64 rows | Lanczos6 rows |
|---|---:|---:|---:|
| C4A, GCC15.2, NEON | 23.44% | 19.61% | 31.15% |
| C4, GCC13.3, AVX2 | 22.52% | 27.68% | 45.25% |
| C4, GCC13.3, AVX-512 | 10.50% | 4.10% | AVX2 fallback |

The C4A explicit two-pass reductions are 12.23%, 11.47% and 18.69%. C4's direct fused 2D route remains AVX2 and improves by approximately 27–30% for these kernels. These are preloaded-frame engine measurements, with three alternating pairs and identical fixed work per pair. Cloud processes and their workers run on CPU0. External float arrays in this fixture are 16 modulo 64; native VS buffers have their own alignment.

Full VSPipe checks use 320 consecutive video frames, 319 uniformly sampled GetFnative candidates, 320 v2 candidates, and 101 SelectKernel candidates. Each workload has two alternating pairs. C4A Lanczos6 video improves 6.38%; C4 automatic Lanczos6 improves 7.15%, and C4 GetFnative improves 5.78%. Small changes remain inconclusive: C4 automatic Lanczos4/Spline64 video estimates are -0.93%/-0.45%, and v2/SelectKernel have mixed pair directions. All raw pairs are retained.

A separate M4 Max / Apple Clang21 check has five pairs: rows improve 39.29%, 34.32% and 44.37%. It is unbound on macOS and is supplementary evidence, not a cross-host speed comparison.

## Validation and profile interpretation

- 141 final cloud stage configurations; 29 real-workflow configurations across C4A and C4.
- 8,840 real outputs match their same-host baseline in pixel SHA256 and PlaneStats float values. Stage hashes match; maximum F32 scalar-reference error is 1.07288e-6 and integer error is one code value.
- Release CTest 6/6 and ASan/UBSan C++ tests 5/5 pass on each cloud architecture; native Mac CTest 5/5 passes. Expanded guard-page cases cover narrow systems, SIMD tails, scratch reuse, padding and buffered/streamed wide-band integer calls.
- PMU event running rates are 100%, with no lost samples in 28 phase and six workflow recordings. A first AVX2 layout attempt missed one integer caller's allocation and failed tests; it was repaired before accepted measurements and retained as an excluded experiment.
- NEON rows retire about 33–47% fewer instructions, but some L1/L2 refill counts increase. AVX-512 rows retire slightly more instructions while using fewer cycles. The same source transformation need not improve every microarchitectural counter.

The updated C4A Lanczos6 2D profile attributes 49.01% of cycles samples to the generic vertical loop and 43.34% to the horizontal window. Next experiments should compare wider B11 column groups, coefficient layout and intermediate-buffer traffic while preserving operation order. In the tested R75 GetNative-like scripts, Expr remains the largest sample source (68.25%/75.77%); optimizing that interpreter-bound pipeline is separate client/runtime work.

Detailed results, the NEON route, raw data, source/binary manifests and cleanup proofs are under `artifacts/cpu-simd-explore-20260907/`, with `report.md` and `neon-route.md` as entry points. Both Spot VMs and auto-delete disks were deleted after archive verification. Final source differs from measured source only by removing three trailing spaces, recorded in `format-only-changes.json`. No commit or deployment is included.
