# Ordered RHS span specialization (2026-09-07)

The F32 AVX2 and NEON horizontal paths now specialize common transposed-weight spans: 6/7 terms for B5, 9/10 for B7, and 13/14 for B11. The shared helper preserves ascending source-term FMA order and uses the existing fallback for every other span. Both fixed AVX2 rows and the generic horizontal path used by fused 2D are covered.

| Kernel | AVX2 rows time reduction | AVX2 2D time reduction | NEON rows time reduction | NEON 2D time reduction |
| --- | --- | --- | --- | --- |
| lanczos3 | 9.80% | 8.32% | 7.40% | 5.10% |
| spline64 | 5.51% | 5.80% | 8.23% | 4.89% |
| lanczos6 | 2.74% | 4.46% | 6.31% | 3.17% |

These are median wall-time reductions from three short same-host pairs on C4 (GCC 13.3, AVX2) and C4A (GCC 15.2, NEON), using the same preloaded real F32 frame, 64-byte aligned buffers, and active height 951.5. Total timing, including rejected/repeated samples, was 19.55 s on C4 and 15.30 s on C4A. This is an engine benchmark, not full video-workflow timing.

Both Linux native CTest suites passed 6/6 and ASan/UBSan passed 5/5. The Mac existing configuration passed 10/10. Fifty-four additional baseline/candidate configurations had byte-identical complete output buffers, including alternate geometries and integer/F64 controls. Guard-page tests explicitly exercise both terms of each specialized span pair, odd strides, tails, and 2D execution.

Full evidence: `artifacts/cpu-rhs-span-20260907/report.md`. Other pending vertical/grouping/writeback experiments are separate from this change.
