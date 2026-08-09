# Ill-Conditioned Descale Boundary Analysis

## Scope

This note explains the numerical behavior exposed by the corrected ARM64
release error sweep. It covers:

- why valid fractional descale geometries can produce an ill-conditioned
  least-squares system;
- why two Float32 implementations of the same mathematical operator can return
  visibly different native outputs while reconstructing almost identically;
- how this relates to the commonly observed bad-border behavior of an
  imperfect descale model; and
- which part of the problem is improved by the condition-aware Float64 backend.

The measurements referenced here come from
[`release-benchmark-arm64.md`](release-benchmark-arm64.md). The throughput
tables and charts are outside the scope of this note.

## Executive Conclusion

The error sweep did not create the problem. It scanned enough candidate
geometries to reach discontinuities where the centered output canvas grows by
two pixels while the active resampling length grows by only a fraction of a
pixel. The newly exposed edge unknowns are weakly observed and almost linearly
dependent on their neighbors.

Descale solves this inverse problem through the normal matrix `G = A^T A`.
Normal equations square the condition number of the forward operator. The
affected plans have estimated normal-matrix reciprocal condition numbers near
`2e-6` to `4e-6`, so Float32-sized perturbations in coefficients, right-hand
side accumulation, or substitution order can become percent-scale changes in
the inferred native image.

The condition-aware backend materially improves this numerical failure. For a
risk plan it retains the original Float64 transpose and LDLT factors, performs
RHS accumulation and the full solve in Double, and keeps a Double intermediate
between the two axes. This makes the result much closer to the intended
unregularized least-squares solution and removes backend-dependent Float32
amplification.

It does not make the inverse problem well-conditioned. It also does not remove
model mismatch, source noise, or the weak observability of boundary pixels.
Consequently, a mathematically more accurate unregularized solution can still
have visually unstable borders and may even look more extreme than an old
Float32 result that accidentally damped the weak mode.

## 1. Mathematical Problem

For one axis, let:

- `x` be the unknown native-resolution samples;
- `A` be the forward resampling matrix determined by geometry, kernel, and
  border mode; and
- `y` be the observed higher-resolution samples.

Descale computes the least-squares estimate:

```text
minimize ||A x - y||^2
```

The current solver forms and factorizes the normal equations:

```text
G x = b
G = A^T A
b = A^T y
```

`src/axis_plan.cpp` constructs the sampling geometry in `build_geometry`, the
forward weights in `make_descale_matrix`, and the symmetric banded normal
matrix in `form_normal_bands`. It solves the banded system with LDLT.

The plan depends on geometry and kernel, not frame content. Frame content only
changes `b`, and therefore determines how strongly a weak direction is
excited.

## 2. Geometry Discontinuities Create Weak Edge Unknowns

The release error runner reproduces the GetNative centered-canvas geometry:

```python
output_height = 1000 - 2 * int((1000 - src_height) / 2)
src_top = (output_height - src_height) / 2
```

The horizontal path applies the same construction after deriving `src_width`
from the source aspect ratio. See `build_geometry` in
`benchmarks/e2e_benchmark.py`.

This output size is a staircase function. Crossing an integer boundary can add
two output samples, one on each side, even when the active length increases by
only `0.1`. The plugin passes the resulting destination size, active length,
and shift directly into `AxisRequest` in `src/vs_plugin.cpp`.

The relevant transitions are:

| Candidate axis | Destination | Active length | Shift | First sample position | Result |
|---|---:|---:|---:|---:|---|
| Lanczos2 `978.0` vertical | 978 | 978.0 | 0.0 | 0.45278 | Edge strongly observed |
| Lanczos2 `978.1` vertical | 980 | 978.1 | 0.95 | 1.40282 | Two weak edge unknowns |
| Bilinear `974.2` horizontal | 1732 | 1731.91111 | 0.04444 | 0.49546 | Edge strongly observed |
| Bilinear `974.3` horizontal | 1734 | 1732.08889 | 0.95556 | 1.40662 | Two weak edge unknowns |

The sampling position used by the planner is:

```text
p_i = (i + 0.5) * active_length / source_size + shift
```

For the `bilinear@974.3` horizontal plan, the first observation is
approximately:

```text
y_0 = 0.09338 x_0 + 0.90662 x_1
```

The perturbation:

```text
delta_x_1 = -(0.09338 / 0.90662) delta_x_0
          = -0.1030 delta_x_0
```

almost cancels the first observation. A decaying sequence of compensating
neighbor values can also keep subsequent observations small. The same behavior
appears symmetrically at the other edge.

The edge column is therefore not zero. It is weak and nearly representable by
neighboring columns. In linear-algebra terms, the matrix has a boundary-local
near-null direction.

A local read-only reconstruction of the same sparse matrix showed that more
than `99.99999999%` of the smallest-mode energy for both conditioned fixtures
was contained in the first and last ten samples. This distinguishes the
observed failure from a global Nyquist-frequency mode. Kernel support and
negative lobes affect severity, but the centered-canvas transition is the
primary trigger in these two cases.

## 3. Why Normal Equations Amplify the Problem

For the forward matrix `A`, let its singular values be `sigma_max` and
`sigma_min`. The normal matrix has eigenvalues `sigma_max^2` and
`sigma_min^2`, so in the 2-norm:

```text
kappa(A^T A) = kappa(A)^2
```

The implementation records an estimated 1-norm reciprocal condition number:

```text
rcond(G) approximately 1 / kappa(G)
```

The corrected sweep found:

| Case | Estimated `rcond(G)` | Approximate `kappa(G)` | `kappa(G) * epsilon_float` |
|---|---:|---:|---:|
| Lanczos2 `978.1` vertical | `3.94802e-6` | 253,000 | 0.0302 |
| Bilinear `974.3` horizontal | `2.26783e-6` | 441,000 | 0.0526 |

Here:

```text
epsilon_float = 2^-23 = 1.19209e-7
```

Formal roundoff analysis often uses the unit roundoff
`u = epsilon_float / 2`. Multiple operations and algorithm-dependent constants
make `kappa * epsilon_float` useful as an order-of-magnitude indicator rather
than an exact pixel-error prediction.

The condition number enters because a computed backend result can be viewed as
the exact solution of a slightly perturbed system:

```text
(G + delta_G)(x + delta_x) = b + delta_b
```

To first order:

```text
G delta_x = delta_b - delta_G x
delta_x = G^-1 (delta_b - delta_G x)
```

The corresponding sensitivity scale is:

```text
||delta_x|| / ||x||
    <= kappa(G) * (
         ||delta_G|| / ||G||
         + ||delta_b|| / ||b||
       )
```

This is a worst-direction relative bound, not an equality and not a direct
per-pixel absolute-error formula.

A normalized two-dimensional example shows the mechanism:

```text
G       = diag(1, 2.27e-6)
delta_b = (0, 1.19e-7)

delta_x_weak = 1.19e-7 / 2.27e-6
             = 0.0524
```

The backend introduced only a Float32-scale perturbation, but division by the
weak eigenvalue produced a value near `0.05`. For normalized image values and
an output norm of order one, that is consistent with an observed componentwise
difference near `0.055`. It does not imply that every such plan must produce
that exact error.

The `bilinear@974.3` plan also has a maximum retained inverse diagonal of
`322,349.67`, corresponding to an LDLT pivot near `3.10e-6`. The raw edge
diagonal is larger; the pivot collapses after eliminating the information
already explained by neighboring columns. This is direct evidence of near
linear dependence rather than a missing matrix row.

## 4. Why Native Output Can Change While Reconstruction Does Not

If `q_min` is a weak right-singular direction, then:

```text
G q_min = sigma_min^2 q_min
```

A large solution difference along that direction still has a small forward
effect:

```text
||A delta_x|| = sigma_min ||delta_x||
```

This is why the corrected sweep can observe both:

- a `0.0553046` maximum old/new native-output difference for
  `bilinear@974.3`; and
- only a `2.98023e-6` reconstruction difference for the same candidate.

The small reconstruction residual does not prove that the inferred native
samples are accurate. A backward-stable computation can have a small residual
and still have a large forward error when the problem itself is ill-conditioned.

## 5. Relation to Imperfect Descale and Bad Borders

Real material is better modeled as:

```text
y = (A + delta_A) x + noise
```

`delta_A` can represent incorrect active dimensions, offsets, kernel
parameters, border extension, cropping, sharpening, or other processing that
is absent from the assumed descale model. Compression and source noise add
further perturbations.

Solving with the idealized `A` gives an error component approximately shaped
by:

```text
delta_x = A^+ (delta_A x + noise)
```

The boundary-local near-null mode makes `A^+` large at the edges. This can
produce the commonly observed bad-border pattern:

- a risky horizontal plan produces left/right edge oscillation;
- a risky vertical plan produces top/bottom edge oscillation; and
- two risky axes can concentrate the largest artifact in the corners.

The real-material model mismatch and the backend numerical error are distinct,
but they excite the same weak directions and can stack:

```text
geometry creates the amplifier
model mismatch and noise provide physical input perturbations
Float32 planning and execution add numerical perturbations
```

## 6. Does the Backend Change Improve the Problem?

### What it improves

Yes. It improves the numerical backend component of the problem.

With the default `f64mode=0`, plans with estimated `rcond < 1e-4`:

- retain Float64 transpose weights, LDLT bands, and inverse diagonals;
- bypass Float32 CPU-plan packing;
- accumulate `A^T y` in Double;
- perform forward substitution, diagonal scaling, and backward substitution
  in Double;
- keep the intermediate between horizontal and vertical solves in Double when
  either axis is risky; and
- use ARM64 NEON `float64x2_t` vectors to batch independent Double solves.

`f64mode=1` forces the existing Float32 plan and solve even below the threshold;
`f64mode=2` retains and uses the Float64 path for every plan. The condition
estimate is still computed in all three modes. These controls change numerical
precision, not the least-squares operator or padding semantics.

This is important: retaining a Float64 plan without a Float64 solve would still
allow the backend arithmetic to inject Float32-sized perturbations. Conversely,
running Double arithmetic on already quantized Float32 factors could not
recover the lost factor precision. The risk path needs both.

The high-precision references show that this is a real improvement rather than
mere agreement with a different backend:

| Fixture | Float32-path error | Condition-aware error | Evidence |
|---|---:|---:|---|
| Lanczos2 `978.1` axis | about `2.51e-4` | about `2.97e-6` | Direct Float64 Householder QR |
| Bilinear `974.3` maximum point | `0.0553096` | `4.96776e-6` | Independent 60-digit solve |

At the bilinear point, the condition-aware result is about 11,100 times closer
to the intended least-squares result.

### What it does not improve

The backend change does not:

- increase the smallest singular value of `A`;
- remove the centered-canvas geometry discontinuity;
- add information about the weak edge samples;
- correct an incorrect kernel, offset, border mode, or source-processing model;
- suppress source noise in the weak mode; or
- guarantee visually natural border values.

The result can therefore be numerically correct under the current
unregularized semantics and still be physically uncertain at the edge. The old
Float32 result may occasionally look less extreme because factor quantization
or solve error acted as accidental damping. That appearance is not evidence
that it was closer to the mathematical solution.

### Backend routing consequence

The current Metal shaders implement the Float32 solve. A retained Float64 plan
is therefore excluded from Metal GPU eligibility and executed by the
heterogeneous scheduler's CPU lane. CUDA and Vulkan reject retained Float64
plans explicitly. Well-conditioned automatic plans and `f64mode=1` plans remain
on the existing Float32 fast paths.

An actual GPU implementation of the same corrected behavior would need to
consume the retained Float64 plan and perform the risk solve at sufficient
precision. Merely sending the existing Float32 plan to another device would
not address the conditioning failure.

## 7. Semantic Alternatives and Their Tradeoffs

The current change preserves the existing unregularized descale operator. Other
approaches can reduce visible bad borders, but they solve a different problem:

- Removing or cropping weak padding samples changes output geometry.
- Constraining boundary samples changes the inverse model.
- Tikhonov or edge regularization solves
  `(A^T A + lambda R) x = A^T y` and intentionally damps weak modes.
- Direct QR or SVD avoids squaring the condition number, but does not create
  information absent from `A` and has a different performance profile.

Those options may be appropriate for a visual reconstruction mode, but they
must not be presented as bit-compatible implementations of the current descale
semantics. The condition-aware Float64 fallback is the narrower change: it
computes the existing least-squares definition more accurately.

## 8. Evidence Boundaries

Established by current source, tests, and release artifacts:

- the exact candidate geometries and condition estimates;
- the boundary-local weak pivots;
- Float64-plan retention and Double CPU execution;
- direct QR and 60-digit reference comparisons;
- unchanged selected candidates and per-family minima; and
- CPU routing for every risk plan observed in the Metal error audit.

Not established by this error sweep:

- that all visually bad descale borders are caused by this mechanism;
- that Float64 removes real-material model mismatch;
- that an unregularized exact solution is always the most visually useful
  output; or
- that Metal has an independent Float64 numerical result for these candidates.
