# dsmvc reference

Detailed argument behavior, the numerical contract, and CPU/planner internals.
The [README](../README.md) has the short version.

## Function signatures

```text
core.dsmvc.Debilinear(clip src, int width, int height, float blur=1.0, float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=None, int force=0, int force_h=force, int force_v=force, int opt=0, data backend="auto", int padding=3, int f64mode=0)

core.dsmvc.Debicubic(clip src, int width, int height, float b=0.0, float c=0.5, float blur=1.0, float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=None, int force=0, int force_h=force, int force_v=force, int opt=0, data backend="auto", int padding=3, int f64mode=0)

core.dsmvc.Delanczos(clip src, int width, int height, int taps=3, float blur=1.0, float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=None, int force=0, int force_h=force, int force_v=force, int opt=0, data backend="auto", int padding=3, int f64mode=0)

core.dsmvc.Despline16(clip src, int width, int height, float blur=1.0, float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=None, int force=0, int force_h=force, int force_v=force, int opt=0, data backend="auto", int padding=3, int f64mode=0)

core.dsmvc.Despline36(...)

core.dsmvc.Despline64(...)

core.dsmvc.Descale(clip src, int width, int height, data kernel=None, int taps=3, float b=0.0, float c=0.5, float blur=1.0, float src_left=0.0, float src_top=0.0, float src_width=width, float src_height=height, int border_handling=None, int force=0, int force_h=force, int force_v=force, int opt=0, func custom=None, int support=None, func custom_kernel=None, data backend="auto", int padding=3, int f64mode=0)
```

`padding` and the legacy `border_handling` argument are mutually exclusive.
Generic `Descale` requires either `kernel` or a custom-kernel callback; custom
kernels also require `taps` or `support`.

The plugin identifier is `com.dsmvc.descale`. It intentionally exports the
API4 `VapourSynthPluginInit2` entry point only; API3 hosts are not supported.
It does not register a `core.descale` alias.

## Arguments

The native functions preserve the baseline arguments, add JET v12-compatible
`blur:float`, then append the optional `backend:data`, `padding:int`, and
`f64mode:int` controls shown above.

`Descale` accepts the baseline custom-kernel forms `custom`/`support` and
`custom_kernel`/`taps`. If both aliases are supplied, baseline precedence is
preserved: `custom` wins over `custom_kernel`, and `taps` wins over `support`.

### `blur`

`blur` defaults to `1.0` and is available on every fixed-kernel entry, generic
`Descale`, and the Python wrappers. It stretches a kernel entirely during plan
construction:

```text
effective_support = ceil(base_support * blur)
weight = kernel(distance / blur)
```

The value must be finite and greater than zero, and the plugin requires it to
be smaller than the minimum input-plane extent. Lanczos continues to use its
original `taps` value for the window even when effective support grows. The
default `blur=1.0` plan follows the original coefficient path without an added
division. Wider supports increase planning, factor, and execution cost, and may
change the normal matrix enough for automatic `f64mode=0` to retain Float64.
There is no extra per-frame convolution, frame, or memory round trip.

### `padding` and `border_handling`

`padding` selects the explicit edge extension and defaults to `3`:

| Value | Behavior |
|---|---|
| `0` | zero padding |
| `1` | repeat the nearest edge sample |
| `2` | periodic reflect101 (`... 2 1 \| 0 1 2 ...`), without duplicating the edge |
| `3` | periodic symmetric (`... 1 0 \| 0 1 2 ...`), with a duplicated edge |

The legacy `border_handling` argument remains available, but cannot be combined
with `padding`. Its `1` and `2` values retain zero and repeat behavior. Value `0`
(and other legacy values) selects the original descale mirror mapping: for a
half-pixel tap center `x`, negative `x` maps to `-x` and right-side `x` maps to
`min(2*N-x, N-0.5)`. This is an edge-duplicating symmetric reflection for the
first image-width only, not a periodic extension. Original descale commit
`8c53f5d` applies no second reflection or bounds guard for extreme custom
support; dsmvc safely discards a tap that remains out of range after the legacy
mapping.

### `f64mode`

`f64mode` controls solve precision: `0` (default) automatically retains and
uses Float64 factors when the normal matrix has estimated `rcond < 1e-4`, `1`
forces the Float32 path, and `2` forces the Float64 path. The condition estimate
is retained in all three modes. Explicit CUDA and Vulkan execute Float64 plans
natively or raise an error; they never fall back to CPU. The plugin-level Metal
scheduler routes a Float64 plan to its CPU fallback.

### `backend`

`backend` accepts `auto`, `cpu`, `metal`, `vulkan`, or `cuda`. A normal build
uses CPU for `auto`. Enabled builds accept `cuda` or `vulkan` on a compatible
device. The opt-in Apple ARM64 Metal build adds explicit fixed-recipe GRAYS and
YUV420P8/P10 routes. Its automatic route requires at least 64 frames, at least
8 core threads, a wide kernel, sufficient measured work, and concurrent or
resident-source admission on a unified-memory Apple M-series device. GRAY as
well as YUV formats may participate when their format and dimensions satisfy
those gates. Short clips, narrow kernels, low concurrency, and unsupported
geometry remain on CPU under `auto`. An unavailable or uncompiled explicit
backend raises an error. Explicit CUDA and Vulkan do not silently fall back;
explicit Metal is a mixed plugin-level route and publishes whether a frame
actually used Metal.

### `opt`

For the CPU backend, `opt=1` selects the scalar path and `opt=2` strictly
requires AVX2/FMA on x86-64 or NEON/FMA on AArch64. On x86-64, `opt=3`
strictly requires AVX-512F/DQ/BW/VL and FMA. The default selects AVX-512,
then AVX2 or NEON, when available. Other numeric values retain baseline
behavior and select automatic dispatch. The Python wrapper accepts either
these integers or `Opt.AUTO`, `Opt.NONE`, `Opt.AVX2`, `Opt.AVX512`, `Opt.NEON`,
and `Opt.SIMD`; `Opt.AVX2`, `Opt.NEON`, and `Opt.SIMD` preserve `opt=2`, while
`Opt.AVX512` is the new explicit value.

### Python wrapper

The Python wrapper is [dsmvc.py](../dsmvc.py). It preserves the
baseline RGB, YUV, GRAY, bit-depth, subsampling, `yuv444`, `gray`, and chroma
conversion behavior while dispatching to `core.dsmvc`.

## Numerical contract

Vulkan Float64 requires a strict device capability contract:
`shaderFloat64`, RTE Float64 rounding, and NaN/Inf/signed-zero preservation.
Devices that do not meet it are rejected with an explicit error, never a
silent fallback. Denorm preservation is probed and recorded; devices without
it (including NVIDIA) run with a documented flush-to-zero boundary for
subnormal intermediates.

The plugin-level Metal scheduler keeps its heterogeneous contract and
routes Float64 plans to its CPU fallback; the direct Metal executor rejects
Float64 plans explicitly.

Float64 float output matches the CPU scalar Float64 reference within 1 output
ULP on every backend. For both F32 and F64 solves, U8/U10/U16 output differs
from the same-precision CPU scalar reference by at most 1 code; pairwise
backend differences are not the contract. Repeated execution on one concrete
route is bit-exact. Optimized CPU routes, including buffered/streamed and
fused/two-pass dynamic routes, are compared within the documented Float32
tolerance because SIMD width and FMA evaluation order may differ. Automatic routing never selects a
Float64 plan on a GPU backend -- that admission requires paired plugin-level
evidence against the optimized CPU Float64 path, which has not yet been
collected.

Retained-F64 entry points reject NaN or infinity in their input or result with
`std::runtime_error`; signed zero and finite subnormals remain valid. Public
matrix executors access only each row's logical width. A caller may allocate
exactly `(rows - 1) * stride + logical_width` elements, without full pitch
padding after the final row, and output padding is never modified.

## CPU paths

The AVX-512 path keeps F32 `Delanczos(taps=3)` and `Despline64` width-axis
solves in 16-row blocks, with ordered specialization of common RHS spans.
Other width-axis kernels stay on AVX2. F32 height-axis solves with
half-bandwidth 5, 7, or 11 share coefficients across 64 columns, retaining
16-column blocks and the existing tail handling for the remainder.
When both F32 axes have one of those half-bandwidths, the output width is at
least 64, and the input height is at least 16, the AVX-512 CPU path uses
separate horizontal and vertical passes with a 64-byte aligned intermediate.
The horizontal pass keeps its existing ISA selection; Lanczos6 horizontal
work still uses AVX2. Other 2D shapes, F64, and integer operations keep their
existing kernels. NEON half-bandwidth-11 height-axis solves share coefficients
across 16 columns. See the [CPU release benchmark](release-benchmark-v0.1.3.md)
for measured gains and validation boundaries.
The AVX2 F32 `Debilinear` / `Debicubic` / `Delanczos(taps=3)` / `Despline64`
width-axis kernels write complete 8-wide tiles directly and keep only the
final partial tile in local storage, avoiding a full padded-output copy for
widths that are not divisible by eight.

At the public engine level, explicitly requesting `CpuPath::avx2`,
`CpuPath::avx512`, or
`CpuPath::neon` also requires that exact path to be compiled and supported by
the current CPU; it raises an error otherwise. Only `CpuPath::automatic` may
fall back to scalar.

## Planning and caching

Planning is deferred until the first requested frame. The inverse-only planner
uses Float64 CSR and banded LDLT construction, stores immutable Float32
coefficients, and retains Float64 factors according to `f64mode`. Built-in plans
use bounded exact-key single-flight LRU caching; the exact blur bit pattern is
part of the plan key. Sampling geometry is cached separately by effective
support, so different blur values in the same support tier can reuse it. SIMD
packed plans are shared by filters that share a canonical Float32 plan.
