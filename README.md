# dsmvc

> **Release performance:** `dsmvc` reaches **13.13x** the original descale
> throughput on the full `getfnative` scan at R1T1, and remains **6.45x faster**
> at R32T32. See the [full release benchmark](docs/release-benchmark.md) for
> E2E, fixed-kernel, BlankClip, and output-compatibility results.

`dsmvc` is a VapourSynth API4 plugin compatible with the public filter API of
Irrational-Encoding-Wizardry/descale. It registers only the `dsmvc` namespace:

```python
core.std.LoadPlugin(path=r"C:\path\to\dsmvc.dll")
output = core.dsmvc.Debicubic(source, 1280, 720, b=0.0, c=0.5)
# CUDA-enabled builds opt in explicitly:
gpu_output = core.dsmvc.Debicubic(
    source, 1280, 720, b=0.0, c=0.5, backend="cuda")
# Vulkan-enabled builds use the same explicit backend contract:
vulkan_output = core.dsmvc.Debicubic(
    source, 1280, 720, b=0.0, c=0.5, backend="vulkan")
```

The plugin identifier is `com.dsmvc.descale`. It intentionally exports the
API4 `VapourSynthPluginInit2` entry point only; API3 hosts are not supported.
It does not register a `core.descale` alias.

## Filters

The following functions preserve the baseline arguments, then append the
optional `backend:data`, `padding:int`, and `f64mode:int` controls:

- `Debilinear`
- `Debicubic`
- `Delanczos`
- `Despline16`
- `Despline36`
- `Despline64`
- `Descale`

`Descale` accepts the baseline custom-kernel forms `custom`/`support` and
`custom_kernel`/`taps`. If both aliases are supplied, baseline precedence is
preserved: `custom` wins over `custom_kernel`, and `taps` wins over `support`.

`padding` selects the explicit edge extension and defaults to `3`:

- `0`: zero padding
- `1`: repeat the nearest edge sample
- `2`: periodic reflect101 (`... 2 1 | 0 1 2 ...`), without duplicating the edge
- `3`: periodic symmetric (`... 1 0 | 0 1 2 ...`), with a duplicated edge

The legacy `border_handling` argument remains available, but cannot be combined
with `padding`. Its `1` and `2` values retain zero and repeat behavior. Value `0`
(and other legacy values) selects the original descale mirror mapping: for a
half-pixel tap center `x`, negative `x` maps to `-x` and right-side `x` maps to
`min(2*N-x, N-0.5)`. This is an edge-duplicating symmetric reflection for the
first image-width only, not a periodic extension. Original descale commit
`8c53f5d` applies no second reflection or bounds guard for extreme custom
support; dsmvc safely discards a tap that remains out of range after the legacy
mapping.

`f64mode` controls CPU solve precision: `0` (default) automatically retains and
uses Float64 factors when the normal matrix has estimated `rcond < 1e-4`, `1`
forces the Float32 path, and `2` forces the Float64 path. The condition estimate
is retained in all three modes. CUDA and Vulkan reject a Float64 plan; the
plugin-level Metal scheduler keeps its heterogeneous contract and routes such a
plan to its CPU fallback.

`backend` accepts `auto`, `cpu`, `metal`, `vulkan`, or `cuda`. A normal build
uses CPU for `auto`. Enabled builds accept `cuda` or `vulkan` on a compatible
device. The opt-in Apple ARM64 Metal build adds explicit fixed-recipe GRAYS and
YUV420P8/P10 routes. Its automatic route remains limited to measured
wide-kernel YUV420 clips with at least 256 frames, at least 16 core threads, and
a 16-callback admission window on a unified-memory Apple M-series device. Short
clips, narrow kernels, GRAYS, low concurrency, and unsupported geometry remain
on CPU under `auto`. An unavailable or uncompiled explicit backend raises an
error. Explicit CUDA and Vulkan do not silently fall back; explicit Metal is a
mixed plugin-level route and publishes whether a frame actually used Metal.

For the CPU backend, `opt=1` selects the scalar path and `opt=2` strictly
requires the architecture's native SIMD path: AVX2/FMA on x86-64 or NEON/FMA
on AArch64. The default selects native SIMD when available. Other numeric
values retain baseline behavior and select automatic dispatch. The Python
wrapper accepts either these integers or `Opt.AUTO`, `Opt.NONE`, `Opt.AVX2`,
`Opt.NEON`, and `Opt.SIMD`; the last three names all preserve the legacy
`opt=2` value.

Planning is deferred until the first requested frame. The inverse-only planner
uses Float64 CSR and banded LDLT construction, stores immutable Float32
coefficients, and retains Float64 factors according to `f64mode`. Built-in plans
use bounded exact-key single-flight LRU caching; sampling geometry is cached
separately so kernel families and precision modes can reuse it. SIMD packed
plans are shared by filters that share a canonical Float32 plan.

The Python wrapper is [dsmvc.py](dsmvc.py). It preserves the
baseline RGB, YUV, GRAY, bit-depth, subsampling, `yuv444`, `gray`, and chroma
conversion behavior while dispatching to `core.dsmvc`.

## Build

Requirements:

- CMake 3.24 or newer
- A C++23 compiler
- VapourSynth API4 headers
- On Windows, Visual Studio 2022 with the x64 C++ workload

Native Apple ARM64 Release and RelWithDebInfo builds use `-O3 -flto=full`
without an Apple-model-specific `-mcpu` setting. CMake selects the NEON source
only for AArch64 and the existing AVX2/FMA source only for x86-64; universal
macOS builds must be produced as separate architecture builds.

The Metal backend is enabled by default only for native Apple ARM64 builds with
the Xcode Metal toolchain. Release packages set the backend and deployment
target explicitly; the supported macOS baseline is 13.3:

```sh
cmake -S . -B build-metal -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=13.3 \
  -DDSMVC_VAPOURSYNTH_SDK=/path/to/vapoursynth \
  -DDSMVC_ENABLE_METAL=ON \
  -DBUILD_TESTING=ON
cmake --build build-metal --parallel
ctest --test-dir build-metal --output-on-failure
```

Non-Apple builds do not compile or link the Metal executor. Pass
`-DDSMVC_ENABLE_METAL=OFF` to produce a CPU-only Apple ARM64 build. Supported
formats, geometry, device admission, batch sizes, and automatic routing remain
limited to validated cases.

The Release DLL is written to `build/Release/dsmvc.dll`. The build uses the
static MSVC runtime so that an older runtime DLL bundled with a host cannot
change the STL synchronization ABI.

The CUDA backend is optional and currently supports Windows and Linux with a
CUDA Toolkit containing the CUDA compiler and static Runtime library.
CUDA-enabled test builds also require `cuobjdump`:

```sh
cmake -S . -B build-cuda \
  -DDSMVC_VAPOURSYNTH_SDK=/path/to/vapoursynth \
  -DDSMVC_ENABLE_CUDA=ON
cmake --build build-cuda --config Release --parallel
```

The build embeds native SM75, SM86, SM89, and SM120 kernels by default, plus
compute_75 and compute_120 PTX fallbacks. The matrix is configurable with
`DSMVC_CUDA_ARCHITECTURES` and `DSMVC_CUDA_PTX_ARCHITECTURES`. Architectures
without a native image use driver JIT compilation and may incur a first-run
delay. The plugin statically links the official CUDA Runtime and therefore does
not require a CUDA Runtime DLL or shared library beside it. CUDA Runtime device
code, streams, and allocations use the device's primary context so they can
coexist with other CUDA filters in the host process. At runtime,
`DSMVC_CUDA_STREAMS` may be set to a
value from 1 through 16. When unset, the runtime uses up to eight concurrent
slots for reused-input 2D plans with half-bandwidth seven or greater, while
narrower 2D and one-axis work remain limited to four. Float32 2D frames use a
separate bounded pinned-staging pool: unique inputs admit two GPU executions
and four host staging phases at once, while input-cache hits can expand to the
full slot count. This keeps host packing and unpacking outside the lifetime of
a GPU execution slot. Integer and one-axis paths retain slot-local staging.
An explicit stream value applies uniformly and disables adaptive slot limits.
Set `DSMVC_CUDA_HOST_TRANSFER=staging` to restore slot-local Float32 staging
for comparison. The `pageable` and `registered` values select diagnostic
direct-copy paths; per-frame driver staging or host registration made both
slower than pinned staging on the reference system.
`DSMVC_CUDA_INPUT_CACHE_MB` bounds the shared immutable-input cache and defaults
to 64 MiB; set it to `0` to disable source upload and first-transpose reuse.
Keep the cache enabled for repeated-source workloads such as GetNative scans;
disable it for long unique-frame runs when reuse is not expected.
`DSMVC_CUDA_PLAN_CACHE_MB` bounds the device-resident packed-plan LRU and
defaults to 16 MiB. This deliberately small default is appropriate for
GetNative-style height scans, where most plans are used once; increase it for a
long-lived workload that repeatedly reuses many distinct plans.
Use an explicit low stream count only when reducing peak device memory is more
important than adaptive mixed-workload throughput.
For a GetNative graph containing tens of thousands of candidate frames, also
set a finite VapourSynth cache before constructing the graph, for example
`core.max_cache_size = 512`. VapourSynth's default can otherwise retain several
GiB of host frames independently of the CUDA cache limits.
For half-bandwidth three and above, RHS evaluation is split from the recursive
solve adaptively by default: the first execution of a packed plan stays fused,
and later executions use the higher-throughput RHS kernel. Set
`DSMVC_CUDA_SPLIT_RHS=0` to disable it or `force` to use it from the first
execution. `DSMVC_CUDA_SPLIT_HORIZONTAL_THREADS` and
`DSMVC_CUDA_SPLIT_VERTICAL_THREADS` accept power-of-two values from 16 through
256 and default to 32. `DSMVC_CUDA_HORIZONTAL_GLOBAL_TRANSPOSE` defaults to
`1`; the row-major `0` mode is available for experiments but is not the tested
default.
CUDA uses hardware fused multiply-add, so Float32 output is numerically checked
against the scalar backend but is not promised to be bit-identical.

The Vulkan 1.2 backend is optional on Windows and Linux. A build requires the
Vulkan SDK headers, loader, and `glslc`; test builds also require `spirv-val`:

```sh
cmake -S . -B build-vulkan \
  -DDSMVC_VAPOURSYNTH_SDK=/path/to/vapoursynth \
  -DDSMVC_ENABLE_VULKAN=ON
cmake --build build-vulkan --config Release --parallel
```

GLSL is compiled with `--target-env=vulkan1.2 -O`, validated in test builds,
and embedded in the plugin. Installed packages need only the system Vulkan
loader and a Vulkan 1.2 driver; they do not load shader sidecars or a runtime
compiler. Devices need a compute queue and the Vulkan core compute limits. No
FP16/FP64, subgroup, narrow storage, descriptor indexing, or synchronization2
feature is required.

By default device selection scores discrete, integrated, virtual, and CPU
devices in that order and prefers a compute-only queue. Set
`DSMVC_VULKAN_DEVICE` to a Vulkan enumeration index or hexadecimal
`vendor_id:device_id`; an invalid or ineligible explicit selection reports all
detected devices and does not fall back. `DSMVC_VULKAN_SLOTS=1..16` overrides
the adaptive four-slot default and eight-slot heavy-plan expansion.
`DSMVC_VULKAN_PLAN_CACHE_MB` and `DSMVC_VULKAN_INPUT_CACHE_MB` default to 16
and 64 MiB. `DSMVC_VULKAN_SPLIT_RHS=0|adaptive|force` controls fused versus
split RHS execution. `DSMVC_VULKAN_VALIDATION=1` enables the Khronos validation
layer when installed. `DSMVC_VULKAN_WORKGROUP=128` forces the core-limit
fallback variants for diagnostics. Float32 results use explicit shader `fma`
and are numerically checked, not promised to be bit-identical to CPU or CUDA.

## Benchmark

The reproducible old/new runner and its case definitions are documented in
[benchmarks/README.md](benchmarks/README.md). A full run uses the fixed input
and baseline hashes, executes each implementation in separate processes, and
writes JSON, CSV, Markdown, command lines, images, difference maps, and error
curves outside the VapourSynth installation.

The API3/API4 CPU regression runner and the opt-in Apple ARM64 Metal validation
workflow are documented in [benchmarks/README.md](benchmarks/README.md).
`benchmarks/validate_api4_apple_arm.sh` is the release gate for this migration;
it records architecture-specific build commands, correctness and identity
checks, paired CPU/Metal results, system-pressure snapshots, and required CI
evidence.

For the real-video end-to-end comparison based on the supplied training HTML,
`test_getfnative*.vpy`, and `test_selectkernel.vpy`, use
`benchmarks/e2e_benchmark.py`. It accepts the
explicit MKV, current plugin, original plugin, VapourSynth Python, and VSPipe
paths, and writes paired performance and error reports.

## License

dsmvc is MIT licensed. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
