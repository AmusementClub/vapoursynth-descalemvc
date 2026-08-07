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
```

The plugin identifier is `com.dsmvc.descale`. It intentionally exports the
API4 `VapourSynthPluginInit2` entry point only; API3 hosts are not supported.
It does not register a `core.descale` alias.

## Filters

The following functions match the baseline argument order and behavior, with
one optional `backend:data` argument appended at the end:

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

`backend` accepts `auto`, `cpu`, `metal`, `vulkan`, or `cuda`. A normal build
uses CPU for `auto`; Metal and Vulkan remain capability stubs. A CUDA-enabled
build accepts `cuda` on a compatible NVIDIA device. The opt-in Apple ARM64
Metal build adds explicit fixed-recipe GRAYS and YUV420P8/P10 routes. Its
automatic route remains limited to measured wide-kernel YUV420 clips with at
least 256 frames, at least 16 core threads, and a 16-callback admission window
on a unified-memory Apple M-series device. Short clips, narrow kernels, GRAYS,
low concurrency, and unsupported geometry remain on CPU under `auto`. An
unavailable or uncompiled explicit backend raises an error and never silently
falls back to CPU.

For the CPU backend, `opt=1` selects the scalar path and `opt=2` strictly
requires the architecture's native SIMD path: AVX2/FMA on x86-64 or NEON/FMA
on AArch64. The default selects native SIMD when available. Other numeric
values retain baseline behavior and select automatic dispatch. The Python
wrapper accepts either these integers or `Opt.AUTO`, `Opt.NONE`, `Opt.AVX2`,
`Opt.NEON`, and `Opt.SIMD`; the last three names all preserve the legacy
`opt=2` value.

Planning is deferred until the first requested frame. The inverse-only planner
uses Float64 CSR and banded LDLT construction, then stores immutable Float32
coefficients. Built-in plans use bounded exact-key single-flight LRU caching;
sampling geometry is cached separately so Bicubic parameter families can reuse
it. AVX2 packed plans are shared by filters that share a canonical plan.

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

Experimental Metal support is off by default and is available only in a native
Apple ARM64 build with the Xcode Metal toolchain:

```sh
cmake -S . -B build-metal -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DDSMVC_VAPOURSYNTH_SDK=/path/to/vapoursynth \
  -DDSMVC_BUILD_METAL_EXPERIMENTS=ON \
  -DBUILD_TESTING=ON
cmake --build build-metal --parallel
ctest --test-dir build-metal --output-on-failure
```

Non-Apple and default builds do not compile or link the Metal executor. The
feature remains experimental: supported formats, geometry, device admission,
batch sizes, and automatic routing are deliberately not expanded beyond the
validated cases.

The Release DLL is written to `build/Release/dsmvc.dll`. The build uses the
static MSVC runtime so that an older runtime DLL bundled with a host cannot
change the STL synchronization ABI.

The CUDA backend is optional and currently supports Windows and Linux with a
CUDA Toolkit containing `nvcc`, `cuobjdump`, and `cuda.h`:

```sh
cmake -S . -B build-cuda \
  -DDSMVC_VAPOURSYNTH_SDK=/path/to/vapoursynth \
  -DDSMVC_ENABLE_CUDA=ON
cmake --build build-cuda --config Release --parallel
```

The build embeds native SM75-SM121 kernels plus PTX fallbacks by default. The
matrix is configurable with `DSMVC_CUDA_ARCHITECTURES` and
`DSMVC_CUDA_PTX_ARCHITECTURES`; `DSMVC_CUDA_MIN_ARCHITECTURE` defaults to 75.
The plugin loads the NVIDIA Driver API dynamically and does not require a CUDA
Runtime DLL beside the plugin. At runtime, `DSMVC_CUDA_STREAMS` may be set to a
value from 1 through 16. When unset, the runtime uses up to eight concurrent
slots for 2D plans with half-bandwidth seven or greater, while narrower 2D and
one-axis work remain limited to four. An explicit value applies uniformly to
all paths. Each slot owns pinned host staging so its transfers and kernels
share one stream.
`DSMVC_CUDA_INPUT_CACHE_MB` bounds the shared immutable-input cache and defaults
to 64 MiB; set it to `0` to disable source upload and first-transpose reuse.
Keep the cache enabled for repeated-source workloads such as GetNative scans;
disable it for long unique-frame runs when reuse is not expected.
`DSMVC_CUDA_PLAN_CACHE_MB` bounds the device-resident packed-plan LRU and
defaults to 16 MiB. This deliberately small default is appropriate for
GetNative-style height scans, where most plans are used once; increase it for a
long-lived workload that repeatedly reuses many distinct plans. Set
`DSMVC_CUDA_STREAMS=4` when reducing peak device memory is more important than
the last few percent of mixed-workload throughput; the adaptive default can
grow from four to eight slots for half-bandwidth-seven work.
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
