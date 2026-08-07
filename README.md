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

`backend` accepts `auto`, `cpu`, `metal`, `vulkan`, or `cuda`. `auto` continues
to select the CPU implementation. A CUDA-enabled build accepts `cuda` on a
compatible NVIDIA device; an unavailable or uncompiled explicit backend raises
an error and never silently falls back to CPU. Metal and Vulkan remain stable
capability stubs.

For the CPU backend, `opt=1` selects the scalar path and `opt=2` strictly
requires AVX2 and FMA. The default selects AVX2/FMA when available. Other
numeric values retain baseline behavior and select automatic dispatch. The
Python wrapper accepts either these integers or `Opt.AUTO`, `Opt.NONE`, and
`Opt.AVX2`.

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

## Benchmark

The reproducible old/new runner and its case definitions are documented in
[benchmarks/README.md](benchmarks/README.md). A full run uses the fixed input
and baseline hashes, executes each implementation in separate processes, and
writes JSON, CSV, Markdown, command lines, images, difference maps, and error
curves outside the VapourSynth installation.

For the real-video end-to-end comparison based on the supplied training HTML,
`test_getfnative*.vpy`, and `test_selectkernel.vpy`, use
`benchmarks/e2e_benchmark.py`. It accepts the
explicit MKV, current plugin, original plugin, VapourSynth Python, and VSPipe
paths, and writes paired performance and error reports.

## License

dsmvc is MIT licensed. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
