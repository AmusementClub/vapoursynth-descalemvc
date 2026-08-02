# dsmvc

`dsmvc` is a VapourSynth API3 plugin compatible with the public filter API of
Irrational-Encoding-Wizardry/descale. It registers only the `dsmvc` namespace:

```python
core.std.LoadPlugin(path=r"C:\path\to\dsmvc.dll")
output = core.dsmvc.Debicubic(source, 1280, 720, b=0.0, c=0.5)
```

The plugin identifier is `com.dsmvc.descale`. It intentionally exports the
API3 `VapourSynthPluginInit` entry point and does not export an API4 plugin
entry point or a `core.descale` alias.

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

`backend` accepts `auto`, `cpu`, `metal`, `vulkan`, or `cuda`. `auto` selects
the CPU implementation. The three GPU names are stable capability stubs and
raise an explicit unsupported error; they never silently fall back to CPU.

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

The Python wrapper is [python/dsmvc.py](python/dsmvc.py). It preserves the
baseline RGB, YUV, GRAY, bit-depth, subsampling, `yuv444`, `gray`, and chroma
conversion behavior while dispatching to `core.dsmvc`.

## Build

Requirements:

- CMake 3.24 or newer
- A C++23 compiler
- VapourSynth API3 headers
- On Windows, Visual Studio 2022 with the x64 C++ workload

Example for the target R57 installation:

```powershell
& 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' `
  -S . -B build -A x64 `
  -DDSMVC_VAPOURSYNTH_SDK='D:\okegui\OKEGui\tools\vapoursynth\sdk' `
  -DDSMVC_VS_PYTHON='D:\okegui\OKEGui\tools\vapoursynth\python.exe' `
  -DDSMVC_BASELINE_PLUGIN='D:\okegui\OKEGui\tools\vapoursynth\vapoursynth64\plugins\descale.dll'

& 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' `
  --build build --config Release --parallel

& 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe' `
  --test-dir build -C Release --output-on-failure
```

If `DSMVC_GETNATIVE_SOURCE_DIR` is not set, CMake fetches the pinned
GetNative-VF commit recorded in [CMakeLists.txt](CMakeLists.txt).

The Release DLL is written to `build/Release/dsmvc.dll`. The build uses the
static MSVC runtime so that an older runtime DLL bundled with a host cannot
change the STL synchronization ABI.

## Benchmark

The reproducible old/new runner and its case definitions are documented in
[benchmarks/README.md](benchmarks/README.md). A full run uses the fixed input
and baseline hashes, executes each implementation in separate processes, and
writes JSON, CSV, Markdown, command lines, images, difference maps, and error
curves outside the VapourSynth installation.

## License

dsmvc is MIT licensed. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
