# Building and tuning dsmvc

Build notes per backend and the runtime environment variables for CUDA and
Vulkan. The [README](../README.md) has the basic build commands. The switches
added in v0.1.4 are listed in [gpu-performance.md](gpu-performance.md#environment-switches)
and [frame-io.md](frame-io.md).

## General notes

Native Apple ARM64 Release and RelWithDebInfo builds use `-O3 -flto=full`
without an Apple-model-specific `-mcpu` setting. CMake selects the NEON source
only for AArch64 and the existing AVX2/FMA source only for x86-64; universal
macOS builds must be produced as separate architecture builds. Set
`DSMVC_ENABLE_NATIVE_CPU_SIMD=OFF` for a scalar-only portability or
contract-test build; normal builds leave it enabled.

The Release DLL is written to `build/Release/dsmvc.dll`. The build uses the
static MSVC runtime so that an older runtime DLL bundled with a host cannot
change the STL synchronization ABI.

The ordered sparse-RHS optimization is part of the normal GPU build. It uses
fixed counts only in the admitted fused Float32 routes: CUDA half-bandwidths
5/7, Vulkan 32-thread fixed-band shaders 5/7/11, and Metal dedicated entrypoints
5/7/11. Other counts and routes retain the dynamic loop. The former three
`DSMVC_*_RHS_SPAN_EXPERIMENT` options have been removed; GPU backend selection
and adaptive routing still use the existing settings. See the
[GPU release benchmark](release-benchmark-gpu-v0.1.3.md) for
measured performance and workload boundaries.

## Metal

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

`DSMVC_ENABLE_METAL` is the single Metal build switch. An existing
`DSMVC_BUILD_METAL_EXPERIMENTS` cache value is migrated once and removed.
The macOS deployment target is passed to both the C++ and Metal compilers.

## CUDA

The CUDA backend is optional and currently supports Windows and Linux with a
CUDA Toolkit containing the CUDA compiler and static Runtime library.
CUDA-enabled test builds also require `cuobjdump`:

```sh
cmake -S . -B build-cuda \
  -DDSMVC_VAPOURSYNTH_SDK=/path/to/vapoursynth \
  -DDSMVC_ENABLE_CUDA=ON
cmake --build build-cuda --config Release --parallel
```

CUDA Release uses CMake's optimization defaults; use RelWithDebInfo for
optimized kernels with `-lineinfo`. Debug builds no longer force `-O3`.

The build embeds native SM75, SM86, SM89, and SM120 kernels by default, plus
compute_75 and compute_120 PTX fallbacks. The matrix is configurable with
`DSMVC_CUDA_ARCHITECTURES` and `DSMVC_CUDA_PTX_ARCHITECTURES`. Architectures
without a native image use driver JIT compilation and may incur a first-run
delay. The plugin statically links the official CUDA Runtime and therefore does
not require a CUDA Runtime DLL or shared library beside it. CUDA Runtime device
code, streams, and allocations use the device's primary context so they can
coexist with other CUDA filters in the host process.

### CUDA environment variables

| Variable | Default | Meaning |
|---|---|---|
| `DSMVC_CUDA_STREAMS` | adaptive | Explicit slot count `1`–`16`. Unset: up to 8 concurrent slots for reused-input 2D plans with half-bandwidth ≥ 7, narrower 2D and one-axis work limited to 4. An explicit value disables adaptive limits. |
| `DSMVC_CUDA_HOST_TRANSFER` | `auto` | 16-bit and Float32 2D frames are copied by the driver from frame memory (`pageable`), 8-bit frames go through the pinned staging pool. `staging-decoupled` uses the staging pool for all formats, `staging` restores slot-local Float32 staging, and `registered` is a diagnostic direct-copy path. |
| `DSMVC_CUDA_INPUT_CACHE_MB` | `64` | Bounds the shared immutable-input cache; `0` disables source upload and first-transpose reuse. Keep enabled for repeated-source workloads (GetNative scans); disable for long unique-frame runs. |
| `DSMVC_CUDA_PLAN_CACHE_MB` | `16` | Bounds the device-resident packed-plan LRU. Deliberately small for use-once height scans; increase for long-lived many-plan workloads. |
| `DSMVC_CUDA_SPLIT_RHS` | adaptive | Half-bandwidth ≥ 3: first execution of a packed plan stays fused, later ones use the higher-throughput split RHS kernel. `0` disables, `force` splits from the first execution. |
| `DSMVC_CUDA_SPLIT_HORIZONTAL_THREADS` / `DSMVC_CUDA_SPLIT_VERTICAL_THREADS` | `32` | Power-of-two values `16`–`256`. |
| `DSMVC_CUDA_HORIZONTAL_GLOBAL_TRANSPOSE` | `1` | The row-major `0` mode is available for experiments but is not the tested default. |

Float32 2D frames use a separate bounded pinned-staging pool: unique inputs
admit two GPU executions and four host staging phases at once, while
input-cache hits can expand to the full slot count. This keeps host packing
and unpacking outside the lifetime of a GPU execution slot. Integer and
one-axis paths retain slot-local staging. Use an explicit low stream count
only when reducing peak device memory is more important than adaptive
mixed-workload throughput. For a GetNative graph containing tens of thousands
of candidate frames, also set a finite VapourSynth cache before constructing
the graph, for example `core.max_cache_size = 512`; VapourSynth's default can
otherwise retain several GiB of host frames independently of the CUDA cache
limits. CUDA uses hardware fused multiply-add, so Float32 output is
numerically checked against the scalar backend but is not promised to be
bit-identical.

## Vulkan

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
compiler. For Float32 work, devices need only a compute queue and the Vulkan
core compute limits — no FP16, subgroup, narrow storage, descriptor indexing,
or synchronization2 feature is required. Explicit Float64 additionally
requires the strict capability contract described in the
[numerical contract](reference.md#numerical-contract), and
F32-only devices continue to build and run normally.

### Vulkan environment variables

| Variable | Default | Meaning |
|---|---|---|
| `DSMVC_VULKAN_DEVICE` | scored selection | Vulkan enumeration index or hexadecimal `vendor_id:device_id`. Default scores discrete, integrated, virtual, and CPU devices in that order and prefers a compute-only queue. An invalid or ineligible explicit selection reports all detected devices and does not fall back. |
| `DSMVC_VULKAN_SLOTS` | adaptive | `1`–`16`; unset uses a four-slot default with eight-slot heavy-plan expansion. |
| `DSMVC_VULKAN_PLAN_CACHE_MB` / `DSMVC_VULKAN_INPUT_CACHE_MB` | `16` / `64` | Device plan LRU and immutable-input cache bounds. |
| `DSMVC_VULKAN_SPLIT_RHS` | adaptive | `0` / `adaptive` / `force` fused versus split RHS execution. |
| `DSMVC_VULKAN_VALIDATION` | off | `1` enables the Khronos validation layer when installed. |
| `DSMVC_VULKAN_WORKGROUP` | — | `128` forces the core-limit fallback variants for diagnostics. |

Float32 results use explicit shader `fma` and are numerically checked, not
promised to be bit-identical to CPU or CUDA.
