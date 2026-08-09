# Release benchmark

This directory contains the reproducible old/current Release comparison used
to produce [`docs/release-benchmark.md`](../docs/release-benchmark.md).

The pipeline is split into four Python runners:

- `release_benchmark.py` consolidates the E2E, fixed-kernel, BlankClip, and
  error results.
- `e2e_benchmark.py` measures the three real-video candidate scans.
- `fixed_kernel_benchmark.py` measures fixed kernels on the supplied video.
- `blank_fixed_kernel_benchmark.py` measures the same kernels from an
  in-memory `std.BlankClip`.

`cpu_api_regression.py` is a focused ABI regression runner for comparing a
preserved API3 plugin with the API4-only build on `backend=cpu`. It runs each
ABI in a separate VSPipe process, alternates their order, gives every measured
process its own warm-up, and flags a cell when median VSPipe throughput drops
past the selected threshold. For example:

```sh
python3 benchmarks/cpu_api_regression.py \
  --api3-plugin /path/to/api3/dsmvc.so \
  --api4-plugin /path/to/api4/dsmvc.so \
  --vspipe /path/to/vspipe \
  --output profile-results/cpu-api4-regression \
  --frames 5000 --warmup-frames 128 --runs 3 \
  --threads 1 8 16 32 \
  --kernels bilinear bicubic_b0_c0_5 spline64
```

It writes raw JSON/CSV samples, a summary CSV and Markdown report, and the
exact warm-up and measurement commands. VSPipe-internal FPS is the primary
metric; the default regression threshold is 3%, and the command returns a
nonzero status when any cell crosses it.

## Numerical contract baseline

`dsmvc_numerical_contract_tests` freezes the shared planner and arithmetic
contract used before backend-specific F32 or F64 work. Its fixtures cover
B1/B3/B5/B7, all four explicit padding modes, SIMD tails, a forced-F32
conditioned plan, forced and automatic F64 plans, retained normal-matrix
metadata, cache accounting, a mixed safe/risky 2D case, and an independent QR
anchor.

`dsmvc_numerical_contract_benchmark` is the matching lightweight axis baseline.
It records ordered-reference, current-scalar, and native-CPU routes for ordinary
F32, automatic-risk F64, and forced F64. It is not a full rows/columns/2D,
integer, or plugin benchmark, and its timings are not release throughput claims.

```sh
cmake -S . -B build-numerical-contract -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DDSMVC_BUILD_BENCHMARKS=ON \
  -DDSMVC_ENABLE_CUDA=OFF \
  -DDSMVC_ENABLE_VULKAN=OFF \
  -DDSMVC_ENABLE_METAL=OFF
cmake --build build-numerical-contract --target \
  dsmvc_engine_tests \
  dsmvc_numerical_contract_tests \
  dsmvc_numerical_contract_benchmark
ctest --test-dir build-numerical-contract --output-on-failure \
  -R '^dsmvc_(engine|numerical_contract)_tests$'
./build-numerical-contract/dsmvc_numerical_contract_benchmark \
  --samples 5 --iterations 32 \
  --source-id "$(git rev-parse HEAD)" \
  --json-out build-numerical-contract/numerical-contract.json
```

Append an explicit dirty/candidate suffix to `--source-id` when the executable
does not come from a clean commit. Preserve the JSON outside a build directory
when it is intended as review or release evidence.

The JSON deliberately exposes two plan identities:

- `f32_contract_identity` hashes the request, padding and precision
  classification, indices, and all ordered F32 planner arrays. This is the
  cross-platform and cross-backend contract and must match exactly.
- `complete_identity` additionally hashes all Double metadata and arrays. It
  binds evidence to a concrete host/toolchain plan, but is not a cross-libm
  bit-identity gate because transcendental coefficient generation can differ in
  low Double bits.

`result_identity` binds one measured route to its output. Strict F32 fixtures
use exact identities. F64 routes additionally use the independent QR,
high-precision error, final-output ULP, and integer-exactness gates defined by
the handover plan. Existing old-descale and published CPU release data are not
rerun by this baseline.

## API4 Apple ARM validation

`validate_api4_apple_arm.sh` is the consolidated migration evaluator. It uses
VapourSynth R57 API3/API4 headers to build a preserved API3 control and the
API4 candidate in fresh Ninja directories, while running both plugins through
the local VapourSynth R78 Python/VSPipe runtime. It verifies effective ARM and
x86 build commands, API4 registration and return signatures, Metal-off and
Metal-on behavior, native-SIMD correctness and stable identities, 12 CPU A/B
cells, and the bounded Metal integration and throughput cases.

```sh
bash benchmarks/validate_api4_apple_arm.sh \
  --api3-control /path/to/api3-control-worktree \
  --venv /path/to/vapoursynth/venv
```

The evaluator stops before additional heavy stages if macOS reports low free
memory or new swap-outs. Its local result cannot satisfy the full release gate
by itself: Windows x64 CUDA, Linux x64 CUDA, and macOS arm64 CI evidence for the
exact candidate commit is mandatory. When a candidate has been pushed under
separate authorization, provide the exported CI evidence through the path
documented by `--help`.

The Metal plugin A/B runner compares API3 and API4 in separate VSPipe processes
with alternating order. The migration gate runs only P8 Bicubic B3 and P10
Spline64 at requests 16/32, 512 frames, and five measured pairs, requiring each
API4 ratio of medians to remain at or above `0.97x`. Correctness is established
separately by `tests/vs_metal_integration.py`, which covers P8/P10, all six
kernel classes, limited/full range, 17/32-frame tails, batch sizes 4/7,
cancellation reuse, and high/low-concurrency automatic routing.

The native Metal profilers and route analyzers are exploratory tools. Their
stored API3 measurements explain the narrow routing policy but are not current
API4 migration evidence; retain JSON, plugin hashes, and command lines with any
new result.

Each runner invokes its matching VapourSynth graph:

- `vspipe_e2e.vpy`
- `vspipe_fixed_kernel.vpy`
- `vspipe_blank_fixed_kernel.vpy`

The graphs use the supplied source and plugin paths passed as VSPipe
arguments. They do not write into the VapourSynth installation. The release
runner uses separate VSPipe processes for each implementation and thread
configuration, then writes machine-readable results and SVG charts under
`benchmark-results/`.

Performance runners execute an untimed throwaway warm-up before each measured
cell. This warms persistent driver/module/page caches and GPU clocks, while the
measured fresh VSPipe process still pays for its own VapourSynth and CUDA
context initialization. Use sufficiently long measured runs and compare both
the external wall time and the VSPipe-internal time recorded in JSON; do not use
the warm-up sample itself as a performance result.

The E2E graph bounds VapourSynth's frame cache with
`--performance-cache-mb` (512 MiB by default). A full GetNative scan therefore
runs as one graph by default, avoiding repeated decoder, VapourSynth, and CUDA
startup. `--performance-batch-size N` splits the candidate interval into fresh
VSPipe processes only as a low-memory fallback; `0` means one complete bounded
graph. `--cuda-plan-cache-mb` controls the CUDA packed-plan LRU (16 MiB by
default), which is intentionally small because height-scan plans are normally
used once. Candidate ranges are half-open and each scaler group is preserved
when a range crosses a scaler boundary.

On the reference 16 GiB RTX 5080, the complete 30,800-candidate GetNative scan
peaked at about 2.7 GiB host RSS and 504 MiB of CUDA process memory with the
bounded single graph and adaptive slots. The same scan with
`DSMVC_CUDA_STREAMS=4` peaked at about 2.6 GiB RSS and 406 MiB of CUDA process
memory. Full-scan timings varied by roughly 3% across runs, so these data do not
establish a stable throughput advantage for either slot policy. The earlier
multi-GiB failure was VapourSynth frame-cache retention (about 12.5 GiB without
the bound), not 30,800 resident CUDA plans. If host memory is still constrained,
use a positive performance batch size; expect repeated process/context startup
and measure the aggregate wall time rather than one batch's FPS.

For the complete Linux/macOS release procedure, see
[`mac_release_benchmark_guide.md`](mac_release_benchmark_guide.md). The
published report and its charts are kept in [`docs/`](../docs/).
