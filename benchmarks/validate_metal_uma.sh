#!/usr/bin/env bash

set -euo pipefail

readonly minimum_free_percent=10
readonly baseline_ref="2d846923f2e866aca2971987a9fbb90aac9bb707"

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv=""
sdk="/private/tmp/dsmvc-vs-sdk-r57-20260806"
source_video="/Users/owen/Downloads/cesh/[LoliHouse] DIGIMON BEATBREAK - 40 [WebRip 1080p HEVC-10bit AAC SRTx2].mkv"
source_plugin="/private/tmp/dsmvc-release-arm-deps-20260806/ffms2/libffms2.dylib"
output=""
skip_performance=0
last_swapouts=""
result_status="failed"

usage() {
    cat <<'EOF'
usage: validate_metal_uma.sh --venv PATH [options]

Required:
  --venv PATH          VapourSynth environment containing Python and VSPipe

Options:
  --sdk PATH           VapourSynth SDK; defaults to the local R57 checkout
  --source PATH        Representative GetFnative source video
  --source-plugin PATH FFMS2 plugin used to open the source
  --output PATH        New artifact directory under .omx/evidence by default
  --skip-performance   Run fresh build/correctness gates only

The full evaluator intentionally runs only two fixed wide kernels over P8/P10
at R16/R32 and one 4-scaler x 64-height GetFnative profile. It never runs the
full GetFnative candidate space. Exit 125 means memory or swap pressure stopped
the run before further heavy work.
EOF
}

while (($# > 0)); do
    case "$1" in
        --venv)
            venv="$2"
            shift 2
            ;;
        --sdk)
            sdk="$2"
            shift 2
            ;;
        --source)
            source_video="$2"
            shift 2
            ;;
        --source-plugin)
            source_plugin="$2"
            shift 2
            ;;
        --output)
            output="$2"
            shift 2
            ;;
        --skip-performance)
            skip_performance=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$venv" ]]; then
    usage >&2
    exit 2
fi

for command in awk cmake ctest file git grep memory_pressure ninja nm python3 \
               shasum sysctl tar uname vm_stat xcrun; do
    if ! command -v "$command" >/dev/null 2>&1; then
        printf 'required command is unavailable: %s\n' "$command" >&2
        exit 2
    fi
done

venv="$(cd -- "$venv" && pwd)"
sdk="$(cd -- "$sdk" && pwd)"
python="$venv/bin/python"
vspipe="$venv/bin/vspipe"
if [[ ! -x "$python" || ! -x "$vspipe" ]]; then
    printf 'VapourSynth Python or VSPipe is missing under %s\n' "$venv" >&2
    exit 2
fi
if [[ ! -f "$sdk/include/VapourSynth4.h" ]]; then
    printf 'VapourSynth4.h is missing under %s\n' "$sdk" >&2
    exit 2
fi
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    printf 'this evaluator requires a native Apple ARM64 host\n' >&2
    exit 2
fi
if ! git -C "$repo_root" cat-file -e "$baseline_ref^{commit}"; then
    printf 'CPU control commit is missing: %s\n' "$baseline_ref" >&2
    exit 2
fi
if ((skip_performance == 0)); then
    if [[ ! -f "$source_video" || ! -f "$source_plugin" ]]; then
        printf 'GetFnative source or source plugin is missing\n' >&2
        exit 2
    fi
fi

if [[ -z "$output" ]]; then
    output="$repo_root/.omx/evidence/metal-uma-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$output" ]]; then
    printf 'artifact directory already exists: %s\n' "$output" >&2
    exit 2
fi
mkdir -p "$output"
output="$(cd -- "$output" && pwd)"

on_exit() {
    local code=$?
    {
        printf 'status=%s\n' "$result_status"
        printf 'exit_code=%s\n' "$code"
        printf 'candidate_head=%s\n' "$(git -C "$repo_root" rev-parse HEAD)"
        printf 'artifacts=%s\n' "$output"
    } > "$output/result.env"
}
trap on_exit EXIT

stage() {
    printf '[metal-uma] %s\n' "$1"
}

run_logged() {
    local log=$1
    shift
    {
        printf '+ '
        printf '%q ' "$@"
        printf '\n'
        "$@"
    } >> "$log" 2>&1
}

pressure_guard() {
    local label=$1
    local free_percent
    local swapouts
    free_percent="$(memory_pressure -Q | awk \
        '/System-wide memory free percentage:/ { gsub(/%/, "", $NF); print $NF }')"
    swapouts="$(vm_stat | awk \
        '/^Swapouts:/ { gsub(/\./, "", $2); print $2 }')"
    if [[ -z "$free_percent" || -z "$swapouts" ]]; then
        printf 'could not read memory pressure at %s\n' "$label" >&2
        result_status="stopped-pressure"
        exit 125
    fi
    {
        printf 'label=%s free_percent=%s swapouts=%s\n' \
            "$label" "$free_percent" "$swapouts"
        memory_pressure -Q
        vm_stat
    } >> "$output/memory-pressure.log"
    if ((free_percent < minimum_free_percent)); then
        printf 'stopping at %s: free memory is %s%%\n' \
            "$label" "$free_percent" >&2
        result_status="stopped-pressure"
        exit 125
    fi
    if [[ -n "$last_swapouts" ]] && ((swapouts > last_swapouts)); then
        printf 'stopping at %s: swapouts grew from %s to %s\n' \
            "$label" "$last_swapouts" "$swapouts" >&2
        result_status="stopped-pressure"
        exit 125
    fi
    last_swapouts="$swapouts"
}

configure_build() {
    local source=$1
    local build=$2
    local architecture=$3
    local metal=$4
    local vs_python=$5
    local log="$output/$(basename "$build").log"
    local args=(
        -S "$source" -B "$build" -G Ninja
        -DCMAKE_BUILD_TYPE=RelWithDebInfo
        -DCMAKE_CXX_COMPILER="$(xcrun --find clang++)"
        -DCMAKE_OSX_ARCHITECTURES="$architecture"
        -DCMAKE_OSX_SYSROOT="$(xcrun --sdk macosx --show-sdk-path)"
        -DDSMVC_VAPOURSYNTH_SDK="$sdk"
        -DDSMVC_BUILD_METAL_EXPERIMENTS="$metal"
        -DDSMVC_BUILD_BENCHMARKS=OFF
        -DBUILD_TESTING=ON)
    if [[ -n "$vs_python" ]]; then
        args+=( -DDSMVC_VS_PYTHON="$vs_python" )
    fi
    run_logged "$log" cmake "${args[@]}"
    run_logged "$log" cmake --build "$build" --parallel 8
    run_logged "$log" ctest --test-dir "$build" \
        --output-on-failure --timeout 360
    ninja -C "$build" -t commands \
        > "$output/$(basename "$build")-commands.txt"
    file "$build/dsmvc.so" > "$output/$(basename "$build")-file.txt"
    shasum -a 256 "$build/dsmvc.so" \
        > "$output/$(basename "$build")-sha256.txt"
}

stage "recording source and host identity"
{
    printf 'candidate_head=%s\n' "$(git -C "$repo_root" rev-parse HEAD)"
    printf 'baseline_ref=%s\n' "$baseline_ref"
    printf 'candidate_index_tree=%s\n' \
        "$(git -C "$repo_root" write-tree)"
    printf 'sdk=%s\n' "$sdk"
    printf 'python=%s\n' "$python"
    printf 'vspipe=%s\n' "$vspipe"
    "$python" -c \
        'import platform, vapoursynth as vs; print(platform.python_version()); print(vs.__version__)'
    sw_vers
    uname -a
    sysctl -n hw.model hw.memsize hw.logicalcpu
    "$(xcrun --find clang++)" --version
} > "$output/identity.txt" 2>&1
pressure_guard pre-build

baseline_source="$output/source-head"
mkdir -p "$baseline_source"
git -C "$repo_root" archive --format=tar "$baseline_ref" \
    | tar -xf - -C "$baseline_source"

baseline_build="$output/build-head-cpu"
metal_build="$output/build-candidate-metal"
metal_off_build="$output/build-candidate-metal-off"
x86_build="$output/build-candidate-x86"

stage "building HEAD CPU control"
configure_build "$baseline_source" "$baseline_build" arm64 OFF "$python"
pressure_guard post-head-build

stage "building and testing candidate Metal-on"
configure_build "$repo_root" "$metal_build" arm64 ON "$python"
pressure_guard post-metal-build

stage "building and testing candidate Metal-off"
configure_build "$repo_root" "$metal_off_build" arm64 OFF "$python"
pressure_guard post-metal-off-build

stage "building and testing candidate x86_64"
configure_build "$repo_root" "$x86_build" x86_64 OFF ""
pressure_guard post-x86-build

stage "checking API4, architecture, AVX2, and CUDA isolation"
nm -gjU "$metal_build/dsmvc.so" > "$output/api4-entrypoints.txt"
if ! grep -Fxq '_VapourSynthPluginInit2' "$output/api4-entrypoints.txt"; then
    printf 'API4 entrypoint is missing\n' >&2
    exit 1
fi
if grep -Fxq '_VapourSynthPluginInit' "$output/api4-entrypoints.txt"; then
    printf 'legacy API3 entrypoint is exported\n' >&2
    exit 1
fi
arm_commands="$output/build-candidate-metal-commands.txt"
off_commands="$output/build-candidate-metal-off-commands.txt"
x86_commands="$output/build-candidate-x86-commands.txt"
if ! grep -q 'cpu_executor_neon.cpp' "$arm_commands" \
   || grep -q 'cpu_executor_avx2.cpp' "$arm_commands"; then
    printf 'ARM build does not preserve NEON isolation\n' >&2
    exit 1
fi
if ! grep -q 'metal_scheduler_apple.mm' "$arm_commands" \
   || grep -q 'metal_scheduler_apple.mm' "$off_commands"; then
    printf 'Metal-on/off source isolation failed\n' >&2
    exit 1
fi
if ! grep -q 'cpu_executor_avx2.cpp' "$x86_commands" \
   || ! grep -q -- '-mavx2' "$x86_commands" \
   || ! grep -q -- '-mfma' "$x86_commands" \
   || grep -q 'cpu_executor_neon.cpp' "$x86_commands"; then
    printf 'x86 AVX2 isolation failed\n' >&2
    exit 1
fi
if ! git -C "$repo_root" diff --quiet HEAD -- src/cpu_executor_avx2.cpp \
   || ! git -C "$repo_root" diff --quiet HEAD -- src/cuda; then
    printf 'AVX2 or CUDA implementation changed in the Metal candidate\n' >&2
    exit 1
fi

if ((skip_performance != 0)); then
    result_status="nonperformance-pass"
    printf '%s\n' \
        'NONPERFORMANCE PASS: fresh Metal-on/off and x86 builds, native tests,' \
        'API4 integration, scheduler lifetime, and isolation gates passed.' \
        | tee "$output/summary.txt"
    exit 0
fi

stage "running bounded CPU throughput regression"
cpu_output="$output/cpu-regression"
run_logged "$output/cpu-regression.log" \
    "$python" "$repo_root/benchmarks/cpu_api_regression.py" \
    --api3-plugin "$baseline_build/dsmvc.so" \
    --api4-plugin "$metal_build/dsmvc.so" \
    --vspipe "$vspipe" --output "$cpu_output" \
    --frames 1024 --warmup-frames 64 --runs 7 \
    --threads 1 16 32 \
    --kernels bilinear bicubic_b0_c0_5 spline64 \
    --regression-threshold 0.03
pressure_guard post-cpu-benchmark

stage "running bounded paired fixed-kernel CPU/auto benchmark"
fixed_json="$output/fixed-kernel.json"
run_logged "$output/fixed-kernel.log" \
    "$python" "$repo_root/benchmarks/metal_plugin_benchmark.py" \
    --plugin "$metal_build/dsmvc.so" --vspipe "$vspipe" \
    --json-out "$fixed_json" --candidate-backend auto \
    --formats p8 p10 --kernels lanczos3 spline64 --requests 16 32 \
    --frames 256 --samples 7 --warmups 1 \
    --bootstrap-resamples 50000
pressure_guard post-fixed-kernel-benchmark

stage "running 256-candidate stratified GetFnative CPU/auto benchmark"
getfnative_json="$output/getfnative-stratified256.json"
run_logged "$output/getfnative-stratified256.log" \
    "$python" "$repo_root/benchmarks/e2e_plugin_ab_benchmark.py" \
    --control-plugin "$metal_build/dsmvc.so" \
    --candidate-plugin "$metal_build/dsmvc.so" \
    --control-backend cpu --candidate-backend auto \
    --vspipe "$vspipe" --source "$source_video" \
    --source-plugin "$source_plugin" --source-filter ffms2 \
    --profile stratified256 --cases getfnative \
    --requests 16 --threads 16 --samples 7 --warmups 1 \
    --bootstrap-resamples 50000 --json-out "$getfnative_json"
pressure_guard post-getfnative-benchmark

stage "validating performance evidence"
run_logged "$output/performance-evidence.log" \
    "$python" "$repo_root/benchmarks/validate_metal_uma_evidence.py" \
    --cpu "$cpu_output/benchmark.json" \
    --fixed "$fixed_json" --getfnative "$getfnative_json" \
    --json-out "$output/performance-evidence.json"

result_status="pass"
printf '%s\n' \
    'PASS: fresh general Metal UMA correctness/isolation gates and bounded' \
    'CPU, fixed-kernel, and stratified256 GetFnative throughput gates passed.' \
    | tee "$output/summary.txt"
