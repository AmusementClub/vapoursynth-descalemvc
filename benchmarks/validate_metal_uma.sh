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
source_fingerprint=""
candidate_plugin_sha256=""

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
  --skip-performance   Run source-bound build/correctness gates only

The scored path runs one p10/spline64/R16 fixed cell and one GetFnative
stratified32x4 case (32 unique cells, 128 observations). Each has one unscored
warm-up pair and six scored pairs ordered C-A, A-C, C-A, A-C, C-A, A-C. A
full paired wall-clock median below 1.03x stops the candidate immediately.
Exit 125 means memory pressure or swap growth stopped further heavy work.
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

for command in awk cmake ctest file git grep memory_pressure ninja nm pmset \
               python3 shasum sysctl tar uname vm_stat xcrun; do
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
        printf 'source_fingerprint=%s\n' "$source_fingerprint"
        printf 'candidate_plugin_sha256=%s\n' "$candidate_plugin_sha256"
        printf 'artifacts=%s\n' "$output"
    } > "$output/result.env"
}
trap on_exit EXIT

stage() {
    printf '[metal-uma] %s\n' "$1"
}

record_command() {
    local path=$1
    shift
    {
        printf '+ '
        printf '%q ' "$@"
        printf '\n'
    } >> "$path"
}

run_logged() {
    local log=$1
    shift
    record_command "$log" "$@"
    local code=0
    "$@" >> "$log" 2>&1 || code=$?
    if ((code == 125)); then
        result_status="stopped-pressure"
    fi
    return "$code"
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
        printf 'label=%s timestamp=%s free_percent=%s swapouts=%s\n' \
            "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            "$free_percent" "$swapouts"
        memory_pressure -Q
        vm_stat
        pmset -g therm || true
        uptime
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

fingerprint_value() {
    "$python" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["source_fingerprint"])' \
        "$1"
}

record_fingerprint() {
    local label=$1
    local path="$output/source-fingerprint-$label.json"
    run_logged "$output/source-fingerprint.log" \
        "$python" "$repo_root/benchmarks/source_fingerprint.py" \
        --repo "$repo_root" --json-out "$path"
}

verify_fingerprint() {
    local label=$1
    record_fingerprint "$label"
    local current
    current="$(fingerprint_value "$output/source-fingerprint-$label.json")"
    if [[ "$current" != "$source_fingerprint" ]]; then
        result_status="stale-source"
        printf 'source fingerprint changed at %s: %s != %s\n' \
            "$label" "$current" "$source_fingerprint" >&2
        exit 4
    fi
}

configure_build() {
    local source=$1
    local build=$2
    local architecture=$3
    local metal=$4
    local vs_python=$5
    local label=$6
    local log="$output/$label.log"
    local invocations="$output/$label-invocations.txt"
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
    : > "$invocations"
    record_command "$invocations" cmake "${args[@]}"
    run_logged "$log" cmake "${args[@]}"
    record_command "$invocations" cmake --build "$build" --parallel 8
    run_logged "$log" cmake --build "$build" --parallel 8
    record_command "$invocations" ctest --test-dir "$build" \
        --output-on-failure --timeout 360
    run_logged "$log" ctest --test-dir "$build" \
        --output-on-failure --timeout 360
    ninja -C "$build" -t commands > "$output/$label-commands.txt"
    file "$build/dsmvc.so" > "$output/$label-file.txt"
    shasum -a 256 "$build/dsmvc.so" > "$output/$label-sha256.txt"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$output/$label-completed-at.txt"
}

write_route_log() {
    local benchmark=$1
    local destination=$2
    "$python" - "$benchmark" "$output/candidate-build-manifest.json" \
        "$destination" <<'PY'
import json
import pathlib
import sys

benchmark_path = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
destination = pathlib.Path(sys.argv[3])
benchmark = json.loads(benchmark_path.read_text())
manifest = json.loads(manifest_path.read_text())
document = {
    "schema": "dsmvc-route-log-v1",
    "benchmark": str(benchmark_path),
    "source_fingerprint": manifest["source_fingerprint"],
    "plugin": manifest["plugin"],
    "runs": [
        {
            "variant": item["variant"],
            "sample": item["sample"],
            "pair_order": item["pair_order"],
            "telemetry": item["telemetry"],
            "frame_properties": item["frame_properties"],
        }
        for item in [
            *benchmark.get("warmup_samples", []),
            *benchmark.get("samples", []),
        ]
    ],
}
destination.write_text(json.dumps(document, indent=2) + "\n")
PY
}

stage "recording source and host identity"
record_fingerprint pre-build
source_fingerprint="$(fingerprint_value \
    "$output/source-fingerprint-pre-build.json")"
{
    printf 'candidate_head=%s\n' "$(git -C "$repo_root" rev-parse HEAD)"
    printf 'baseline_ref=%s\n' "$baseline_ref"
    printf 'source_fingerprint=%s\n' "$source_fingerprint"
    printf 'sdk=%s\n' "$sdk"
    printf 'python=%s\n' "$python"
    printf 'vspipe=%s\n' "$vspipe"
    "$python" -c \
        'import platform, vapoursynth as vs; print(platform.python_version()); print(vs.__version__)'
    sw_vers
    uname -a
    sysctl -n hw.model hw.memsize hw.logicalcpu
    "$(xcrun --find clang++)" --version
    "$(xcrun --find metal-nm)" --version || true
} > "$output/identity.txt" 2>&1
pressure_guard pre-build

baseline_source="$output/source-frozen-control"
mkdir -p "$baseline_source"
git -C "$repo_root" archive --format=tar "$baseline_ref" \
    | tar -xf - -C "$baseline_source"
baseline_tree="$(git -C "$repo_root" rev-parse "$baseline_ref^{tree}")"

fingerprint_short="${source_fingerprint:0:16}"
baseline_build="$output/build-control-${baseline_ref:0:16}"
metal_build="$output/build-candidate-metal-$fingerprint_short"
metal_off_build="$output/build-candidate-metal-off-$fingerprint_short"
x86_build="$output/build-candidate-x86-$fingerprint_short"

stage "building frozen CPU control"
configure_build "$baseline_source" "$baseline_build" arm64 OFF "$python" \
    build-control
verify_fingerprint post-control-build
pressure_guard post-control-build

stage "building and testing latest candidate Metal-on"
configure_build "$repo_root" "$metal_build" arm64 ON "$python" \
    build-candidate-metal
verify_fingerprint post-metal-build
pressure_guard post-metal-build

stage "building and testing latest candidate Metal-off"
configure_build "$repo_root" "$metal_off_build" arm64 OFF "$python" \
    build-candidate-metal-off
verify_fingerprint post-metal-off-build
pressure_guard post-metal-off-build

stage "building and testing latest candidate x86_64"
configure_build "$repo_root" "$x86_build" x86_64 OFF "" \
    build-candidate-x86
verify_fingerprint post-x86-build
pressure_guard post-x86-build

stage "checking Metal symbols, API4, architecture, AVX2, and CUDA isolation"
metallib="$metal_build/generated/metal-routes/dsmvc_metal_routes.metallib"
metal_nm="$(xcrun --find metal-nm)"
run_logged "$output/metal-nm.log" "$metal_nm" "$metallib"
required_symbols=(
    inverse_axis_transposed_generic
    inverse_axis_transposed_h1
    inverse_axis_transposed_h3
    inverse_axis_transposed_h5
    inverse_axis_transposed_h7
    inverse_axis_transposed_batch_generic
    inverse_axis_transposed_batch_h1
    inverse_axis_transposed_batch_h3
    inverse_axis_transposed_batch_h5
    inverse_axis_transposed_batch_h7)
: > "$output/metal-symbol-inventory.txt"
for symbol in "${required_symbols[@]}"; do
    if ! grep -Fq "$symbol" "$output/metal-nm.log"; then
        printf 'MISSING %s\n' "$symbol" \
            | tee -a "$output/metal-symbol-inventory.txt" >&2
        exit 1
    fi
    printf 'FOUND %s\n' "$symbol" >> "$output/metal-symbol-inventory.txt"
done

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

run_logged "$output/build-manifest.log" \
    "$python" "$repo_root/benchmarks/build_manifest.py" \
    --kind candidate \
    --source-fingerprint-json "$output/source-fingerprint-pre-build.json" \
    --build-dir "$metal_build" --plugin "$metal_build/dsmvc.so" \
    --metallib "$metallib" \
    --invocations "$output/build-candidate-metal-invocations.txt" \
    --compile-commands "$arm_commands" \
    --completed-at "$(<"$output/build-candidate-metal-completed-at.txt")" \
    --json-out "$output/candidate-build-manifest.json"
run_logged "$output/build-manifest.log" \
    "$python" "$repo_root/benchmarks/build_manifest.py" \
    --kind frozen-control --control-ref "$baseline_ref" \
    --control-tree "$baseline_tree" --build-dir "$baseline_build" \
    --plugin "$baseline_build/dsmvc.so" \
    --invocations "$output/build-control-invocations.txt" \
    --compile-commands "$output/build-control-commands.txt" \
    --completed-at "$(<"$output/build-control-completed-at.txt")" \
    --json-out "$output/control-build-manifest.json"
candidate_plugin_sha256="$(shasum -a 256 "$metal_build/dsmvc.so" \
    | awk '{print $1}')"

if ((skip_performance != 0)); then
    verify_fingerprint final-nonperformance
    result_status="nonperformance-pass"
    printf '%s\n' \
        'NONPERFORMANCE PASS: source-bound Metal-on/off and x86 builds, tests,' \
        '10/10 Metal symbols, API4 integration, lifetime, and isolation gates passed.' \
        | tee "$output/summary.txt"
    exit 0
fi

stage "running the one-cell fixed H7 paired gate"
fixed_json="$output/fixed-p10-spline64-r16.json"
if run_logged "$output/fixed-p10-spline64-r16.log" \
    "$python" "$repo_root/benchmarks/metal_plugin_benchmark.py" \
    --plugin "$metal_build/dsmvc.so" --vspipe "$vspipe" \
    --build-manifest "$output/candidate-build-manifest.json" \
    --telemetry-dir "$output/fixed-frame-properties" \
    --json-out "$fixed_json" --candidate-backend auto \
    --formats p10 --kernels spline64 --requests 16 \
    --frames 256 --samples 6 --warmups 1 \
    --minimum-free-percent "$minimum_free_percent" \
    --bootstrap-resamples 50000; then
    :
else
    code=$?
    ((code == 125)) || result_status="fixed-run-failed"
    exit "$code"
fi
pressure_guard post-fixed-kernel-benchmark
write_route_log "$fixed_json" "$output/fixed-route-log.json"
if run_logged "$output/fixed-evidence.log" \
    "$python" "$repo_root/benchmarks/validate_metal_uma_evidence.py" \
    --build-manifest "$output/candidate-build-manifest.json" \
    --fixed "$fixed_json" --json-out "$output/fixed-evidence.json"; then
    :
else
    code=$?
    result_status="fixed-gate-failed"
    printf 'fixed paired median did not admit the candidate; stopping\n' \
        | tee "$output/summary.txt" >&2
    exit "$code"
fi

stage "running the 32-cell / 128-observation GetFnative paired gate"
getfnative_json="$output/getfnative-stratified32x4.json"
if run_logged "$output/getfnative-stratified32x4.log" \
    "$python" "$repo_root/benchmarks/e2e_plugin_ab_benchmark.py" \
    --control-plugin "$metal_build/dsmvc.so" \
    --candidate-plugin "$metal_build/dsmvc.so" \
    --control-backend cpu --candidate-backend auto \
    --build-manifest "$output/candidate-build-manifest.json" \
    --telemetry-dir "$output/getfnative-frame-properties" \
    --vspipe "$vspipe" --source "$source_video" \
    --source-plugin "$source_plugin" --source-filter ffms2 \
    --profile stratified32x4 --cases getfnative \
    --requests 16 --threads 16 --samples 6 --warmups 1 \
    --minimum-free-percent "$minimum_free_percent" \
    --bootstrap-resamples 50000 --json-out "$getfnative_json"; then
    :
else
    code=$?
    ((code == 125)) || result_status="getfnative-run-failed"
    exit "$code"
fi
pressure_guard post-getfnative-benchmark
write_route_log "$getfnative_json" "$output/getfnative-route-log.json"
if run_logged "$output/getfnative-evidence.log" \
    "$python" "$repo_root/benchmarks/validate_metal_uma_evidence.py" \
    --build-manifest "$output/candidate-build-manifest.json" \
    --getfnative "$getfnative_json" \
    --json-out "$output/getfnative-evidence.json"; then
    :
else
    code=$?
    result_status="getfnative-gate-failed"
    printf 'GetFnative paired median did not admit the candidate; stopping\n' \
        | tee "$output/summary.txt" >&2
    exit "$code"
fi

stage "running bounded CPU-only regression after both admission gates"
cpu_output="$output/cpu-regression"
if run_logged "$output/cpu-regression.log" \
    "$python" "$repo_root/benchmarks/cpu_api_regression.py" \
    --api3-plugin "$baseline_build/dsmvc.so" \
    --api4-plugin "$metal_build/dsmvc.so" \
    --vspipe "$vspipe" --output "$cpu_output" \
    --frames 1024 --warmup-frames 64 --runs 3 \
    --threads 1 16 32 \
    --kernels bilinear bicubic_b0_c0_5 spline64 \
    --regression-threshold 0.03 \
    --minimum-free-percent "$minimum_free_percent"; then
    :
else
    code=$?
    ((code == 125)) || result_status="cpu-regression-failed"
    exit "$code"
fi
pressure_guard post-cpu-benchmark

stage "capturing one bounded attribution trace after both 1.03x gates"
xcrun xctrace list templates > "$output/xctrace-templates.txt"
if ! grep -Fq 'Metal System Trace' "$output/xctrace-templates.txt"; then
    printf 'Metal System Trace template is unavailable\n' >&2
    result_status="attribution-unavailable"
    exit 1
fi
attribution_trace="$output/attribution-metal-system.trace"
attribution_props="$output/attribution-frame-properties.json"
if run_logged "$output/attribution-xctrace.log" \
    xcrun xctrace record --template "Metal System Trace" --no-prompt \
    --time-limit 30s --output "$attribution_trace" \
    --env DSMVC_METAL_PROFILE_SIGNPOSTS=1 --launch -- \
    "$vspipe" --arg "plugin=$metal_build/dsmvc.so" \
    --arg format=p10 --arg kernel=spline64 --arg backend=auto \
    --arg frames=64 --arg threads=16 --requests 16 --start 0 --end 63 \
    --json "$attribution_props" --filter-time \
    "$repo_root/benchmarks/vspipe_metal_plugin.vpy" --; then
    :
else
    result_status="attribution-failed"
    exit 1
fi
run_logged "$output/attribution-export.log" \
    xcrun xctrace export "$attribution_trace" --toc \
    --output "$output/attribution-toc.xml"
"$python" - "$repo_root/benchmarks" "$attribution_props" \
    "$output/candidate-build-manifest.json" "$attribution_trace" \
    "$output/attribution-toc.xml" "$output/attribution-summary.json" <<'PY'
import json
import pathlib
import sys

sys.path.insert(0, sys.argv[1])
from paired_benchmark_support import (
    load_frame_properties, sha256_file, sha256_tree,
    summarize_frame_properties,
)

props = pathlib.Path(sys.argv[2])
manifest_path = pathlib.Path(sys.argv[3])
trace = pathlib.Path(sys.argv[4])
toc = pathlib.Path(sys.argv[5])
output = pathlib.Path(sys.argv[6])
manifest = json.loads(manifest_path.read_text())
telemetry = summarize_frame_properties(load_frame_properties(props, 64))
if telemetry["metal_frames"] <= 0:
    raise SystemExit("attribution run did not route any Metal frames")
document = {
    "schema": "dsmvc-metal-attribution-v1",
    "source_fingerprint": manifest["source_fingerprint"],
    "plugin": manifest["plugin"],
    "telemetry": telemetry,
    "frame_properties": {"path": str(props), "sha256": sha256_file(props)},
    "trace": {"path": str(trace), "sha256": sha256_tree(trace)},
    "toc": {"path": str(toc), "sha256": sha256_file(toc)},
}
output.write_text(json.dumps(document, indent=2) + "\n")
PY
pressure_guard post-attribution

stage "validating complete performance evidence"
run_logged "$output/performance-evidence.log" \
    "$python" "$repo_root/benchmarks/validate_metal_uma_evidence.py" \
    --build-manifest "$output/candidate-build-manifest.json" \
    --cpu "$cpu_output/benchmark.json" \
    --fixed "$fixed_json" --getfnative "$getfnative_json" \
    --json-out "$output/performance-evidence.json"
verify_fingerprint final

result_status="pass"
printf '%s\n' \
    'PASS: source-bound correctness/isolation, fixed and GetFnative 1.03x' \
    'paired gates, CPU-only regression, and post-gate attribution passed.' \
    | tee "$output/summary.txt"
