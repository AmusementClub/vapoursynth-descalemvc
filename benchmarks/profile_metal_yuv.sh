#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="/tmp/dsmvc-metal-yuv-profile-build"
output_root="/tmp/dsmvc-metal-yuv-profile-results"
vapoursynth_sdk="${DSMVC_VAPOURSYNTH_SDK:-}"
samples=31
warmups=7
iterations=200
batch_size=16
narrow_cpu_frames=12
wide_cpu_frames=9
narrow_cpu_concurrency=16
wide_cpu_concurrency=8
threadgroup=128
profile_case="yuv420p10-b7-spline64"

while (($# > 0)); do
    case "$1" in
        --build-dir)
            build_dir="$2"
            shift 2
            ;;
        --output)
            output_root="$2"
            shift 2
            ;;
        --vapoursynth-sdk)
            vapoursynth_sdk="$2"
            shift 2
            ;;
        --samples)
            samples="$2"
            shift 2
            ;;
        --warmups)
            warmups="$2"
            shift 2
            ;;
        --iterations)
            iterations="$2"
            shift 2
            ;;
        --batch-size)
            batch_size="$2"
            shift 2
            ;;
        --narrow-cpu-frames)
            narrow_cpu_frames="$2"
            shift 2
            ;;
        --wide-cpu-frames)
            wide_cpu_frames="$2"
            shift 2
            ;;
        --narrow-cpu-concurrency)
            narrow_cpu_concurrency="$2"
            shift 2
            ;;
        --wide-cpu-concurrency)
            wide_cpu_concurrency="$2"
            shift 2
            ;;
        --threadgroup)
            threadgroup="$2"
            shift 2
            ;;
        --profile-case)
            profile_case="$2"
            shift 2
            ;;
        *)
            printf 'unknown option: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

for value in "$samples" "$warmups" "$iterations" "$batch_size" \
             "$narrow_cpu_frames" "$wide_cpu_frames" \
             "$narrow_cpu_concurrency" "$wide_cpu_concurrency" \
             "$threadgroup"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        printf 'numeric options must be positive integers; got %s\n' "$value" >&2
        exit 2
    fi
done
if ((narrow_cpu_frames > batch_size || wide_cpu_frames > batch_size)); then
    printf 'CPU frame split exceeds batch size %s\n' "$batch_size" >&2
    exit 2
fi
if [[ "$(uname -m)" != "arm64" ]]; then
    printf 'profile_metal_yuv.sh requires native arm64\n' >&2
    exit 2
fi

if [[ -z "$vapoursynth_sdk" && -f "$build_dir/CMakeCache.txt" ]]; then
    vapoursynth_sdk="$(sed -n \
        's/^DSMVC_VAPOURSYNTH_SDK:PATH=//p' "$build_dir/CMakeCache.txt")"
fi
if [[ ! -f "$vapoursynth_sdk/include/VapourSynth4.h" ]]; then
    printf 'VapourSynth API4 headers not found; pass --vapoursynth-sdk PATH\n' >&2
    exit 2
fi

xctrace="$(xcrun --find xctrace)"
clangxx="$(xcrun --find clang++)"
macos_sdk="$(xcrun --sdk macosx --show-sdk-path)"
for template in "CPU Counters" "Metal System Trace"; do
    if ! "$xctrace" list templates | rg -Fxq "$template"; then
        printf 'xctrace template is unavailable: %s\n' "$template" >&2
        exit 2
    fi
done

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
run_dir="$output_root/$run_id"
mkdir -p "$run_dir"

head_id="$(git -C "$repo_root" rev-parse HEAD)"
worktree_id="$({
    git -C "$repo_root" diff --no-ext-diff --binary
    shasum -a 256 \
        "$repo_root/CMakeLists.txt" \
        "$repo_root/benchmarks/metal_yuv_benchmark.mm" \
        "$repo_root/src/metal/metal_routes.metal" \
        "$repo_root/benchmarks/analyze_metal_profile.py" \
        "$repo_root/benchmarks/analyze_metal_routes.py" \
        "$repo_root/benchmarks/profile_metal_yuv.sh"
} | shasum -a 256 | cut -d' ' -f1)"
source_id="${head_id}+dirty-${worktree_id:0:16}"

{
    printf 'run_id=%s\n' "$run_id"
    printf 'source_id=%s\n' "$source_id"
    printf 'repo_root=%s\n' "$repo_root"
    printf 'build_dir=%s\n' "$build_dir"
    printf 'samples=%s\n' "$samples"
    printf 'warmups=%s\n' "$warmups"
    printf 'iterations=%s\n' "$iterations"
    printf 'batch_size=%s\n' "$batch_size"
    printf 'narrow_cpu_frames=%s\n' "$narrow_cpu_frames"
    printf 'wide_cpu_frames=%s\n' "$wide_cpu_frames"
    printf 'narrow_cpu_concurrency=%s\n' "$narrow_cpu_concurrency"
    printf 'wide_cpu_concurrency=%s\n' "$wide_cpu_concurrency"
    printf 'threadgroup=%s\n' "$threadgroup"
    printf 'profile_case=%s\n' "$profile_case"
    sw_vers
    xcodebuild -version
    "$clangxx" --version | sed -n '1p'
    sysctl -n hw.model hw.ncpu \
        hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu
} > "$run_dir/metadata.txt"

cmake -S "$repo_root" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_SYSROOT="$macos_sdk" \
    -DDSMVC_VAPOURSYNTH_SDK="$vapoursynth_sdk" \
    -DDSMVC_BUILD_METAL_EXPERIMENTS=ON \
    -DBUILD_TESTING=ON \
    > "$run_dir/cmake-configure.log"
cmake --build "$build_dir" \
    --target dsmvc_metal_yuv_benchmark dsmvc_engine_tests \
    --parallel > "$run_dir/cmake-build.log"
ninja -C "$build_dir" -t commands dsmvc_metal_yuv_benchmark \
    > "$run_dir/effective-build-commands.log"
ctest --test-dir "$build_dir" -C RelWithDebInfo \
    --output-on-failure -R '^dsmvc_engine_tests$' > "$run_dir/ctest.log"

benchmark="$build_dir/dsmvc_metal_yuv_benchmark"
if [[ ! -x "$benchmark" ]]; then
    printf 'benchmark executable was not produced: %s\n' "$benchmark" >&2
    exit 1
fi
file "$benchmark" > "$run_dir/benchmark-file.txt"
if ! rg -q 'arm64' "$run_dir/benchmark-file.txt"; then
    printf 'benchmark is not arm64: %s\n' "$benchmark" >&2
    exit 1
fi

common_args=(
    --warmups "$warmups"
    --threads-per-threadgroup "$threadgroup"
    --batch-size "$batch_size"
    --narrow-cpu-frames "$narrow_cpu_frames"
    --wide-cpu-frames "$wide_cpu_frames"
    --narrow-cpu-concurrency "$narrow_cpu_concurrency"
    --wide-cpu-concurrency "$wide_cpu_concurrency"
)

benchmark_json="$run_dir/routes.json"
"$benchmark" "${common_args[@]}" --samples "$samples" \
    --json-out "$benchmark_json" --assert > "$run_dir/routes.txt"
python3 "$repo_root/benchmarks/analyze_metal_routes.py" \
    "$benchmark_json" --baseline-route neon \
    --json-out "$run_dir/route-analysis.json" \
    --assert-route neon+metal --assert-case "$profile_case" \
    > "$run_dir/route-analysis.txt"

profile_cpu_frames="$(python3 -c \
    'import json, sys
document = json.load(open(sys.argv[1], encoding="utf-8"))
matches = [case["heterogeneous_cpu_frames"] for case in document["cases"] if case["name"] == sys.argv[2]]
if len(matches) != 1:
    raise SystemExit("profile case is missing or duplicated in benchmark JSON")
print(matches[0])' \
    "$benchmark_json" "$profile_case")"

record_cpu() {
    local name="$1"
    local route="$2"
    "$xctrace" record --template "CPU Counters" --no-prompt \
        --output "$run_dir/$name.trace" --launch -- "$benchmark" \
        "${common_args[@]}" --profile-iterations "$iterations" \
        --profile-route "$route" --profile-case "$profile_case" \
        > "$run_dir/$name-record.log" 2>&1
    "$xctrace" export "$run_dir/$name.trace" --toc \
        --output "$run_dir/$name-toc.xml" \
        > "$run_dir/$name-toc-export.log" 2>&1
    "$xctrace" export "$run_dir/$name.trace" \
        --xpath '/trace-toc/run[@number="1"]/data/table[@schema="CounterMetricAggregatedForProcess"]' \
        --output "$run_dir/$name-counters.xml" \
        > "$run_dir/$name-counters-export.log" 2>&1
    "$xctrace" export "$run_dir/$name.trace" \
        --xpath '/trace-toc/run[@number="1"]/data/table[@schema="os-signpost"]' \
        --output "$run_dir/$name-signposts.xml" \
        > "$run_dir/$name-signposts-export.log" 2>&1
}

record_cpu cpu-neon neon
record_cpu cpu-mixed neon+metal

metal_trace="$run_dir/metal-mixed.trace"
"$xctrace" record --template "Metal System Trace" --no-prompt \
    --output "$metal_trace" --launch -- "$benchmark" \
    "${common_args[@]}" --profile-iterations "$iterations" \
    --profile-route neon+metal --profile-case "$profile_case" \
    > "$run_dir/metal-mixed-record.log" 2>&1
"$xctrace" export "$metal_trace" --toc \
    --output "$run_dir/metal-mixed-toc.xml" \
    > "$run_dir/metal-mixed-toc-export.log" 2>&1

export_metal_table() {
    local schema="$1"
    local output="$2"
    "$xctrace" export "$metal_trace" \
        --xpath "/trace-toc/run[@number=\"1\"]/data/table[@schema=\"$schema\"]" \
        --output "$run_dir/$output" \
        > "$run_dir/$output.export.log" 2>&1
}

export_metal_table \
    metal-application-command-buffer-submissions metal-submissions.xml
export_metal_table metal-application-encoders-list metal-encoders.xml
export_metal_table metal-gpu-intervals metal-gpu-intervals.xml
export_metal_table metal-current-allocated-size metal-allocated.xml

python3 "$repo_root/benchmarks/analyze_metal_profile.py" \
    --target-process dsmvc_metal_yuv_benchmark \
    --submissions "$run_dir/metal-submissions.xml" \
    --encoders "$run_dir/metal-encoders.xml" \
    --gpu-intervals "$run_dir/metal-gpu-intervals.xml" \
    --allocated "$run_dir/metal-allocated.xml" \
    --cpu-neon "$run_dir/cpu-neon-counters.xml" \
    --cpu-neon-signposts "$run_dir/cpu-neon-signposts.xml" \
    --cpu-mixed "$run_dir/cpu-mixed-counters.xml" \
    --cpu-mixed-signposts "$run_dir/cpu-mixed-signposts.xml" \
    --warmups "$warmups" --iterations "$iterations" \
    --batch-size "$batch_size" --cpu-frames "$profile_cpu_frames" \
    --json-out "$run_dir/profile-analysis.json" \
    > "$run_dir/profile-analysis.txt"

printf 'metal_yuv_profile_run=%s\n' "$run_dir"
printf 'route_results=%s\n' "$benchmark_json"
printf 'route_analysis=%s\n' "$run_dir/route-analysis.json"
printf 'profile_analysis=%s\n' "$run_dir/profile-analysis.json"
