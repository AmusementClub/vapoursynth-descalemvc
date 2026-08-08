#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="/tmp/dsmvc-profile-build"
output_dir="/tmp/dsmvc-profile-results"
baseline_json="/tmp/dsmvc-profile-baseline.json"
samples=21
assert_mode=0

while (($# > 0)); do
    case "$1" in
        --build-dir)
            build_dir="$2"
            shift 2
            ;;
        --output)
            output_dir="$2"
            shift 2
            ;;
        --baseline-json)
            baseline_json="$2"
            shift 2
            ;;
        --samples)
            samples="$2"
            shift 2
            ;;
        --assert)
            assert_mode=1
            shift
            ;;
        *)
            printf 'unknown option: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

if [[ "$(uname -m)" != "arm64" ]]; then
    printf 'profile_macos.sh requires a native arm64 process; got %s\n' "$(uname -m)" >&2
    exit 2
fi

clangxx="$(xcrun --find clang++)"
clang="$(xcrun --find clang)"
macos_sdk="$(xcrun --sdk macosx --show-sdk-path)"
xctrace="$(xcrun --find xctrace)"
if ! xcrun xctrace list templates | grep -Fxq 'Time Profiler'; then
    printf 'the local Xcode does not provide the Time Profiler template\n' >&2
    exit 2
fi

vapoursynth_sdk="${DSMVC_VAPOURSYNTH_SDK:-/tmp/dsmvc-vs-r57.LcaRcH/vapoursynth}"
if [[ ! -f "$vapoursynth_sdk/include/VapourSynth4.h" ]]; then
    printf 'VapourSynth API4 headers not found under %s\n' "$vapoursynth_sdk" >&2
    exit 2
fi

mkdir -p "$output_dir"
mkdir -p "$(dirname -- "$baseline_json")"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
run_dir="$output_dir/$run_id"
mkdir -p "$run_dir"

head_id="$(git -C "$repo_root" rev-parse HEAD)"
worktree_id="$({ git -C "$repo_root" diff --no-ext-diff --binary; git -C "$repo_root" status --porcelain=v1; } | shasum -a 256 | cut -d' ' -f1)"
source_id="${head_id}+dirty-${worktree_id:0:16}"

{
    printf 'run_id=%s\n' "$run_id"
    printf 'source_id=%s\n' "$source_id"
    printf 'repo_root=%s\n' "$repo_root"
    printf 'build_dir=%s\n' "$build_dir"
    printf 'clang=%s\n' "$clang"
    printf 'clangxx=%s\n' "$clangxx"
    printf 'macos_sdk=%s\n' "$macos_sdk"
    xcodebuild -version
    "$clang" --version | sed -n '1p'
    sysctl -n hw.model hw.ncpu hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu
} > "$run_dir/host-build-metadata.txt"

cmake -S "$repo_root" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_COMPILER="$clangxx" \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_SYSROOT="$macos_sdk" \
    -DDSMVC_VAPOURSYNTH_SDK="$vapoursynth_sdk" \
    -DDSMVC_BUILD_BENCHMARKS=ON \
    -DBUILD_TESTING=ON \
    > "$run_dir/cmake-configure.log"

cmake --build "$build_dir" \
    --target dsmvc_cpu_profile_benchmark dsmvc_engine_tests \
    --parallel > "$run_dir/cmake-build.log"

ctest --test-dir "$build_dir" -C RelWithDebInfo \
    --output-on-failure -R '^dsmvc_engine_tests$' \
    > "$run_dir/ctest.log"

benchmark="$build_dir/dsmvc_cpu_profile_benchmark"
if [[ ! -x "$benchmark" ]]; then
    printf 'benchmark executable was not produced: %s\n' "$benchmark" >&2
    exit 1
fi
file "$benchmark" > "$run_dir/benchmark-file.txt"
if ! grep -Fq 'arm64' "$run_dir/benchmark-file.txt"; then
    printf 'benchmark is not an arm64 executable: %s\n' "$run_dir/benchmark-file.txt" >&2
    exit 1
fi

compare_json="$run_dir/compare.json"
assert_args=()
if ((assert_mode)); then
    assert_args+=(--assert)
fi

"$benchmark" --mode compare --samples "$samples" \
    --source-id "$source_id" --build-type RelWithDebInfo \
    --json-out "$compare_json" "${assert_args[@]}" \
    > "$run_dir/compare.txt"

baseline_created=0
if [[ ! -e "$baseline_json" ]]; then
    cp "$compare_json" "$baseline_json"
    baseline_created=1
fi

python3 - "$compare_json" "$baseline_json" "$assert_mode" <<'PY'
import json
import pathlib
import sys

current_path = pathlib.Path(sys.argv[1])
baseline_path = pathlib.Path(sys.argv[2])
assert_mode = sys.argv[3] == "1"
current = json.loads(current_path.read_text())
baseline = json.loads(baseline_path.read_text())

def fail(message):
    print(f"profile evaluator failure: {message}", file=sys.stderr)
    raise SystemExit(1)

for document, label in ((current, "candidate"), (baseline, "baseline")):
    recipe = document.get("fixed_recipe", {})
    if recipe.get("source") != "1920x1080" or recipe.get("destination") != "1692x952":
        fail(f"{label} fixed recipe identity changed")
    if not recipe.get("plan_prepared_outside_timing"):
        fail(f"{label} measured planner work")

current_cases = current.get("cases", [])
baseline_cases = baseline.get("cases", [])
if [item.get("name") for item in current_cases] != [item.get("name") for item in baseline_cases]:
    fail("candidate and baseline case sets differ")
if not current_cases:
    fail("candidate has no benchmark cases")
if not current.get("correctness", {}).get("pass"):
    fail("candidate correctness or result-identity check failed")
if not current.get("correctness", {}).get("identity_stable"):
    fail("candidate result identity is not stable")

aggregate = current.get("aggregate", {})
neon = aggregate.get("neon_ms", {})
baseline_neon = baseline.get("aggregate", {}).get("neon_ms", {})
candidate_median = float(neon.get("median", 0.0))
baseline_median = float(baseline_neon.get("median", 0.0))
relative_mad = float(aggregate.get("neon_relative_mad", float("inf")))
if candidate_median <= 0.0 or baseline_median <= 0.0:
    fail("candidate or baseline NEON median is not positive")
if relative_mad > 0.10:
    fail(f"candidate NEON relative MAD {relative_mad:.6f} exceeds 0.10")
regression = candidate_median > baseline_median * 1.05
print(
    f"candidate_neon_median_ms={candidate_median:.6f} "
    f"frozen_baseline_neon_median_ms={baseline_median:.6f} "
    f"relative_mad={relative_mad:.6f} "
    f"regression={'fail' if regression else 'pass'}"
)
if assert_mode and regression:
    fail("candidate NEON median is more than 5% slower than the frozen baseline")
PY

trace="$run_dir/time-profiler.trace"
toc="$run_dir/time-profiler-toc.xml"
profile_table="$run_dir/time-profile-table.xml"
profile_json="$run_dir/profile-run.json"
"$xctrace" record \
    --template "Time Profiler" \
    --no-prompt \
    --output "$trace" \
    --launch -- "$benchmark" \
    --mode neon --iterations 20 \
    --source-id "$source_id" --build-type RelWithDebInfo \
    --json-out "$profile_json" \
    > "$run_dir/xctrace-record.log" 2>&1

"$xctrace" export "$trace" --toc --output "$toc" \
    > "$run_dir/xctrace-export.log" 2>&1

"$xctrace" export "$trace" \
    --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
    --output "$profile_table" \
    > "$run_dir/time-profile-export.log" 2>&1

if [[ ! -s "$trace" || ! -s "$toc" || ! -s "$profile_table" || ! -s "$profile_json" ]]; then
    printf 'xctrace did not produce all required profile artifacts under %s\n' "$run_dir" >&2
    exit 1
fi

printf 'profile_run=%s\n' "$run_dir"
printf 'compare_json=%s\n' "$compare_json"
printf 'baseline_json=%s\n' "$baseline_json"
printf 'baseline_created=%s\n' "$baseline_created"
printf 'trace=%s\n' "$trace"
printf 'trace_toc=%s\n' "$toc"
printf 'time_profile_table=%s\n' "$profile_table"
