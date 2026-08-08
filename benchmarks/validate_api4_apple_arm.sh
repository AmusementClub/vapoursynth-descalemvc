#!/usr/bin/env bash

set -euo pipefail

readonly baseline_sha="59a3b7fdd28d52f80bed26298e8001aa84d3071e"
readonly control_sha="09fa5c9ea7141106b0d451f2c03fd0d5e047e86d"
readonly sdk_sha="325756ed04588b31840fdb74479537cddcba4bf7"
readonly minimum_free_percent=10

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
api3_control=""
venv=""
sdk="/private/tmp/dsmvc-vs-sdk-r57-20260806"
output=""
ci_evidence="${DSMVC_API4_CI_EVIDENCE:-}"
last_swapouts=""
result_status="failed"

usage() {
    cat <<'EOF'
usage: validate_api4_apple_arm.sh --api3-control PATH --venv PATH [options]

Required:
  --api3-control PATH  Clean worktree at the preserved API3 control 09fa5c9
  --venv PATH          VapourSynth R78 virtual environment

Options:
  --sdk PATH           VapourSynth R57 source/SDK checkout
  --output PATH        New artifact directory; defaults under .omx/evidence
  --ci-evidence PATH   JSON for the exact clean candidate commit, or set
                       DSMVC_API4_CI_EVIDENCE. Required for a full PASS.

CI JSON schema:
  {"commit":"<sha>","tree":"<tree>","jobs":{
    "dsmvc-windows-x64":"success",
    "dsmvc-linux-x64":"success",
    "dsmvc-macos-arm64":"success"}}

Exit status 2 means all local gates passed but required CI evidence is absent.
EOF
}

while (($# > 0)); do
    case "$1" in
        --api3-control)
            api3_control="$2"
            shift 2
            ;;
        --venv)
            venv="$2"
            shift 2
            ;;
        --sdk)
            sdk="$2"
            shift 2
            ;;
        --output)
            output="$2"
            shift 2
            ;;
        --ci-evidence)
            ci_evidence="$2"
            shift 2
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

if [[ -z "$api3_control" || -z "$venv" ]]; then
    usage >&2
    exit 2
fi

api3_control="$(cd -- "$api3_control" && pwd)"
venv="$(cd -- "$venv" && pwd)"
sdk="$(cd -- "$sdk" && pwd)"

for command in arch awk cmake ctest file git memory_pressure ninja nm python3 \
               shasum sysctl uname vm_stat xcrun; do
    if ! command -v "$command" >/dev/null 2>&1; then
        printf 'required command is unavailable: %s\n' "$command" >&2
        exit 2
    fi
done

python="$venv/bin/python"
vspipe="$venv/bin/vspipe"
if [[ ! -x "$python" || ! -x "$vspipe" ]]; then
    printf 'VapourSynth Python or VSPipe is missing under %s\n' "$venv" >&2
    exit 2
fi
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    printf 'this evaluator requires a native Apple ARM64 host\n' >&2
    exit 2
fi
if [[ "$(git -C "$api3_control" rev-parse HEAD)" != "$control_sha" ]]; then
    printf 'API3 control is not at %s\n' "$control_sha" >&2
    exit 2
fi
if ! git -C "$repo_root" merge-base --is-ancestor "$baseline_sha" HEAD; then
    printf 'candidate does not descend from API4 baseline %s\n' "$baseline_sha" >&2
    exit 2
fi
if [[ "$(git -C "$sdk" rev-parse HEAD)" != "$sdk_sha" ]]; then
    printf 'VapourSynth SDK is not at R57 commit %s\n' "$sdk_sha" >&2
    exit 2
fi
if [[ ! -f "$sdk/include/VapourSynth.h" \
      || ! -f "$sdk/include/VapourSynth4.h" ]]; then
    printf 'R57 API3/API4 headers are incomplete under %s\n' "$sdk" >&2
    exit 2
fi

if ! git -C "$repo_root" diff --quiet; then
    printf 'candidate has unstaged tracked changes; stage the intended tree first\n' >&2
    exit 2
fi
if ! git -C "$api3_control" diff --quiet \
   || ! git -C "$api3_control" diff --cached --quiet; then
    printf 'API3 control has tracked changes\n' >&2
    exit 2
fi
unexpected_untracked="$(git -C "$repo_root" ls-files --others --exclude-standard \
    | awk '!/^\.omx\// { print }')"
if [[ -n "$unexpected_untracked" ]]; then
    printf 'candidate has unstaged untracked files:\n%s\n' \
        "$unexpected_untracked" >&2
    exit 2
fi
control_untracked="$(git -C "$api3_control" ls-files --others --exclude-standard \
    | awk '!/^\.omx\// { print }')"
if [[ -n "$control_untracked" ]]; then
    printf 'API3 control has unexpected untracked files:\n%s\n' \
        "$control_untracked" >&2
    exit 2
fi

candidate_tree="$(git -C "$repo_root" write-tree)"
if [[ -z "$output" ]]; then
    output="$repo_root/.omx/evidence/api4-apple-arm-${candidate_tree:0:12}-$(date -u +%Y%m%dT%H%M%SZ)"
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
        printf 'candidate_tree=%s\n' "$candidate_tree"
        printf 'artifacts=%s\n' "$output"
    } > "$output/result.env"
}
trap on_exit EXIT

stage() {
    printf '[api4-arm] %s\n' "$1"
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
    free_percent="$(memory_pressure -Q \
        | awk '/System-wide memory free percentage:/ { gsub(/%/, "", $NF); print $NF }')"
    swapouts="$(vm_stat \
        | awk '/^Swapouts:/ { gsub(/\./, "", $2); print $2 }')"
    if [[ -z "$free_percent" || -z "$swapouts" ]]; then
        printf 'could not read macOS memory pressure at %s\n' "$label" >&2
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
    local benchmarks=$5
    local vs_python=$6
    local log="$output/$(basename "$build").log"
    local args=(
        -S "$source" -B "$build" -G Ninja
        -DCMAKE_BUILD_TYPE=RelWithDebInfo
        -DCMAKE_CXX_COMPILER="$(xcrun --find clang++)"
        -DCMAKE_OSX_ARCHITECTURES="$architecture"
        -DCMAKE_OSX_SYSROOT="$(xcrun --sdk macosx --show-sdk-path)"
        -DDSMVC_VAPOURSYNTH_SDK="$sdk"
        -DDSMVC_BUILD_METAL_EXPERIMENTS="$metal"
        -DDSMVC_BUILD_BENCHMARKS="$benchmarks"
        -DBUILD_TESTING=ON)
    if [[ -n "$vs_python" ]]; then
        args+=( -DDSMVC_VS_PYTHON="$vs_python" )
    fi
    run_logged "$log" cmake "${args[@]}"
    run_logged "$log" cmake --build "$build" --parallel 8
    run_logged "$log" ctest --test-dir "$build" \
        --output-on-failure --timeout 360
    ninja -C "$build" -t commands > "$output/$(basename "$build")-commands.txt"
    file "$build/dsmvc.so" > "$output/$(basename "$build")-file.txt"
    shasum -a 256 "$build/dsmvc.so" \
        > "$output/$(basename "$build")-sha256.txt"
}

stage "recording source and host identity"
{
    printf 'baseline_sha=%s\n' "$baseline_sha"
    printf 'control_sha=%s\n' "$control_sha"
    printf 'candidate_head=%s\n' "$(git -C "$repo_root" rev-parse HEAD)"
    printf 'candidate_tree=%s\n' "$candidate_tree"
    printf 'candidate_branch=%s\n' "$(git -C "$repo_root" branch --show-current)"
    printf 'sdk_sha=%s\n' "$(git -C "$sdk" rev-parse HEAD)"
    printf 'python=%s\n' "$python"
    printf 'vspipe=%s\n' "$vspipe"
    "$python" -c 'import platform, vapoursynth as vs; print(platform.python_version()); print(vs.__version__)'
    sw_vers
    uname -a
    sysctl -n hw.model hw.memsize hw.logicalcpu
    xcodebuild -version
    "$(xcrun --find clang++)" --version
} > "$output/identity.txt" 2>&1
pressure_guard pre-build

api3_build="$output/build-api3-control"
api4_metal_build="$output/build-api4-metal"
api4_default_build="$output/build-api4-default"
api4_x86_build="$output/build-api4-x86"

stage "building and testing API3 Metal control"
configure_build "$api3_control" "$api3_build" arm64 ON ON "$python"
pressure_guard post-api3-build

stage "building and testing API4 Metal candidate"
configure_build "$repo_root" "$api4_metal_build" arm64 ON ON "$python"
pressure_guard post-api4-metal-build

stage "building and testing API4 default Metal-off candidate"
configure_build "$repo_root" "$api4_default_build" arm64 OFF OFF "$python"
pressure_guard post-api4-default-build

stage "building and testing API4 Rosetta x86 candidate"
configure_build "$repo_root" "$api4_x86_build" x86_64 OFF OFF ""
pressure_guard post-api4-x86-build

stage "checking effective architecture commands and API entrypoint"
python3 - \
    "$output/build-api3-control-commands.txt" \
    "$output/build-api4-metal-commands.txt" \
    "$output/build-api4-x86-commands.txt" \
    "$repo_root" "$baseline_sha" <<'PY'
from pathlib import Path
import subprocess
import sys

control = Path(sys.argv[1]).read_text(encoding="utf-8")
arm = Path(sys.argv[2]).read_text(encoding="utf-8")
x86 = Path(sys.argv[3]).read_text(encoding="utf-8")
repo = Path(sys.argv[4])
baseline = sys.argv[5]

def require(condition, message):
    if not condition:
        raise SystemExit(f"architecture assertion failed: {message}")

for text, label in ((control, "API3 ARM"), (arm, "API4 ARM")):
    require("cpu_executor_neon.cpp" in text, f"{label} omits NEON")
    require("cpu_executor_avx2.cpp" not in text, f"{label} compiles AVX2")
    require("DSMVC_HAS_NEON_OBJECT=1" in text, f"{label} omits NEON define")
    compile_lines = [line for line in text.splitlines()
                     if " -c " in line and "/src/" in line
                     and "xcrun -sdk macosx metal" not in line]
    require(compile_lines, f"{label} has no source compile commands")
    require(all("-O3" in line and "-flto=full" in line
                for line in compile_lines),
            f"{label} source compile misses -O3/-flto=full")
    plugin_links = [line for line in text.splitlines()
                    if " -o dsmvc.so " in line]
    require(plugin_links and all("-flto=full" in line
                                 for line in plugin_links),
            f"{label} plugin link misses -flto=full")

require("cpu_executor_avx2.cpp" in x86, "API4 x86 omits AVX2")
require("-mavx2" in x86 and "-mfma" in x86, "API4 x86 omits AVX2/FMA flags")
require("cpu_executor_neon.cpp" not in x86, "API4 x86 compiles NEON")
require("DSMVC_HAS_NEON_OBJECT" not in x86, "API4 x86 defines NEON")
require("-mcpu=" not in arm and "-mcpu=" not in x86,
        "build contains a model-specific CPU flag")

working_blob = subprocess.check_output(
    ["git", "-C", str(repo), "hash-object", "src/cpu_executor_avx2.cpp"],
    text=True).strip()
baseline_blob = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse",
     f"{baseline}:src/cpu_executor_avx2.cpp"], text=True).strip()
require(working_blob == baseline_blob, "AVX2 blob differs from API4 baseline")
PY

nm -gjU "$api4_metal_build/dsmvc.so" > "$output/api4-entrypoints.txt"
if ! grep -Fxq '_VapourSynthPluginInit2' "$output/api4-entrypoints.txt"; then
    printf 'API4 entrypoint is missing\n' >&2
    exit 1
fi
if grep -Fxq '_VapourSynthPluginInit' "$output/api4-entrypoints.txt"; then
    printf 'legacy API3 entrypoint is exported by the API4 candidate\n' >&2
    exit 1
fi

stage "running native SIMD correctness and identity evaluator"
run_logged "$output/cpu-fixed-recipe.log" \
    "$api4_metal_build/dsmvc_cpu_profile_benchmark" \
    --mode compare --samples 7 --source-id "$candidate_tree" \
    --build-type RelWithDebInfo \
    --json-out "$output/cpu-fixed-recipe.json" --assert
python3 - "$output/cpu-fixed-recipe.json" <<'PY'
import json
import sys

document = json.load(open(sys.argv[1], encoding="utf-8"))
correctness = document.get("correctness", {})
if not correctness.get("pass") or not correctness.get("identity_stable"):
    raise SystemExit("native SIMD correctness or identity gate failed")
error = float(correctness.get("maximum_absolute_error", "inf"))
if error > 1.5e-6:
    raise SystemExit(f"native SIMD maximum error {error} exceeds 1.5e-6")
if len(document.get("cases", [])) < 4:
    raise SystemExit("native SIMD benchmark case set is incomplete")
PY
pressure_guard post-cpu-correctness

stage "checking deterministic API3/API4 CPU output hashes"
hash_output() {
    local plugin=$1
    local kernel=$2
    local label=$3
    local log="$output/hash-${label}-${kernel}.log"
    local digest
    digest="$("$vspipe" \
        --arg implementation=new \
        --arg kernel="$kernel" \
        --arg plugin="$plugin" \
        --arg old_plugin="$plugin" \
        --arg frames=4 \
        --arg threads=8 \
        --arg backend=cpu \
        --arg opt=2 \
        --arg src_height=810 \
        --arg base_height=1000 \
        --requests 8 --start 0 --end 3 \
        "$repo_root/benchmarks/vspipe_blank_fixed_kernel.vpy" - \
        2>> "$log" | shasum -a 256 | awk '{print $1}')"
    printf '%s\n' "$digest"
}

: > "$output/cpu-output-hashes.tsv"
for kernel in bilinear bicubic_b0_c0_5 spline64; do
    api3_hash="$(hash_output "$api3_build/dsmvc.so" "$kernel" api3)"
    api4_hash="$(hash_output "$api4_default_build/dsmvc.so" "$kernel" api4-a)"
    repeat_hash="$(hash_output "$api4_default_build/dsmvc.so" "$kernel" api4-b)"
    printf '%s\t%s\t%s\t%s\n' \
        "$kernel" "$api3_hash" "$api4_hash" "$repeat_hash" \
        >> "$output/cpu-output-hashes.tsv"
    if [[ "$api3_hash" != "$api4_hash" || "$api4_hash" != "$repeat_hash" ]]; then
        printf 'CPU output identity differs for %s\n' "$kernel" >&2
        exit 1
    fi
done
pressure_guard post-cpu-hashes

stage "running 12-cell API3/API4 CPU throughput gate"
run_logged "$output/cpu-api-regression.log" \
    "$python" "$repo_root/benchmarks/cpu_api_regression.py" \
    --api3-plugin "$api3_build/dsmvc.so" \
    --api4-plugin "$api4_default_build/dsmvc.so" \
    --vspipe "$vspipe" \
    --output "$output/cpu-api-regression" \
    --frames 5000 --warmup-frames 128 --runs 5 \
    --threads 1 8 16 32 \
    --kernels bilinear bicubic_b0_c0_5 spline64 \
    --regression-threshold 0.03
python3 - "$output/cpu-api-regression/benchmark.json" <<'PY'
import json
import sys

document = json.load(open(sys.argv[1], encoding="utf-8"))
summary = document.get("summary", [])
if len(summary) != 12:
    raise SystemExit(f"expected 12 CPU API cells, found {len(summary)}")
regressions = [case for case in summary if case.get("regression")
               or float(case.get("vspipe_ratio", 0.0)) < 0.97]
if regressions:
    labels = [f"{case['kernel']}/R{case['threads']}" for case in regressions]
    raise SystemExit("CPU API regression: " + ", ".join(labels))
PY
pressure_guard post-cpu-api-regression

stage "running bounded API3/API4 Metal throughput gates"
run_metal_ab() {
    local format=$1
    local kernel=$2
    local json="$output/metal-api-ab-${format}-${kernel}.json"
    run_logged "$output/metal-api-ab-${format}-${kernel}.log" \
        "$python" "$repo_root/benchmarks/metal_plugin_ab_benchmark.py" \
        --control-plugin "$api3_build/dsmvc.so" \
        --candidate-plugin "$api4_metal_build/dsmvc.so" \
        --control-backend metal --candidate-backend metal \
        --vspipe "$vspipe" --json-out "$json" \
        --formats "$format" --kernels "$kernel" \
        --requests 16 32 --frames 512 --samples 5 --warmups 1
    python3 - "$json" <<'PY'
import json
import sys

document = json.load(open(sys.argv[1], encoding="utf-8"))
comparisons = document.get("comparisons", [])
if len(comparisons) != 2:
    raise SystemExit(f"expected two Metal request cells, found {len(comparisons)}")
failed = [item for item in comparisons if float(item.get("speedup", 0.0)) < 0.97]
if failed:
    details = [f"R{item['requests']}={item['speedup']:.4f}x" for item in failed]
    raise SystemExit("Metal API4 throughput regression: " + ", ".join(details))
PY
}

run_metal_ab p8 bicubic
pressure_guard post-metal-p8-b3
run_metal_ab p10 spline64
pressure_guard post-metal-p10-spline64

stage "checking required remote CI evidence"
if [[ -z "$ci_evidence" || ! -f "$ci_evidence" ]]; then
    result_status="blocked-ci"
    printf '%s\n' \
        'LOCAL PASS: builds, correctness, identities, CPU A/B, and Metal A/B passed.' \
        'BLOCKED: exact-commit Windows x64 CUDA, Linux x64 CUDA, and macOS arm64 CI evidence is absent.' \
        | tee "$output/summary.txt"
    exit 2
fi
if ! git -C "$repo_root" diff --cached --quiet; then
    printf 'CI evidence cannot apply to an uncommitted staged candidate tree\n' >&2
    exit 1
fi
candidate_head="$(git -C "$repo_root" rev-parse HEAD)"
python3 - "$ci_evidence" "$candidate_head" "$candidate_tree" <<'PY'
import json
import sys

document = json.load(open(sys.argv[1], encoding="utf-8"))
if document.get("commit") != sys.argv[2]:
    raise SystemExit("CI evidence commit does not match candidate HEAD")
if document.get("tree") != sys.argv[3]:
    raise SystemExit("CI evidence tree does not match candidate tree")
expected = {
    "dsmvc-windows-x64": "success",
    "dsmvc-linux-x64": "success",
    "dsmvc-macos-arm64": "success",
}
jobs = document.get("jobs", {})
if any(jobs.get(name) != status for name, status in expected.items()):
    raise SystemExit(f"required CI jobs are not green: {jobs}")
PY

result_status="pass"
printf '%s\n' \
    'PASS: local API4 Apple ARM migration gates and exact-commit CI evidence passed.' \
    | tee "$output/summary.txt"
