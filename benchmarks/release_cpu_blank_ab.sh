#!/usr/bin/env bash
set -euo pipefail

# Build and run the same BlankClip-equivalent benchmark at the pre-tail-tile
# baseline and at the current checkout. The baseline is isolated in a temporary
# worktree and is removed automatically on exit.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
baseline=${BASELINE_REF:-8fbed39}
samples=${SAMPLES:-5}
iterations=${ITERATIONS:-3}
build_type=${CMAKE_BUILD_TYPE:-Release}
sdk=${DSMVC_VAPOURSYNTH_SDK:-}
if [[ -z "$sdk" && -f "$repo_root/build/codex-arm-release/CMakeCache.txt" ]]; then
    sdk=$(sed -n 's/^DSMVC_VAPOURSYNTH_SDK:PATH=//p' \
        "$repo_root/build/codex-arm-release/CMakeCache.txt" | head -1)
fi
worktree=$(mktemp -d "${TMPDIR:-/tmp}/dsmvc-release-ab.XXXXXX")
trap 'git -C "$repo_root" worktree remove --force "$worktree/baseline" >/dev/null 2>&1 || true; rm -rf "$worktree"' EXIT

git -C "$repo_root" worktree add --detach "$worktree/baseline" "$baseline" >/dev/null

build_one() {
    local source_dir=$1
    local build_dir=$2
    local -a sdk_arg=()
    if [[ -n "$sdk" ]]; then sdk_arg+=("-DDSMVC_VAPOURSYNTH_SDK=$sdk"); fi
    local -a configure=(cmake -S "$source_dir" -B "$build_dir"
        "-DCMAKE_BUILD_TYPE=$build_type"
        -DDSMVC_BUILD_BENCHMARKS=ON -DDSMVC_ENABLE_NATIVE_CPU_SIMD=ON)
    configure+=("${sdk_arg[@]}")
    "${configure[@]}" >/dev/null
    cmake --build "$build_dir" --target dsmvc_cpu_avx512_benchmark -j2 >/dev/null
}

build_one "$worktree/baseline" "$worktree/baseline-build"
build_one "$repo_root" "$worktree/current-build"

echo "baseline_ref=$baseline current_ref=$(git -C "$repo_root" rev-parse --short HEAD)"
echo "samples=$samples iterations=$iterations workload=BlankClip-equivalent"
echo "--- baseline AVX2 (pre-optimization) ---"
"$worktree/baseline-build/dsmvc_cpu_avx512_benchmark" "$samples" "$iterations"
echo "--- current AVX2 vs AVX-512 ---"
"$worktree/current-build/dsmvc_cpu_avx512_benchmark" "$samples" "$iterations"
