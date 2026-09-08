#!/usr/bin/env bash
set -euo pipefail

# Keep frozen sources, effective compiler commands, binaries, and output proofs.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output=${OUTPUT_DIR:-"$repo_root/artifacts/release-cpu-ab-$(date -u +%Y%m%dT%H%M%SZ)"}
exec python3 "$repo_root/benchmarks/release_cpu_version_ab.py" \
    --baseline "${BASELINE_REF:-v0.1.1}" \
    --output "$output" --samples "${SAMPLES:-5}" --iterations "${ITERATIONS:-3}"
