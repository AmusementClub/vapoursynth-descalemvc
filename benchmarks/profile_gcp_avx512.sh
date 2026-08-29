#!/usr/bin/env bash
set -euo pipefail

# Lightweight in-VM runner. Provisioning, source fingerprinting and cleanup are
# intentionally kept outside this script so it can be used on C4, C4D and C3D.
samples="${1:-3}"
iterations="${2:-1}"
test "${samples}" -gt 0
test "${iterations}" -gt 0

printf 'host=%s\n' "$(hostname)"
printf 'cpu=%s\n' "$(lscpu | awk -F: '/Model name/ {gsub(/^ +/,"",$2); print $2; exit}')"
printf 'source=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
printf 'compiler=%s\n' "$(c++ --version | head -n 1)"

if command -v perf >/dev/null 2>&1 && perf stat -e cycles -- true >/tmp/dsmvc-perf-check 2>&1; then
    perf stat -e cycles,instructions,branches,branch-misses -- \
        ./build/dsmvc_cpu_avx512_benchmark "${samples}" "${iterations}"
else
    ./build/dsmvc_cpu_avx512_benchmark "${samples}" "${iterations}"
fi
