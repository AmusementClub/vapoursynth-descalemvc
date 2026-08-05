#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

if [[ ${EUID} -eq 0 ]]; then
    echo "Run this wrapper as the normal user; it invokes sudo only for sysctl." >&2
    exit 2
fi

original_paranoid=$(< /proc/sys/kernel/perf_event_paranoid)
restored=0

restore_paranoid() {
    if [[ $restored -eq 0 ]]; then
        sudo -n sysctl -q -w "kernel.perf_event_paranoid=${original_paranoid}" \
            >/dev/null || sudo sysctl -q -w \
            "kernel.perf_event_paranoid=${original_paranoid}" >/dev/null
        restored=1
        echo "Restored kernel.perf_event_paranoid=${original_paranoid}" >&2
    fi
}

sudo -v
trap restore_paranoid EXIT
trap 'exit 130' INT TERM

if [[ ${original_paranoid} -gt 0 ]]; then
    sudo sysctl -q -w kernel.perf_event_paranoid=0 >/dev/null
    echo "Temporarily set kernel.perf_event_paranoid=0" >&2
fi

export CACHE_PROFILE_PARANOID_BEFORE="$original_paranoid"
python3 "$repo_root/benchmarks/profile_cache_hypothesis.py" "$@"
