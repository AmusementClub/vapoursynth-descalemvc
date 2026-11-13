#!/usr/bin/env bash
# Staged vs direct host-block I/O on a vapoursynth-classic core: for each route
# the output digest must be identical, and the registry statistics show whether
# the direct path actually ran.
#
# usage: [SW=1920 SH=1080 DW=1280 DH=720] classic_direct_io_matrix.sh <classic_frame_bench> <dsmvc plugin> <cuda|vulkan> [frames] [extra bench args...]
# SW/SH is the source size and DW/DH the descaled size of the 2D cases; the
# one-axis cases keep the other dimension. Validation-layer messages (set
# DSMVC_VULKAN_VALIDATION=1) are counted and make the run fail.
set -u
bench=$1; plugin=$2; backend=$3; frames=${4:-300}; shift 4 2>/dev/null || shift $#
upper=$(echo "$backend" | tr a-z A-Z)
SW=${SW:-1920}; SH=${SH:-1080}; DW=${DW:-1280}; DH=${DH:-720}
fail=0
run() { # label, env-value, bench args...
  local label=$1 mode=$2; shift 2
  env "DSMVC_${upper}_DIRECT_IO=$mode" "DSMVC_${upper}_DIRECT_IO_STATS=1" \
    "$bench" --plugin "$plugin" --backend "$backend" --frames "$frames" --hash "$@" "${EXTRA[@]}" 2>&1
}
EXTRA=("$@")
case_run() { # label, bench args...
  local label=$1; shift
  local staged direct
  staged=$(run "$label" 0 "$@" | grep -o 'digest=[0-9a-f]*\|fps=[0-9.]*\|FAILED.*' | tr '\n' ' ')
  direct=$(run "$label" 1 "$@")
  local d_line s_digest d_digest validation
  validation=$(echo "$direct" | grep -ciE 'validation|VUID')
  d_line=$(echo "$direct" | grep -o 'digest=[0-9a-f]*\|fps=[0-9.]*\|FAILED.*' | tr '\n' ' ')
  s_digest=$(echo "$staged" | grep -o 'digest=[0-9a-f]*'); d_digest=$(echo "$d_line" | grep -o 'digest=[0-9a-f]*')
  local verdict=OK
  if [ -z "$s_digest" ] || [ "$s_digest" != "$d_digest" ]; then verdict=MISMATCH; fail=1; fi
  if [ "$validation" != 0 ]; then verdict="$verdict+VALIDATION($validation)"; fail=1; echo "$direct" | grep -iE 'validation|VUID' | head -5; fi
  printf '%-28s %-8s staged: %s\n%-28s %-8s direct: %s\n    %s\n' "$label" "$verdict" "$staged" "" "" "$d_line" \
    "$(echo "$direct" | grep 'direct I/O' | sed 's/^/stats: /')"
}
case_run "2D float gray"          --format gray      --width $SW --height $SH --dst-width $DW --dst-height $DH
case_run "2D float rgb"           --format rgb       --width $SW --height $SH --dst-width $DW --dst-height $DH
case_run "rows only (H)"          --format gray      --width $SW --height $SH --dst-width $DW --dst-height $SH
case_run "columns only (V)"       --format gray      --width $SW --height $SH --dst-width $SW --dst-height $DH
case_run "2D u8 yuv420"           --format yuv420p8  --width $SW --height $SH --dst-width $DW --dst-height $DH
case_run "2D u16 yuv420"          --format yuv420p16 --width $SW --height $SH --dst-width $DW --dst-height $DH
case_run "2D float gray Float64"   --format gray     --width $SW --height $SH --dst-width $DW --dst-height $DH --arg f64mode=2
case_run "rows only Float64"      --format gray      --width $SW --height $SH --dst-width $DW --dst-height $SH --arg f64mode=2
case_run "2D u8 yuv420 Float64"   --format yuv420p8  --width $SW --height $SH --dst-width $DW --dst-height $DH --arg f64mode=2
case_run "pool churn (cache 64MB)" --format gray     --width $SW --height $SH --dst-width $DW --dst-height $DH --cache 64
exit $fail
