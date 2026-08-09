#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
    echo "usage: $0 FATBIN CUOBJDUMP OUTPUT_DIR [NVCC] [SOURCE_SHA]" >&2
    exit 2
fi

fatbin=$1
cuobjdump=$2
output_dir=$3
nvcc=${4:-nvcc}
source_sha=${5:-$(git rev-parse HEAD)}

if [[ ! -f "$fatbin" ]]; then
    echo "CUDA fatbin does not exist: $fatbin" >&2
    exit 2
fi
if [[ ! -x "$cuobjdump" ]]; then
    echo "cuobjdump is not executable: $cuobjdump" >&2
    exit 2
fi

mkdir -p "$output_dir"

f64_functions=(
    dsmvc_cuda_transpose_f32_f64
    dsmvc_cuda_transpose_u8_f64
    dsmvc_cuda_transpose_u16_f64
    dsmvc_cuda_promote_f32_f64
    dsmvc_cuda_inverse_horizontal_f64
    dsmvc_cuda_inverse_vertical_f64
    dsmvc_cuda_rhs_horizontal_f64
    dsmvc_cuda_rhs_vertical_f64
    dsmvc_cuda_solve_horizontal_f64
    dsmvc_cuda_solve_vertical_f64
    dsmvc_cuda_convert_f64_f32
    dsmvc_cuda_convert_f64_u8
    dsmvc_cuda_convert_f64_u16
)
f32_functions=(
    dsmvc_cuda_transpose_f32
    dsmvc_cuda_transpose_u8
    dsmvc_cuda_transpose_u16
    dsmvc_cuda_rhs_horizontal
    dsmvc_cuda_rhs_horizontal_column_major
    dsmvc_cuda_rhs_vertical
    dsmvc_cuda_inverse_horizontal
    dsmvc_cuda_inverse_horizontal_column_major
    dsmvc_cuda_inverse_vertical
    dsmvc_cuda_solve_horizontal
    dsmvc_cuda_solve_horizontal_column_major
    dsmvc_cuda_solve_vertical
    dsmvc_cuda_convert_u8
    dsmvc_cuda_convert_u16
)

join_functions() {
    local IFS=,
    echo "$*"
}

"$cuobjdump" --list-elf "$fatbin" >"$output_dir/list-elf.txt"
"$cuobjdump" --list-ptx "$fatbin" >"$output_dir/list-ptx.txt"
"$cuobjdump" --dump-elf-symbols "$fatbin" >"$output_dir/elf-symbols.txt"
"$cuobjdump" --dump-resource-usage "$fatbin" >"$output_dir/resource-usage.txt"
"$cuobjdump" --dump-sass --function "$(join_functions "${f64_functions[@]}")" \
    "$fatbin" >"$output_dir/f64.sass.txt"
"$cuobjdump" --dump-sass --function "$(join_functions "${f32_functions[@]}")" \
    "$fatbin" >"$output_dir/f32.sass.txt"
"$cuobjdump" --dump-ptx --function "$(join_functions "${f64_functions[@]}")" \
    "$fatbin" >"$output_dir/f64.ptx.txt"

native_count=$(rg -o '\.sm_[0-9]+\.cubin' "$output_dir/list-elf.txt" \
    | sort -u | wc -l)
ptx_count=$(rg -o '\.sm_[0-9]+\.ptx' "$output_dir/list-ptx.txt" \
    | sort -u | wc -l)
if [[ $native_count -eq 0 || $ptx_count -eq 0 ]]; then
    echo "fatbin must contain both native cubins and PTX" >&2
    exit 1
fi

for function in "${f64_functions[@]}"; do
    symbol_count=$(rg -c "STO_ENTRY +${function}$" \
        "$output_dir/elf-symbols.txt")
    if [[ $symbol_count -ne $native_count ]]; then
        echo "$function is missing from a native cubin" >&2
        exit 1
    fi
    if ! rg -q "\.entry ${function}\(" "$output_dir/f64.ptx.txt"; then
        echo "$function is missing from PTX" >&2
        exit 1
    fi
done

if ! rg -q '\bDFMA\b' "$output_dir/f64.sass.txt"; then
    echo "F64 SASS does not contain Double FMA instructions" >&2
    exit 1
fi
if ! rg -q '\bdiv\.rn\.f64\b' "$output_dir/f64.ptx.txt"; then
    echo "F64 PTX does not contain round-to-nearest Double division" >&2
    exit 1
fi
if rg -q '\b(DADD|DFMA|DMUL|DSETP)\b' "$output_dir/f32.sass.txt"; then
    echo "an F32 entrypoint unexpectedly contains Double instructions" >&2
    exit 1
fi
if rg -q 'STACK:[1-9][0-9]*|LOCAL:[1-9][0-9]*' \
    "$output_dir/resource-usage.txt"; then
    echo "CUDA artifact contains stack or local-memory use" >&2
    exit 1
fi

binary_sha=$(sha256sum "$fatbin" | cut -d' ' -f1)
f32_sass_sha=$(
    rg 'Function :|/\* 0x[0-9a-f]{16} \*/' "$output_dir/f32.sass.txt" \
        | sha256sum | cut -d' ' -f1
)
{
    echo "source_sha=$source_sha"
    echo "binary_sha256=$binary_sha"
    echo "f32_sass_sha256=$f32_sass_sha"
    echo "native_cubin_count=$native_count"
    echo "ptx_image_count=$ptx_count"
    echo "command=$0 $*"
    echo "uname=$(uname -a)"
    "$nvcc" --version | sed 's/^/nvcc=/'
    if nvidia-smi --query-gpu=name,compute_cap,driver_version \
        --format=csv,noheader >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,compute_cap,driver_version \
            --format=csv,noheader | sed 's/^/gpu=/'
    else
        echo "gpu=unavailable"
    fi
} >"$output_dir/provenance.txt"

gzip -9 --force \
    "$output_dir/f32.sass.txt" \
    "$output_dir/f64.sass.txt" \
    "$output_dir/f64.ptx.txt"
sha256sum \
    "$output_dir/elf-symbols.txt" \
    "$output_dir/f32.sass.txt.gz" \
    "$output_dir/f64.ptx.txt.gz" \
    "$output_dir/f64.sass.txt.gz" \
    "$output_dir/list-elf.txt" \
    "$output_dir/list-ptx.txt" \
    "$output_dir/provenance.txt" \
    "$output_dir/resource-usage.txt" \
    >"$output_dir/artifact-sha256.txt"
echo "CUDA F64 artifact inventory passed: $output_dir"
