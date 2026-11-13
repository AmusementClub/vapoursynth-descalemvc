#include "cuda_launch.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace kernel = dsmvc::cuda_kernel;

namespace {

__device__ __forceinline__ std::uint32_t minimum(
    std::uint32_t left, std::uint32_t right) {
    return left < right ? left : right;
}

template <std::uint32_t Count>
__device__ __forceinline__ float accumulate_rhs_fixed(
    const std::int32_t *__restrict__ indices,
    const float *__restrict__ weights, const float *__restrict__ input,
    std::uint32_t input_stride, std::uint32_t begin) {
    float sum = 0.0F;
#pragma unroll
    for (std::uint32_t tap = 0U; tap < Count; ++tap) {
        const auto source = static_cast<std::uint32_t>(indices[begin + tap]);
        sum = fmaf(weights[begin + tap], input[source * input_stride], sum);
    }
    return sum;
}

// Count sparse entries, not the CPU's dense source span. Keep the original
// entry order and one accumulator; neither zero padding nor reassociation is
// permitted. Unlisted bands/counts retain the original dynamic loop. This is
// limited to fused inverse kernels; standalone RHS kernels keep their smaller
// register footprint.
template <std::uint32_t FixedBandwidth>
__device__ __forceinline__ float accumulate_rhs_span(
    const std::int32_t *__restrict__ indices,
    const float *__restrict__ weights, const float *__restrict__ input,
    std::uint32_t input_stride, std::uint32_t begin, std::uint32_t end) {
    const std::uint32_t count = end - begin;
    if constexpr (FixedBandwidth == 5U) {
        if (count == 7U) return accumulate_rhs_fixed<7U>(
            indices, weights, input, input_stride, begin);
        if (count == 6U) return accumulate_rhs_fixed<6U>(
            indices, weights, input, input_stride, begin);
    }
    if constexpr (FixedBandwidth == 7U) {
        if (count == 9U) return accumulate_rhs_fixed<9U>(
            indices, weights, input, input_stride, begin);
        if (count == 10U) return accumulate_rhs_fixed<10U>(
            indices, weights, input, input_stride, begin);
    }
    float sum = 0.0F;
    for (std::uint32_t position = begin; position < end; ++position) {
        const auto source = static_cast<std::uint32_t>(indices[position]);
        sum = fmaf(weights[position], input[source * input_stride], sum);
    }
    return sum;
}

template <std::uint32_t Bandwidth, bool UpperAscending, bool Precomputed>
__device__ __forceinline__ void inverse_axis_ring(
    const kernel::AxisPlanDescriptor &plan,
    const std::uint32_t *__restrict__ transpose_offsets,
    const std::int32_t *__restrict__ transpose_indices,
    const float *__restrict__ transpose_weights,
    const float *__restrict__ lower_ld,
    const float *__restrict__ upper_l,
    const float *__restrict__ inverse_diagonal,
    const float *__restrict__ input,
    std::uint32_t input_stride,
    float *__restrict__ output,
    std::uint32_t output_stride) {
    const std::uint32_t size = plan.destination_size;
    float recent[Bandwidth]{};
    for (std::uint32_t index = 0U; index < size; ++index) {
        float sum = 0.0F;
        if constexpr (Precomputed) {
            sum = output[index * output_stride];
        } else {
            const std::uint32_t begin = transpose_offsets[index];
            const std::uint32_t end = transpose_offsets[index + 1U];
            sum = accumulate_rhs_span<Bandwidth>(
                transpose_indices, transpose_weights, input, input_stride,
                begin, end);
        }
        const std::uint32_t available = minimum(Bandwidth, index);
#pragma unroll
        for (std::uint32_t distance = Bandwidth; distance > 0U; --distance) {
            if (distance <= available) {
                sum = fmaf(
                    -lower_ld[(distance - 1U) * size + index],
                    recent[distance - 1U], sum);
            }
        }
        const float current = sum * inverse_diagonal[index];
        output[index * output_stride] = current;
#pragma unroll
        for (std::uint32_t distance = Bandwidth - 1U;
             distance > 0U; --distance) {
            recent[distance] = recent[distance - 1U];
        }
        recent[0] = current;
    }

    if (size < 2U) return;
    recent[0] = output[(size - 1U) * output_stride];
#pragma unroll
    for (std::uint32_t distance = 1U; distance < Bandwidth; ++distance) {
        recent[distance] = 0.0F;
    }
    for (std::uint32_t reverse = size - 1U; reverse > 0U; --reverse) {
        const std::uint32_t index = reverse - 1U;
        const std::uint32_t available = minimum(
            Bandwidth, size - index - 1U);
        float sum = 0.0F;
        if constexpr (UpperAscending) {
#pragma unroll
            for (std::uint32_t distance = 1U;
                 distance <= Bandwidth; ++distance) {
                if (distance <= available) {
                    sum = fmaf(
                        upper_l[(distance - 1U) * size + index],
                        recent[distance - 1U], sum);
                }
            }
        } else {
#pragma unroll
            for (std::uint32_t distance = Bandwidth;
                 distance > 0U; --distance) {
                if (distance <= available) {
                    sum = fmaf(
                        upper_l[(distance - 1U) * size + index],
                        recent[distance - 1U], sum);
                }
            }
        }
        const float current = output[index * output_stride] - sum;
        output[index * output_stride] = current;
#pragma unroll
        for (std::uint32_t distance = Bandwidth - 1U;
             distance > 0U; --distance) {
            recent[distance] = recent[distance - 1U];
        }
        recent[0] = current;
    }
}

template <std::uint32_t Bandwidth, bool UpperAscending, bool Precomputed>
__device__ __forceinline__ void inverse_axis_row_major_tiled(
    const kernel::AxisPlanDescriptor &plan,
    const std::uint32_t *__restrict__ transpose_offsets,
    const std::int32_t *__restrict__ transpose_indices,
    const float *__restrict__ transpose_weights,
    const float *__restrict__ lower_ld,
    const float *__restrict__ upper_l,
    const float *__restrict__ inverse_diagonal,
    const float *__restrict__ input,
    std::uint32_t input_stride,
    float *__restrict__ output,
    std::uint32_t vector_index,
    std::uint32_t vector_count,
    float *tile) {
    constexpr std::uint32_t tile_width = 32U;
    constexpr std::uint32_t tile_pitch = tile_width + 1U;
    const std::uint32_t size = plan.destination_size;
    const bool active = vector_index < vector_count;
    float recent[Bandwidth]{};
    float last = 0.0F;

    for (std::uint32_t tile_base = 0U; tile_base < size;
         tile_base += tile_width) {
#pragma unroll
        for (std::uint32_t offset = 0U; offset < tile_width; ++offset) {
            const std::uint32_t index = tile_base + offset;
            if (active && index < size) {
                float sum = 0.0F;
                if constexpr (Precomputed) {
                    sum = output[vector_index * size + index];
                } else {
                    const std::uint32_t begin = transpose_offsets[index];
                    const std::uint32_t end = transpose_offsets[index + 1U];
                    sum = accumulate_rhs_span<Bandwidth>(
                        transpose_indices, transpose_weights, input,
                        input_stride, begin, end);
                }
                const std::uint32_t available = minimum(Bandwidth, index);
#pragma unroll
                for (std::uint32_t distance = Bandwidth;
                     distance > 0U; --distance) {
                    if (distance <= available) {
                        sum = fmaf(
                            -lower_ld[(distance - 1U) * size + index],
                            recent[distance - 1U], sum);
                    }
                }
                const float current = sum * inverse_diagonal[index];
                tile[threadIdx.x * tile_pitch + offset] = current;
                last = current;
#pragma unroll
                for (std::uint32_t distance = Bandwidth - 1U;
                     distance > 0U; --distance) {
                    recent[distance] = recent[distance - 1U];
                }
                recent[0] = current;
            }
        }
        __syncthreads();

        const std::uint32_t tile_elements = blockDim.x * tile_width;
        for (std::uint32_t linear = threadIdx.x; linear < tile_elements;
             linear += blockDim.x) {
            const std::uint32_t local_vector = linear / tile_width;
            const std::uint32_t offset = linear - local_vector * tile_width;
            const std::uint32_t destination_index = tile_base + offset;
            const std::uint32_t destination_vector =
                blockIdx.x * blockDim.x + local_vector;
            if (destination_vector < vector_count
                && destination_index < size) {
                output[destination_vector * size + destination_index] =
                    tile[local_vector * tile_pitch + offset];
            }
        }
        __syncthreads();
    }

    if (size < 2U) return;
    recent[0] = last;
#pragma unroll
    for (std::uint32_t distance = 1U; distance < Bandwidth; ++distance) {
        recent[distance] = 0.0F;
    }
    for (std::uint32_t tile_end = size - 1U; tile_end > 0U;) {
        const std::uint32_t tile_count = minimum(tile_width, tile_end);
        const std::uint32_t tile_begin = tile_end - tile_count;
        const std::uint32_t tile_elements = blockDim.x * tile_width;
        for (std::uint32_t linear = threadIdx.x; linear < tile_elements;
             linear += blockDim.x) {
            const std::uint32_t local_vector = linear / tile_width;
            const std::uint32_t offset = linear - local_vector * tile_width;
            const std::uint32_t source_vector =
                blockIdx.x * blockDim.x + local_vector;
            if (source_vector < vector_count && offset < tile_count) {
                tile[local_vector * tile_pitch + offset] =
                    output[source_vector * size + tile_begin + offset];
            }
        }
        __syncthreads();

        if (active) {
            for (std::uint32_t reverse = tile_end;
                 reverse > tile_begin; --reverse) {
                const std::uint32_t index = reverse - 1U;
                const std::uint32_t available = minimum(
                    Bandwidth, size - index - 1U);
                float sum = 0.0F;
                if constexpr (UpperAscending) {
#pragma unroll
                    for (std::uint32_t distance = 1U;
                         distance <= Bandwidth; ++distance) {
                        if (distance <= available) {
                            sum = fmaf(
                                upper_l[(distance - 1U) * size + index],
                                recent[distance - 1U], sum);
                        }
                    }
                } else {
#pragma unroll
                    for (std::uint32_t distance = Bandwidth;
                         distance > 0U; --distance) {
                        if (distance <= available) {
                            sum = fmaf(
                                upper_l[(distance - 1U) * size + index],
                                recent[distance - 1U], sum);
                        }
                    }
                }
                const std::uint32_t offset = index - tile_begin;
                const float current =
                    tile[threadIdx.x * tile_pitch + offset] - sum;
                tile[threadIdx.x * tile_pitch + offset] = current;
#pragma unroll
                for (std::uint32_t distance = Bandwidth - 1U;
                     distance > 0U; --distance) {
                    recent[distance] = recent[distance - 1U];
                }
                recent[0] = current;
            }
        }
        __syncthreads();

        for (std::uint32_t linear = threadIdx.x; linear < tile_elements;
             linear += blockDim.x) {
            const std::uint32_t local_vector = linear / tile_width;
            const std::uint32_t offset = linear - local_vector * tile_width;
            const std::uint32_t destination_vector =
                blockIdx.x * blockDim.x + local_vector;
            if (destination_vector < vector_count && offset < tile_count) {
                output[destination_vector * size + tile_begin + offset] =
                    tile[local_vector * tile_pitch + offset];
            }
        }
        __syncthreads();
        tile_end = tile_begin;
    }
}

__device__ __forceinline__ void inverse_axis(
    const kernel::AxisPlanDescriptor &plan,
    const std::uint32_t *__restrict__ transpose_offsets,
    const std::int32_t *__restrict__ transpose_indices,
    const float *__restrict__ transpose_weights,
    const float *__restrict__ lower_ld,
    const float *__restrict__ upper_l,
    const float *__restrict__ inverse_diagonal,
    const float *__restrict__ input,
    std::uint32_t input_stride,
    float *__restrict__ output,
    std::uint32_t output_stride) {
    if (plan.half_bandwidth == 1U) {
        inverse_axis_ring<1U, false, false>(
            plan, transpose_offsets, transpose_indices, transpose_weights,
            lower_ld, upper_l, inverse_diagonal,
            input, input_stride, output, output_stride);
        return;
    }
    if (plan.half_bandwidth == 3U) {
        inverse_axis_ring<3U, true, false>(
            plan, transpose_offsets, transpose_indices, transpose_weights,
            lower_ld, upper_l, inverse_diagonal,
            input, input_stride, output, output_stride);
        return;
    }
    if (plan.half_bandwidth == 5U) {
        inverse_axis_ring<5U, false, false>(
            plan, transpose_offsets, transpose_indices, transpose_weights,
            lower_ld, upper_l, inverse_diagonal,
            input, input_stride, output, output_stride);
        return;
    }
    if (plan.half_bandwidth == 7U) {
        inverse_axis_ring<7U, false, false>(
            plan, transpose_offsets, transpose_indices, transpose_weights,
            lower_ld, upper_l, inverse_diagonal,
            input, input_stride, output, output_stride);
        return;
    }

    const std::uint32_t size = plan.destination_size;
    for (std::uint32_t index = 0U; index < size; ++index) {
        float sum = 0.0F;
        const std::uint32_t begin = transpose_offsets[index];
        const std::uint32_t end = transpose_offsets[index + 1U];
        // Wide and uncommon bands keep the compact dynamic RHS. Expanding
        // this shared path regresses large downscales and unrelated bands.
        for (std::uint32_t position = begin; position < end; ++position) {
            const auto source = static_cast<std::uint32_t>(
                transpose_indices[position]);
            sum = fmaf(
                transpose_weights[position], input[source * input_stride], sum);
        }
        const std::uint32_t available = minimum(plan.half_bandwidth, index);
        for (std::uint32_t distance = available; distance > 0U; --distance) {
            sum = fmaf(
                -lower_ld[(distance - 1U) * size + index],
                output[(index - distance) * output_stride], sum);
        }
        output[index * output_stride] = sum * inverse_diagonal[index];
    }
    if (size < 2U) return;
    for (std::uint32_t reverse = size - 1U; reverse > 0U; --reverse) {
        const std::uint32_t index = reverse - 1U;
        const std::uint32_t available = minimum(
            plan.half_bandwidth, size - index - 1U);
        float sum = 0.0F;
        if (plan.half_bandwidth == 3U) {
            for (std::uint32_t distance = 1U;
                 distance <= available; ++distance) {
                sum = fmaf(
                    upper_l[(distance - 1U) * size + index],
                    output[(index + distance) * output_stride], sum);
            }
        } else {
            for (std::uint32_t distance = available;
                 distance > 0U; --distance) {
                sum = fmaf(
                    upper_l[(distance - 1U) * size + index],
                    output[(index + distance) * output_stride], sum);
            }
        }
        output[index * output_stride] -= sum;
    }
}

__device__ __forceinline__ void solve_axis(
    const kernel::AxisPlanDescriptor &plan,
    const float *__restrict__ lower_ld,
    const float *__restrict__ upper_l,
    const float *__restrict__ inverse_diagonal,
    float *__restrict__ output,
    std::uint32_t output_stride) {
    if (plan.half_bandwidth == 1U) {
        inverse_axis_ring<1U, false, true>(
            plan, nullptr, nullptr, nullptr,
            lower_ld, upper_l, inverse_diagonal,
            nullptr, 0U, output, output_stride);
        return;
    }
    if (plan.half_bandwidth == 3U) {
        inverse_axis_ring<3U, true, true>(
            plan, nullptr, nullptr, nullptr,
            lower_ld, upper_l, inverse_diagonal,
            nullptr, 0U, output, output_stride);
        return;
    }
    if (plan.half_bandwidth == 5U) {
        inverse_axis_ring<5U, false, true>(
            plan, nullptr, nullptr, nullptr,
            lower_ld, upper_l, inverse_diagonal,
            nullptr, 0U, output, output_stride);
        return;
    }
    if (plan.half_bandwidth == 7U) {
        inverse_axis_ring<7U, false, true>(
            plan, nullptr, nullptr, nullptr,
            lower_ld, upper_l, inverse_diagonal,
            nullptr, 0U, output, output_stride);
        return;
    }

    const std::uint32_t size = plan.destination_size;
    for (std::uint32_t index = 0U; index < size; ++index) {
        float sum = output[index * output_stride];
        const std::uint32_t available = minimum(plan.half_bandwidth, index);
        for (std::uint32_t distance = available; distance > 0U; --distance) {
            sum = fmaf(
                -lower_ld[(distance - 1U) * size + index],
                output[(index - distance) * output_stride], sum);
        }
        output[index * output_stride] = sum * inverse_diagonal[index];
    }
    if (size < 2U) return;
    for (std::uint32_t reverse = size - 1U; reverse > 0U; --reverse) {
        const std::uint32_t index = reverse - 1U;
        const std::uint32_t available = minimum(
            plan.half_bandwidth, size - index - 1U);
        float sum = 0.0F;
        if (plan.half_bandwidth == 3U) {
            for (std::uint32_t distance = 1U;
                 distance <= available; ++distance) {
                sum = fmaf(
                    upper_l[(distance - 1U) * size + index],
                    output[(index + distance) * output_stride], sum);
            }
        } else {
            for (std::uint32_t distance = available;
                 distance > 0U; --distance) {
                sum = fmaf(
                    upper_l[(distance - 1U) * size + index],
                    output[(index + distance) * output_stride], sum);
            }
        }
        output[index * output_stride] -= sum;
    }
}

__device__ __forceinline__ void solve_axis_row_major_tiled(
    const kernel::AxisPlanDescriptor &plan,
    const float *__restrict__ lower_ld,
    const float *__restrict__ upper_l,
    const float *__restrict__ inverse_diagonal,
    float *__restrict__ output,
    std::uint32_t vector_index,
    std::uint32_t vector_count,
    float *tile) {
    if (plan.half_bandwidth == 1U) {
        inverse_axis_row_major_tiled<1U, false, true>(
            plan, nullptr, nullptr, nullptr,
            lower_ld, upper_l, inverse_diagonal,
            nullptr, 0U, output, vector_index, vector_count, tile);
        return;
    }
    if (plan.half_bandwidth == 3U) {
        inverse_axis_row_major_tiled<3U, true, true>(
            plan, nullptr, nullptr, nullptr,
            lower_ld, upper_l, inverse_diagonal,
            nullptr, 0U, output, vector_index, vector_count, tile);
        return;
    }
    if (plan.half_bandwidth == 5U) {
        inverse_axis_row_major_tiled<5U, false, true>(
            plan, nullptr, nullptr, nullptr,
            lower_ld, upper_l, inverse_diagonal,
            nullptr, 0U, output, vector_index, vector_count, tile);
        return;
    }
    if (plan.half_bandwidth == 7U) {
        inverse_axis_row_major_tiled<7U, false, true>(
            plan, nullptr, nullptr, nullptr,
            lower_ld, upper_l, inverse_diagonal,
            nullptr, 0U, output, vector_index, vector_count, tile);
        return;
    }
    if (vector_index < vector_count) {
        solve_axis(
            plan, lower_ld, upper_l, inverse_diagonal,
            output + vector_index * plan.destination_size, 1U);
    }
}

template <bool ColumnMajorOutput>
__device__ __forceinline__ void rhs_axis_2d(
    const kernel::AxisPlanDescriptor &plan,
    const std::uint32_t *__restrict__ transpose_offsets,
    const std::int32_t *__restrict__ transpose_indices,
    const float *__restrict__ transpose_weights,
    const float *__restrict__ input,
    std::uint32_t vector_count,
    float *__restrict__ output) {
    const std::uint32_t vector = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t index = blockIdx.y * blockDim.y + threadIdx.y;
    if (vector >= vector_count || index >= plan.destination_size) return;

    float sum = 0.0F;
    const std::uint32_t begin = transpose_offsets[index];
    const std::uint32_t end = transpose_offsets[index + 1U];
    for (std::uint32_t position = begin; position < end; ++position) {
        const auto source = static_cast<std::uint32_t>(
            transpose_indices[position]);
        sum = fmaf(
            transpose_weights[position],
            input[source * vector_count + vector], sum);
    }
    if constexpr (ColumnMajorOutput) {
        output[index * vector_count + vector] = sum;
    } else {
        output[vector * plan.destination_size + index] = sum;
    }
}

template <class Sample>
__device__ __forceinline__ float load_source(
    Sample value, const kernel::IntegerConversionDescriptor &conversion) {
    return (static_cast<float>(value) - conversion.input_offset)
        * conversion.input_scale;
}

template <class Sample, bool Convert>
__device__ __forceinline__ void transpose_source(
    const Sample *__restrict__ source,
    std::uint32_t source_width,
    std::uint32_t source_height,
    kernel::IntegerConversionDescriptor conversion,
    float *__restrict__ transposed,
    float (&tile)[32][33]) {
    const std::uint32_t x = blockIdx.x * 32U + threadIdx.x;
    const std::uint32_t y = blockIdx.y * 32U + threadIdx.y;
#pragma unroll
    for (std::uint32_t offset = 0U; offset < 32U; offset += 8U) {
        if (x < source_width && y + offset < source_height) {
            const Sample value = source[(y + offset) * source_width + x];
            if constexpr (Convert) {
                tile[threadIdx.y + offset][threadIdx.x] =
                    load_source(value, conversion);
            } else {
                tile[threadIdx.y + offset][threadIdx.x] =
                    static_cast<float>(value);
            }
        }
    }
    __syncthreads();

    const std::uint32_t output_x = blockIdx.y * 32U + threadIdx.x;
    const std::uint32_t output_y = blockIdx.x * 32U + threadIdx.y;
#pragma unroll
    for (std::uint32_t offset = 0U; offset < 32U; offset += 8U) {
        if (output_x < source_height && output_y + offset < source_width) {
            transposed[(output_y + offset) * source_height + output_x] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

template <class Sample>
__device__ __forceinline__ Sample convert_output(
    float value, const kernel::IntegerConversionDescriptor &conversion) {
    value = fmaf(value, conversion.output_scale, conversion.output_offset);
    value = fminf(
        fmaxf(value, 0.0F), static_cast<float>(conversion.output_maximum));
    return static_cast<Sample>(__float2uint_rn(value));
}

__device__ __forceinline__ double plan_weight_f64(
    kernel::PlanPrecision precision, std::uint32_t position,
    const float *__restrict__ weights_f32,
    const double *__restrict__ weights_f64) {
    return precision == kernel::PlanPrecision::float64
        ? weights_f64[position]
        : static_cast<double>(weights_f32[position]);
}

__device__ __forceinline__ double lower_factor_f64(
    kernel::PlanPrecision precision, std::uint32_t size,
    std::uint32_t index, std::uint32_t distance,
    const float *__restrict__ lower_f32,
    const double *__restrict__ bands_f64) {
    return precision == kernel::PlanPrecision::float64
        ? bands_f64[distance * size + index - distance]
        : static_cast<double>(
              lower_f32[(distance - 1U) * size + index]);
}

__device__ __forceinline__ double upper_factor_f64(
    kernel::PlanPrecision precision, std::uint32_t size,
    std::uint32_t index, std::uint32_t distance,
    const float *__restrict__ upper_f32,
    const double *__restrict__ bands_f64) {
    return precision == kernel::PlanPrecision::float64
        ? bands_f64[distance * size + index]
        : static_cast<double>(
              upper_f32[(distance - 1U) * size + index]);
}

__device__ __forceinline__ void solve_axis_f64(
    const kernel::AxisPlanDescriptor &plan,
    const float *__restrict__ lower_f32,
    const float *__restrict__ upper_f32,
    const float *__restrict__ diagonal_f32,
    const double *__restrict__ bands_f64,
    kernel::PlanPrecision precision,
    double *__restrict__ output,
    std::uint32_t output_stride) {
    const std::uint32_t size = plan.destination_size;
    for (std::uint32_t index = 0U; index < size; ++index) {
        double value = output[index * output_stride];
        const std::uint32_t available = minimum(plan.half_bandwidth, index);
        for (std::uint32_t distance = available;
             distance > 0U; --distance) {
            value = __dsub_rn(
                value,
                __dmul_rn(
                    lower_factor_f64(
                        precision, size, index, distance,
                        lower_f32, bands_f64),
                    output[(index - distance) * output_stride]));
        }
        if (precision == kernel::PlanPrecision::float64) {
            output[index * output_stride] = value;
        } else {
            output[index * output_stride] = __dmul_rn(
                value, static_cast<double>(diagonal_f32[index]));
        }
    }
    if (precision == kernel::PlanPrecision::float64) {
        for (std::uint32_t index = 0U; index < size; ++index) {
            output[index * output_stride] = __ddiv_rn(
                output[index * output_stride],
                __dadd_rn(bands_f64[index], DBL_EPSILON));
        }
    }
    if (size < 2U) return;
    for (std::uint32_t reverse = size - 1U; reverse > 0U; --reverse) {
        const std::uint32_t index = reverse - 1U;
        const std::uint32_t available = minimum(
            plan.half_bandwidth, size - index - 1U);
        double value = output[index * output_stride];
        for (std::uint32_t distance = available;
             distance > 0U; --distance) {
            value = __dsub_rn(
                value,
                __dmul_rn(
                    upper_factor_f64(
                        precision, size, index, distance,
                        upper_f32, bands_f64),
                    output[(index + distance) * output_stride]));
        }
        output[index * output_stride] = value;
    }
}

__device__ __forceinline__ void inverse_axis_f64(
    const kernel::AxisPlanDescriptor &plan,
    const std::uint32_t *__restrict__ transpose_offsets,
    const std::int32_t *__restrict__ transpose_indices,
    const float *__restrict__ weights_f32,
    const float *__restrict__ lower_f32,
    const float *__restrict__ upper_f32,
    const float *__restrict__ diagonal_f32,
    const double *__restrict__ weights_f64,
    const double *__restrict__ bands_f64,
    kernel::PlanPrecision precision,
    const double *__restrict__ input,
    std::uint32_t input_stride,
    double *__restrict__ output,
    std::uint32_t output_stride) {
    for (std::uint32_t index = 0U;
         index < plan.destination_size; ++index) {
        double value = 0.0;
        const std::uint32_t begin = transpose_offsets[index];
        const std::uint32_t end = transpose_offsets[index + 1U];
        for (std::uint32_t position = begin; position < end; ++position) {
            const auto source = static_cast<std::uint32_t>(
                transpose_indices[position]);
            value = __dadd_rn(
                value,
                __dmul_rn(
                    plan_weight_f64(
                        precision, position, weights_f32, weights_f64),
                    input[source * input_stride]));
        }
        output[index * output_stride] = value;
    }
    solve_axis_f64(
        plan, lower_f32, upper_f32, diagonal_f32, bands_f64,
        precision, output, output_stride);
}

template <bool ColumnMajorOutput>
__device__ __forceinline__ void rhs_axis_2d_f64(
    const kernel::AxisPlanDescriptor &plan,
    const std::uint32_t *__restrict__ transpose_offsets,
    const std::int32_t *__restrict__ transpose_indices,
    const float *__restrict__ weights_f32,
    const double *__restrict__ weights_f64,
    kernel::PlanPrecision precision,
    const double *__restrict__ input,
    std::uint32_t vector_count,
    double *__restrict__ output) {
    const std::uint32_t vector = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t index = blockIdx.y * blockDim.y + threadIdx.y;
    if (vector >= vector_count || index >= plan.destination_size) return;

    double value = 0.0;
    const std::uint32_t begin = transpose_offsets[index];
    const std::uint32_t end = transpose_offsets[index + 1U];
    for (std::uint32_t position = begin; position < end; ++position) {
        const auto source = static_cast<std::uint32_t>(
            transpose_indices[position]);
        value = __dadd_rn(
            value,
            __dmul_rn(
                plan_weight_f64(
                    precision, position, weights_f32, weights_f64),
                input[source * vector_count + vector]));
    }
    if constexpr (ColumnMajorOutput) {
        output[index * vector_count + vector] = value;
    } else {
        output[vector * plan.destination_size + index] = value;
    }
}

template <class Sample, bool Convert>
__device__ __forceinline__ void transpose_source_f64(
    const Sample *__restrict__ source,
    std::uint32_t source_width,
    std::uint32_t source_height,
    kernel::IntegerConversionDescriptor conversion,
    double *__restrict__ transposed,
    double (&tile)[32][33]) {
    const std::uint32_t x = blockIdx.x * 32U + threadIdx.x;
    const std::uint32_t y = blockIdx.y * 32U + threadIdx.y;
#pragma unroll
    for (std::uint32_t offset = 0U; offset < 32U; offset += 8U) {
        if (x < source_width && y + offset < source_height) {
            const Sample value = source[(y + offset) * source_width + x];
            if constexpr (Convert) {
                tile[threadIdx.y + offset][threadIdx.x] = __dmul_rn(
                    __dsub_rn(
                        static_cast<double>(value),
                        static_cast<double>(conversion.input_offset)),
                    static_cast<double>(conversion.input_scale));
            } else {
                tile[threadIdx.y + offset][threadIdx.x] =
                    static_cast<double>(value);
            }
        }
    }
    __syncthreads();

    const std::uint32_t output_x = blockIdx.y * 32U + threadIdx.x;
    const std::uint32_t output_y = blockIdx.x * 32U + threadIdx.y;
#pragma unroll
    for (std::uint32_t offset = 0U; offset < 32U; offset += 8U) {
        if (output_x < source_height && output_y + offset < source_width) {
            transposed[(output_y + offset) * source_height + output_x] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

template <class Sample>
__device__ __forceinline__ Sample convert_output_f64(
    double value, const kernel::IntegerConversionDescriptor &conversion,
    std::uint32_t *__restrict__ nonfinite) {
    if (!isfinite(value)) {
        atomicExch(nonfinite, 1U);
        return Sample{};
    }
    value = __dadd_rn(
        __dmul_rn(value, static_cast<double>(conversion.output_scale)),
        static_cast<double>(conversion.output_offset));
    if (!isfinite(value)) {
        atomicExch(nonfinite, 1U);
        return Sample{};
    }
    value = fmin(
        fmax(value, 0.0), static_cast<double>(conversion.output_maximum));
    return static_cast<Sample>(__double2uint_rn(value));
}

// ---- Partitioned solve (see src/solve_partition.hpp) ----------------------
// Column-major data, solved in place: element i of vector v at i * V + v.
// A block holds 32 vectors (threadIdx.x) and blockDim.y chunks of each.
// Arithmetic per element is that of inverse_axis_ring<W, UA, true>.

constexpr std::uint32_t chunk_batch = 8U;

template <std::uint32_t W>
__device__ __forceinline__ void shift_in(float (&recent)[W], float value) {
#pragma unroll
    for (std::uint32_t distance = W - 1U; distance > 0U; --distance) {
        recent[distance] = recent[distance - 1U];
    }
    recent[0] = value;
}

template <std::uint32_t W>
__device__ __forceinline__ void chunk_forward(
    const float *__restrict__ lower_ld,
    const float *__restrict__ inverse_diagonal,
    float *__restrict__ column, std::uint32_t stride, std::uint32_t size,
    std::uint32_t begin, std::uint32_t end, float (&recent)[W], bool store) {
    for (std::uint32_t base = begin; base < end; base += chunk_batch) {
        float values[chunk_batch];
#pragma unroll
        for (std::uint32_t offset = 0U; offset < chunk_batch; ++offset) {
            if (base + offset < end) {
                values[offset] = column[(base + offset) * stride];
            }
        }
#pragma unroll
        for (std::uint32_t offset = 0U; offset < chunk_batch; ++offset) {
            const std::uint32_t index = base + offset;
            if (index < end) {
                float sum = values[offset];
                const std::uint32_t available = minimum(W, index);
#pragma unroll
                for (std::uint32_t distance = W; distance > 0U; --distance) {
                    if (distance <= available) {
                        sum = fmaf(
                            -lower_ld[(distance - 1U) * size + index],
                            recent[distance - 1U], sum);
                    }
                }
                const float current = sum * inverse_diagonal[index];
                if (store) column[index * stride] = current;
                shift_in(recent, current);
            }
        }
    }
}

// Indices end - 1 down to begin.
template <std::uint32_t W, bool UpperAscending>
__device__ __forceinline__ void chunk_backward(
    const float *__restrict__ upper_l, float *__restrict__ column,
    std::uint32_t stride, std::uint32_t size, std::uint32_t begin,
    std::uint32_t end, float (&recent)[W], bool store) {
    for (std::uint32_t top = end; top > begin;) {
        const std::uint32_t count = minimum(chunk_batch, top - begin);
        float values[chunk_batch];
#pragma unroll
        for (std::uint32_t offset = 0U; offset < chunk_batch; ++offset) {
            if (offset < count) {
                values[offset] = column[(top - 1U - offset) * stride];
            }
        }
#pragma unroll
        for (std::uint32_t offset = 0U; offset < chunk_batch; ++offset) {
            if (offset < count) {
                const std::uint32_t index = top - 1U - offset;
                const std::uint32_t available = minimum(W, size - index - 1U);
                float sum = 0.0F;
                if constexpr (UpperAscending) {
#pragma unroll
                    for (std::uint32_t distance = 1U; distance <= W; ++distance) {
                        if (distance <= available) {
                            sum = fmaf(
                                upper_l[(distance - 1U) * size + index],
                                recent[distance - 1U], sum);
                        }
                    }
                } else {
#pragma unroll
                    for (std::uint32_t distance = W; distance > 0U; --distance) {
                        if (distance <= available) {
                            sum = fmaf(
                                upper_l[(distance - 1U) * size + index],
                                recent[distance - 1U], sum);
                        }
                    }
                }
                const float current = values[offset] - sum;
                if (store) column[index * stride] = current;
                shift_in(recent, current);
            }
        }
        top -= count;
    }
}

// carry[(chunk * W + k) * 32 + lane] holds, per chunk, first its end state
// from a zero carry-in and then its true carry-in. One thread per vector
// turns the former into the latter with the transfer matrices.
template <std::uint32_t W>
__device__ __forceinline__ void carry_across_chunks(
    float *carry, const float *__restrict__ transfers, std::uint32_t lane,
    std::uint32_t chunks, bool reverse) {
    const auto at = [&](std::uint32_t chunk, std::uint32_t k) -> float & {
        return carry[(chunk * W + k) * 32U + lane];
    };
    float incoming[W];
    const std::uint32_t first = reverse ? chunks - 1U : 0U;
#pragma unroll
    for (std::uint32_t k = 0U; k < W; ++k) incoming[k] = at(first, k);
    if (!reverse) {
#pragma unroll
        for (std::uint32_t k = 0U; k < W; ++k) at(0U, k) = 0.0F;
    }
    for (std::uint32_t step = 1U; step < chunks; ++step) {
        const std::uint32_t chunk = reverse ? chunks - 1U - step : step;
        const float *matrix = transfers + chunk * W * W;
        float next[W];
#pragma unroll
        for (std::uint32_t row = 0U; row < W; ++row) {
            float sum = at(chunk, row);
#pragma unroll
            for (std::uint32_t k = 0U; k < W; ++k) {
                sum = fmaf(matrix[row * W + k], incoming[k], sum);
            }
            next[row] = sum;
        }
#pragma unroll
        for (std::uint32_t k = 0U; k < W; ++k) {
            at(chunk, k) = incoming[k];
            incoming[k] = next[k];
        }
    }
}

template <std::uint32_t W, bool UpperAscending>
__device__ __forceinline__ void solve_axis_chunked(
    std::uint32_t vector_count, const kernel::AxisPlanDescriptor &plan,
    std::uint32_t chunk_length, const float *__restrict__ lower_ld,
    const float *__restrict__ upper_l,
    const float *__restrict__ inverse_diagonal,
    const float *__restrict__ transfers, float *__restrict__ output) {
    extern __shared__ float carry[];
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t chunk = threadIdx.y;
    const std::uint32_t chunks = blockDim.y;
    const std::uint32_t size = plan.destination_size;
    const std::uint32_t vector = blockIdx.x * 32U + lane;
    const bool active = vector < vector_count;
    const bool last = chunk + 1U == chunks;
    float *column = output + (active ? vector : 0U);
    const std::uint32_t begin = minimum(size, chunk * chunk_length);
    const std::uint32_t end = minimum(size, begin + chunk_length);
    float *own = carry + chunk * W * 32U + lane;

    // Forward substitution. The first chunk's carry-in is zero; the last
    // chunk's end state is never needed.
    float recent[W];
#pragma unroll
    for (std::uint32_t k = 0U; k < W; ++k) recent[k] = 0.0F;
    if (active && !last) {
        chunk_forward<W>(
            lower_ld, inverse_diagonal, column, vector_count, size,
            begin, end, recent, false);
    }
#pragma unroll
    for (std::uint32_t k = 0U; k < W; ++k) own[k * 32U] = recent[k];
    __syncthreads();
    if (chunk == 0U) {
        carry_across_chunks<W>(carry, transfers, lane, chunks, false);
    }
    __syncthreads();
#pragma unroll
    for (std::uint32_t k = 0U; k < W; ++k) recent[k] = own[k * 32U];
    if (active) {
        chunk_forward<W>(
            lower_ld, inverse_diagonal, column, vector_count, size,
            begin, end, recent, true);
    }
    __syncthreads();

    // Backward substitution. The last sample is final; the last chunk starts
    // from it, and the first chunk's end state is never needed.
    const std::uint32_t backward_end = minimum(end, size - 1U);
    const auto start_state = [&] {
#pragma unroll
        for (std::uint32_t k = 0U; k < W; ++k) recent[k] = 0.0F;
        if (last && active) recent[0] = column[(size - 1U) * vector_count];
    };
    start_state();
    if (active && chunk != 0U) {
        chunk_backward<W, UpperAscending>(
            upper_l, column, vector_count, size, begin, backward_end,
            recent, false);
    }
#pragma unroll
    for (std::uint32_t k = 0U; k < W; ++k) own[k * 32U] = recent[k];
    __syncthreads();
    if (chunk == 0U) {
        carry_across_chunks<W>(
            carry, transfers + chunks * W * W, lane, chunks, true);
    }
    __syncthreads();
    if (last) {
        start_state();
    } else {
#pragma unroll
        for (std::uint32_t k = 0U; k < W; ++k) recent[k] = own[k * 32U];
    }
    if (active) {
        chunk_backward<W, UpperAscending>(
            upper_l, column, vector_count, size, begin, backward_end,
            recent, true);
    }
}

} // namespace

extern "C" __global__ void dsmvc_cuda_transpose_f32(
    const float *__restrict__ source,
    std::uint32_t source_width,
    std::uint32_t source_height,
    float *__restrict__ transposed) {
    __shared__ float tile[32][33];
    transpose_source<float, false>(
        source, source_width, source_height, {}, transposed, tile);
}

extern "C" __global__ void dsmvc_cuda_transpose_u8(
    const std::uint8_t *__restrict__ source,
    std::uint32_t source_width,
    std::uint32_t source_height,
    kernel::IntegerConversionDescriptor conversion,
    float *__restrict__ transposed) {
    __shared__ float tile[32][33];
    transpose_source<std::uint8_t, true>(
        source, source_width, source_height, conversion, transposed, tile);
}

extern "C" __global__ void dsmvc_cuda_transpose_u16(
    const std::uint16_t *__restrict__ source,
    std::uint32_t source_width,
    std::uint32_t source_height,
    kernel::IntegerConversionDescriptor conversion,
    float *__restrict__ transposed) {
    __shared__ float tile[32][33];
    transpose_source<std::uint16_t, true>(
        source, source_width, source_height, conversion, transposed, tile);
}

extern "C" __global__ void dsmvc_cuda_rhs_horizontal(
    const float *__restrict__ transposed_source,
    std::uint32_t vector_count,
    kernel::AxisPlanDescriptor plan,
    const std::uint32_t *__restrict__ transpose_offsets,
    const std::int32_t *__restrict__ transpose_indices,
    const float *__restrict__ transpose_weights,
    float *__restrict__ output) {
    rhs_axis_2d<false>(
        plan, transpose_offsets, transpose_indices, transpose_weights,
        transposed_source, vector_count, output);
}

extern "C" __global__ void dsmvc_cuda_rhs_horizontal_column_major(
    const float *__restrict__ transposed_source,
    std::uint32_t vector_count,
    kernel::AxisPlanDescriptor plan,
    const std::uint32_t *__restrict__ transpose_offsets,
    const std::int32_t *__restrict__ transpose_indices,
    const float *__restrict__ transpose_weights,
    float *__restrict__ output) {
    rhs_axis_2d<true>(
        plan, transpose_offsets, transpose_indices, transpose_weights,
        transposed_source, vector_count, output);
}

extern "C" __global__ void dsmvc_cuda_rhs_vertical(
    const float *__restrict__ source,
    std::uint32_t source_width,
    kernel::AxisPlanDescriptor plan,
    const std::uint32_t *__restrict__ transpose_offsets,
    const std::int32_t *__restrict__ transpose_indices,
    const float *__restrict__ transpose_weights,
    float *__restrict__ output) {
    rhs_axis_2d<true>(
        plan, transpose_offsets, transpose_indices, transpose_weights,
        source, source_width, output);
}

extern "C" __global__ void dsmvc_cuda_inverse_horizontal(
    const float *__restrict__ transposed_source,
    std::uint32_t vector_count,
    kernel::AxisPlanDescriptor plan,
    const std::uint32_t *__restrict__ transpose_offsets,
    const std::int32_t *__restrict__ transpose_indices,
    const float *__restrict__ transpose_weights,
    const float *__restrict__ lower_ld,
    const float *__restrict__ upper_l,
    const float *__restrict__ inverse_diagonal,
    float *__restrict__ output) {
    const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float tile[];
    if (plan.half_bandwidth == 1U) {
        inverse_axis_row_major_tiled<1U, false, false>(
            plan, transpose_offsets, transpose_indices, transpose_weights,
            lower_ld, upper_l, inverse_diagonal,
            transposed_source + (row < vector_count ? row : 0U), vector_count,
            output, row, vector_count, tile);
        return;
    }
    if (plan.half_bandwidth == 3U) {
        inverse_axis_row_major_tiled<3U, true, false>(
            plan, transpose_offsets, transpose_indices, transpose_weights,
            lower_ld, upper_l, inverse_diagonal,
            transposed_source + (row < vector_count ? row : 0U), vector_count,
            output, row, vector_count, tile);
        return;
    }
    if (plan.half_bandwidth == 5U) {
        inverse_axis_row_major_tiled<5U, false, false>(
            plan, transpose_offsets, transpose_indices, transpose_weights,
            lower_ld, upper_l, inverse_diagonal,
            transposed_source + (row < vector_count ? row : 0U), vector_count,
            output, row, vector_count, tile);
        return;
    }
    if (plan.half_bandwidth == 7U) {
        inverse_axis_row_major_tiled<7U, false, false>(
            plan, transpose_offsets, transpose_indices, transpose_weights,
            lower_ld, upper_l, inverse_diagonal,
            transposed_source + (row < vector_count ? row : 0U), vector_count,
            output, row, vector_count, tile);
        return;
    }
    if (row >= vector_count) return;
    inverse_axis(
        plan, transpose_offsets, transpose_indices, transpose_weights,
        lower_ld, upper_l, inverse_diagonal,
        transposed_source + row, vector_count,
        output + row * plan.destination_size, 1U);
}

extern "C" __global__ void dsmvc_cuda_inverse_horizontal_column_major(
    const float *__restrict__ transposed_source,
    std::uint32_t vector_count,
    kernel::AxisPlanDescriptor plan,
    const std::uint32_t *__restrict__ transpose_offsets,
    const std::int32_t *__restrict__ transpose_indices,
    const float *__restrict__ transpose_weights,
    const float *__restrict__ lower_ld,
    const float *__restrict__ upper_l,
    const float *__restrict__ inverse_diagonal,
    float *__restrict__ output) {
    const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= vector_count) return;
    inverse_axis(
        plan, transpose_offsets, transpose_indices, transpose_weights,
        lower_ld, upper_l, inverse_diagonal,
        transposed_source + row, vector_count,
        output + row, vector_count);
}

extern "C" __global__ void dsmvc_cuda_inverse_vertical(
    const float *__restrict__ source,
    std::uint32_t source_width,
    kernel::AxisPlanDescriptor plan,
    const std::uint32_t *__restrict__ transpose_offsets,
    const std::int32_t *__restrict__ transpose_indices,
    const float *__restrict__ transpose_weights,
    const float *__restrict__ lower_ld,
    const float *__restrict__ upper_l,
    const float *__restrict__ inverse_diagonal,
    float *__restrict__ output) {
    const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= source_width) return;
    inverse_axis(
        plan, transpose_offsets, transpose_indices, transpose_weights,
        lower_ld, upper_l, inverse_diagonal,
        source + column, source_width, output + column, source_width);
}

extern "C" __global__ void dsmvc_cuda_solve_horizontal(
    std::uint32_t vector_count,
    kernel::AxisPlanDescriptor plan,
    const float *__restrict__ lower_ld,
    const float *__restrict__ upper_l,
    const float *__restrict__ inverse_diagonal,
    float *__restrict__ output) {
    const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float tile[];
    solve_axis_row_major_tiled(
        plan, lower_ld, upper_l, inverse_diagonal,
        output, row, vector_count, tile);
}

extern "C" __global__ void dsmvc_cuda_solve_horizontal_column_major(
    std::uint32_t vector_count,
    kernel::AxisPlanDescriptor plan,
    const float *__restrict__ lower_ld,
    const float *__restrict__ upper_l,
    const float *__restrict__ inverse_diagonal,
    float *__restrict__ output) {
    const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= vector_count) return;
    solve_axis(
        plan, lower_ld, upper_l, inverse_diagonal,
        output + row, vector_count);
}

extern "C" __global__ void dsmvc_cuda_solve_vertical(
    std::uint32_t source_width,
    kernel::AxisPlanDescriptor plan,
    const float *__restrict__ lower_ld,
    const float *__restrict__ upper_l,
    const float *__restrict__ inverse_diagonal,
    float *__restrict__ output) {
    const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= source_width) return;
    solve_axis(
        plan, lower_ld, upper_l, inverse_diagonal,
        output + column, source_width);
}

#define DSMVC_SOLVE_CHUNKED(name, W, UA) \
    extern "C" __global__ void __launch_bounds__(1024) name( \
        std::uint32_t vector_count, kernel::AxisPlanDescriptor plan, \
        std::uint32_t chunk_length, const float *__restrict__ lower_ld, \
        const float *__restrict__ upper_l, \
        const float *__restrict__ inverse_diagonal, \
        const float *__restrict__ transfers, float *__restrict__ output) { \
        solve_axis_chunked<W, UA>( \
            vector_count, plan, chunk_length, lower_ld, upper_l, \
            inverse_diagonal, transfers, output); \
    }
DSMVC_SOLVE_CHUNKED(dsmvc_cuda_solve_chunked_w1, 1U, false)
DSMVC_SOLVE_CHUNKED(dsmvc_cuda_solve_chunked_w3, 3U, true)
DSMVC_SOLVE_CHUNKED(dsmvc_cuda_solve_chunked_w5, 5U, false)
DSMVC_SOLVE_CHUNKED(dsmvc_cuda_solve_chunked_w7, 7U, false)
#undef DSMVC_SOLVE_CHUNKED

extern "C" __global__ void dsmvc_cuda_convert_u8(
    const float *__restrict__ source,
    std::uint32_t element_count,
    kernel::IntegerConversionDescriptor conversion,
    std::uint8_t *__restrict__ output) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = convert_output<std::uint8_t>(source[index], conversion);
    }
}

extern "C" __global__ void dsmvc_cuda_convert_u16(
    const float *__restrict__ source,
    std::uint32_t element_count,
    kernel::IntegerConversionDescriptor conversion,
    std::uint16_t *__restrict__ output) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = convert_output<std::uint16_t>(source[index], conversion);
    }
}

extern "C" __global__ void dsmvc_cuda_transpose_f32_f64(
    const float *__restrict__ source,
    std::uint32_t source_width,
    std::uint32_t source_height,
    double *__restrict__ transposed) {
    __shared__ double tile[32][33];
    transpose_source_f64<float, false>(
        source, source_width, source_height, {}, transposed, tile);
}

extern "C" __global__ void dsmvc_cuda_transpose_u8_f64(
    const std::uint8_t *__restrict__ source,
    std::uint32_t source_width,
    std::uint32_t source_height,
    kernel::IntegerConversionDescriptor conversion,
    double *__restrict__ transposed) {
    __shared__ double tile[32][33];
    transpose_source_f64<std::uint8_t, true>(
        source, source_width, source_height, conversion, transposed, tile);
}

extern "C" __global__ void dsmvc_cuda_transpose_u16_f64(
    const std::uint16_t *__restrict__ source,
    std::uint32_t source_width,
    std::uint32_t source_height,
    kernel::IntegerConversionDescriptor conversion,
    double *__restrict__ transposed) {
    __shared__ double tile[32][33];
    transpose_source_f64<std::uint16_t, true>(
        source, source_width, source_height, conversion, transposed, tile);
}

extern "C" __global__ void dsmvc_cuda_promote_f32_f64(
    const float *__restrict__ source,
    std::uint32_t element_count,
    double *__restrict__ destination) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < element_count) {
        destination[index] = static_cast<double>(source[index]);
    }
}

extern "C" __global__ void dsmvc_cuda_check_finite_f64(
    const double *__restrict__ source,
    std::uint32_t element_count,
    std::uint32_t *__restrict__ nonfinite) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < element_count && !isfinite(source[index])) {
        atomicExch(nonfinite, 1U);
    }
}

extern "C" __global__ void dsmvc_cuda_inverse_horizontal_f64(
    const double *__restrict__ source,
    std::uint32_t vector_count,
    kernel::AxisPlanDescriptor plan,
    const std::uint32_t *__restrict__ offsets,
    const std::int32_t *__restrict__ indices,
    const float *__restrict__ weights_f32,
    const float *__restrict__ lower_f32,
    const float *__restrict__ upper_f32,
    const float *__restrict__ diagonal_f32,
    const double *__restrict__ weights_f64,
    const double *__restrict__ bands_f64,
    kernel::PlanPrecision precision,
    double *__restrict__ output) {
    const std::uint32_t vector = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector >= vector_count) return;
    inverse_axis_f64(
        plan, offsets, indices, weights_f32, lower_f32, upper_f32,
        diagonal_f32, weights_f64, bands_f64, precision,
        source + vector, vector_count,
        output + vector * plan.destination_size, 1U);
}

extern "C" __global__ void dsmvc_cuda_inverse_vertical_f64(
    const double *__restrict__ source,
    std::uint32_t source_width,
    kernel::AxisPlanDescriptor plan,
    const std::uint32_t *__restrict__ offsets,
    const std::int32_t *__restrict__ indices,
    const float *__restrict__ weights_f32,
    const float *__restrict__ lower_f32,
    const float *__restrict__ upper_f32,
    const float *__restrict__ diagonal_f32,
    const double *__restrict__ weights_f64,
    const double *__restrict__ bands_f64,
    kernel::PlanPrecision precision,
    double *__restrict__ output) {
    const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= source_width) return;
    inverse_axis_f64(
        plan, offsets, indices, weights_f32, lower_f32, upper_f32,
        diagonal_f32, weights_f64, bands_f64, precision,
        source + column, source_width, output + column, source_width);
}

extern "C" __global__ void dsmvc_cuda_rhs_horizontal_f64(
    const double *__restrict__ source,
    std::uint32_t vector_count,
    kernel::AxisPlanDescriptor plan,
    const std::uint32_t *__restrict__ offsets,
    const std::int32_t *__restrict__ indices,
    const float *__restrict__ weights_f32,
    const double *__restrict__ weights_f64,
    kernel::PlanPrecision precision,
    double *__restrict__ output) {
    rhs_axis_2d_f64<false>(
        plan, offsets, indices, weights_f32, weights_f64, precision,
        source, vector_count, output);
}

extern "C" __global__ void dsmvc_cuda_rhs_vertical_f64(
    const double *__restrict__ source,
    std::uint32_t source_width,
    kernel::AxisPlanDescriptor plan,
    const std::uint32_t *__restrict__ offsets,
    const std::int32_t *__restrict__ indices,
    const float *__restrict__ weights_f32,
    const double *__restrict__ weights_f64,
    kernel::PlanPrecision precision,
    double *__restrict__ output) {
    rhs_axis_2d_f64<true>(
        plan, offsets, indices, weights_f32, weights_f64, precision,
        source, source_width, output);
}

extern "C" __global__ void dsmvc_cuda_solve_horizontal_f64(
    std::uint32_t vector_count,
    kernel::AxisPlanDescriptor plan,
    const float *__restrict__ lower_f32,
    const float *__restrict__ upper_f32,
    const float *__restrict__ diagonal_f32,
    const double *__restrict__ bands_f64,
    kernel::PlanPrecision precision,
    double *__restrict__ output) {
    const std::uint32_t vector = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector >= vector_count) return;
    solve_axis_f64(
        plan, lower_f32, upper_f32, diagonal_f32, bands_f64, precision,
        output + vector * plan.destination_size, 1U);
}

extern "C" __global__ void dsmvc_cuda_solve_vertical_f64(
    std::uint32_t source_width,
    kernel::AxisPlanDescriptor plan,
    const float *__restrict__ lower_f32,
    const float *__restrict__ upper_f32,
    const float *__restrict__ diagonal_f32,
    const double *__restrict__ bands_f64,
    kernel::PlanPrecision precision,
    double *__restrict__ output) {
    const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= source_width) return;
    solve_axis_f64(
        plan, lower_f32, upper_f32, diagonal_f32, bands_f64, precision,
        output + column, source_width);
}

extern "C" __global__ void dsmvc_cuda_convert_f64_f32(
    const double *__restrict__ source,
    std::uint32_t element_count,
    float *__restrict__ output,
    std::uint32_t *__restrict__ nonfinite) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < element_count) {
        const double value = source[index];
        const float converted = static_cast<float>(value);
        if (!isfinite(value) || !isfinite(converted)) {
            atomicExch(nonfinite, 1U);
            output[index] = 0.0F;
        } else {
            output[index] = converted;
        }
    }
}

extern "C" __global__ void dsmvc_cuda_convert_f64_u8(
    const double *__restrict__ source,
    std::uint32_t element_count,
    kernel::IntegerConversionDescriptor conversion,
    std::uint8_t *__restrict__ output,
    std::uint32_t *__restrict__ nonfinite) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = convert_output_f64<std::uint8_t>(
            source[index], conversion, nonfinite);
    }
}

extern "C" __global__ void dsmvc_cuda_convert_f64_u16(
    const double *__restrict__ source,
    std::uint32_t element_count,
    kernel::IntegerConversionDescriptor conversion,
    std::uint16_t *__restrict__ output,
    std::uint32_t *__restrict__ nonfinite) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = convert_output_f64<std::uint16_t>(
            source[index], conversion, nonfinite);
    }
}

namespace dsmvc::cuda_detail::cuda_launch {
namespace {

constexpr unsigned int rhs_vector_threads = 32U;
constexpr unsigned int rhs_index_threads = 8U;
constexpr unsigned int conversion_threads = 256U;

[[nodiscard]] constexpr unsigned int divide_up(
    std::uint32_t value, unsigned int divisor) noexcept {
    return (value + divisor - 1U) / divisor;
}

[[nodiscard]] cudaError_t launch_status() noexcept {
    return cudaGetLastError();
}

} // namespace

cudaError_t transpose(
    const float *source, std::uint32_t width, std::uint32_t height,
    float *destination, cudaStream_t stream) {
    dsmvc_cuda_transpose_f32<<<
        dim3(divide_up(width, 32U), divide_up(height, 32U)), dim3(32U, 8U),
        0U, stream>>>(source, width, height, destination);
    return launch_status();
}

cudaError_t transpose(
    const std::uint8_t *source, std::uint32_t width, std::uint32_t height,
    cuda_kernel::IntegerConversionDescriptor conversion,
    float *destination, cudaStream_t stream) {
    dsmvc_cuda_transpose_u8<<<
        dim3(divide_up(width, 32U), divide_up(height, 32U)), dim3(32U, 8U),
        0U, stream>>>(source, width, height, conversion, destination);
    return launch_status();
}

cudaError_t transpose(
    const std::uint16_t *source, std::uint32_t width, std::uint32_t height,
    cuda_kernel::IntegerConversionDescriptor conversion,
    float *destination, cudaStream_t stream) {
    dsmvc_cuda_transpose_u16<<<
        dim3(divide_up(width, 32U), divide_up(height, 32U)), dim3(32U, 8U),
        0U, stream>>>(source, width, height, conversion, destination);
    return launch_status();
}

cudaError_t transpose_f64(
    const float *source, std::uint32_t width, std::uint32_t height,
    double *destination, cudaStream_t stream) {
    dsmvc_cuda_transpose_f32_f64<<<
        dim3(divide_up(width, 32U), divide_up(height, 32U)), dim3(32U, 8U),
        0U, stream>>>(source, width, height, destination);
    return launch_status();
}

cudaError_t transpose_f64(
    const std::uint8_t *source, std::uint32_t width, std::uint32_t height,
    cuda_kernel::IntegerConversionDescriptor conversion,
    double *destination, cudaStream_t stream) {
    dsmvc_cuda_transpose_u8_f64<<<
        dim3(divide_up(width, 32U), divide_up(height, 32U)), dim3(32U, 8U),
        0U, stream>>>(source, width, height, conversion, destination);
    return launch_status();
}

cudaError_t transpose_f64(
    const std::uint16_t *source, std::uint32_t width, std::uint32_t height,
    cuda_kernel::IntegerConversionDescriptor conversion,
    double *destination, cudaStream_t stream) {
    dsmvc_cuda_transpose_u16_f64<<<
        dim3(divide_up(width, 32U), divide_up(height, 32U)), dim3(32U, 8U),
        0U, stream>>>(source, width, height, conversion, destination);
    return launch_status();
}

cudaError_t promote_f64(
    const float *source, std::uint32_t element_count,
    double *destination, cudaStream_t stream) {
    dsmvc_cuda_promote_f32_f64<<<
        divide_up(element_count, conversion_threads), conversion_threads,
        0U, stream>>>(source, element_count, destination);
    return launch_status();
}

cudaError_t check_finite_f64(
    const double *source, std::uint32_t element_count,
    std::uint32_t *nonfinite, cudaStream_t stream) {
    dsmvc_cuda_check_finite_f64<<<
        divide_up(element_count, conversion_threads), conversion_threads,
        0U, stream>>>(source, element_count, nonfinite);
    return launch_status();
}

cudaError_t inverse_horizontal(
    const float *source, std::uint32_t vector_count,
    cuda_kernel::AxisPlanDescriptor plan,
    const std::uint32_t *offsets, const std::int32_t *indices,
    const float *weights, const float *lower, const float *upper,
    const float *diagonal, float *output, bool column_major,
    unsigned int threads, unsigned int shared_bytes, cudaStream_t stream) {
    const dim3 grid(divide_up(vector_count, threads));
    const dim3 block(threads);
    if (column_major) {
        dsmvc_cuda_inverse_horizontal_column_major<<<grid, block, 0U, stream>>>(
            source, vector_count, plan, offsets, indices, weights,
            lower, upper, diagonal, output);
    } else {
        dsmvc_cuda_inverse_horizontal<<<grid, block, shared_bytes, stream>>>(
            source, vector_count, plan, offsets, indices, weights,
            lower, upper, diagonal, output);
    }
    return launch_status();
}

cudaError_t inverse_vertical(
    const float *source, std::uint32_t source_width,
    cuda_kernel::AxisPlanDescriptor plan,
    const std::uint32_t *offsets, const std::int32_t *indices,
    const float *weights, const float *lower, const float *upper,
    const float *diagonal, float *output, unsigned int threads,
    cudaStream_t stream) {
    dsmvc_cuda_inverse_vertical<<<divide_up(source_width, threads), threads,
        0U, stream>>>(
        source, source_width, plan, offsets, indices, weights,
        lower, upper, diagonal, output);
    return launch_status();
}

cudaError_t rhs_horizontal(
    const float *source, std::uint32_t vector_count,
    cuda_kernel::AxisPlanDescriptor plan,
    const std::uint32_t *offsets, const std::int32_t *indices,
    const float *weights, float *output, bool column_major,
    cudaStream_t stream) {
    const dim3 grid(
        divide_up(vector_count, rhs_vector_threads),
        divide_up(plan.destination_size, rhs_index_threads));
    const dim3 block(rhs_vector_threads, rhs_index_threads);
    if (column_major) {
        dsmvc_cuda_rhs_horizontal_column_major<<<grid, block, 0U, stream>>>(
            source, vector_count, plan, offsets, indices, weights, output);
    } else {
        dsmvc_cuda_rhs_horizontal<<<grid, block, 0U, stream>>>(
            source, vector_count, plan, offsets, indices, weights, output);
    }
    return launch_status();
}

cudaError_t rhs_vertical(
    const float *source, std::uint32_t source_width,
    cuda_kernel::AxisPlanDescriptor plan,
    const std::uint32_t *offsets, const std::int32_t *indices,
    const float *weights, float *output, cudaStream_t stream) {
    const dim3 grid(
        divide_up(source_width, rhs_vector_threads),
        divide_up(plan.destination_size, rhs_index_threads));
    dsmvc_cuda_rhs_vertical<<<grid,
        dim3(rhs_vector_threads, rhs_index_threads), 0U, stream>>>(
        source, source_width, plan, offsets, indices, weights, output);
    return launch_status();
}

cudaError_t solve_horizontal(
    std::uint32_t vector_count, cuda_kernel::AxisPlanDescriptor plan,
    const float *lower, const float *upper, const float *diagonal,
    float *output, bool column_major, unsigned int threads,
    unsigned int shared_bytes, cudaStream_t stream) {
    const dim3 grid(divide_up(vector_count, threads));
    if (column_major) {
        dsmvc_cuda_solve_horizontal_column_major<<<grid, threads, 0U, stream>>>(
            vector_count, plan, lower, upper, diagonal, output);
    } else {
        dsmvc_cuda_solve_horizontal<<<grid, threads, shared_bytes, stream>>>(
            vector_count, plan, lower, upper, diagonal, output);
    }
    return launch_status();
}

cudaError_t solve_vertical(
    std::uint32_t source_width, cuda_kernel::AxisPlanDescriptor plan,
    const float *lower, const float *upper, const float *diagonal,
    float *output, unsigned int threads, cudaStream_t stream) {
    dsmvc_cuda_solve_vertical<<<divide_up(source_width, threads), threads,
        0U, stream>>>(
        source_width, plan, lower, upper, diagonal, output);
    return launch_status();
}

cudaError_t solve_chunked(
    std::uint32_t vector_count, cuda_kernel::AxisPlanDescriptor plan,
    std::uint32_t chunks, std::uint32_t chunk_length,
    const float *lower, const float *upper, const float *diagonal,
    const float *transfers, float *output, cudaStream_t stream) {
    const dim3 block(32U, chunks);
    const dim3 grid(divide_up(vector_count, 32U));
    const std::size_t shared_bytes = static_cast<std::size_t>(chunks)
        * plan.half_bandwidth * 32U * sizeof(float);
    switch (plan.half_bandwidth) {
    case 1U:
        dsmvc_cuda_solve_chunked_w1<<<grid, block, shared_bytes, stream>>>(
            vector_count, plan, chunk_length, lower, upper, diagonal,
            transfers, output);
        break;
    case 3U:
        dsmvc_cuda_solve_chunked_w3<<<grid, block, shared_bytes, stream>>>(
            vector_count, plan, chunk_length, lower, upper, diagonal,
            transfers, output);
        break;
    case 5U:
        dsmvc_cuda_solve_chunked_w5<<<grid, block, shared_bytes, stream>>>(
            vector_count, plan, chunk_length, lower, upper, diagonal,
            transfers, output);
        break;
    case 7U:
        dsmvc_cuda_solve_chunked_w7<<<grid, block, shared_bytes, stream>>>(
            vector_count, plan, chunk_length, lower, upper, diagonal,
            transfers, output);
        break;
    default:
        return cudaErrorInvalidValue;
    }
    return launch_status();
}

cudaError_t inverse_horizontal_f64(
    const double *source, std::uint32_t vector_count,
    cuda_kernel::AxisPlanDescriptor plan,
    const std::uint32_t *offsets, const std::int32_t *indices,
    const float *weights_f32, const float *lower_f32,
    const float *upper_f32, const float *diagonal_f32,
    const double *weights_f64, const double *bands_f64,
    cuda_kernel::PlanPrecision precision, double *output,
    unsigned int threads, cudaStream_t stream) {
    dsmvc_cuda_inverse_horizontal_f64<<<
        divide_up(vector_count, threads), threads, 0U, stream>>>(
        source, vector_count, plan, offsets, indices,
        weights_f32, lower_f32, upper_f32, diagonal_f32,
        weights_f64, bands_f64, precision, output);
    return launch_status();
}

cudaError_t inverse_vertical_f64(
    const double *source, std::uint32_t source_width,
    cuda_kernel::AxisPlanDescriptor plan,
    const std::uint32_t *offsets, const std::int32_t *indices,
    const float *weights_f32, const float *lower_f32,
    const float *upper_f32, const float *diagonal_f32,
    const double *weights_f64, const double *bands_f64,
    cuda_kernel::PlanPrecision precision, double *output,
    unsigned int threads, cudaStream_t stream) {
    dsmvc_cuda_inverse_vertical_f64<<<
        divide_up(source_width, threads), threads, 0U, stream>>>(
        source, source_width, plan, offsets, indices,
        weights_f32, lower_f32, upper_f32, diagonal_f32,
        weights_f64, bands_f64, precision, output);
    return launch_status();
}

cudaError_t rhs_horizontal_f64(
    const double *source, std::uint32_t vector_count,
    cuda_kernel::AxisPlanDescriptor plan,
    const std::uint32_t *offsets, const std::int32_t *indices,
    const float *weights_f32, const double *weights_f64,
    cuda_kernel::PlanPrecision precision, double *output,
    cudaStream_t stream) {
    const dim3 grid(
        divide_up(vector_count, rhs_vector_threads),
        divide_up(plan.destination_size, rhs_index_threads));
    dsmvc_cuda_rhs_horizontal_f64<<<
        grid, dim3(rhs_vector_threads, rhs_index_threads), 0U, stream>>>(
        source, vector_count, plan, offsets, indices,
        weights_f32, weights_f64, precision, output);
    return launch_status();
}

cudaError_t rhs_vertical_f64(
    const double *source, std::uint32_t source_width,
    cuda_kernel::AxisPlanDescriptor plan,
    const std::uint32_t *offsets, const std::int32_t *indices,
    const float *weights_f32, const double *weights_f64,
    cuda_kernel::PlanPrecision precision, double *output,
    cudaStream_t stream) {
    const dim3 grid(
        divide_up(source_width, rhs_vector_threads),
        divide_up(plan.destination_size, rhs_index_threads));
    dsmvc_cuda_rhs_vertical_f64<<<
        grid, dim3(rhs_vector_threads, rhs_index_threads), 0U, stream>>>(
        source, source_width, plan, offsets, indices,
        weights_f32, weights_f64, precision, output);
    return launch_status();
}

cudaError_t solve_horizontal_f64(
    std::uint32_t vector_count, cuda_kernel::AxisPlanDescriptor plan,
    const float *lower_f32, const float *upper_f32,
    const float *diagonal_f32, const double *bands_f64,
    cuda_kernel::PlanPrecision precision, double *output,
    unsigned int threads, cudaStream_t stream) {
    dsmvc_cuda_solve_horizontal_f64<<<
        divide_up(vector_count, threads), threads, 0U, stream>>>(
        vector_count, plan, lower_f32, upper_f32, diagonal_f32,
        bands_f64, precision, output);
    return launch_status();
}

cudaError_t solve_vertical_f64(
    std::uint32_t source_width, cuda_kernel::AxisPlanDescriptor plan,
    const float *lower_f32, const float *upper_f32,
    const float *diagonal_f32, const double *bands_f64,
    cuda_kernel::PlanPrecision precision, double *output,
    unsigned int threads, cudaStream_t stream) {
    dsmvc_cuda_solve_vertical_f64<<<
        divide_up(source_width, threads), threads, 0U, stream>>>(
        source_width, plan, lower_f32, upper_f32, diagonal_f32,
        bands_f64, precision, output);
    return launch_status();
}

cudaError_t convert(
    const float *source, std::uint32_t element_count,
    cuda_kernel::IntegerConversionDescriptor conversion,
    std::uint8_t *output, cudaStream_t stream) {
    dsmvc_cuda_convert_u8<<<divide_up(element_count, conversion_threads),
        conversion_threads, 0U, stream>>>(
        source, element_count, conversion, output);
    return launch_status();
}

cudaError_t convert(
    const float *source, std::uint32_t element_count,
    cuda_kernel::IntegerConversionDescriptor conversion,
    std::uint16_t *output, cudaStream_t stream) {
    dsmvc_cuda_convert_u16<<<divide_up(element_count, conversion_threads),
        conversion_threads, 0U, stream>>>(
        source, element_count, conversion, output);
    return launch_status();
}

cudaError_t convert_f64(
    const double *source, std::uint32_t element_count,
    float *output, std::uint32_t *nonfinite, cudaStream_t stream) {
    dsmvc_cuda_convert_f64_f32<<<divide_up(element_count, conversion_threads),
        conversion_threads, 0U, stream>>>(
            source, element_count, output, nonfinite);
    return launch_status();
}

cudaError_t convert_f64(
    const double *source, std::uint32_t element_count,
    cuda_kernel::IntegerConversionDescriptor conversion,
    std::uint8_t *output, std::uint32_t *nonfinite, cudaStream_t stream) {
    dsmvc_cuda_convert_f64_u8<<<divide_up(element_count, conversion_threads),
        conversion_threads, 0U, stream>>>(
        source, element_count, conversion, output, nonfinite);
    return launch_status();
}

cudaError_t convert_f64(
    const double *source, std::uint32_t element_count,
    cuda_kernel::IntegerConversionDescriptor conversion,
    std::uint16_t *output, std::uint32_t *nonfinite, cudaStream_t stream) {
    dsmvc_cuda_convert_f64_u16<<<divide_up(element_count, conversion_threads),
        conversion_threads, 0U, stream>>>(
        source, element_count, conversion, output, nonfinite);
    return launch_status();
}

} // namespace dsmvc::cuda_detail::cuda_launch
