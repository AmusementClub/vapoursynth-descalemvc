#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "axis_plan_internal.hpp"
#include "cpu_packed.hpp"

namespace dsmvc {

void inverse_rows_avx512(
    const AxisPlan &plan, const detail::PackedCpuPlan &packed,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t row_count) {
    if (row_count < 16) {
        for (std::int32_t row = 0; row < row_count; ++row) {
            inverse_axis_f32(
                plan, input + static_cast<std::ptrdiff_t>(row) * input_row_stride,
                1, output + static_cast<std::ptrdiff_t>(row) * output_row_stride,
                1);
        }
        return;
    }

    const auto source_size = packed.padded_source_size;
    const auto destination_size = packed.padded_destination_size;
    const auto factor_stride = static_cast<std::size_t>(destination_size);
    thread_local std::vector<float> scratch;
    thread_local std::vector<float> work;
    scratch.resize(static_cast<std::size_t>(source_size) * 16U);
    work.resize(static_cast<std::size_t>(destination_size) * 16U);

    for (std::int32_t block = 0; block + 16 <= row_count; block += 16) {
        for (std::int32_t source = 0; source < source_size; ++source) {
            auto *lane = scratch.data() + static_cast<std::size_t>(source) * 16U;
            for (std::int32_t row = 0; row < 16; ++row) {
                lane[row] = source < plan.source_size
                    ? input[static_cast<std::ptrdiff_t>(block + row)
                            * input_row_stride + source]
                    : 0.0F;
            }
        }

        for (std::int32_t i = 0; i < destination_size; ++i) {
            __m512 value = _mm512_setzero_ps();
            if (i < plan.destination_size) {
                const auto left = packed.weights_left[static_cast<std::size_t>(i)];
                const auto right = packed.weights_right[static_cast<std::size_t>(i)];
                const auto base = static_cast<std::size_t>(i)
                    * static_cast<std::size_t>(packed.weights_columns);
                for (std::int32_t source = left; source < right; ++source) {
                    value = _mm512_fmadd_ps(
                        _mm512_set1_ps(packed.weights[base
                            + static_cast<std::size_t>(source - left)]),
                        _mm512_loadu_ps(scratch.data()
                            + static_cast<std::size_t>(source) * 16U), value);
                }
                const auto available = std::min(plan.half_bandwidth, i);
                for (std::int32_t distance = available; distance >= 1; --distance) {
                    value = _mm512_fnmadd_ps(
                        _mm512_set1_ps(packed.lower_ld[
                            static_cast<std::size_t>(distance - 1) * factor_stride
                            + static_cast<std::size_t>(i)]),
                        _mm512_loadu_ps(work.data()
                            + static_cast<std::size_t>(i - distance) * 16U), value);
                }
                value = _mm512_mul_ps(value, _mm512_set1_ps(
                    packed.inverse_diagonal[static_cast<std::size_t>(i)]));
            }
            _mm512_storeu_ps(work.data() + static_cast<std::size_t>(i) * 16U,
                             value);
        }
        for (std::int32_t i = plan.destination_size - 2; i >= 0; --i) {
            __m512 value = _mm512_loadu_ps(
                work.data() + static_cast<std::size_t>(i) * 16U);
            const auto available = std::min(plan.half_bandwidth,
                plan.destination_size - i - 1);
            for (std::int32_t distance = available; distance >= 1; --distance) {
                value = _mm512_fnmadd_ps(
                    _mm512_set1_ps(packed.upper_l[
                        static_cast<std::size_t>(distance - 1) * factor_stride
                        + static_cast<std::size_t>(i)]),
                    _mm512_loadu_ps(work.data()
                        + static_cast<std::size_t>(i + distance) * 16U), value);
            }
            _mm512_storeu_ps(work.data() + static_cast<std::size_t>(i) * 16U,
                             value);
        }
        for (std::int32_t row = 0; row < 16; ++row) {
            for (std::int32_t i = 0; i < plan.destination_size; ++i) {
                output[static_cast<std::ptrdiff_t>(block + row) * output_row_stride + i]
                    = work[static_cast<std::size_t>(i) * 16U + row];
            }
        }
    }
    const auto complete = row_count & ~15;
    for (std::int32_t row = complete; row < row_count; ++row) {
        inverse_axis_f32(
            plan, input + static_cast<std::ptrdiff_t>(row) * input_row_stride, 1,
            output + static_cast<std::ptrdiff_t>(row) * output_row_stride, 1);
    }
}

void inverse_columns_avx512(
    const AxisPlan &plan, const detail::PackedCpuPlan &packed,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t column_count) {
    const auto factor_stride = static_cast<std::size_t>(
        packed.padded_destination_size);
    const auto vector_columns = column_count & ~15;
    for (std::int32_t column = 0; column < vector_columns; column += 16) {
        for (std::int32_t row = 0; row < plan.destination_size; ++row) {
            const auto left = packed.weights_left[static_cast<std::size_t>(row)];
            const auto right = packed.weights_right[static_cast<std::size_t>(row)];
            const auto weight_base = static_cast<std::size_t>(row)
                * static_cast<std::size_t>(packed.weights_columns);
            __m512 value = _mm512_setzero_ps();
            for (std::int32_t source = left; source < right; ++source) {
                value = _mm512_fmadd_ps(
                    _mm512_set1_ps(packed.weights[weight_base
                        + static_cast<std::size_t>(source - left)]),
                    _mm512_loadu_ps(input
                        + static_cast<std::ptrdiff_t>(source) * input_row_stride
                        + column), value);
            }
            const auto available = std::min(plan.half_bandwidth, row);
            for (std::int32_t distance = available; distance >= 1; --distance) {
                value = _mm512_fnmadd_ps(
                    _mm512_set1_ps(packed.lower_ld[
                        static_cast<std::size_t>(distance - 1) * factor_stride
                        + static_cast<std::size_t>(row)]),
                    _mm512_loadu_ps(output
                        + static_cast<std::ptrdiff_t>(row - distance)
                            * output_row_stride + column), value);
            }
            value = _mm512_mul_ps(value, _mm512_set1_ps(
                packed.inverse_diagonal[static_cast<std::size_t>(row)]));
            _mm512_storeu_ps(output
                + static_cast<std::ptrdiff_t>(row) * output_row_stride + column,
                value);
        }
        for (std::int32_t row = plan.destination_size - 2; row >= 0; --row) {
            __m512 value = _mm512_loadu_ps(output
                + static_cast<std::ptrdiff_t>(row) * output_row_stride + column);
            const auto available = std::min(plan.half_bandwidth,
                plan.destination_size - row - 1);
            for (std::int32_t distance = available; distance >= 1; --distance) {
                value = _mm512_fnmadd_ps(
                    _mm512_set1_ps(packed.upper_l[
                        static_cast<std::size_t>(distance - 1) * factor_stride
                        + static_cast<std::size_t>(row)]),
                    _mm512_loadu_ps(output
                        + static_cast<std::ptrdiff_t>(row + distance)
                            * output_row_stride + column), value);
            }
            _mm512_storeu_ps(output
                + static_cast<std::ptrdiff_t>(row) * output_row_stride + column,
                value);
        }
    }
    for (std::int32_t column = vector_columns; column < column_count; ++column) {
        dsmvc::inverse_axis_f32(plan, input + column, input_row_stride,
                                output + column, output_row_stride);
    }
}

// Keeps the AVX-512 translation unit and its ISA requirement explicit.  The
// executor currently delegates kernels without a measured 512-bit variant to
// the AVX2 implementation; this marker is also used by the runtime path test
// to ensure that an AVX-512 object was linked.
void avx512_path_marker() noexcept {
    alignas(64) static volatile float source[16]{};
    volatile __m512 value = _mm512_loadu_ps(const_cast<const float *>(source));
    volatile float lane = _mm512_cvtss_f32(value);
    (void)lane;
}

} // namespace dsmvc
