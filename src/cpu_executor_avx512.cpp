#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "axis_plan_internal.hpp"
#include "cpu_packed.hpp"

namespace dsmvc {
namespace {

#if defined(_MSC_VER)
#define DSMVC_FORCE_INLINE __forceinline
#else
#define DSMVC_FORCE_INLINE inline __attribute__((always_inline))
#endif

void transpose8(__m256 (&rows)[8]) noexcept {
    const __m256 t0 = _mm256_unpacklo_ps(rows[0], rows[1]);
    const __m256 t1 = _mm256_unpackhi_ps(rows[0], rows[1]);
    const __m256 t2 = _mm256_unpacklo_ps(rows[2], rows[3]);
    const __m256 t3 = _mm256_unpackhi_ps(rows[2], rows[3]);
    const __m256 t4 = _mm256_unpacklo_ps(rows[4], rows[5]);
    const __m256 t5 = _mm256_unpackhi_ps(rows[4], rows[5]);
    const __m256 t6 = _mm256_unpacklo_ps(rows[6], rows[7]);
    const __m256 t7 = _mm256_unpackhi_ps(rows[6], rows[7]);
    const __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
    const __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    const __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
    const __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    const __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
    const __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    const __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
    const __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    rows[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
    rows[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
    rows[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
    rows[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
    rows[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
    rows[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
    rows[6] = _mm256_permute2f128_ps(s2, s6, 0x31);
    rows[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
}

void pack_rows_16(const AxisPlan &plan, const float *input,
                  std::ptrdiff_t input_row_stride, float *scratch) noexcept {
    const auto vector_source = plan.source_size & ~7;
    for (std::int32_t source = 0; source < vector_source; source += 8) {
        __m256 lanes[8];
        for (std::int32_t row = 0; row < 8; ++row) {
            lanes[row] = _mm256_loadu_ps(
                input + static_cast<std::ptrdiff_t>(row) * input_row_stride
                + source);
        }
        transpose8(lanes);
        for (std::int32_t column = 0; column < 8; ++column) {
            _mm256_storeu_ps(
                scratch + static_cast<std::size_t>(source + column) * 16U,
                lanes[column]);
        }
        for (std::int32_t row = 0; row < 8; ++row) {
            lanes[row] = _mm256_loadu_ps(
                input + static_cast<std::ptrdiff_t>(row + 8) * input_row_stride
                + source);
        }
        transpose8(lanes);
        for (std::int32_t column = 0; column < 8; ++column) {
            _mm256_storeu_ps(
                scratch + static_cast<std::size_t>(source + column) * 16U + 8U,
                lanes[column]);
        }
    }
    for (std::int32_t source = vector_source;
         source < plan.source_size; ++source) {
        auto *lanes = scratch + static_cast<std::size_t>(source) * 16U;
        for (std::int32_t row = 0; row < 16; ++row) {
            lanes[row] = input[
                static_cast<std::ptrdiff_t>(row) * input_row_stride + source];
        }
    }
}

void unpack_rows_16(const AxisPlan &plan, const float *work,
                    float *output, std::ptrdiff_t output_row_stride) noexcept {
    const auto vector_destination = plan.destination_size & ~7;
    for (std::int32_t destination = 0;
         destination < vector_destination; destination += 8) {
        __m256 lanes[8];
        for (std::int32_t column = 0; column < 8; ++column) {
            lanes[column] = _mm256_loadu_ps(
                work + static_cast<std::size_t>(destination + column) * 16U);
        }
        transpose8(lanes);
        for (std::int32_t row = 0; row < 8; ++row) {
            _mm256_storeu_ps(
                output + static_cast<std::ptrdiff_t>(row) * output_row_stride
                + destination, lanes[row]);
        }
        for (std::int32_t column = 0; column < 8; ++column) {
            lanes[column] = _mm256_loadu_ps(
                work + static_cast<std::size_t>(destination + column) * 16U + 8U);
        }
        transpose8(lanes);
        for (std::int32_t row = 0; row < 8; ++row) {
            _mm256_storeu_ps(
                output + static_cast<std::ptrdiff_t>(row + 8) * output_row_stride
                + destination, lanes[row]);
        }
    }
    for (std::int32_t destination = vector_destination;
         destination < plan.destination_size; ++destination) {
        for (std::int32_t row = 0; row < 16; ++row) {
            output[static_cast<std::ptrdiff_t>(row) * output_row_stride
                   + destination]
                = work[static_cast<std::size_t>(destination) * 16U + row];
        }
    }
}

[[nodiscard]] DSMVC_FORCE_INLINE __m512 multiply_transposed_row(
    const detail::PackedCpuPlan &packed, const float *scratch,
    std::int32_t destination) noexcept {
    __m512 value = _mm512_setzero_ps();
    const auto left = packed.weights_left[
        static_cast<std::size_t>(destination)];
    const auto right = packed.weights_right[
        static_cast<std::size_t>(destination)];
    const auto base = static_cast<std::size_t>(destination)
        * static_cast<std::size_t>(packed.weights_columns);
    for (std::int32_t source = left; source < right; ++source) {
        value = _mm512_fmadd_ps(
            _mm512_set1_ps(packed.weights[
                base + static_cast<std::size_t>(source - left)]),
            _mm512_loadu_ps(
                scratch + static_cast<std::size_t>(source) * 16U), value);
    }
    return value;
}

template <std::int32_t Distance>
[[nodiscard]] DSMVC_FORCE_INLINE __m512 forward_fixed_b5(
    const detail::PackedCpuPlan &packed, const float *work,
    std::size_t factor_stride, std::int32_t destination,
    __m512 value) noexcept {
    value = _mm512_fnmadd_ps(
        _mm512_set1_ps(packed.lower_ld[
            static_cast<std::size_t>(Distance - 1) * factor_stride
            + static_cast<std::size_t>(destination)]),
        _mm512_loadu_ps(work
            + static_cast<std::size_t>(destination - Distance) * 16U), value);
    if constexpr (Distance > 1) {
        return forward_fixed_b5<Distance - 1>(
            packed, work, factor_stride, destination, value);
    }
    return value;
}

template <std::int32_t Distance>
[[nodiscard]] DSMVC_FORCE_INLINE __m512 backward_fixed_b5(
    const detail::PackedCpuPlan &packed, const float *work,
    std::size_t factor_stride, std::int32_t destination,
    __m512 value) noexcept {
    value = _mm512_fnmadd_ps(
        _mm512_set1_ps(packed.upper_l[
            static_cast<std::size_t>(Distance - 1) * factor_stride
            + static_cast<std::size_t>(destination)]),
        _mm512_loadu_ps(work
            + static_cast<std::size_t>(destination + Distance) * 16U), value);
    if constexpr (Distance > 1) {
        return backward_fixed_b5<Distance - 1>(
            packed, work, factor_stride, destination, value);
    }
    return value;
}

void solve_horizontal_b5(const AxisPlan &plan,
                         const detail::PackedCpuPlan &packed,
                         const float *scratch, float *work) noexcept {
    constexpr std::int32_t bandwidth = 5;
    const auto n = plan.destination_size;
    const auto factor_stride = static_cast<std::size_t>(
        packed.padded_destination_size);
    const auto forward_boundary = std::min(bandwidth, n);
    for (std::int32_t i = 0; i < forward_boundary; ++i) {
        __m512 value = multiply_transposed_row(packed, scratch, i);
        for (std::int32_t distance = i; distance >= 1; --distance) {
            value = _mm512_fnmadd_ps(
                _mm512_set1_ps(packed.lower_ld[
                    static_cast<std::size_t>(distance - 1) * factor_stride
                    + static_cast<std::size_t>(i)]),
                _mm512_loadu_ps(work
                    + static_cast<std::size_t>(i - distance) * 16U), value);
        }
        value = _mm512_mul_ps(value, _mm512_set1_ps(
            packed.inverse_diagonal[static_cast<std::size_t>(i)]));
        _mm512_storeu_ps(work + static_cast<std::size_t>(i) * 16U, value);
    }
    for (std::int32_t i = bandwidth; i < n; ++i) {
        __m512 value = forward_fixed_b5<bandwidth>(
            packed, work, factor_stride, i,
            multiply_transposed_row(packed, scratch, i));
        value = _mm512_mul_ps(value, _mm512_set1_ps(
            packed.inverse_diagonal[static_cast<std::size_t>(i)]));
        _mm512_storeu_ps(work + static_cast<std::size_t>(i) * 16U, value);
    }
    for (std::int32_t i = n; i < packed.padded_destination_size; ++i) {
        _mm512_storeu_ps(
            work + static_cast<std::size_t>(i) * 16U, _mm512_setzero_ps());
    }

    const auto backward_boundary = std::max(n - bandwidth, 0);
    for (std::int32_t i = n - 2; i >= backward_boundary; --i) {
        __m512 value = _mm512_loadu_ps(
            work + static_cast<std::size_t>(i) * 16U);
        const auto available = n - i - 1;
        for (std::int32_t distance = available; distance >= 1; --distance) {
            value = _mm512_fnmadd_ps(
                _mm512_set1_ps(packed.upper_l[
                    static_cast<std::size_t>(distance - 1) * factor_stride
                    + static_cast<std::size_t>(i)]),
                _mm512_loadu_ps(work
                    + static_cast<std::size_t>(i + distance) * 16U), value);
        }
        _mm512_storeu_ps(work + static_cast<std::size_t>(i) * 16U, value);
    }
    for (std::int32_t i = n - bandwidth - 1; i >= 0; --i) {
        const __m512 value = backward_fixed_b5<bandwidth>(
            packed, work, factor_stride, i,
            _mm512_loadu_ps(work + static_cast<std::size_t>(i) * 16U));
        _mm512_storeu_ps(work + static_cast<std::size_t>(i) * 16U, value);
    }
}

} // namespace

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

    const auto solve_block = [&](std::int32_t block) {
        pack_rows_16(
            plan, input + static_cast<std::ptrdiff_t>(block) * input_row_stride,
            input_row_stride, scratch.data());
        for (std::int32_t source = plan.source_size;
             source < source_size; ++source) {
            _mm512_storeu_ps(
                scratch.data() + static_cast<std::size_t>(source) * 16U,
                _mm512_setzero_ps());
        }

        if (plan.half_bandwidth == 5) {
            solve_horizontal_b5(plan, packed, scratch.data(), work.data());
        } else {
            for (std::int32_t i = 0; i < destination_size; ++i) {
                __m512 value = _mm512_setzero_ps();
                if (i < plan.destination_size) {
                    const auto left = packed.weights_left[
                        static_cast<std::size_t>(i)];
                    const auto right = packed.weights_right[
                        static_cast<std::size_t>(i)];
                    const auto base = static_cast<std::size_t>(i)
                        * static_cast<std::size_t>(packed.weights_columns);
                    for (std::int32_t source = left;
                         source < right; ++source) {
                        value = _mm512_fmadd_ps(
                            _mm512_set1_ps(packed.weights[base
                                + static_cast<std::size_t>(source - left)]),
                            _mm512_loadu_ps(scratch.data()
                                + static_cast<std::size_t>(source) * 16U),
                            value);
                    }
                    const auto available = std::min(plan.half_bandwidth, i);
                    for (std::int32_t distance = available;
                         distance >= 1; --distance) {
                        value = _mm512_fnmadd_ps(
                            _mm512_set1_ps(packed.lower_ld[
                                static_cast<std::size_t>(distance - 1)
                                    * factor_stride
                                + static_cast<std::size_t>(i)]),
                            _mm512_loadu_ps(work.data()
                                + static_cast<std::size_t>(i - distance) * 16U),
                            value);
                    }
                    value = _mm512_mul_ps(value, _mm512_set1_ps(
                        packed.inverse_diagonal[static_cast<std::size_t>(i)]));
                }
                _mm512_storeu_ps(
                    work.data() + static_cast<std::size_t>(i) * 16U, value);
            }
            for (std::int32_t i = plan.destination_size - 2; i >= 0; --i) {
                __m512 value = _mm512_loadu_ps(
                    work.data() + static_cast<std::size_t>(i) * 16U);
                const auto available = std::min(plan.half_bandwidth,
                    plan.destination_size - i - 1);
                for (std::int32_t distance = available;
                     distance >= 1; --distance) {
                    value = _mm512_fnmadd_ps(
                        _mm512_set1_ps(packed.upper_l[
                            static_cast<std::size_t>(distance - 1) * factor_stride
                            + static_cast<std::size_t>(i)]),
                        _mm512_loadu_ps(work.data()
                            + static_cast<std::size_t>(i + distance) * 16U),
                        value);
                }
                _mm512_storeu_ps(
                    work.data() + static_cast<std::size_t>(i) * 16U, value);
            }
        }
        unpack_rows_16(
            plan, work.data(),
            output + static_cast<std::ptrdiff_t>(block) * output_row_stride,
            output_row_stride);
    };
    for (std::int32_t block = 0; block + 16 <= row_count; block += 16) {
        solve_block(block);
    }
    const auto complete = row_count & ~15;
    if (complete != row_count) solve_block(row_count - 16);
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
    const __m512 value = _mm512_loadu_ps(const_cast<const float *>(source));
    volatile float lane = _mm512_cvtss_f32(value);
    (void)lane;
}

} // namespace dsmvc

#undef DSMVC_FORCE_INLINE
