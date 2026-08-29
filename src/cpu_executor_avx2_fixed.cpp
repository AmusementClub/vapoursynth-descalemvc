#include <dsmvc/engine.hpp>

#include "cpu_packed.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <vector>

namespace dsmvc {
namespace {

#if defined(_MSC_VER)
#define DSMVC_FORCE_INLINE __forceinline
#else
#define DSMVC_FORCE_INLINE inline __attribute__((always_inline))
#endif

struct alignas(32) ScratchVector {
    float lanes[8];
};

DSMVC_FORCE_INLINE void transpose8(
    __m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3,
    __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7) noexcept {
    const __m256 t0 = _mm256_unpacklo_ps(row0, row1);
    const __m256 t1 = _mm256_unpackhi_ps(row0, row1);
    const __m256 t2 = _mm256_unpacklo_ps(row2, row3);
    const __m256 t3 = _mm256_unpackhi_ps(row2, row3);
    const __m256 t4 = _mm256_unpacklo_ps(row4, row5);
    const __m256 t5 = _mm256_unpackhi_ps(row4, row5);
    const __m256 t6 = _mm256_unpacklo_ps(row6, row7);
    const __m256 t7 = _mm256_unpackhi_ps(row6, row7);
    const __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
    const __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    const __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
    const __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    const __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
    const __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    const __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
    const __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    row0 = _mm256_permute2f128_ps(s0, s4, 0x20);
    row1 = _mm256_permute2f128_ps(s1, s5, 0x20);
    row2 = _mm256_permute2f128_ps(s2, s6, 0x20);
    row3 = _mm256_permute2f128_ps(s3, s7, 0x20);
    row4 = _mm256_permute2f128_ps(s0, s4, 0x31);
    row5 = _mm256_permute2f128_ps(s1, s5, 0x31);
    row6 = _mm256_permute2f128_ps(s2, s6, 0x31);
    row7 = _mm256_permute2f128_ps(s3, s7, 0x31);
}

void transpose_source(const float *input, std::ptrdiff_t stride,
                      std::int32_t logical_width,
                      std::int32_t padded_width, float *scratch) noexcept {
    for (std::int32_t column = 0; column < padded_width; column += 8) {
        const auto load_row = [&](std::int32_t row) {
            const auto remaining = std::clamp(logical_width - column, 0, 8);
            const auto *source = input
                + static_cast<std::ptrdiff_t>(row) * stride + column;
            std::array<float, 8> tail{};
            if (remaining != 8) {
                std::memcpy(tail.data(), source,
                    static_cast<std::size_t>(remaining) * sizeof(float));
                source = tail.data();
            }
            return _mm256_loadu_ps(source);
        };
        __m256 x0 = load_row(0);
        __m256 x1 = load_row(1);
        __m256 x2 = load_row(2);
        __m256 x3 = load_row(3);
        __m256 x4 = load_row(4);
        __m256 x5 = load_row(5);
        __m256 x6 = load_row(6);
        __m256 x7 = load_row(7);
        transpose8(x0, x1, x2, x3, x4, x5, x6, x7);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 0) * 8U, x0);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 1) * 8U, x1);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 2) * 8U, x2);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 3) * 8U, x3);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 4) * 8U, x4);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 5) * 8U, x5);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 6) * 8U, x6);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 7) * 8U, x7);
    }
}

[[nodiscard]] DSMVC_FORCE_INLINE __m256 multiply_transpose(
    const detail::PackedCpuPlan &packed, const float *scratch,
    std::int32_t row) noexcept {
    __m256 sum = _mm256_setzero_ps();
    const auto left = packed.weights_left[static_cast<std::size_t>(row)];
    const auto right = packed.weights_right[static_cast<std::size_t>(row)];
    const auto base = static_cast<std::size_t>(row)
        * static_cast<std::size_t>(packed.weights_columns);
    for (std::int32_t source = left; source < right; ++source) {
        sum = _mm256_fmadd_ps(
            _mm256_set1_ps(packed.weights[
                base + static_cast<std::size_t>(source - left)]),
            _mm256_load_ps(
                scratch + static_cast<std::size_t>(source) * 8U), sum);
    }
    return sum;
}

[[nodiscard]] float *transposed_output(
    float *output, std::ptrdiff_t stride, std::int32_t index) noexcept {
    return output + static_cast<std::ptrdiff_t>(index & 7) * stride
        + static_cast<std::ptrdiff_t>(index & ~7);
}

template <std::int32_t Distance>
[[nodiscard]] DSMVC_FORCE_INLINE __m256 forward_fixed_band(
    const detail::PackedCpuPlan &packed, float *output,
    std::ptrdiff_t stride, std::size_t factor_stride,
    std::int32_t i, __m256 value) noexcept {
    value = _mm256_fnmadd_ps(
        _mm256_set1_ps(packed.lower_ld[
            static_cast<std::size_t>(Distance - 1) * factor_stride
            + static_cast<std::size_t>(i)]),
        _mm256_loadu_ps(transposed_output(output, stride, i - Distance)),
        value);
    if constexpr (Distance > 1) {
        return forward_fixed_band<Distance - 1>(
            packed, output, stride, factor_stride, i, value);
    }
    return value;
}

template <std::int32_t Distance>
[[nodiscard]] DSMVC_FORCE_INLINE __m256 backward_fixed_band(
    const detail::PackedCpuPlan &packed, float *output,
    std::ptrdiff_t stride, std::size_t factor_stride,
    std::int32_t i, __m256 value) noexcept {
    value = _mm256_fnmadd_ps(
        _mm256_set1_ps(packed.upper_l[
            static_cast<std::size_t>(Distance - 1) * factor_stride
            + static_cast<std::size_t>(i)]),
        _mm256_loadu_ps(transposed_output(output, stride, i + Distance)),
        value);
    if constexpr (Distance > 1) {
        return backward_fixed_band<Distance - 1>(
            packed, output, stride, factor_stride, i, value);
    }
    return value;
}

void transpose_output(const detail::PackedCpuPlan &packed, float *output,
                      std::ptrdiff_t stride) noexcept {
    for (std::int32_t j = 0; j < packed.padded_destination_size; j += 8) {
        __m256 x0 = _mm256_loadu_ps(output + j);
        __m256 x1 = _mm256_loadu_ps(output + stride + j);
        __m256 x2 = _mm256_loadu_ps(output + 2 * stride + j);
        __m256 x3 = _mm256_loadu_ps(output + 3 * stride + j);
        __m256 x4 = _mm256_loadu_ps(output + 4 * stride + j);
        __m256 x5 = _mm256_loadu_ps(output + 5 * stride + j);
        __m256 x6 = _mm256_loadu_ps(output + 6 * stride + j);
        __m256 x7 = _mm256_loadu_ps(output + 7 * stride + j);
        transpose8(x0, x1, x2, x3, x4, x5, x6, x7);
        _mm256_storeu_ps(output + j, x0);
        _mm256_storeu_ps(output + stride + j, x1);
        _mm256_storeu_ps(output + 2 * stride + j, x2);
        _mm256_storeu_ps(output + 3 * stride + j, x3);
        _mm256_storeu_ps(output + 4 * stride + j, x4);
        _mm256_storeu_ps(output + 5 * stride + j, x5);
        _mm256_storeu_ps(output + 6 * stride + j, x6);
        _mm256_storeu_ps(output + 7 * stride + j, x7);
    }
}

[[nodiscard]] DSMVC_FORCE_INLINE __m256 load_tail_mapped(
    float *output, std::ptrdiff_t stride, const float *tail,
    std::int32_t full_destination, std::int32_t index) noexcept {
    if (index < full_destination) {
        return _mm256_loadu_ps(transposed_output(output, stride, index));
    }
    return _mm256_load_ps(
        tail + static_cast<std::size_t>(index - full_destination) * 8U);
}

DSMVC_FORCE_INLINE void store_tail_mapped(
    float *output, std::ptrdiff_t stride, float *tail,
    std::int32_t full_destination, std::int32_t index,
    __m256 value) noexcept {
    if (index < full_destination) {
        _mm256_storeu_ps(transposed_output(output, stride, index), value);
    } else {
        _mm256_store_ps(
            tail + static_cast<std::size_t>(index - full_destination) * 8U,
            value);
    }
}

template <std::int32_t Bandwidth>
void solve_fixed(const AxisPlan &plan, const detail::PackedCpuPlan &packed,
                 const float *scratch, float *output,
                 std::ptrdiff_t stride) noexcept {
    const auto n = plan.destination_size;
    const auto factor_stride = static_cast<std::size_t>(
        packed.padded_destination_size);
    const auto forward_boundary = std::min(Bandwidth, n);
    for (std::int32_t i = 0; i < forward_boundary; ++i) {
        __m256 value = multiply_transpose(packed, scratch, i);
        for (std::int32_t distance = i; distance >= 1; --distance) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.lower_ld[
                    static_cast<std::size_t>(distance - 1) * factor_stride
                    + static_cast<std::size_t>(i)]),
                _mm256_loadu_ps(
                    transposed_output(output, stride, i - distance)), value);
        }
        value = _mm256_mul_ps(value, _mm256_set1_ps(
            packed.inverse_diagonal[static_cast<std::size_t>(i)]));
        _mm256_storeu_ps(transposed_output(output, stride, i), value);
    }
    for (std::int32_t i = Bandwidth; i < n; ++i) {
        __m256 value = forward_fixed_band<Bandwidth>(
            packed, output, stride, factor_stride, i,
            multiply_transpose(packed, scratch, i));
        value = _mm256_mul_ps(value, _mm256_set1_ps(
            packed.inverse_diagonal[static_cast<std::size_t>(i)]));
        _mm256_storeu_ps(transposed_output(output, stride, i), value);
    }
    for (std::int32_t i = n; i < packed.padded_destination_size; ++i) {
        _mm256_storeu_ps(
            transposed_output(output, stride, i), _mm256_setzero_ps());
    }
    const auto backward_boundary = std::max(n - Bandwidth, 0);
    for (std::int32_t i = n - 2; i >= backward_boundary; --i) {
        __m256 value = _mm256_loadu_ps(transposed_output(output, stride, i));
        const auto available = n - i - 1;
        for (std::int32_t distance = available; distance >= 1; --distance) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.upper_l[
                    static_cast<std::size_t>(distance - 1) * factor_stride
                    + static_cast<std::size_t>(i)]),
                _mm256_loadu_ps(
                    transposed_output(output, stride, i + distance)), value);
        }
        _mm256_storeu_ps(transposed_output(output, stride, i), value);
    }
    for (std::int32_t i = n - Bandwidth - 1; i >= 0; --i) {
        const __m256 value = backward_fixed_band<Bandwidth>(
            packed, output, stride, factor_stride, i,
            _mm256_loadu_ps(transposed_output(output, stride, i)));
        _mm256_storeu_ps(transposed_output(output, stride, i), value);
    }
    transpose_output(packed, output, stride);
}

template <std::int32_t Bandwidth>
void solve_fixed_tail(const AxisPlan &plan,
                      const detail::PackedCpuPlan &packed,
                      const float *scratch, float *output,
                      std::ptrdiff_t stride) noexcept {
    const auto n = plan.destination_size;
    const auto full_destination = n & ~7;
    const auto factor_stride = static_cast<std::size_t>(
        packed.padded_destination_size);
    alignas(32) float tail_tile[64]{};

    const auto forward_boundary = std::min(Bandwidth, n);
    for (std::int32_t i = 0; i < forward_boundary; ++i) {
        __m256 value = multiply_transpose(packed, scratch, i);
        for (std::int32_t distance = i; distance >= 1; --distance) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.lower_ld[
                    static_cast<std::size_t>(distance - 1) * factor_stride
                    + static_cast<std::size_t>(i)]),
                load_tail_mapped(
                    output, stride, tail_tile, full_destination, i - distance),
                value);
        }
        value = _mm256_mul_ps(value, _mm256_set1_ps(
            packed.inverse_diagonal[static_cast<std::size_t>(i)]));
        store_tail_mapped(
            output, stride, tail_tile, full_destination, i, value);
    }
    for (std::int32_t i = Bandwidth; i < full_destination; ++i) {
        __m256 value = forward_fixed_band<Bandwidth>(
            packed, output, stride, factor_stride, i,
            multiply_transpose(packed, scratch, i));
        value = _mm256_mul_ps(value, _mm256_set1_ps(
            packed.inverse_diagonal[static_cast<std::size_t>(i)]));
        _mm256_storeu_ps(transposed_output(output, stride, i), value);
    }
    for (std::int32_t i = std::max(Bandwidth, full_destination); i < n; ++i) {
        __m256 value = multiply_transpose(packed, scratch, i);
        for (std::int32_t distance = Bandwidth; distance >= 1; --distance) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.lower_ld[
                    static_cast<std::size_t>(distance - 1) * factor_stride
                    + static_cast<std::size_t>(i)]),
                load_tail_mapped(
                    output, stride, tail_tile, full_destination, i - distance),
                value);
        }
        value = _mm256_mul_ps(value, _mm256_set1_ps(
            packed.inverse_diagonal[static_cast<std::size_t>(i)]));
        store_tail_mapped(
            output, stride, tail_tile, full_destination, i, value);
    }

    const auto tail_backward_boundary = std::max(full_destination - Bandwidth, 0);
    for (std::int32_t i = n - 2; i >= tail_backward_boundary; --i) {
        __m256 value = load_tail_mapped(
            output, stride, tail_tile, full_destination, i);
        const auto available = std::min(Bandwidth, n - i - 1);
        for (std::int32_t distance = available; distance >= 1; --distance) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.upper_l[
                    static_cast<std::size_t>(distance - 1) * factor_stride
                    + static_cast<std::size_t>(i)]),
                load_tail_mapped(
                    output, stride, tail_tile, full_destination, i + distance),
                value);
        }
        store_tail_mapped(
            output, stride, tail_tile, full_destination, i, value);
    }
    for (std::int32_t i = full_destination - Bandwidth - 1; i >= 0; --i) {
        const __m256 value = backward_fixed_band<Bandwidth>(
            packed, output, stride, factor_stride, i,
            _mm256_loadu_ps(transposed_output(output, stride, i)));
        _mm256_storeu_ps(transposed_output(output, stride, i), value);
    }

    for (std::int32_t j = 0; j < full_destination; j += 8) {
        __m256 x0 = _mm256_loadu_ps(output + j);
        __m256 x1 = _mm256_loadu_ps(output + stride + j);
        __m256 x2 = _mm256_loadu_ps(output + 2 * stride + j);
        __m256 x3 = _mm256_loadu_ps(output + 3 * stride + j);
        __m256 x4 = _mm256_loadu_ps(output + 4 * stride + j);
        __m256 x5 = _mm256_loadu_ps(output + 5 * stride + j);
        __m256 x6 = _mm256_loadu_ps(output + 6 * stride + j);
        __m256 x7 = _mm256_loadu_ps(output + 7 * stride + j);
        transpose8(x0, x1, x2, x3, x4, x5, x6, x7);
        _mm256_storeu_ps(output + j, x0);
        _mm256_storeu_ps(output + stride + j, x1);
        _mm256_storeu_ps(output + 2 * stride + j, x2);
        _mm256_storeu_ps(output + 3 * stride + j, x3);
        _mm256_storeu_ps(output + 4 * stride + j, x4);
        _mm256_storeu_ps(output + 5 * stride + j, x5);
        _mm256_storeu_ps(output + 6 * stride + j, x6);
        _mm256_storeu_ps(output + 7 * stride + j, x7);
    }

    __m256 x0 = _mm256_load_ps(tail_tile + 0U * 8U);
    __m256 x1 = _mm256_load_ps(tail_tile + 1U * 8U);
    __m256 x2 = _mm256_load_ps(tail_tile + 2U * 8U);
    __m256 x3 = _mm256_load_ps(tail_tile + 3U * 8U);
    __m256 x4 = _mm256_load_ps(tail_tile + 4U * 8U);
    __m256 x5 = _mm256_load_ps(tail_tile + 5U * 8U);
    __m256 x6 = _mm256_load_ps(tail_tile + 6U * 8U);
    __m256 x7 = _mm256_load_ps(tail_tile + 7U * 8U);
    transpose8(x0, x1, x2, x3, x4, x5, x6, x7);
    const auto remaining = n - full_destination;
    const __m256i mask = _mm256_setr_epi32(
        remaining > 0 ? -1 : 0, remaining > 1 ? -1 : 0,
        remaining > 2 ? -1 : 0, remaining > 3 ? -1 : 0,
        remaining > 4 ? -1 : 0, remaining > 5 ? -1 : 0,
        remaining > 6 ? -1 : 0, remaining > 7 ? -1 : 0);
    _mm256_maskstore_ps(output + full_destination, mask, x0);
    _mm256_maskstore_ps(output + stride + full_destination, mask, x1);
    _mm256_maskstore_ps(output + 2 * stride + full_destination, mask, x2);
    _mm256_maskstore_ps(output + 3 * stride + full_destination, mask, x3);
    _mm256_maskstore_ps(output + 4 * stride + full_destination, mask, x4);
    _mm256_maskstore_ps(output + 5 * stride + full_destination, mask, x5);
    _mm256_maskstore_ps(output + 6 * stride + full_destination, mask, x6);
    _mm256_maskstore_ps(output + 7 * stride + full_destination, mask, x7);
}

} // namespace

void inverse_rows_fixed_avx2(
    const AxisPlan &plan, const detail::PackedCpuPlan &packed,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t row_count) {
    if (row_count < 8) {
        for (std::int32_t row = 0; row < row_count; ++row) {
            inverse_axis_f32(
                plan, input + static_cast<std::ptrdiff_t>(row) * input_row_stride,
                1, output + static_cast<std::ptrdiff_t>(row) * output_row_stride,
                1);
        }
        return;
    }

    thread_local std::vector<ScratchVector> scratch;
    scratch.resize(static_cast<std::size_t>(packed.padded_source_size));
    const auto solve_block = [&](std::int32_t row) {
        auto *block_output =
            output + static_cast<std::ptrdiff_t>(row) * output_row_stride;
        transpose_source(
            input + static_cast<std::ptrdiff_t>(row) * input_row_stride,
            input_row_stride, plan.source_size,
            packed.padded_source_size, scratch.front().lanes);
        if (plan.half_bandwidth == 5) {
            if (plan.destination_size == packed.padded_destination_size) {
                solve_fixed<5>(
                    plan, packed, scratch.front().lanes,
                    block_output, output_row_stride);
            } else {
                solve_fixed_tail<5>(
                    plan, packed, scratch.front().lanes,
                    block_output, output_row_stride);
            }
        } else {
            if (plan.destination_size == packed.padded_destination_size) {
                solve_fixed<7>(
                    plan, packed, scratch.front().lanes,
                    block_output, output_row_stride);
            } else {
                solve_fixed_tail<7>(
                    plan, packed, scratch.front().lanes,
                    block_output, output_row_stride);
            }
        }
    };
    const auto complete_rows = row_count & ~7;
    for (std::int32_t row = 0; row < complete_rows; row += 8) {
        solve_block(row);
    }
    if (complete_rows != row_count) solve_block(row_count - 8);
}

} // namespace dsmvc

#undef DSMVC_FORCE_INLINE
