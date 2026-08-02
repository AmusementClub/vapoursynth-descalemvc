#include <dsmvc/engine.hpp>

#include "cpu_packed.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

DSMVC_FORCE_INLINE void transpose8(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3,
                __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7) noexcept {
    const __m256 t0 = _mm256_unpacklo_ps(row0, row1);
    const __m256 t1 = _mm256_unpackhi_ps(row0, row1);
    const __m256 t2 = _mm256_unpacklo_ps(row2, row3);
    const __m256 t3 = _mm256_unpackhi_ps(row2, row3);
    const __m256 t4 = _mm256_unpacklo_ps(row4, row5);
    const __m256 t5 = _mm256_unpackhi_ps(row4, row5);
    const __m256 t6 = _mm256_unpacklo_ps(row6, row7);
    const __m256 t7 = _mm256_unpackhi_ps(row6, row7);
    const __m256 u0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    row0 = _mm256_permute2f128_ps(u0, u4, 0x20);
    row1 = _mm256_permute2f128_ps(u1, u5, 0x20);
    row2 = _mm256_permute2f128_ps(u2, u6, 0x20);
    row3 = _mm256_permute2f128_ps(u3, u7, 0x20);
    row4 = _mm256_permute2f128_ps(u0, u4, 0x31);
    row5 = _mm256_permute2f128_ps(u1, u5, 0x31);
    row6 = _mm256_permute2f128_ps(u2, u6, 0x31);
    row7 = _mm256_permute2f128_ps(u3, u7, 0x31);
}

void transpose_source(const float *input, std::ptrdiff_t stride,
                      std::int32_t padded_width, float *scratch) noexcept {
    for (std::int32_t column = 0; column < padded_width; column += 8) {
        __m256 x0 = _mm256_loadu_ps(input + column);
        __m256 x1 = _mm256_loadu_ps(input + stride + column);
        __m256 x2 = _mm256_loadu_ps(input + 2 * stride + column);
        __m256 x3 = _mm256_loadu_ps(input + 3 * stride + column);
        __m256 x4 = _mm256_loadu_ps(input + 4 * stride + column);
        __m256 x5 = _mm256_loadu_ps(input + 5 * stride + column);
        __m256 x6 = _mm256_loadu_ps(input + 6 * stride + column);
        __m256 x7 = _mm256_loadu_ps(input + 7 * stride + column);
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

[[nodiscard]] DSMVC_FORCE_INLINE __m256 multiply_transpose(const detail::PackedCpuPlan &packed,
                                        const float *scratch,
                                        std::int32_t row) noexcept {
    __m256 sum = _mm256_setzero_ps();
    const auto left = packed.weights_left[static_cast<std::size_t>(row)];
    const auto right = packed.weights_right[static_cast<std::size_t>(row)];
    const auto base = static_cast<std::size_t>(row)
        * static_cast<std::size_t>(packed.weights_columns);
    for (std::int32_t source = left; source < right; ++source) {
        const __m256 weight = _mm256_set1_ps(
            packed.weights[base + static_cast<std::size_t>(source - left)]);
        sum = _mm256_fmadd_ps(
            weight, _mm256_load_ps(
                        scratch + static_cast<std::size_t>(source) * 8U), sum);
    }
    return sum;
}

[[nodiscard]] DSMVC_FORCE_INLINE __m256 forward_b1(const detail::PackedCpuPlan &packed,
                                std::int32_t i, __m256 value,
                                __m256 previous) noexcept {
    value = _mm256_fnmadd_ps(
        _mm256_set1_ps(packed.lower_ld[static_cast<std::size_t>(i)]),
        previous, value);
    return _mm256_mul_ps(
        value, _mm256_set1_ps(
                   packed.inverse_diagonal[static_cast<std::size_t>(i)]));
}

[[nodiscard]] DSMVC_FORCE_INLINE __m256 backward_b1(const detail::PackedCpuPlan &packed,
                                 std::int32_t i, __m256 value,
                                 __m256 next) noexcept {
    return _mm256_fnmadd_ps(
        _mm256_set1_ps(packed.upper_l[static_cast<std::size_t>(i)]),
        next, value);
}

[[nodiscard]] DSMVC_FORCE_INLINE __m256 forward_b3(const detail::PackedCpuPlan &packed,
                                std::int32_t i, __m256 value,
                                __m256 previous1, __m256 previous2,
                                __m256 previous3) noexcept {
    const auto stride = static_cast<std::size_t>(packed.padded_destination_size);
    const auto index = static_cast<std::size_t>(i);
    value = _mm256_fnmadd_ps(_mm256_set1_ps(packed.lower_ld[2U * stride + index]),
                             previous3, value);
    value = _mm256_fnmadd_ps(_mm256_set1_ps(packed.lower_ld[stride + index]),
                             previous2, value);
    value = _mm256_fnmadd_ps(_mm256_set1_ps(packed.lower_ld[index]),
                             previous1, value);
    return _mm256_mul_ps(value, _mm256_set1_ps(packed.inverse_diagonal[index]));
}

[[nodiscard]] DSMVC_FORCE_INLINE __m256 backward_b3(const detail::PackedCpuPlan &packed,
                                 std::int32_t i, __m256 value,
                                 __m256 next1, __m256 next2,
                                 __m256 next3) noexcept {
    const auto stride = static_cast<std::size_t>(packed.padded_destination_size);
    const auto index = static_cast<std::size_t>(i);
    value = _mm256_fnmadd_ps(_mm256_set1_ps(packed.upper_l[index]), next1, value);
    value = _mm256_fnmadd_ps(_mm256_set1_ps(packed.upper_l[stride + index]),
                             next2, value);
    return _mm256_fnmadd_ps(
        _mm256_set1_ps(packed.upper_l[2U * stride + index]), next3, value);
}

void solve_horizontal_b1(const detail::PackedCpuPlan &packed, const float *scratch,
                         float *output, std::ptrdiff_t stride) noexcept {
    __m256 previous = _mm256_setzero_ps();
    const auto padded = packed.padded_destination_size;
    for (std::int32_t j = 0; j < padded; j += 8) {
        __m256 x0 = forward_b1(packed, j + 0, multiply_transpose(packed, scratch, j + 0), previous);
        __m256 x1 = forward_b1(packed, j + 1, multiply_transpose(packed, scratch, j + 1), x0);
        __m256 x2 = forward_b1(packed, j + 2, multiply_transpose(packed, scratch, j + 2), x1);
        __m256 x3 = forward_b1(packed, j + 3, multiply_transpose(packed, scratch, j + 3), x2);
        __m256 x4 = forward_b1(packed, j + 4, multiply_transpose(packed, scratch, j + 4), x3);
        __m256 x5 = forward_b1(packed, j + 5, multiply_transpose(packed, scratch, j + 5), x4);
        __m256 x6 = forward_b1(packed, j + 6, multiply_transpose(packed, scratch, j + 6), x5);
        __m256 x7 = forward_b1(packed, j + 7, multiply_transpose(packed, scratch, j + 7), x6);
        previous = x7;
        _mm256_storeu_ps(output + j, x0);
        _mm256_storeu_ps(output + stride + j, x1);
        _mm256_storeu_ps(output + 2 * stride + j, x2);
        _mm256_storeu_ps(output + 3 * stride + j, x3);
        _mm256_storeu_ps(output + 4 * stride + j, x4);
        _mm256_storeu_ps(output + 5 * stride + j, x5);
        _mm256_storeu_ps(output + 6 * stride + j, x6);
        _mm256_storeu_ps(output + 7 * stride + j, x7);
    }

    __m256 next = _mm256_setzero_ps();
    for (std::int32_t j = padded - 8; j >= 0; j -= 8) {
        __m256 x0 = _mm256_loadu_ps(output + j);
        __m256 x1 = _mm256_loadu_ps(output + stride + j);
        __m256 x2 = _mm256_loadu_ps(output + 2 * stride + j);
        __m256 x3 = _mm256_loadu_ps(output + 3 * stride + j);
        __m256 x4 = _mm256_loadu_ps(output + 4 * stride + j);
        __m256 x5 = _mm256_loadu_ps(output + 5 * stride + j);
        __m256 x6 = _mm256_loadu_ps(output + 6 * stride + j);
        __m256 x7 = _mm256_loadu_ps(output + 7 * stride + j);
        x7 = backward_b1(packed, j + 7, x7, next);
        x6 = backward_b1(packed, j + 6, x6, x7);
        x5 = backward_b1(packed, j + 5, x5, x6);
        x4 = backward_b1(packed, j + 4, x4, x5);
        x3 = backward_b1(packed, j + 3, x3, x4);
        x2 = backward_b1(packed, j + 2, x2, x3);
        x1 = backward_b1(packed, j + 1, x1, x2);
        x0 = backward_b1(packed, j + 0, x0, x1);
        next = x0;
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

void solve_horizontal_b3(const detail::PackedCpuPlan &packed, const float *scratch,
                         float *output, std::ptrdiff_t stride) noexcept {
    __m256 previous1 = _mm256_setzero_ps();
    __m256 previous2 = _mm256_setzero_ps();
    __m256 previous3 = _mm256_setzero_ps();
    const auto padded = packed.padded_destination_size;
    for (std::int32_t j = 0; j < padded; j += 8) {
#define DSMVC_FORWARD3(LANE, PREVIOUS1, PREVIOUS2, PREVIOUS3) \
        __m256 x##LANE = forward_b3( \
            packed, j + LANE, multiply_transpose(packed, scratch, j + LANE), \
            PREVIOUS1, PREVIOUS2, PREVIOUS3)
        DSMVC_FORWARD3(0, previous1, previous2, previous3);
        DSMVC_FORWARD3(1, x0, previous1, previous2);
        DSMVC_FORWARD3(2, x1, x0, previous1);
        DSMVC_FORWARD3(3, x2, x1, x0);
        DSMVC_FORWARD3(4, x3, x2, x1);
        DSMVC_FORWARD3(5, x4, x3, x2);
        DSMVC_FORWARD3(6, x5, x4, x3);
        DSMVC_FORWARD3(7, x6, x5, x4);
#undef DSMVC_FORWARD3
        previous1 = x7;
        previous2 = x6;
        previous3 = x5;
        _mm256_storeu_ps(output + j, x0);
        _mm256_storeu_ps(output + stride + j, x1);
        _mm256_storeu_ps(output + 2 * stride + j, x2);
        _mm256_storeu_ps(output + 3 * stride + j, x3);
        _mm256_storeu_ps(output + 4 * stride + j, x4);
        _mm256_storeu_ps(output + 5 * stride + j, x5);
        _mm256_storeu_ps(output + 6 * stride + j, x6);
        _mm256_storeu_ps(output + 7 * stride + j, x7);
    }

    __m256 next1 = _mm256_setzero_ps();
    __m256 next2 = _mm256_setzero_ps();
    __m256 next3 = _mm256_setzero_ps();
    for (std::int32_t j = padded - 8; j >= 0; j -= 8) {
        __m256 x0 = _mm256_loadu_ps(output + j);
        __m256 x1 = _mm256_loadu_ps(output + stride + j);
        __m256 x2 = _mm256_loadu_ps(output + 2 * stride + j);
        __m256 x3 = _mm256_loadu_ps(output + 3 * stride + j);
        __m256 x4 = _mm256_loadu_ps(output + 4 * stride + j);
        __m256 x5 = _mm256_loadu_ps(output + 5 * stride + j);
        __m256 x6 = _mm256_loadu_ps(output + 6 * stride + j);
        __m256 x7 = _mm256_loadu_ps(output + 7 * stride + j);
        x7 = backward_b3(packed, j + 7, x7, next1, next2, next3);
        x6 = backward_b3(packed, j + 6, x6, x7, next1, next2);
        x5 = backward_b3(packed, j + 5, x5, x6, x7, next1);
        x4 = backward_b3(packed, j + 4, x4, x5, x6, x7);
        x3 = backward_b3(packed, j + 3, x3, x4, x5, x6);
        x2 = backward_b3(packed, j + 2, x2, x3, x4, x5);
        x1 = backward_b3(packed, j + 1, x1, x2, x3, x4);
        x0 = backward_b3(packed, j + 0, x0, x1, x2, x3);
        next1 = x0;
        next2 = x1;
        next3 = x2;
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

[[nodiscard]] float *transposed_output(float *output, std::ptrdiff_t stride,
                                       std::int32_t index) noexcept {
    return output + static_cast<std::ptrdiff_t>(index & 7) * stride
        + static_cast<std::ptrdiff_t>(index & ~7);
}

void solve_horizontal_generic(const getnative::AxisPlan &plan,
                              const detail::PackedCpuPlan &packed,
                              const float *scratch, float *output,
                              std::ptrdiff_t stride) noexcept {
    const auto n = plan.destination_size;
    const auto factor_stride = static_cast<std::size_t>(packed.padded_destination_size);
    for (std::int32_t i = 0; i < n; ++i) {
        __m256 value = multiply_transpose(packed, scratch, i);
        const auto available = std::min(plan.half_bandwidth, i);
        for (std::int32_t distance = available; distance >= 1; --distance) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.lower_ld[
                    static_cast<std::size_t>(distance - 1) * factor_stride
                    + static_cast<std::size_t>(i)]),
                _mm256_loadu_ps(transposed_output(output, stride, i - distance)),
                value);
        }
        value = _mm256_mul_ps(
            value, _mm256_set1_ps(
                       packed.inverse_diagonal[static_cast<std::size_t>(i)]));
        _mm256_storeu_ps(transposed_output(output, stride, i), value);
    }
    for (std::int32_t i = n; i < packed.padded_destination_size; ++i) {
        _mm256_storeu_ps(transposed_output(output, stride, i), _mm256_setzero_ps());
    }
    for (std::int32_t i = n - 2; i >= 0; --i) {
        __m256 value = _mm256_loadu_ps(transposed_output(output, stride, i));
        const auto available = std::min(plan.half_bandwidth, n - i - 1);
        for (std::int32_t distance = available; distance >= 1; --distance) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.upper_l[
                    static_cast<std::size_t>(distance - 1) * factor_stride
                    + static_cast<std::size_t>(i)]),
                _mm256_loadu_ps(transposed_output(output, stride, i + distance)),
                value);
        }
        _mm256_storeu_ps(transposed_output(output, stride, i), value);
    }
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

void solve_horizontal_block(const getnative::AxisPlan &plan,
                            const detail::PackedCpuPlan &packed,
                            const float *input, std::ptrdiff_t input_stride,
                            float *output, std::ptrdiff_t output_stride,
                            float *scratch) noexcept {
    transpose_source(input, input_stride, packed.padded_source_size, scratch);
    if (plan.half_bandwidth == 1) {
        solve_horizontal_b1(packed, scratch, output, output_stride);
    } else if (plan.half_bandwidth == 3) {
        solve_horizontal_b3(packed, scratch, output, output_stride);
    } else {
        solve_horizontal_generic(plan, packed, scratch, output, output_stride);
    }
}

DSMVC_FORCE_INLINE void multiply_columns_pair(
    const detail::PackedCpuPlan &packed,
    const float *input, std::ptrdiff_t input_stride,
    std::int32_t row, std::int32_t column,
    __m256 &value0, __m256 &value1) noexcept {
    const auto left = packed.weights_left[static_cast<std::size_t>(row)];
    const auto right = packed.weights_right[static_cast<std::size_t>(row)];
    const auto weight_base = static_cast<std::size_t>(row)
        * static_cast<std::size_t>(packed.weights_columns);
    value0 = _mm256_setzero_ps();
    value1 = _mm256_setzero_ps();

    if (right - left == 2) {
        const auto *source = input + static_cast<std::ptrdiff_t>(left)
            * input_stride + column;
        const auto *weights = packed.weights.data() + weight_base;
#define DSMVC_ACCUMULATE_PAIR_2(TAP) \
        { \
            const __m256 weight = _mm256_set1_ps(weights[TAP]); \
            const auto *tap_source = source \
                + static_cast<std::ptrdiff_t>(TAP) * input_stride; \
            value0 = _mm256_fmadd_ps( \
                weight, _mm256_loadu_ps(tap_source), value0); \
            value1 = _mm256_fmadd_ps( \
                weight, _mm256_loadu_ps(tap_source + 8), value1); \
        }
        DSMVC_ACCUMULATE_PAIR_2(0);
        DSMVC_ACCUMULATE_PAIR_2(1);
#undef DSMVC_ACCUMULATE_PAIR_2
        return;
    }

    if (right - left == 4) {
        const auto *source = input + static_cast<std::ptrdiff_t>(left)
            * input_stride + column;
        const auto *weights = packed.weights.data() + weight_base;
#define DSMVC_ACCUMULATE_PAIR(TAP) \
        { \
            const __m256 weight = _mm256_set1_ps(weights[TAP]); \
            const auto *tap_source = source \
                + static_cast<std::ptrdiff_t>(TAP) * input_stride; \
            value0 = _mm256_fmadd_ps( \
                weight, _mm256_loadu_ps(tap_source), value0); \
            value1 = _mm256_fmadd_ps( \
                weight, _mm256_loadu_ps(tap_source + 8), value1); \
        }
        DSMVC_ACCUMULATE_PAIR(0);
        DSMVC_ACCUMULATE_PAIR(1);
        DSMVC_ACCUMULATE_PAIR(2);
        DSMVC_ACCUMULATE_PAIR(3);
#undef DSMVC_ACCUMULATE_PAIR
        return;
    }

    for (std::int32_t source = left; source < right; ++source) {
        const __m256 weight = _mm256_set1_ps(
            packed.weights[weight_base
                + static_cast<std::size_t>(source - left)]);
        const auto *tap_source = input + static_cast<std::ptrdiff_t>(source)
            * input_stride + column;
        value0 = _mm256_fmadd_ps(
            weight, _mm256_loadu_ps(tap_source), value0);
        value1 = _mm256_fmadd_ps(
            weight, _mm256_loadu_ps(tap_source + 8), value1);
    }
}

DSMVC_FORCE_INLINE __m256 multiply_columns_single(
    const detail::PackedCpuPlan &packed,
    const float *input, std::ptrdiff_t input_stride,
    std::int32_t row, std::int32_t column) noexcept {
    const auto left = packed.weights_left[static_cast<std::size_t>(row)];
    const auto right = packed.weights_right[static_cast<std::size_t>(row)];
    const auto weight_base = static_cast<std::size_t>(row)
        * static_cast<std::size_t>(packed.weights_columns);
    __m256 value = _mm256_setzero_ps();

    if (right - left == 2) {
        const auto *source = input + static_cast<std::ptrdiff_t>(left)
            * input_stride + column;
        const auto *weights = packed.weights.data() + weight_base;
#define DSMVC_ACCUMULATE_SINGLE_2(TAP) \
        value = _mm256_fmadd_ps( \
            _mm256_set1_ps(weights[TAP]), \
            _mm256_loadu_ps(source \
                + static_cast<std::ptrdiff_t>(TAP) * input_stride), value)
        DSMVC_ACCUMULATE_SINGLE_2(0);
        DSMVC_ACCUMULATE_SINGLE_2(1);
#undef DSMVC_ACCUMULATE_SINGLE_2
        return value;
    }

    if (right - left == 4) {
        const auto *source = input + static_cast<std::ptrdiff_t>(left)
            * input_stride + column;
        const auto *weights = packed.weights.data() + weight_base;
#define DSMVC_ACCUMULATE_SINGLE(TAP) \
        value = _mm256_fmadd_ps( \
            _mm256_set1_ps(weights[TAP]), \
            _mm256_loadu_ps(source \
                + static_cast<std::ptrdiff_t>(TAP) * input_stride), value)
        DSMVC_ACCUMULATE_SINGLE(0);
        DSMVC_ACCUMULATE_SINGLE(1);
        DSMVC_ACCUMULATE_SINGLE(2);
        DSMVC_ACCUMULATE_SINGLE(3);
#undef DSMVC_ACCUMULATE_SINGLE
        return value;
    }

    for (std::int32_t source = left; source < right; ++source) {
        value = _mm256_fmadd_ps(
            _mm256_set1_ps(packed.weights[weight_base
                + static_cast<std::size_t>(source - left)]),
            _mm256_loadu_ps(input + static_cast<std::ptrdiff_t>(source)
                * input_stride + column), value);
    }
    return value;
}

void solve_columns_b3_pair(const detail::PackedCpuPlan &packed,
                           const float *input, std::ptrdiff_t input_stride,
                           float *output, std::ptrdiff_t output_stride,
                           std::int32_t column,
                           std::int32_t destination_size) noexcept {
    const auto factor_stride = static_cast<std::size_t>(
        packed.padded_destination_size);
    __m256 previous10 = _mm256_setzero_ps();
    __m256 previous11 = _mm256_setzero_ps();
    __m256 previous20 = _mm256_setzero_ps();
    __m256 previous21 = _mm256_setzero_ps();
    __m256 previous30 = _mm256_setzero_ps();
    __m256 previous31 = _mm256_setzero_ps();

    for (std::int32_t i = 0; i < destination_size; ++i) {
        __m256 value0;
        __m256 value1;
        multiply_columns_pair(
            packed, input, input_stride, i, column, value0, value1);
        const auto index = static_cast<std::size_t>(i);
        if (i >= 3) {
            const __m256 lower3 = _mm256_set1_ps(
                packed.lower_ld[2U * factor_stride + index]);
            value0 = _mm256_fnmadd_ps(lower3, previous30, value0);
            value1 = _mm256_fnmadd_ps(lower3, previous31, value1);
        }
        if (i >= 2) {
            const __m256 lower2 = _mm256_set1_ps(
                packed.lower_ld[factor_stride + index]);
            value0 = _mm256_fnmadd_ps(lower2, previous20, value0);
            value1 = _mm256_fnmadd_ps(lower2, previous21, value1);
        }
        if (i >= 1) {
            const __m256 lower1 = _mm256_set1_ps(packed.lower_ld[index]);
            value0 = _mm256_fnmadd_ps(lower1, previous10, value0);
            value1 = _mm256_fnmadd_ps(lower1, previous11, value1);
        }
        const __m256 inverse = _mm256_set1_ps(packed.inverse_diagonal[index]);
        value0 = _mm256_mul_ps(value0, inverse);
        value1 = _mm256_mul_ps(value1, inverse);
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        _mm256_storeu_ps(destination, value0);
        _mm256_storeu_ps(destination + 8, value1);
        previous30 = previous20;
        previous31 = previous21;
        previous20 = previous10;
        previous21 = previous11;
        previous10 = value0;
        previous11 = value1;
    }

    if (destination_size < 2) return;
    const auto *last = output + static_cast<std::ptrdiff_t>(destination_size - 1)
        * output_stride + column;
    __m256 next10 = _mm256_loadu_ps(last);
    __m256 next11 = _mm256_loadu_ps(last + 8);
    __m256 next20 = _mm256_setzero_ps();
    __m256 next21 = _mm256_setzero_ps();
    __m256 next30 = _mm256_setzero_ps();
    __m256 next31 = _mm256_setzero_ps();
    for (std::int32_t i = destination_size - 2; i >= 0; --i) {
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        __m256 value0 = _mm256_loadu_ps(destination);
        __m256 value1 = _mm256_loadu_ps(destination + 8);
        const auto index = static_cast<std::size_t>(i);
        const __m256 upper1 = _mm256_set1_ps(packed.upper_l[index]);
        value0 = _mm256_fnmadd_ps(upper1, next10, value0);
        value1 = _mm256_fnmadd_ps(upper1, next11, value1);
        if (i + 2 < destination_size) {
            const __m256 upper2 = _mm256_set1_ps(
                packed.upper_l[factor_stride + index]);
            value0 = _mm256_fnmadd_ps(upper2, next20, value0);
            value1 = _mm256_fnmadd_ps(upper2, next21, value1);
        }
        if (i + 3 < destination_size) {
            const __m256 upper3 = _mm256_set1_ps(
                packed.upper_l[2U * factor_stride + index]);
            value0 = _mm256_fnmadd_ps(upper3, next30, value0);
            value1 = _mm256_fnmadd_ps(upper3, next31, value1);
        }
        _mm256_storeu_ps(destination, value0);
        _mm256_storeu_ps(destination + 8, value1);
        next30 = next20;
        next31 = next21;
        next20 = next10;
        next21 = next11;
        next10 = value0;
        next11 = value1;
    }
}

void solve_columns_b3_single(const detail::PackedCpuPlan &packed,
                             const float *input, std::ptrdiff_t input_stride,
                             float *output, std::ptrdiff_t output_stride,
                             std::int32_t column,
                             std::int32_t destination_size) noexcept {
    const auto factor_stride = static_cast<std::size_t>(
        packed.padded_destination_size);
    __m256 previous1 = _mm256_setzero_ps();
    __m256 previous2 = _mm256_setzero_ps();
    __m256 previous3 = _mm256_setzero_ps();
    for (std::int32_t i = 0; i < destination_size; ++i) {
        __m256 value = multiply_columns_single(
            packed, input, input_stride, i, column);
        const auto index = static_cast<std::size_t>(i);
        if (i >= 3) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.lower_ld[2U * factor_stride + index]),
                previous3, value);
        }
        if (i >= 2) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.lower_ld[factor_stride + index]),
                previous2, value);
        }
        if (i >= 1) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.lower_ld[index]), previous1, value);
        }
        value = _mm256_mul_ps(
            value, _mm256_set1_ps(packed.inverse_diagonal[index]));
        _mm256_storeu_ps(output + static_cast<std::ptrdiff_t>(i)
                         * output_stride + column, value);
        previous3 = previous2;
        previous2 = previous1;
        previous1 = value;
    }

    if (destination_size < 2) return;
    __m256 next1 = _mm256_loadu_ps(
        output + static_cast<std::ptrdiff_t>(destination_size - 1)
            * output_stride + column);
    __m256 next2 = _mm256_setzero_ps();
    __m256 next3 = _mm256_setzero_ps();
    for (std::int32_t i = destination_size - 2; i >= 0; --i) {
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        __m256 value = _mm256_loadu_ps(destination);
        const auto index = static_cast<std::size_t>(i);
        value = _mm256_fnmadd_ps(
            _mm256_set1_ps(packed.upper_l[index]), next1, value);
        if (i + 2 < destination_size) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.upper_l[factor_stride + index]),
                next2, value);
        }
        if (i + 3 < destination_size) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.upper_l[2U * factor_stride + index]),
                next3, value);
        }
        _mm256_storeu_ps(destination, value);
        next3 = next2;
        next2 = next1;
        next1 = value;
    }
}

void solve_columns_b3(const getnative::AxisPlan &plan,
                      const detail::PackedCpuPlan &packed,
                      const float *input, std::ptrdiff_t input_stride,
                      float *output, std::ptrdiff_t output_stride,
                      std::int32_t vector_columns) noexcept {
    const auto paired_columns = vector_columns & ~15;
    for (std::int32_t column = 0; column < paired_columns; column += 16) {
        solve_columns_b3_pair(
            packed, input, input_stride, output, output_stride,
            column, plan.destination_size);
    }
    if (paired_columns != vector_columns) {
        solve_columns_b3_single(
            packed, input, input_stride, output, output_stride,
            paired_columns, plan.destination_size);
    }
}

void solve_columns_b1_pair(const detail::PackedCpuPlan &packed,
                           const float *input, std::ptrdiff_t input_stride,
                           float *output, std::ptrdiff_t output_stride,
                           std::int32_t column,
                           std::int32_t destination_size) noexcept {
    __m256 previous0 = _mm256_setzero_ps();
    __m256 previous1 = _mm256_setzero_ps();
    for (std::int32_t i = 0; i < destination_size; ++i) {
        __m256 value0;
        __m256 value1;
        multiply_columns_pair(
            packed, input, input_stride, i, column, value0, value1);
        const auto index = static_cast<std::size_t>(i);
        if (i >= 1) {
            const __m256 lower = _mm256_set1_ps(packed.lower_ld[index]);
            value0 = _mm256_fnmadd_ps(lower, previous0, value0);
            value1 = _mm256_fnmadd_ps(lower, previous1, value1);
        }
        const __m256 inverse = _mm256_set1_ps(packed.inverse_diagonal[index]);
        value0 = _mm256_mul_ps(value0, inverse);
        value1 = _mm256_mul_ps(value1, inverse);
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        _mm256_storeu_ps(destination, value0);
        _mm256_storeu_ps(destination + 8, value1);
        previous0 = value0;
        previous1 = value1;
    }

    if (destination_size < 2) return;
    const auto *last = output + static_cast<std::ptrdiff_t>(destination_size - 1)
        * output_stride + column;
    __m256 next0 = _mm256_loadu_ps(last);
    __m256 next1 = _mm256_loadu_ps(last + 8);
    for (std::int32_t i = destination_size - 2; i >= 0; --i) {
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        __m256 value0 = _mm256_loadu_ps(destination);
        __m256 value1 = _mm256_loadu_ps(destination + 8);
        const __m256 upper = _mm256_set1_ps(
            packed.upper_l[static_cast<std::size_t>(i)]);
        value0 = _mm256_fnmadd_ps(upper, next0, value0);
        value1 = _mm256_fnmadd_ps(upper, next1, value1);
        _mm256_storeu_ps(destination, value0);
        _mm256_storeu_ps(destination + 8, value1);
        next0 = value0;
        next1 = value1;
    }
}

void solve_columns_b1_single(const detail::PackedCpuPlan &packed,
                             const float *input, std::ptrdiff_t input_stride,
                             float *output, std::ptrdiff_t output_stride,
                             std::int32_t column,
                             std::int32_t destination_size) noexcept {
    __m256 previous = _mm256_setzero_ps();
    for (std::int32_t i = 0; i < destination_size; ++i) {
        __m256 value = multiply_columns_single(
            packed, input, input_stride, i, column);
        if (i >= 1) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.lower_ld[static_cast<std::size_t>(i)]),
                previous, value);
        }
        value = _mm256_mul_ps(
            value, _mm256_set1_ps(
                       packed.inverse_diagonal[static_cast<std::size_t>(i)]));
        _mm256_storeu_ps(output + static_cast<std::ptrdiff_t>(i)
                         * output_stride + column, value);
        previous = value;
    }

    if (destination_size < 2) return;
    __m256 next = _mm256_loadu_ps(
        output + static_cast<std::ptrdiff_t>(destination_size - 1)
            * output_stride + column);
    for (std::int32_t i = destination_size - 2; i >= 0; --i) {
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        __m256 value = _mm256_loadu_ps(destination);
        value = _mm256_fnmadd_ps(
            _mm256_set1_ps(packed.upper_l[static_cast<std::size_t>(i)]),
            next, value);
        _mm256_storeu_ps(destination, value);
        next = value;
    }
}

void solve_columns_b1(const getnative::AxisPlan &plan,
                      const detail::PackedCpuPlan &packed,
                      const float *input, std::ptrdiff_t input_stride,
                      float *output, std::ptrdiff_t output_stride,
                      std::int32_t vector_columns) noexcept {
    const auto paired_columns = vector_columns & ~15;
    for (std::int32_t column = 0; column < paired_columns; column += 16) {
        solve_columns_b1_pair(
            packed, input, input_stride, output, output_stride,
            column, plan.destination_size);
    }
    if (paired_columns != vector_columns) {
        solve_columns_b1_single(
            packed, input, input_stride, output, output_stride,
            paired_columns, plan.destination_size);
    }
}

template <int FixedBandwidth>
void solve_columns_vector(const getnative::AxisPlan &plan,
                          const detail::PackedCpuPlan &packed,
                          const float *input, std::ptrdiff_t input_stride,
                          float *output, std::ptrdiff_t output_stride,
                          std::int32_t vector_columns) noexcept {
    // Keep one input/output column tile in the per-core L2 while the forward
    // and backward recurrences consume it.  A full-frame forward pass leaves
    // the intermediate rows cold before the backward pass at high concurrency.
    constexpr std::int32_t l2_column_tile = 32;
    constexpr std::int32_t frame_parallel_threshold = 1024;
    const auto column_tile = vector_columns >= frame_parallel_threshold
        ? l2_column_tile : vector_columns;
    const auto n = plan.destination_size;
    const auto factor_stride = static_cast<std::size_t>(packed.padded_destination_size);
    const auto bandwidth = FixedBandwidth == 0 ? plan.half_bandwidth : FixedBandwidth;
    for (std::int32_t tile = 0; tile < vector_columns; tile += column_tile) {
        const auto tile_end = std::min(tile + column_tile, vector_columns);
        for (std::int32_t i = 0; i < n; ++i) {
            const auto left = packed.weights_left[static_cast<std::size_t>(i)];
            const auto right = packed.weights_right[static_cast<std::size_t>(i)];
            const auto weight_base = static_cast<std::size_t>(i)
                * static_cast<std::size_t>(packed.weights_columns);
            for (std::int32_t column = tile; column < tile_end; column += 8) {
                __m256 value = _mm256_setzero_ps();
                for (std::int32_t source = left; source < right; ++source) {
                    value = _mm256_fmadd_ps(
                        _mm256_set1_ps(packed.weights[
                            weight_base + static_cast<std::size_t>(source - left)]),
                        _mm256_loadu_ps(input + static_cast<std::ptrdiff_t>(source)
                                        * input_stride + column), value);
                }
                const auto available = std::min(bandwidth, i);
                for (std::int32_t distance = available; distance >= 1; --distance) {
                    value = _mm256_fnmadd_ps(
                        _mm256_set1_ps(packed.lower_ld[
                            static_cast<std::size_t>(distance - 1) * factor_stride
                            + static_cast<std::size_t>(i)]),
                        _mm256_loadu_ps(output + static_cast<std::ptrdiff_t>(i - distance)
                                        * output_stride + column), value);
                }
                value = _mm256_mul_ps(
                    value, _mm256_set1_ps(
                               packed.inverse_diagonal[static_cast<std::size_t>(i)]));
                _mm256_storeu_ps(output + static_cast<std::ptrdiff_t>(i)
                                 * output_stride + column, value);
            }
        }
        for (std::int32_t i = n - 2; i >= 0; --i) {
            const auto available = std::min(bandwidth, n - i - 1);
            for (std::int32_t column = tile; column < tile_end; column += 8) {
                __m256 value = _mm256_loadu_ps(
                    output + static_cast<std::ptrdiff_t>(i) * output_stride + column);
                if constexpr (FixedBandwidth == 3) {
                    for (std::int32_t distance = 1; distance <= available; ++distance) {
                        value = _mm256_fnmadd_ps(
                            _mm256_set1_ps(packed.upper_l[
                                static_cast<std::size_t>(distance - 1) * factor_stride
                                + static_cast<std::size_t>(i)]),
                            _mm256_loadu_ps(output
                                + static_cast<std::ptrdiff_t>(i + distance)
                                    * output_stride + column), value);
                    }
                } else {
                    for (std::int32_t distance = available; distance >= 1; --distance) {
                        value = _mm256_fnmadd_ps(
                            _mm256_set1_ps(packed.upper_l[
                                static_cast<std::size_t>(distance - 1) * factor_stride
                                + static_cast<std::size_t>(i)]),
                            _mm256_loadu_ps(output
                                + static_cast<std::ptrdiff_t>(i + distance)
                                    * output_stride + column), value);
                    }
                }
                _mm256_storeu_ps(output + static_cast<std::ptrdiff_t>(i)
                                 * output_stride + column, value);
            }
        }
    }
}

} // namespace

void inverse_rows_avx2(const getnative::AxisPlan &plan,
                       const detail::PackedCpuPlan &packed,
                       const float *input, std::ptrdiff_t input_row_stride,
                       float *output, std::ptrdiff_t output_row_stride,
                       std::int32_t row_count) {
    if (row_count < 8
        || input_row_stride < packed.padded_source_size
        || output_row_stride < packed.padded_destination_size) {
        for (std::int32_t row = 0; row < row_count; ++row) {
            getnative::inverse_axis_f32(
                plan, input + static_cast<std::ptrdiff_t>(row) * input_row_stride, 1,
                output + static_cast<std::ptrdiff_t>(row) * output_row_stride, 1);
        }
        return;
    }

    thread_local std::vector<ScratchVector> scratch;
    scratch.resize(static_cast<std::size_t>(packed.padded_source_size));
    auto *scratch_data = scratch.front().lanes;
    const auto complete_rows = row_count & ~7;
    for (std::int32_t row = 0; row < complete_rows; row += 8) {
        solve_horizontal_block(
            plan, packed,
            input + static_cast<std::ptrdiff_t>(row) * input_row_stride,
            input_row_stride,
            output + static_cast<std::ptrdiff_t>(row) * output_row_stride,
            output_row_stride, scratch_data);
    }
    if (complete_rows != row_count) {
        const auto row = row_count - 8;
        solve_horizontal_block(
            plan, packed,
            input + static_cast<std::ptrdiff_t>(row) * input_row_stride,
            input_row_stride,
            output + static_cast<std::ptrdiff_t>(row) * output_row_stride,
            output_row_stride, scratch_data);
    }
}

void inverse_columns_avx2(const getnative::AxisPlan &plan,
                          const detail::PackedCpuPlan &packed,
                          const float *input, std::ptrdiff_t input_row_stride,
                          float *output, std::ptrdiff_t output_row_stride,
                          std::int32_t column_count) {
    const auto padded_columns = (column_count + 7) & ~7;
    const auto vector_columns = input_row_stride >= padded_columns
            && output_row_stride >= padded_columns
        ? padded_columns : (column_count & ~7);
    if (plan.half_bandwidth == 1) {
        solve_columns_b1(plan, packed, input, input_row_stride, output,
                         output_row_stride, vector_columns);
    } else if (plan.half_bandwidth == 3) {
        solve_columns_b3(plan, packed, input, input_row_stride, output,
                         output_row_stride, vector_columns);
    } else {
        solve_columns_vector<0>(plan, packed, input, input_row_stride, output,
                                output_row_stride, vector_columns);
    }
    for (std::int32_t column = vector_columns; column < column_count; ++column) {
        getnative::inverse_axis_f32(plan, input + column, input_row_stride,
                                    output + column, output_row_stride);
    }
}

} // namespace dsmvc

#undef DSMVC_FORCE_INLINE
