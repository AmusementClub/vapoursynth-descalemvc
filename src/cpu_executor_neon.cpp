#include <dsmvc/engine.hpp>

#include "cpu_packed.hpp"

#include <algorithm>
#include <arm_neon.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace dsmvc {

namespace {

#if defined(_MSC_VER)
#define DSMVC_FORCE_INLINE __forceinline
#else
#define DSMVC_FORCE_INLINE inline __attribute__((always_inline))
#endif

struct alignas(16) ScratchVector {
    float lanes[4];
};

struct NeonPair {
    float32x4_t first;
    float32x4_t second;
};

DSMVC_FORCE_INLINE void transpose4(float32x4_t &row0, float32x4_t &row1,
                                   float32x4_t &row2,
                                   float32x4_t &row3) noexcept {
    const float32x4x2_t t0 = vtrnq_f32(row0, row1);
    const float32x4x2_t t1 = vtrnq_f32(row2, row3);
    row0 = vcombine_f32(vget_low_f32(t0.val[0]), vget_low_f32(t1.val[0]));
    row1 = vcombine_f32(vget_low_f32(t0.val[1]), vget_low_f32(t1.val[1]));
    row2 = vcombine_f32(vget_high_f32(t0.val[0]), vget_high_f32(t1.val[0]));
    row3 = vcombine_f32(vget_high_f32(t0.val[1]), vget_high_f32(t1.val[1]));
}

void transpose_source(const float *input, std::ptrdiff_t stride,
                      std::int32_t padded_width, float *scratch) noexcept {
    for (std::int32_t column = 0; column < padded_width; column += 4) {
        float32x4_t x0 = vld1q_f32(input + column);
        float32x4_t x1 = vld1q_f32(input + stride + column);
        float32x4_t x2 = vld1q_f32(input + 2 * stride + column);
        float32x4_t x3 = vld1q_f32(input + 3 * stride + column);
        transpose4(x0, x1, x2, x3);
        vst1q_f32(scratch + static_cast<std::size_t>(column + 0) * 4U, x0);
        vst1q_f32(scratch + static_cast<std::size_t>(column + 1) * 4U, x1);
        vst1q_f32(scratch + static_cast<std::size_t>(column + 2) * 4U, x2);
        vst1q_f32(scratch + static_cast<std::size_t>(column + 3) * 4U, x3);
    }
}

[[nodiscard]] DSMVC_FORCE_INLINE float32x4_t multiply_transpose(
    const detail::PackedCpuPlan &packed, const float *scratch,
    std::int32_t row) noexcept {
    float32x4_t sum = vdupq_n_f32(0.0F);
    const auto left = packed.weights_left[static_cast<std::size_t>(row)];
    const auto right = packed.weights_right[static_cast<std::size_t>(row)];
    const auto base = static_cast<std::size_t>(row)
        * static_cast<std::size_t>(packed.weights_columns);

    // Built-in kernels use 2, 4, 6, or 8 packed taps. Keep the fallback for
    // boundary rows with a non-contiguous span and for custom kernels.
    const auto tap_count = right - left;
    if (tap_count == 2 || tap_count == 4
        || tap_count == 6 || tap_count == 8) {
        const auto *weights = packed.weights.data() + base;
        const auto *source = scratch + static_cast<std::size_t>(left) * 4U;
#define DSMVC_ACCUMULATE_TRANSPOSE(TAP) \
        sum = vfmaq_f32( \
            sum, vdupq_n_f32(weights[TAP]), \
            vld1q_f32(source + static_cast<std::size_t>(TAP) * 4U))
        if (tap_count >= 2) {
            DSMVC_ACCUMULATE_TRANSPOSE(0);
            DSMVC_ACCUMULATE_TRANSPOSE(1);
        }
        if (tap_count >= 4) {
            DSMVC_ACCUMULATE_TRANSPOSE(2);
            DSMVC_ACCUMULATE_TRANSPOSE(3);
        }
        if (tap_count >= 6) {
            DSMVC_ACCUMULATE_TRANSPOSE(4);
            DSMVC_ACCUMULATE_TRANSPOSE(5);
        }
        if (tap_count >= 8) {
            DSMVC_ACCUMULATE_TRANSPOSE(6);
            DSMVC_ACCUMULATE_TRANSPOSE(7);
        }
#undef DSMVC_ACCUMULATE_TRANSPOSE
        return sum;
    }

    for (std::int32_t source = left; source < right; ++source) {
        const auto weight = vdupq_n_f32(
            packed.weights[base + static_cast<std::size_t>(source - left)]);
        sum = vfmaq_f32(
            sum, weight,
            vld1q_f32(scratch + static_cast<std::size_t>(source) * 4U));
    }
    return sum;
}

[[nodiscard]] DSMVC_FORCE_INLINE NeonPair multiply_transpose_pair(
    const detail::PackedCpuPlan &packed, const float *scratch_first,
    const float *scratch_second, std::int32_t row) noexcept {
    NeonPair sum{vdupq_n_f32(0.0F), vdupq_n_f32(0.0F)};
    const auto left = packed.weights_left[static_cast<std::size_t>(row)];
    const auto right = packed.weights_right[static_cast<std::size_t>(row)];
    const auto base = static_cast<std::size_t>(row)
        * static_cast<std::size_t>(packed.weights_columns);
    const auto tap_count = right - left;

    if (tap_count == 2 || tap_count == 4
        || tap_count == 6 || tap_count == 8) {
        const auto *weights = packed.weights.data() + base;
        const auto *source_first = scratch_first
            + static_cast<std::size_t>(left) * 4U;
        const auto *source_second = scratch_second
            + static_cast<std::size_t>(left) * 4U;
#define DSMVC_ACCUMULATE_TRANSPOSE_PAIR(TAP) \
        do { \
            const auto weight = vdupq_n_f32(weights[TAP]); \
            sum.first = vfmaq_f32( \
                sum.first, weight, \
                vld1q_f32(source_first \
                          + static_cast<std::size_t>(TAP) * 4U)); \
            sum.second = vfmaq_f32( \
                sum.second, weight, \
                vld1q_f32(source_second \
                          + static_cast<std::size_t>(TAP) * 4U)); \
        } while (false)
        if (tap_count >= 2) {
            DSMVC_ACCUMULATE_TRANSPOSE_PAIR(0);
            DSMVC_ACCUMULATE_TRANSPOSE_PAIR(1);
        }
        if (tap_count >= 4) {
            DSMVC_ACCUMULATE_TRANSPOSE_PAIR(2);
            DSMVC_ACCUMULATE_TRANSPOSE_PAIR(3);
        }
        if (tap_count >= 6) {
            DSMVC_ACCUMULATE_TRANSPOSE_PAIR(4);
            DSMVC_ACCUMULATE_TRANSPOSE_PAIR(5);
        }
        if (tap_count >= 8) {
            DSMVC_ACCUMULATE_TRANSPOSE_PAIR(6);
            DSMVC_ACCUMULATE_TRANSPOSE_PAIR(7);
        }
#undef DSMVC_ACCUMULATE_TRANSPOSE_PAIR
        return sum;
    }

    for (std::int32_t source = left; source < right; ++source) {
        const auto weight = vdupq_n_f32(
            packed.weights[base + static_cast<std::size_t>(source - left)]);
        const auto offset = static_cast<std::size_t>(source) * 4U;
        sum.first = vfmaq_f32(
            sum.first, weight, vld1q_f32(scratch_first + offset));
        sum.second = vfmaq_f32(
            sum.second, weight, vld1q_f32(scratch_second + offset));
    }
    return sum;
}

[[nodiscard]] DSMVC_FORCE_INLINE float32x4_t forward_b1(
    const detail::PackedCpuPlan &packed, std::int32_t i, float32x4_t value,
    float32x4_t previous) noexcept {
    value = vfmsq_f32(
        value, vdupq_n_f32(packed.lower_ld[static_cast<std::size_t>(i)]),
        previous);
    return vmulq_f32(
        value,
        vdupq_n_f32(packed.inverse_diagonal[static_cast<std::size_t>(i)]));
}

[[nodiscard]] DSMVC_FORCE_INLINE float32x4_t backward_b1(
    const detail::PackedCpuPlan &packed, std::int32_t i, float32x4_t value,
    float32x4_t next) noexcept {
    return vfmsq_f32(
        value, vdupq_n_f32(packed.upper_l[static_cast<std::size_t>(i)]), next);
}

[[nodiscard]] DSMVC_FORCE_INLINE float32x4_t forward_b3(
    const detail::PackedCpuPlan &packed, std::int32_t i, float32x4_t value,
    float32x4_t previous1, float32x4_t previous2,
    float32x4_t previous3) noexcept {
    const auto stride = static_cast<std::size_t>(packed.padded_destination_size);
    const auto index = static_cast<std::size_t>(i);
    value = vfmsq_f32(
        value, vdupq_n_f32(packed.lower_ld[2U * stride + index]), previous3);
    value = vfmsq_f32(
        value, vdupq_n_f32(packed.lower_ld[stride + index]), previous2);
    value = vfmsq_f32(
        value, vdupq_n_f32(packed.lower_ld[index]), previous1);
    return vmulq_f32(value, vdupq_n_f32(packed.inverse_diagonal[index]));
}

[[nodiscard]] DSMVC_FORCE_INLINE float32x4_t backward_b3(
    const detail::PackedCpuPlan &packed, std::int32_t i, float32x4_t value,
    float32x4_t next1, float32x4_t next2, float32x4_t next3) noexcept {
    const auto stride = static_cast<std::size_t>(packed.padded_destination_size);
    const auto index = static_cast<std::size_t>(i);
    value = vfmsq_f32(value, vdupq_n_f32(packed.upper_l[index]), next1);
    value = vfmsq_f32(
        value, vdupq_n_f32(packed.upper_l[stride + index]), next2);
    return vfmsq_f32(
        value, vdupq_n_f32(packed.upper_l[2U * stride + index]), next3);
}

[[nodiscard]] DSMVC_FORCE_INLINE NeonPair forward_b3_pair(
    const detail::PackedCpuPlan &packed, std::int32_t i, NeonPair value,
    NeonPair previous1, NeonPair previous2, NeonPair previous3) noexcept {
    const auto stride = static_cast<std::size_t>(packed.padded_destination_size);
    const auto index = static_cast<std::size_t>(i);
    auto coefficient = vdupq_n_f32(packed.lower_ld[2U * stride + index]);
    value.first = vfmsq_f32(value.first, coefficient, previous3.first);
    value.second = vfmsq_f32(value.second, coefficient, previous3.second);
    coefficient = vdupq_n_f32(packed.lower_ld[stride + index]);
    value.first = vfmsq_f32(value.first, coefficient, previous2.first);
    value.second = vfmsq_f32(value.second, coefficient, previous2.second);
    coefficient = vdupq_n_f32(packed.lower_ld[index]);
    value.first = vfmsq_f32(value.first, coefficient, previous1.first);
    value.second = vfmsq_f32(value.second, coefficient, previous1.second);
    const auto inverse = vdupq_n_f32(packed.inverse_diagonal[index]);
    return {
        vmulq_f32(value.first, inverse),
        vmulq_f32(value.second, inverse),
    };
}

[[nodiscard]] DSMVC_FORCE_INLINE NeonPair backward_b3_pair(
    const detail::PackedCpuPlan &packed, std::int32_t i, NeonPair value,
    NeonPair next1, NeonPair next2, NeonPair next3) noexcept {
    const auto stride = static_cast<std::size_t>(packed.padded_destination_size);
    const auto index = static_cast<std::size_t>(i);
    auto coefficient = vdupq_n_f32(packed.upper_l[index]);
    value.first = vfmsq_f32(value.first, coefficient, next1.first);
    value.second = vfmsq_f32(value.second, coefficient, next1.second);
    coefficient = vdupq_n_f32(packed.upper_l[stride + index]);
    value.first = vfmsq_f32(value.first, coefficient, next2.first);
    value.second = vfmsq_f32(value.second, coefficient, next2.second);
    coefficient = vdupq_n_f32(packed.upper_l[2U * stride + index]);
    return {
        vfmsq_f32(value.first, coefficient, next3.first),
        vfmsq_f32(value.second, coefficient, next3.second),
    };
}

void solve_horizontal_b1(const detail::PackedCpuPlan &packed,
                         const float *scratch, float *output,
                         std::ptrdiff_t stride) noexcept {
    float32x4_t previous = vdupq_n_f32(0.0F);
    const auto padded = packed.padded_destination_size;
    for (std::int32_t j = 0; j < padded; j += 4) {
        float32x4_t x0 = forward_b1(
            packed, j + 0, multiply_transpose(packed, scratch, j + 0), previous);
        float32x4_t x1 = forward_b1(
            packed, j + 1, multiply_transpose(packed, scratch, j + 1), x0);
        float32x4_t x2 = forward_b1(
            packed, j + 2, multiply_transpose(packed, scratch, j + 2), x1);
        float32x4_t x3 = forward_b1(
            packed, j + 3, multiply_transpose(packed, scratch, j + 3), x2);
        previous = x3;
        vst1q_f32(output + j, x0);
        vst1q_f32(output + stride + j, x1);
        vst1q_f32(output + 2 * stride + j, x2);
        vst1q_f32(output + 3 * stride + j, x3);
    }

    float32x4_t next = vdupq_n_f32(0.0F);
    for (std::int32_t j = padded - 4; j >= 0; j -= 4) {
        float32x4_t x0 = vld1q_f32(output + j);
        float32x4_t x1 = vld1q_f32(output + stride + j);
        float32x4_t x2 = vld1q_f32(output + 2 * stride + j);
        float32x4_t x3 = vld1q_f32(output + 3 * stride + j);
        x3 = backward_b1(packed, j + 3, x3, next);
        x2 = backward_b1(packed, j + 2, x2, x3);
        x1 = backward_b1(packed, j + 1, x1, x2);
        x0 = backward_b1(packed, j + 0, x0, x1);
        next = x0;
        transpose4(x0, x1, x2, x3);
        vst1q_f32(output + j, x0);
        vst1q_f32(output + stride + j, x1);
        vst1q_f32(output + 2 * stride + j, x2);
        vst1q_f32(output + 3 * stride + j, x3);
    }
}

void solve_horizontal_b3(const detail::PackedCpuPlan &packed,
                         const float *scratch, float *output,
                         std::ptrdiff_t stride) noexcept {
    float32x4_t previous1 = vdupq_n_f32(0.0F);
    float32x4_t previous2 = vdupq_n_f32(0.0F);
    float32x4_t previous3 = vdupq_n_f32(0.0F);
    const auto padded = packed.padded_destination_size;
    for (std::int32_t j = 0; j < padded; j += 4) {
#define DSMVC_FORWARD3(LANE, PREVIOUS1, PREVIOUS2, PREVIOUS3) \
        float32x4_t x##LANE = forward_b3( \
            packed, j + LANE, multiply_transpose(packed, scratch, j + LANE), \
            PREVIOUS1, PREVIOUS2, PREVIOUS3)
        DSMVC_FORWARD3(0, previous1, previous2, previous3);
        DSMVC_FORWARD3(1, x0, previous1, previous2);
        DSMVC_FORWARD3(2, x1, x0, previous1);
        DSMVC_FORWARD3(3, x2, x1, x0);
#undef DSMVC_FORWARD3
        previous1 = x3;
        previous2 = x2;
        previous3 = x1;
        vst1q_f32(output + j, x0);
        vst1q_f32(output + stride + j, x1);
        vst1q_f32(output + 2 * stride + j, x2);
        vst1q_f32(output + 3 * stride + j, x3);
    }

    float32x4_t next1 = vdupq_n_f32(0.0F);
    float32x4_t next2 = vdupq_n_f32(0.0F);
    float32x4_t next3 = vdupq_n_f32(0.0F);
    for (std::int32_t j = padded - 4; j >= 0; j -= 4) {
        float32x4_t x0 = vld1q_f32(output + j);
        float32x4_t x1 = vld1q_f32(output + stride + j);
        float32x4_t x2 = vld1q_f32(output + 2 * stride + j);
        float32x4_t x3 = vld1q_f32(output + 3 * stride + j);
        x3 = backward_b3(packed, j + 3, x3, next1, next2, next3);
        x2 = backward_b3(packed, j + 2, x2, x3, next1, next2);
        x1 = backward_b3(packed, j + 1, x1, x2, x3, next1);
        x0 = backward_b3(packed, j + 0, x0, x1, x2, x3);
        next1 = x0;
        next2 = x1;
        next3 = x2;
        transpose4(x0, x1, x2, x3);
        vst1q_f32(output + j, x0);
        vst1q_f32(output + stride + j, x1);
        vst1q_f32(output + 2 * stride + j, x2);
        vst1q_f32(output + 3 * stride + j, x3);
    }
}

void solve_horizontal_b3_pair(const detail::PackedCpuPlan &packed,
                              const float *scratch_first,
                              const float *scratch_second, float *output,
                              std::ptrdiff_t stride) noexcept {
    const NeonPair zero{vdupq_n_f32(0.0F), vdupq_n_f32(0.0F)};
    NeonPair previous1 = zero;
    NeonPair previous2 = zero;
    NeonPair previous3 = zero;
    const auto padded = packed.padded_destination_size;
    auto *output_second = output + 4 * stride;
    for (std::int32_t j = 0; j < padded; j += 4) {
#define DSMVC_FORWARD3_PAIR(LANE, PREVIOUS1, PREVIOUS2, PREVIOUS3) \
        NeonPair x##LANE = forward_b3_pair( \
            packed, j + LANE, \
            multiply_transpose_pair( \
                packed, scratch_first, scratch_second, j + LANE), \
            PREVIOUS1, PREVIOUS2, PREVIOUS3)
        DSMVC_FORWARD3_PAIR(0, previous1, previous2, previous3);
        DSMVC_FORWARD3_PAIR(1, x0, previous1, previous2);
        DSMVC_FORWARD3_PAIR(2, x1, x0, previous1);
        DSMVC_FORWARD3_PAIR(3, x2, x1, x0);
#undef DSMVC_FORWARD3_PAIR
        previous1 = x3;
        previous2 = x2;
        previous3 = x1;
        vst1q_f32(output + j, x0.first);
        vst1q_f32(output + stride + j, x1.first);
        vst1q_f32(output + 2 * stride + j, x2.first);
        vst1q_f32(output + 3 * stride + j, x3.first);
        vst1q_f32(output_second + j, x0.second);
        vst1q_f32(output_second + stride + j, x1.second);
        vst1q_f32(output_second + 2 * stride + j, x2.second);
        vst1q_f32(output_second + 3 * stride + j, x3.second);
    }

    NeonPair next1 = zero;
    NeonPair next2 = zero;
    NeonPair next3 = zero;
    for (std::int32_t j = padded - 4; j >= 0; j -= 4) {
        NeonPair x0{
            vld1q_f32(output + j),
            vld1q_f32(output_second + j),
        };
        NeonPair x1{
            vld1q_f32(output + stride + j),
            vld1q_f32(output_second + stride + j),
        };
        NeonPair x2{
            vld1q_f32(output + 2 * stride + j),
            vld1q_f32(output_second + 2 * stride + j),
        };
        NeonPair x3{
            vld1q_f32(output + 3 * stride + j),
            vld1q_f32(output_second + 3 * stride + j),
        };
        x3 = backward_b3_pair(packed, j + 3, x3, next1, next2, next3);
        x2 = backward_b3_pair(packed, j + 2, x2, x3, next1, next2);
        x1 = backward_b3_pair(packed, j + 1, x1, x2, x3, next1);
        x0 = backward_b3_pair(packed, j + 0, x0, x1, x2, x3);
        next1 = x0;
        next2 = x1;
        next3 = x2;
        transpose4(x0.first, x1.first, x2.first, x3.first);
        transpose4(x0.second, x1.second, x2.second, x3.second);
        vst1q_f32(output + j, x0.first);
        vst1q_f32(output + stride + j, x1.first);
        vst1q_f32(output + 2 * stride + j, x2.first);
        vst1q_f32(output + 3 * stride + j, x3.first);
        vst1q_f32(output_second + j, x0.second);
        vst1q_f32(output_second + stride + j, x1.second);
        vst1q_f32(output_second + 2 * stride + j, x2.second);
        vst1q_f32(output_second + 3 * stride + j, x3.second);
    }
}

[[nodiscard]] float *transposed_output(float *output, std::ptrdiff_t stride,
                                       std::int32_t index) noexcept {
    return output + static_cast<std::ptrdiff_t>(index & 3) * stride
        + static_cast<std::ptrdiff_t>(index & ~3);
}

template <int Distance>
DSMVC_FORCE_INLINE void subtract_lower_descending(
    const detail::PackedCpuPlan &packed, std::size_t factor_stride,
    std::int32_t index, float32x4_t &value, float *output,
    std::ptrdiff_t stride) noexcept {
    if constexpr (Distance > 0) {
        value = vfmsq_f32(
            value,
            vdupq_n_f32(packed.lower_ld[
                static_cast<std::size_t>(Distance - 1) * factor_stride
                + static_cast<std::size_t>(index)]),
            vld1q_f32(transposed_output(output, stride, index - Distance)));
        subtract_lower_descending<Distance - 1>(
            packed, factor_stride, index, value, output, stride);
    }
}

template <int Distance>
DSMVC_FORCE_INLINE void subtract_upper_descending(
    const detail::PackedCpuPlan &packed, std::size_t factor_stride,
    std::int32_t index, float32x4_t &value, float *output,
    std::ptrdiff_t stride) noexcept {
    if constexpr (Distance > 0) {
        value = vfmsq_f32(
            value,
            vdupq_n_f32(packed.upper_l[
                static_cast<std::size_t>(Distance - 1) * factor_stride
                + static_cast<std::size_t>(index)]),
            vld1q_f32(transposed_output(output, stride, index + Distance)));
        subtract_upper_descending<Distance - 1>(
            packed, factor_stride, index, value, output, stride);
    }
}

template <int FixedBandwidth>
void solve_horizontal_generic(const AxisPlan &plan,
                              const detail::PackedCpuPlan &packed,
                              const float *scratch, float *output,
                              std::ptrdiff_t stride) noexcept {
    const auto n = plan.destination_size;
    const auto factor_stride = static_cast<std::size_t>(
        packed.padded_destination_size);
    for (std::int32_t i = 0; i < n; ++i) {
        float32x4_t value = multiply_transpose(packed, scratch, i);
        const auto available = std::min(plan.half_bandwidth, i);
        if constexpr (FixedBandwidth > 0) {
            if (i >= FixedBandwidth) {
                subtract_lower_descending<FixedBandwidth>(
                    packed, factor_stride, i, value, output, stride);
            } else {
                for (std::int32_t distance = available;
                     distance >= 1; --distance) {
                    value = vfmsq_f32(
                        value,
                        vdupq_n_f32(packed.lower_ld[
                            static_cast<std::size_t>(distance - 1)
                                * factor_stride
                            + static_cast<std::size_t>(i)]),
                        vld1q_f32(transposed_output(
                            output, stride, i - distance)));
                }
            }
        } else {
            for (std::int32_t distance = available; distance >= 1; --distance) {
                value = vfmsq_f32(
                    value,
                    vdupq_n_f32(packed.lower_ld[
                        static_cast<std::size_t>(distance - 1) * factor_stride
                        + static_cast<std::size_t>(i)]),
                    vld1q_f32(transposed_output(output, stride, i - distance)));
            }
        }
        value = vmulq_f32(
            value,
            vdupq_n_f32(packed.inverse_diagonal[static_cast<std::size_t>(i)]));
        vst1q_f32(transposed_output(output, stride, i), value);
    }
    for (std::int32_t i = n; i < packed.padded_destination_size; ++i) {
        vst1q_f32(
            transposed_output(output, stride, i), vdupq_n_f32(0.0F));
    }
    for (std::int32_t i = n - 2; i >= 0; --i) {
        float32x4_t value = vld1q_f32(transposed_output(output, stride, i));
        const auto available = std::min(plan.half_bandwidth, n - i - 1);
        if constexpr (FixedBandwidth > 0) {
            if (i + FixedBandwidth < n) {
                subtract_upper_descending<FixedBandwidth>(
                    packed, factor_stride, i, value, output, stride);
            } else {
                for (std::int32_t distance = available;
                     distance >= 1; --distance) {
                    value = vfmsq_f32(
                        value,
                        vdupq_n_f32(packed.upper_l[
                            static_cast<std::size_t>(distance - 1)
                                * factor_stride
                            + static_cast<std::size_t>(i)]),
                        vld1q_f32(transposed_output(
                            output, stride, i + distance)));
                }
            }
        } else {
            for (std::int32_t distance = available; distance >= 1; --distance) {
                value = vfmsq_f32(
                    value,
                    vdupq_n_f32(packed.upper_l[
                        static_cast<std::size_t>(distance - 1) * factor_stride
                        + static_cast<std::size_t>(i)]),
                    vld1q_f32(transposed_output(output, stride, i + distance)));
            }
        }
        vst1q_f32(transposed_output(output, stride, i), value);
    }
    for (std::int32_t j = 0; j < packed.padded_destination_size; j += 4) {
        float32x4_t x0 = vld1q_f32(output + j);
        float32x4_t x1 = vld1q_f32(output + stride + j);
        float32x4_t x2 = vld1q_f32(output + 2 * stride + j);
        float32x4_t x3 = vld1q_f32(output + 3 * stride + j);
        transpose4(x0, x1, x2, x3);
        vst1q_f32(output + j, x0);
        vst1q_f32(output + stride + j, x1);
        vst1q_f32(output + 2 * stride + j, x2);
        vst1q_f32(output + 3 * stride + j, x3);
    }
}

void solve_horizontal_block(const AxisPlan &plan,
                            const detail::PackedCpuPlan &packed,
                            const float *input, std::ptrdiff_t input_stride,
                            float *output, std::ptrdiff_t output_stride,
                            float *scratch) noexcept {
    transpose_source(input, input_stride, packed.padded_source_size, scratch);
    if (plan.half_bandwidth == 1) {
        solve_horizontal_b1(packed, scratch, output, output_stride);
    } else if (plan.half_bandwidth == 3) {
        solve_horizontal_b3(packed, scratch, output, output_stride);
    } else if (plan.half_bandwidth == 5) {
        solve_horizontal_generic<5>(
            plan, packed, scratch, output, output_stride);
    } else if (plan.half_bandwidth == 7) {
        solve_horizontal_generic<7>(
            plan, packed, scratch, output, output_stride);
    } else {
        solve_horizontal_generic<0>(
            plan, packed, scratch, output, output_stride);
    }
}

void solve_horizontal_b3_pair_block(
    const detail::PackedCpuPlan &packed,
    const float *input, std::ptrdiff_t input_stride,
    float *output, std::ptrdiff_t output_stride,
    float *scratch) noexcept {
    const auto scratch_stride = static_cast<std::size_t>(
        packed.padded_source_size) * 4U;
    auto *scratch_second = scratch + scratch_stride;
    transpose_source(
        input, input_stride, packed.padded_source_size, scratch);
    transpose_source(
        input + 4 * input_stride, input_stride,
        packed.padded_source_size, scratch_second);
    solve_horizontal_b3_pair(
        packed, scratch, scratch_second, output, output_stride);
}

DSMVC_FORCE_INLINE void multiply_columns_pair(
    const detail::PackedCpuPlan &packed,
    const float *input, std::ptrdiff_t input_stride,
    std::int32_t row, std::int32_t column,
    float32x4_t &value0, float32x4_t &value1) noexcept {
    const auto left = packed.weights_left[static_cast<std::size_t>(row)];
    const auto right = packed.weights_right[static_cast<std::size_t>(row)];
    const auto weight_base = static_cast<std::size_t>(row)
        * static_cast<std::size_t>(packed.weights_columns);
    value0 = vdupq_n_f32(0.0F);
    value1 = vdupq_n_f32(0.0F);

    if (right - left == 2) {
        const auto *source = input + static_cast<std::ptrdiff_t>(left)
            * input_stride + column;
        const auto *weights = packed.weights.data() + weight_base;
#define DSMVC_ACCUMULATE_PAIR_2(TAP) \
        { \
            const auto weight = vdupq_n_f32(weights[TAP]); \
            const auto *tap_source = source \
                + static_cast<std::ptrdiff_t>(TAP) * input_stride; \
            value0 = vfmaq_f32(value0, weight, vld1q_f32(tap_source)); \
            value1 = vfmaq_f32(value1, weight, vld1q_f32(tap_source + 4)); \
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
            const auto weight = vdupq_n_f32(weights[TAP]); \
            const auto *tap_source = source \
                + static_cast<std::ptrdiff_t>(TAP) * input_stride; \
            value0 = vfmaq_f32(value0, weight, vld1q_f32(tap_source)); \
            value1 = vfmaq_f32(value1, weight, vld1q_f32(tap_source + 4)); \
        }
        DSMVC_ACCUMULATE_PAIR(0);
        DSMVC_ACCUMULATE_PAIR(1);
        DSMVC_ACCUMULATE_PAIR(2);
        DSMVC_ACCUMULATE_PAIR(3);
#undef DSMVC_ACCUMULATE_PAIR
        return;
    }

    for (std::int32_t source = left; source < right; ++source) {
        const auto weight = vdupq_n_f32(
            packed.weights[weight_base
                + static_cast<std::size_t>(source - left)]);
        const auto *tap_source = input + static_cast<std::ptrdiff_t>(source)
            * input_stride + column;
        value0 = vfmaq_f32(value0, weight, vld1q_f32(tap_source));
        value1 = vfmaq_f32(value1, weight, vld1q_f32(tap_source + 4));
    }
}

DSMVC_FORCE_INLINE void multiply_columns_quad(
    const detail::PackedCpuPlan &packed,
    const float *input, std::ptrdiff_t input_stride,
    std::int32_t row, std::int32_t column,
    float32x4_t &value0, float32x4_t &value1,
    float32x4_t &value2, float32x4_t &value3) noexcept {
    const auto left = packed.weights_left[static_cast<std::size_t>(row)];
    const auto right = packed.weights_right[static_cast<std::size_t>(row)];
    const auto weight_base = static_cast<std::size_t>(row)
        * static_cast<std::size_t>(packed.weights_columns);
    value0 = vdupq_n_f32(0.0F);
    value1 = vdupq_n_f32(0.0F);
    value2 = vdupq_n_f32(0.0F);
    value3 = vdupq_n_f32(0.0F);

    if (right - left == 2) {
        const auto *source = input + static_cast<std::ptrdiff_t>(left)
            * input_stride + column;
        const auto *weights = packed.weights.data() + weight_base;
#define DSMVC_ACCUMULATE_QUAD_2(TAP) \
        { \
            const auto weight = vdupq_n_f32(weights[TAP]); \
            const auto *tap_source = source \
                + static_cast<std::ptrdiff_t>(TAP) * input_stride; \
            value0 = vfmaq_f32(value0, weight, vld1q_f32(tap_source)); \
            value1 = vfmaq_f32(value1, weight, vld1q_f32(tap_source + 4)); \
            value2 = vfmaq_f32(value2, weight, vld1q_f32(tap_source + 8)); \
            value3 = vfmaq_f32(value3, weight, vld1q_f32(tap_source + 12)); \
        }
        DSMVC_ACCUMULATE_QUAD_2(0);
        DSMVC_ACCUMULATE_QUAD_2(1);
#undef DSMVC_ACCUMULATE_QUAD_2
        return;
    }

    if (right - left == 4) {
        const auto *source = input + static_cast<std::ptrdiff_t>(left)
            * input_stride + column;
        const auto *weights = packed.weights.data() + weight_base;
#define DSMVC_ACCUMULATE_QUAD(TAP) \
        { \
            const auto weight = vdupq_n_f32(weights[TAP]); \
            const auto *tap_source = source \
                + static_cast<std::ptrdiff_t>(TAP) * input_stride; \
            value0 = vfmaq_f32(value0, weight, vld1q_f32(tap_source)); \
            value1 = vfmaq_f32(value1, weight, vld1q_f32(tap_source + 4)); \
            value2 = vfmaq_f32(value2, weight, vld1q_f32(tap_source + 8)); \
            value3 = vfmaq_f32(value3, weight, vld1q_f32(tap_source + 12)); \
        }
        DSMVC_ACCUMULATE_QUAD(0);
        DSMVC_ACCUMULATE_QUAD(1);
        DSMVC_ACCUMULATE_QUAD(2);
        DSMVC_ACCUMULATE_QUAD(3);
#undef DSMVC_ACCUMULATE_QUAD
        return;
    }

    for (std::int32_t source = left; source < right; ++source) {
        const auto weight = vdupq_n_f32(
            packed.weights[weight_base
                + static_cast<std::size_t>(source - left)]);
        const auto *tap_source = input + static_cast<std::ptrdiff_t>(source)
            * input_stride + column;
        value0 = vfmaq_f32(value0, weight, vld1q_f32(tap_source));
        value1 = vfmaq_f32(value1, weight, vld1q_f32(tap_source + 4));
        value2 = vfmaq_f32(value2, weight, vld1q_f32(tap_source + 8));
        value3 = vfmaq_f32(value3, weight, vld1q_f32(tap_source + 12));
    }
}

[[nodiscard]] DSMVC_FORCE_INLINE float32x4_t multiply_columns_single(
    const detail::PackedCpuPlan &packed,
    const float *input, std::ptrdiff_t input_stride,
    std::int32_t row, std::int32_t column) noexcept {
    const auto left = packed.weights_left[static_cast<std::size_t>(row)];
    const auto right = packed.weights_right[static_cast<std::size_t>(row)];
    const auto weight_base = static_cast<std::size_t>(row)
        * static_cast<std::size_t>(packed.weights_columns);
    float32x4_t value = vdupq_n_f32(0.0F);

    if (right - left == 2) {
        const auto *source = input + static_cast<std::ptrdiff_t>(left)
            * input_stride + column;
        const auto *weights = packed.weights.data() + weight_base;
#define DSMVC_ACCUMULATE_SINGLE_2(TAP) \
        value = vfmaq_f32( \
            value, vdupq_n_f32(weights[TAP]), \
            vld1q_f32(source \
                + static_cast<std::ptrdiff_t>(TAP) * input_stride))
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
        value = vfmaq_f32( \
            value, vdupq_n_f32(weights[TAP]), \
            vld1q_f32(source \
                + static_cast<std::ptrdiff_t>(TAP) * input_stride))
        DSMVC_ACCUMULATE_SINGLE(0);
        DSMVC_ACCUMULATE_SINGLE(1);
        DSMVC_ACCUMULATE_SINGLE(2);
        DSMVC_ACCUMULATE_SINGLE(3);
#undef DSMVC_ACCUMULATE_SINGLE
        return value;
    }

    for (std::int32_t source = left; source < right; ++source) {
        value = vfmaq_f32(
            value,
            vdupq_n_f32(packed.weights[weight_base
                + static_cast<std::size_t>(source - left)]),
            vld1q_f32(input + static_cast<std::ptrdiff_t>(source)
                * input_stride + column));
    }
    return value;
}

void solve_columns_b3_quad(const detail::PackedCpuPlan &packed,
                           const float *input, std::ptrdiff_t input_stride,
                           float *output, std::ptrdiff_t output_stride,
                           std::int32_t column,
                           std::int32_t destination_size) noexcept {
    const auto factor_stride = static_cast<std::size_t>(
        packed.padded_destination_size);
    float32x4_t previous10 = vdupq_n_f32(0.0F);
    float32x4_t previous11 = vdupq_n_f32(0.0F);
    float32x4_t previous12 = vdupq_n_f32(0.0F);
    float32x4_t previous13 = vdupq_n_f32(0.0F);
    float32x4_t previous20 = vdupq_n_f32(0.0F);
    float32x4_t previous21 = vdupq_n_f32(0.0F);
    float32x4_t previous22 = vdupq_n_f32(0.0F);
    float32x4_t previous23 = vdupq_n_f32(0.0F);
    float32x4_t previous30 = vdupq_n_f32(0.0F);
    float32x4_t previous31 = vdupq_n_f32(0.0F);
    float32x4_t previous32 = vdupq_n_f32(0.0F);
    float32x4_t previous33 = vdupq_n_f32(0.0F);

    for (std::int32_t i = 0; i < destination_size; ++i) {
        float32x4_t value0;
        float32x4_t value1;
        float32x4_t value2;
        float32x4_t value3;
        multiply_columns_quad(
            packed, input, input_stride, i, column,
            value0, value1, value2, value3);
        const auto index = static_cast<std::size_t>(i);
        if (i >= 3) {
            const auto lower3 = vdupq_n_f32(
                packed.lower_ld[2U * factor_stride + index]);
            value0 = vfmsq_f32(value0, lower3, previous30);
            value1 = vfmsq_f32(value1, lower3, previous31);
            value2 = vfmsq_f32(value2, lower3, previous32);
            value3 = vfmsq_f32(value3, lower3, previous33);
        }
        if (i >= 2) {
            const auto lower2 = vdupq_n_f32(
                packed.lower_ld[factor_stride + index]);
            value0 = vfmsq_f32(value0, lower2, previous20);
            value1 = vfmsq_f32(value1, lower2, previous21);
            value2 = vfmsq_f32(value2, lower2, previous22);
            value3 = vfmsq_f32(value3, lower2, previous23);
        }
        if (i >= 1) {
            const auto lower1 = vdupq_n_f32(packed.lower_ld[index]);
            value0 = vfmsq_f32(value0, lower1, previous10);
            value1 = vfmsq_f32(value1, lower1, previous11);
            value2 = vfmsq_f32(value2, lower1, previous12);
            value3 = vfmsq_f32(value3, lower1, previous13);
        }
        const auto inverse = vdupq_n_f32(packed.inverse_diagonal[index]);
        value0 = vmulq_f32(value0, inverse);
        value1 = vmulq_f32(value1, inverse);
        value2 = vmulq_f32(value2, inverse);
        value3 = vmulq_f32(value3, inverse);
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        vst1q_f32(destination, value0);
        vst1q_f32(destination + 4, value1);
        vst1q_f32(destination + 8, value2);
        vst1q_f32(destination + 12, value3);
        previous30 = previous20;
        previous31 = previous21;
        previous32 = previous22;
        previous33 = previous23;
        previous20 = previous10;
        previous21 = previous11;
        previous22 = previous12;
        previous23 = previous13;
        previous10 = value0;
        previous11 = value1;
        previous12 = value2;
        previous13 = value3;
    }

    if (destination_size < 2) return;
    const auto *last = output + static_cast<std::ptrdiff_t>(destination_size - 1)
        * output_stride + column;
    float32x4_t next10 = vld1q_f32(last);
    float32x4_t next11 = vld1q_f32(last + 4);
    float32x4_t next12 = vld1q_f32(last + 8);
    float32x4_t next13 = vld1q_f32(last + 12);
    float32x4_t next20 = vdupq_n_f32(0.0F);
    float32x4_t next21 = vdupq_n_f32(0.0F);
    float32x4_t next22 = vdupq_n_f32(0.0F);
    float32x4_t next23 = vdupq_n_f32(0.0F);
    float32x4_t next30 = vdupq_n_f32(0.0F);
    float32x4_t next31 = vdupq_n_f32(0.0F);
    float32x4_t next32 = vdupq_n_f32(0.0F);
    float32x4_t next33 = vdupq_n_f32(0.0F);
    for (std::int32_t i = destination_size - 2; i >= 0; --i) {
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        float32x4_t value0 = vld1q_f32(destination);
        float32x4_t value1 = vld1q_f32(destination + 4);
        float32x4_t value2 = vld1q_f32(destination + 8);
        float32x4_t value3 = vld1q_f32(destination + 12);
        const auto index = static_cast<std::size_t>(i);
        const auto upper1 = vdupq_n_f32(packed.upper_l[index]);
        value0 = vfmsq_f32(value0, upper1, next10);
        value1 = vfmsq_f32(value1, upper1, next11);
        value2 = vfmsq_f32(value2, upper1, next12);
        value3 = vfmsq_f32(value3, upper1, next13);
        if (i + 2 < destination_size) {
            const auto upper2 = vdupq_n_f32(
                packed.upper_l[factor_stride + index]);
            value0 = vfmsq_f32(value0, upper2, next20);
            value1 = vfmsq_f32(value1, upper2, next21);
            value2 = vfmsq_f32(value2, upper2, next22);
            value3 = vfmsq_f32(value3, upper2, next23);
        }
        if (i + 3 < destination_size) {
            const auto upper3 = vdupq_n_f32(
                packed.upper_l[2U * factor_stride + index]);
            value0 = vfmsq_f32(value0, upper3, next30);
            value1 = vfmsq_f32(value1, upper3, next31);
            value2 = vfmsq_f32(value2, upper3, next32);
            value3 = vfmsq_f32(value3, upper3, next33);
        }
        vst1q_f32(destination, value0);
        vst1q_f32(destination + 4, value1);
        vst1q_f32(destination + 8, value2);
        vst1q_f32(destination + 12, value3);
        next30 = next20;
        next31 = next21;
        next32 = next22;
        next33 = next23;
        next20 = next10;
        next21 = next11;
        next22 = next12;
        next23 = next13;
        next10 = value0;
        next11 = value1;
        next12 = value2;
        next13 = value3;
    }
}

void solve_columns_b3_pair(const detail::PackedCpuPlan &packed,
                           const float *input, std::ptrdiff_t input_stride,
                           float *output, std::ptrdiff_t output_stride,
                           std::int32_t column,
                           std::int32_t destination_size) noexcept {
    const auto factor_stride = static_cast<std::size_t>(
        packed.padded_destination_size);
    float32x4_t previous10 = vdupq_n_f32(0.0F);
    float32x4_t previous11 = vdupq_n_f32(0.0F);
    float32x4_t previous20 = vdupq_n_f32(0.0F);
    float32x4_t previous21 = vdupq_n_f32(0.0F);
    float32x4_t previous30 = vdupq_n_f32(0.0F);
    float32x4_t previous31 = vdupq_n_f32(0.0F);

    for (std::int32_t i = 0; i < destination_size; ++i) {
        float32x4_t value0;
        float32x4_t value1;
        multiply_columns_pair(
            packed, input, input_stride, i, column, value0, value1);
        const auto index = static_cast<std::size_t>(i);
        if (i >= 3) {
            const auto lower3 = vdupq_n_f32(
                packed.lower_ld[2U * factor_stride + index]);
            value0 = vfmsq_f32(value0, lower3, previous30);
            value1 = vfmsq_f32(value1, lower3, previous31);
        }
        if (i >= 2) {
            const auto lower2 = vdupq_n_f32(
                packed.lower_ld[factor_stride + index]);
            value0 = vfmsq_f32(value0, lower2, previous20);
            value1 = vfmsq_f32(value1, lower2, previous21);
        }
        if (i >= 1) {
            const auto lower1 = vdupq_n_f32(packed.lower_ld[index]);
            value0 = vfmsq_f32(value0, lower1, previous10);
            value1 = vfmsq_f32(value1, lower1, previous11);
        }
        const auto inverse = vdupq_n_f32(packed.inverse_diagonal[index]);
        value0 = vmulq_f32(value0, inverse);
        value1 = vmulq_f32(value1, inverse);
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        vst1q_f32(destination, value0);
        vst1q_f32(destination + 4, value1);
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
    float32x4_t next10 = vld1q_f32(last);
    float32x4_t next11 = vld1q_f32(last + 4);
    float32x4_t next20 = vdupq_n_f32(0.0F);
    float32x4_t next21 = vdupq_n_f32(0.0F);
    float32x4_t next30 = vdupq_n_f32(0.0F);
    float32x4_t next31 = vdupq_n_f32(0.0F);
    for (std::int32_t i = destination_size - 2; i >= 0; --i) {
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        float32x4_t value0 = vld1q_f32(destination);
        float32x4_t value1 = vld1q_f32(destination + 4);
        const auto index = static_cast<std::size_t>(i);
        const auto upper1 = vdupq_n_f32(packed.upper_l[index]);
        value0 = vfmsq_f32(value0, upper1, next10);
        value1 = vfmsq_f32(value1, upper1, next11);
        if (i + 2 < destination_size) {
            const auto upper2 = vdupq_n_f32(
                packed.upper_l[factor_stride + index]);
            value0 = vfmsq_f32(value0, upper2, next20);
            value1 = vfmsq_f32(value1, upper2, next21);
        }
        if (i + 3 < destination_size) {
            const auto upper3 = vdupq_n_f32(
                packed.upper_l[2U * factor_stride + index]);
            value0 = vfmsq_f32(value0, upper3, next30);
            value1 = vfmsq_f32(value1, upper3, next31);
        }
        vst1q_f32(destination, value0);
        vst1q_f32(destination + 4, value1);
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
    float32x4_t previous1 = vdupq_n_f32(0.0F);
    float32x4_t previous2 = vdupq_n_f32(0.0F);
    float32x4_t previous3 = vdupq_n_f32(0.0F);
    for (std::int32_t i = 0; i < destination_size; ++i) {
        float32x4_t value = multiply_columns_single(
            packed, input, input_stride, i, column);
        const auto index = static_cast<std::size_t>(i);
        if (i >= 3) {
            value = vfmsq_f32(
                value,
                vdupq_n_f32(packed.lower_ld[2U * factor_stride + index]),
                previous3);
        }
        if (i >= 2) {
            value = vfmsq_f32(
                value, vdupq_n_f32(packed.lower_ld[factor_stride + index]),
                previous2);
        }
        if (i >= 1) {
            value = vfmsq_f32(
                value, vdupq_n_f32(packed.lower_ld[index]), previous1);
        }
        value = vmulq_f32(
            value, vdupq_n_f32(packed.inverse_diagonal[index]));
        vst1q_f32(output + static_cast<std::ptrdiff_t>(i)
                      * output_stride + column,
                  value);
        previous3 = previous2;
        previous2 = previous1;
        previous1 = value;
    }

    if (destination_size < 2) return;
    float32x4_t next1 = vld1q_f32(
        output + static_cast<std::ptrdiff_t>(destination_size - 1)
            * output_stride + column);
    float32x4_t next2 = vdupq_n_f32(0.0F);
    float32x4_t next3 = vdupq_n_f32(0.0F);
    for (std::int32_t i = destination_size - 2; i >= 0; --i) {
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        float32x4_t value = vld1q_f32(destination);
        const auto index = static_cast<std::size_t>(i);
        value = vfmsq_f32(
            value, vdupq_n_f32(packed.upper_l[index]), next1);
        if (i + 2 < destination_size) {
            value = vfmsq_f32(
                value, vdupq_n_f32(packed.upper_l[factor_stride + index]),
                next2);
        }
        if (i + 3 < destination_size) {
            value = vfmsq_f32(
                value,
                vdupq_n_f32(packed.upper_l[2U * factor_stride + index]),
                next3);
        }
        vst1q_f32(destination, value);
        next3 = next2;
        next2 = next1;
        next1 = value;
    }
}

void solve_columns_b3(const AxisPlan &plan,
                      const detail::PackedCpuPlan &packed,
                      const float *input, std::ptrdiff_t input_stride,
                      float *output, std::ptrdiff_t output_stride,
                      std::int32_t vector_columns) noexcept {
    const auto quad_columns = vector_columns & ~15;
    for (std::int32_t column = 0; column < quad_columns; column += 16) {
        solve_columns_b3_quad(
            packed, input, input_stride, output, output_stride,
            column, plan.destination_size);
    }
    auto remaining_column = quad_columns;
    if (vector_columns - remaining_column >= 8) {
        solve_columns_b3_pair(
            packed, input, input_stride, output, output_stride,
            remaining_column, plan.destination_size);
        remaining_column += 8;
    }
    if (remaining_column != vector_columns) {
        solve_columns_b3_single(
            packed, input, input_stride, output, output_stride,
            remaining_column, plan.destination_size);
    }
}

void solve_columns_b1_pair(const detail::PackedCpuPlan &packed,
                           const float *input, std::ptrdiff_t input_stride,
                           float *output, std::ptrdiff_t output_stride,
                           std::int32_t column,
                           std::int32_t destination_size) noexcept {
    float32x4_t previous0 = vdupq_n_f32(0.0F);
    float32x4_t previous1 = vdupq_n_f32(0.0F);
    for (std::int32_t i = 0; i < destination_size; ++i) {
        float32x4_t value0;
        float32x4_t value1;
        multiply_columns_pair(
            packed, input, input_stride, i, column, value0, value1);
        const auto index = static_cast<std::size_t>(i);
        if (i >= 1) {
            const auto lower = vdupq_n_f32(packed.lower_ld[index]);
            value0 = vfmsq_f32(value0, lower, previous0);
            value1 = vfmsq_f32(value1, lower, previous1);
        }
        const auto inverse = vdupq_n_f32(packed.inverse_diagonal[index]);
        value0 = vmulq_f32(value0, inverse);
        value1 = vmulq_f32(value1, inverse);
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        vst1q_f32(destination, value0);
        vst1q_f32(destination + 4, value1);
        previous0 = value0;
        previous1 = value1;
    }

    if (destination_size < 2) return;
    const auto *last = output + static_cast<std::ptrdiff_t>(destination_size - 1)
        * output_stride + column;
    float32x4_t next0 = vld1q_f32(last);
    float32x4_t next1 = vld1q_f32(last + 4);
    for (std::int32_t i = destination_size - 2; i >= 0; --i) {
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        float32x4_t value0 = vld1q_f32(destination);
        float32x4_t value1 = vld1q_f32(destination + 4);
        const auto upper = vdupq_n_f32(
            packed.upper_l[static_cast<std::size_t>(i)]);
        value0 = vfmsq_f32(value0, upper, next0);
        value1 = vfmsq_f32(value1, upper, next1);
        vst1q_f32(destination, value0);
        vst1q_f32(destination + 4, value1);
        next0 = value0;
        next1 = value1;
    }
}

void solve_columns_b1_single(const detail::PackedCpuPlan &packed,
                             const float *input, std::ptrdiff_t input_stride,
                             float *output, std::ptrdiff_t output_stride,
                             std::int32_t column,
                             std::int32_t destination_size) noexcept {
    float32x4_t previous = vdupq_n_f32(0.0F);
    for (std::int32_t i = 0; i < destination_size; ++i) {
        float32x4_t value = multiply_columns_single(
            packed, input, input_stride, i, column);
        if (i >= 1) {
            value = vfmsq_f32(
                value,
                vdupq_n_f32(packed.lower_ld[static_cast<std::size_t>(i)]),
                previous);
        }
        value = vmulq_f32(
            value,
            vdupq_n_f32(packed.inverse_diagonal[static_cast<std::size_t>(i)]));
        vst1q_f32(output + static_cast<std::ptrdiff_t>(i)
                      * output_stride + column,
                  value);
        previous = value;
    }

    if (destination_size < 2) return;
    float32x4_t next = vld1q_f32(
        output + static_cast<std::ptrdiff_t>(destination_size - 1)
            * output_stride + column);
    for (std::int32_t i = destination_size - 2; i >= 0; --i) {
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        float32x4_t value = vld1q_f32(destination);
        value = vfmsq_f32(
            value,
            vdupq_n_f32(packed.upper_l[static_cast<std::size_t>(i)]), next);
        vst1q_f32(destination, value);
        next = value;
    }
}

void solve_columns_b1(const AxisPlan &plan,
                      const detail::PackedCpuPlan &packed,
                      const float *input, std::ptrdiff_t input_stride,
                      float *output, std::ptrdiff_t output_stride,
                      std::int32_t vector_columns) noexcept {
    const auto paired_columns = vector_columns & ~7;
    for (std::int32_t column = 0; column < paired_columns; column += 8) {
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
void solve_columns_vector(const AxisPlan &plan,
                          const detail::PackedCpuPlan &packed,
                          const float *input, std::ptrdiff_t input_stride,
                          float *output, std::ptrdiff_t output_stride,
                          std::int32_t vector_columns) noexcept {
    constexpr std::int32_t l2_column_tile = 32;
    constexpr std::int32_t frame_parallel_threshold = 1024;
    const auto column_tile = vector_columns >= frame_parallel_threshold
        ? l2_column_tile : vector_columns;
    const auto n = plan.destination_size;
    const auto factor_stride = static_cast<std::size_t>(
        packed.padded_destination_size);
    const auto bandwidth = FixedBandwidth == 0
        ? plan.half_bandwidth : FixedBandwidth;
    for (std::int32_t tile = 0; tile < vector_columns; tile += column_tile) {
        const auto tile_end = std::min(tile + column_tile, vector_columns);
        for (std::int32_t i = 0; i < n; ++i) {
            const auto left = packed.weights_left[static_cast<std::size_t>(i)];
            const auto right = packed.weights_right[static_cast<std::size_t>(i)];
            const auto weight_base = static_cast<std::size_t>(i)
                * static_cast<std::size_t>(packed.weights_columns);
            for (std::int32_t column = tile; column < tile_end; column += 4) {
                float32x4_t value = vdupq_n_f32(0.0F);
                for (std::int32_t source = left; source < right; ++source) {
                    value = vfmaq_f32(
                        value,
                        vdupq_n_f32(packed.weights[
                            weight_base
                            + static_cast<std::size_t>(source - left)]),
                        vld1q_f32(input + static_cast<std::ptrdiff_t>(source)
                            * input_stride + column));
                }
                const auto available = std::min(bandwidth, i);
                for (std::int32_t distance = available;
                     distance >= 1; --distance) {
                    value = vfmsq_f32(
                        value,
                        vdupq_n_f32(packed.lower_ld[
                            static_cast<std::size_t>(distance - 1) * factor_stride
                            + static_cast<std::size_t>(i)]),
                        vld1q_f32(output
                            + static_cast<std::ptrdiff_t>(i - distance)
                                * output_stride + column));
                }
                value = vmulq_f32(
                    value,
                    vdupq_n_f32(
                        packed.inverse_diagonal[static_cast<std::size_t>(i)]));
                vst1q_f32(output + static_cast<std::ptrdiff_t>(i)
                              * output_stride + column,
                          value);
            }
        }
        for (std::int32_t i = n - 2; i >= 0; --i) {
            const auto available = std::min(bandwidth, n - i - 1);
            for (std::int32_t column = tile; column < tile_end; column += 4) {
                float32x4_t value = vld1q_f32(
                    output + static_cast<std::ptrdiff_t>(i)
                        * output_stride + column);
                if constexpr (FixedBandwidth == 3) {
                    for (std::int32_t distance = 1;
                         distance <= available; ++distance) {
                        value = vfmsq_f32(
                            value,
                            vdupq_n_f32(packed.upper_l[
                                static_cast<std::size_t>(distance - 1)
                                    * factor_stride
                                + static_cast<std::size_t>(i)]),
                            vld1q_f32(output
                                + static_cast<std::ptrdiff_t>(i + distance)
                                    * output_stride + column));
                    }
                } else {
                    for (std::int32_t distance = available;
                         distance >= 1; --distance) {
                        value = vfmsq_f32(
                            value,
                            vdupq_n_f32(packed.upper_l[
                                static_cast<std::size_t>(distance - 1)
                                    * factor_stride
                                + static_cast<std::size_t>(i)]),
                            vld1q_f32(output
                                + static_cast<std::ptrdiff_t>(i + distance)
                                    * output_stride + column));
                    }
                }
                vst1q_f32(output + static_cast<std::ptrdiff_t>(i)
                              * output_stride + column,
                          value);
            }
        }
    }
}

void solve_columns_pair(const AxisPlan &plan,
                        const detail::PackedCpuPlan &packed,
                        const float *input, std::ptrdiff_t input_stride,
                        float *output, std::ptrdiff_t output_stride,
                        std::int32_t vector_columns) noexcept {
    const auto paired_columns = vector_columns & ~7;
    constexpr std::int32_t l2_column_tile = 32;
    constexpr std::int32_t frame_parallel_threshold = 1024;
    const auto column_tile = paired_columns >= frame_parallel_threshold
        ? l2_column_tile : paired_columns;
    const auto n = plan.destination_size;
    const auto factor_stride = static_cast<std::size_t>(
        packed.padded_destination_size);

    for (std::int32_t tile = 0; tile < paired_columns; tile += column_tile) {
        const auto tile_end = std::min(tile + column_tile, paired_columns);
        for (std::int32_t i = 0; i < n; ++i) {
            const auto index = static_cast<std::size_t>(i);
            const auto available = std::min(plan.half_bandwidth, i);
            for (std::int32_t column = tile; column < tile_end; column += 8) {
                float32x4_t value0;
                float32x4_t value1;
                multiply_columns_pair(
                    packed, input, input_stride, i, column, value0, value1);
                for (std::int32_t distance = available;
                     distance >= 1; --distance) {
                    const auto lower = vdupq_n_f32(packed.lower_ld[
                        static_cast<std::size_t>(distance - 1) * factor_stride
                        + index]);
                    const auto *previous = output
                        + static_cast<std::ptrdiff_t>(i - distance)
                            * output_stride + column;
                    value0 = vfmsq_f32(value0, lower, vld1q_f32(previous));
                    value1 = vfmsq_f32(value1, lower, vld1q_f32(previous + 4));
                }
                const auto inverse = vdupq_n_f32(
                    packed.inverse_diagonal[index]);
                value0 = vmulq_f32(value0, inverse);
                value1 = vmulq_f32(value1, inverse);
                auto *destination = output
                    + static_cast<std::ptrdiff_t>(i) * output_stride + column;
                vst1q_f32(destination, value0);
                vst1q_f32(destination + 4, value1);
            }
        }

        for (std::int32_t i = n - 2; i >= 0; --i) {
            const auto index = static_cast<std::size_t>(i);
            const auto available = std::min(plan.half_bandwidth, n - i - 1);
            for (std::int32_t column = tile; column < tile_end; column += 8) {
                auto *destination = output
                    + static_cast<std::ptrdiff_t>(i) * output_stride + column;
                float32x4_t value0 = vld1q_f32(destination);
                float32x4_t value1 = vld1q_f32(destination + 4);
                for (std::int32_t distance = available;
                     distance >= 1; --distance) {
                    const auto upper = vdupq_n_f32(packed.upper_l[
                        static_cast<std::size_t>(distance - 1) * factor_stride
                        + index]);
                    const auto *next = output
                        + static_cast<std::ptrdiff_t>(i + distance)
                            * output_stride + column;
                    value0 = vfmsq_f32(value0, upper, vld1q_f32(next));
                    value1 = vfmsq_f32(value1, upper, vld1q_f32(next + 4));
                }
                vst1q_f32(destination, value0);
                vst1q_f32(destination + 4, value1);
            }
        }
    }

    if (paired_columns != vector_columns) {
        solve_columns_vector<0>(
            plan, packed, input + paired_columns, input_stride,
            output + paired_columns, output_stride,
            vector_columns - paired_columns);
    }
}

template <class Sample>
void normalize_integer_block(
    const Sample *input, std::ptrdiff_t input_stride,
    std::int32_t source_size, std::int32_t padded_source_size,
    const IntegerConversion &conversion, float *output) noexcept {
    const auto offset = vdupq_n_f32(conversion.input_offset);
    const auto scale = vdupq_n_f32(conversion.input_scale);
    for (std::int32_t row = 0; row < 4; ++row) {
        const auto *source = input
            + static_cast<std::ptrdiff_t>(row) * input_stride;
        auto *destination = output
            + static_cast<std::ptrdiff_t>(row) * padded_source_size;
        std::int32_t column = 0;
        if constexpr (std::is_same_v<Sample, std::uint8_t>) {
            for (; column + 16 <= source_size; column += 16) {
                const auto bytes = vld1q_u8(source + column);
                const auto low16 = vmovl_u8(vget_low_u8(bytes));
                const auto high16 = vmovl_u8(vget_high_u8(bytes));
                const auto store4 = [&](std::int32_t target,
                                        uint32x4_t values) noexcept {
                    auto converted = vcvtq_f32_u32(values);
                    converted = vmulq_f32(vsubq_f32(converted, offset), scale);
                    vst1q_f32(destination + target, converted);
                };
                store4(column + 0, vmovl_u16(vget_low_u16(low16)));
                store4(column + 4, vmovl_u16(vget_high_u16(low16)));
                store4(column + 8, vmovl_u16(vget_low_u16(high16)));
                store4(column + 12, vmovl_u16(vget_high_u16(high16)));
            }
        } else {
            for (; column + 8 <= source_size; column += 8) {
                const auto words = vld1q_u16(source + column);
                auto low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(words)));
                auto high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(words)));
                low = vmulq_f32(vsubq_f32(low, offset), scale);
                high = vmulq_f32(vsubq_f32(high, offset), scale);
                vst1q_f32(destination + column, low);
                vst1q_f32(destination + column + 4, high);
            }
        }
        for (; column < source_size; ++column) {
            destination[column] =
                (static_cast<float>(source[column]) - conversion.input_offset)
                * conversion.input_scale;
        }
        std::fill(destination + source_size,
                  destination + padded_source_size, 0.0F);
    }
}

template <class Sample>
void forward_2d_integer_rhs(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_horizontal,
    const AxisPlan &vertical,
    const detail::PackedCpuPlan &packed_vertical,
    const Sample *input, std::ptrdiff_t input_row_stride,
    const IntegerConversion &conversion,
    float *output, std::ptrdiff_t output_row_stride) {
    const auto padded_columns = packed_horizontal.padded_destination_size;
    const auto cache_blocks = static_cast<std::size_t>(
        std::max(packed_vertical.streaming_cache_blocks, 1));

    thread_local std::vector<float> normalized_block;
    thread_local std::vector<ScratchVector> transpose_scratch;
    thread_local std::vector<ScratchVector> horizontal_cache;
    thread_local std::vector<std::int32_t> cache_rows;
    thread_local std::vector<std::uint64_t> cache_ages;
    thread_local std::vector<const float *> source_rows;
    normalized_block.resize(
        4U * static_cast<std::size_t>(packed_horizontal.padded_source_size));
    transpose_scratch.resize(
        static_cast<std::size_t>(packed_horizontal.padded_source_size));
    horizontal_cache.resize(
        cache_blocks * static_cast<std::size_t>(padded_columns));
    cache_rows.assign(cache_blocks, -1);
    cache_ages.assign(cache_blocks, 0U);
    auto *transpose_data = transpose_scratch.front().lanes;
    std::uint64_t age = 0U;

    const auto tail_block = vertical.source_size & 3
        ? vertical.source_size - 4 : vertical.source_size;
    const auto source_block = [&](std::int32_t source) noexcept {
        return source >= tail_block ? tail_block : source & ~3;
    };
    const auto get_source_row = [&](std::int32_t source) -> const float * {
        const auto block_row = source_block(source);
        std::size_t slot = cache_blocks;
        for (std::size_t candidate = 0; candidate < cache_blocks; ++candidate) {
            if (cache_rows[candidate] == block_row) {
                slot = candidate;
                break;
            }
        }
        if (slot == cache_blocks) {
            slot = 0U;
            for (std::size_t candidate = 0; candidate < cache_blocks; ++candidate) {
                if (cache_rows[candidate] < 0) {
                    slot = candidate;
                    break;
                }
                if (cache_ages[candidate] < cache_ages[slot]) slot = candidate;
            }
            normalize_integer_block(
                input + static_cast<std::ptrdiff_t>(block_row) * input_row_stride,
                input_row_stride, horizontal.source_size,
                packed_horizontal.padded_source_size,
                conversion, normalized_block.data());
            auto *horizontal_rows = horizontal_cache[
                slot * static_cast<std::size_t>(padded_columns)].lanes;
            solve_horizontal_block(
                horizontal, packed_horizontal, normalized_block.data(),
                packed_horizontal.padded_source_size,
                horizontal_rows, padded_columns, transpose_data);
            cache_rows[slot] = block_row;
        }
        cache_ages[slot] = ++age;
        return horizontal_cache[
            slot * static_cast<std::size_t>(padded_columns)].lanes
            + static_cast<std::ptrdiff_t>(source - block_row) * padded_columns;
    };

    const auto factor_stride = static_cast<std::size_t>(
        packed_vertical.padded_destination_size);
    for (std::int32_t row = 0; row < vertical.destination_size; ++row) {
        const auto begin = vertical.transpose_offsets[static_cast<std::size_t>(row)];
        const auto end = vertical.transpose_offsets[static_cast<std::size_t>(row) + 1U];
        source_rows.resize(static_cast<std::size_t>(end - begin));
        for (auto offset_index = begin; offset_index < end; ++offset_index) {
            source_rows[static_cast<std::size_t>(offset_index - begin)] =
                get_source_row(vertical.transpose_indices[offset_index]);
        }

        auto *destination = output
            + static_cast<std::ptrdiff_t>(row) * output_row_stride;
        const auto available = std::min(vertical.half_bandwidth, row);
        for (std::int32_t column = 0;
             column < padded_columns; column += 4) {
            auto value = vdupq_n_f32(0.0F);
            for (auto offset_index = begin; offset_index < end; ++offset_index) {
                value = vfmaq_f32(
                    value,
                    vdupq_n_f32(vertical.transpose_weights[offset_index]),
                    vld1q_f32(source_rows[
                        static_cast<std::size_t>(offset_index - begin)] + column));
            }
            for (std::int32_t distance = available;
                 distance >= 1; --distance) {
                value = vfmsq_f32(
                    value,
                    vdupq_n_f32(packed_vertical.lower_ld[
                        static_cast<std::size_t>(distance - 1) * factor_stride
                        + static_cast<std::size_t>(row)]),
                    vld1q_f32(output
                        + static_cast<std::ptrdiff_t>(row - distance)
                            * output_row_stride + column));
            }
            value = vmulq_f32(
                value, vdupq_n_f32(packed_vertical.inverse_diagonal[
                    static_cast<std::size_t>(row)]));
            vst1q_f32(destination + column, value);
        }
    }
}

void backward_2d_rhs(
    const AxisPlan &plan, const detail::PackedCpuPlan &packed,
    float *values, std::ptrdiff_t stride,
    std::int32_t padded_columns) noexcept {
    const auto n = plan.destination_size;
    const auto factor_stride = static_cast<std::size_t>(
        packed.padded_destination_size);
    for (std::int32_t row = n - 2; row >= 0; --row) {
        const auto available = std::min(plan.half_bandwidth, n - row - 1);
        for (std::int32_t column = 0; column < padded_columns; column += 8) {
            auto *destination = values
                + static_cast<std::ptrdiff_t>(row) * stride + column;
            auto value0 = vld1q_f32(destination);
            auto value1 = vld1q_f32(destination + 4);
            const auto apply_distance = [&](std::int32_t distance) noexcept {
                const auto upper = vdupq_n_f32(packed.upper_l[
                    static_cast<std::size_t>(distance - 1) * factor_stride
                    + static_cast<std::size_t>(row)]);
                const auto *next = values
                    + static_cast<std::ptrdiff_t>(row + distance) * stride
                    + column;
                value0 = vfmsq_f32(value0, upper, vld1q_f32(next));
                value1 = vfmsq_f32(value1, upper, vld1q_f32(next + 4));
            };
            if (plan.half_bandwidth == 3) {
                for (std::int32_t distance = 1;
                     distance <= available; ++distance) {
                    apply_distance(distance);
                }
            } else {
                for (std::int32_t distance = available;
                     distance >= 1; --distance) {
                    apply_distance(distance);
                }
            }
            vst1q_f32(destination, value0);
            vst1q_f32(destination + 4, value1);
        }
    }
}

[[nodiscard]] DSMVC_FORCE_INLINE uint32x4_t convert_output_vector(
    const float *source, const IntegerConversion &conversion) noexcept {
    auto value = vfmaq_f32(
        vdupq_n_f32(conversion.output_offset), vld1q_f32(source),
        vdupq_n_f32(conversion.output_scale));
    value = vmaxq_f32(value, vdupq_n_f32(0.0F));
    value = vminq_f32(
        value, vdupq_n_f32(static_cast<float>(conversion.output_maximum)));
    return vcvtnq_u32_f32(value);
}

template <class Sample>
void convert_2d_output(
    const float *input, std::ptrdiff_t input_stride,
    Sample *output, std::ptrdiff_t output_stride,
    std::int32_t rows, std::int32_t columns,
    const IntegerConversion &conversion) noexcept {
    for (std::int32_t row = 0; row < rows; ++row) {
        const auto *source = input
            + static_cast<std::ptrdiff_t>(row) * input_stride;
        auto *destination = output
            + static_cast<std::ptrdiff_t>(row) * output_stride;
        std::int32_t column = 0;
        if constexpr (std::is_same_v<Sample, std::uint8_t>) {
            for (; column + 16 <= columns; column += 16) {
                const auto low0 = vqmovn_u32(
                    convert_output_vector(source + column + 0, conversion));
                const auto low1 = vqmovn_u32(
                    convert_output_vector(source + column + 4, conversion));
                const auto high0 = vqmovn_u32(
                    convert_output_vector(source + column + 8, conversion));
                const auto high1 = vqmovn_u32(
                    convert_output_vector(source + column + 12, conversion));
                const auto bytes0 = vqmovn_u16(vcombine_u16(low0, low1));
                const auto bytes1 = vqmovn_u16(vcombine_u16(high0, high1));
                vst1q_u8(destination + column, vcombine_u8(bytes0, bytes1));
            }
        } else {
            for (; column + 8 <= columns; column += 8) {
                const auto low = vqmovn_u32(
                    convert_output_vector(source + column, conversion));
                const auto high = vqmovn_u32(
                    convert_output_vector(source + column + 4, conversion));
                vst1q_u16(destination + column, vcombine_u16(low, high));
            }
        }
        for (; column < columns; ++column) {
            const auto scaled = std::clamp(
                std::fma(source[column], conversion.output_scale,
                         conversion.output_offset),
                0.0F, static_cast<float>(conversion.output_maximum));
            destination[column] = static_cast<Sample>(std::nearbyint(scaled));
        }
    }
}

template <class Sample>
void normalize_2d_input(
    const Sample *input, std::ptrdiff_t input_stride,
    float *output, std::ptrdiff_t output_stride,
    std::int32_t rows, std::int32_t columns,
    const IntegerConversion &conversion) noexcept {
    if (rows < 4) {
        for (std::int32_t row = 0; row < rows; ++row) {
            auto *destination = output
                + static_cast<std::ptrdiff_t>(row) * output_stride;
            const auto *source = input
                + static_cast<std::ptrdiff_t>(row) * input_stride;
            for (std::int32_t column = 0; column < columns; ++column) {
                destination[column] =
                    (static_cast<float>(source[column]) - conversion.input_offset)
                    * conversion.input_scale;
            }
            std::fill(destination + columns,
                      destination + output_stride, 0.0F);
        }
        return;
    }
    const auto complete_rows = rows & ~3;
    for (std::int32_t row = 0; row < complete_rows; row += 4) {
        normalize_integer_block(
            input + static_cast<std::ptrdiff_t>(row) * input_stride,
            input_stride, columns, static_cast<std::int32_t>(output_stride),
            conversion,
            output + static_cast<std::ptrdiff_t>(row) * output_stride);
    }
    if (complete_rows != rows) {
        const auto row = rows - 4;
        normalize_integer_block(
            input + static_cast<std::ptrdiff_t>(row) * input_stride,
            input_stride, columns, static_cast<std::int32_t>(output_stride),
            conversion,
            output + static_cast<std::ptrdiff_t>(row) * output_stride);
    }
}

template <class Sample>
void inverse_2d_integer_neon(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_horizontal,
    const AxisPlan &vertical,
    const detail::PackedCpuPlan &packed_vertical,
    const Sample *input, std::ptrdiff_t input_row_stride,
    Sample *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion) {
    const auto stride = packed_horizontal.padded_destination_size;
    thread_local std::vector<float> rhs;
    rhs.resize(
        static_cast<std::size_t>(vertical.destination_size)
        * static_cast<std::size_t>(stride));
    forward_2d_integer_rhs(
        horizontal, packed_horizontal, vertical, packed_vertical,
        input, input_row_stride, conversion, rhs.data(), stride);
    backward_2d_rhs(
        vertical, packed_vertical, rhs.data(), stride, stride);
    convert_2d_output(
        rhs.data(), stride, output, output_row_stride,
        vertical.destination_size, horizontal.destination_size, conversion);
}

} // namespace

void inverse_rows_neon(const AxisPlan &plan,
                       const detail::PackedCpuPlan &packed,
                       const float *input, std::ptrdiff_t input_row_stride,
                       float *output, std::ptrdiff_t output_row_stride,
                       std::int32_t row_count) {
    if (row_count < 4
        || input_row_stride < packed.padded_source_size
        || output_row_stride < packed.padded_destination_size) {
        for (std::int32_t row = 0; row < row_count; ++row) {
            dsmvc::inverse_axis_f32(
                plan,
                input + static_cast<std::ptrdiff_t>(row) * input_row_stride, 1,
                output + static_cast<std::ptrdiff_t>(row) * output_row_stride,
                1);
        }
        return;
    }

    const auto complete_rows = row_count & ~3;
    const bool pair_b3 = plan.half_bandwidth == 3 && complete_rows >= 8;
    const auto scratch_blocks = pair_b3 ? 2U : 1U;
    thread_local std::vector<ScratchVector> scratch;
    scratch.resize(
        static_cast<std::size_t>(packed.padded_source_size) * scratch_blocks);
    auto *scratch_data = scratch.front().lanes;
    std::int32_t row = 0;
    if (pair_b3) {
        const auto paired_rows = complete_rows & ~7;
        for (; row < paired_rows; row += 8) {
            solve_horizontal_b3_pair_block(
                packed,
                input + static_cast<std::ptrdiff_t>(row) * input_row_stride,
                input_row_stride,
                output + static_cast<std::ptrdiff_t>(row) * output_row_stride,
                output_row_stride, scratch_data);
        }
    }
    for (; row < complete_rows; row += 4) {
        solve_horizontal_block(
            plan, packed,
            input + static_cast<std::ptrdiff_t>(row) * input_row_stride,
            input_row_stride,
            output + static_cast<std::ptrdiff_t>(row) * output_row_stride,
            output_row_stride, scratch_data);
    }
    if (complete_rows != row_count) {
        const auto row = row_count - 4;
        solve_horizontal_block(
            plan, packed,
            input + static_cast<std::ptrdiff_t>(row) * input_row_stride,
            input_row_stride,
            output + static_cast<std::ptrdiff_t>(row) * output_row_stride,
            output_row_stride, scratch_data);
    }
}

void inverse_columns_neon(const AxisPlan &plan,
                          const detail::PackedCpuPlan &packed,
                          const float *input, std::ptrdiff_t input_row_stride,
                          float *output, std::ptrdiff_t output_row_stride,
                          std::int32_t column_count) {
    const auto padded_columns = (column_count + 3) & ~3;
    const auto vector_columns = input_row_stride >= padded_columns
            && output_row_stride >= padded_columns
        ? padded_columns : (column_count & ~3);
    if (plan.half_bandwidth == 1) {
        solve_columns_b1(plan, packed, input, input_row_stride, output,
                         output_row_stride, vector_columns);
    } else if (plan.half_bandwidth == 3) {
        solve_columns_b3(plan, packed, input, input_row_stride, output,
                         output_row_stride, vector_columns);
    } else if (plan.half_bandwidth == 5 || plan.half_bandwidth == 7) {
        solve_columns_pair(plan, packed, input, input_row_stride, output,
                           output_row_stride, vector_columns);
    } else {
        solve_columns_vector<0>(plan, packed, input, input_row_stride, output,
                                output_row_stride, vector_columns);
    }
    for (std::int32_t column = vector_columns;
         column < column_count; ++column) {
        dsmvc::inverse_axis_f32(plan, input + column, input_row_stride,
                               output + column, output_row_stride);
    }
}

void inverse_2d_u8_neon(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_horizontal,
    const AxisPlan &vertical,
    const detail::PackedCpuPlan &packed_vertical,
    const std::uint8_t *input, std::ptrdiff_t input_row_stride,
    std::uint8_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion) {
    inverse_2d_integer_neon(
        horizontal, packed_horizontal, vertical, packed_vertical,
        input, input_row_stride, output, output_row_stride, conversion);
}

void inverse_2d_u16_neon(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_horizontal,
    const AxisPlan &vertical,
    const detail::PackedCpuPlan &packed_vertical,
    const std::uint16_t *input, std::ptrdiff_t input_row_stride,
    std::uint16_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion) {
    inverse_2d_integer_neon(
        horizontal, packed_horizontal, vertical, packed_vertical,
        input, input_row_stride, output, output_row_stride, conversion);
}

void normalize_u8_neon(
    const std::uint8_t *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t rows, std::int32_t columns,
    const IntegerConversion &conversion) {
    normalize_2d_input(
        input, input_row_stride, output, output_row_stride,
        rows, columns, conversion);
}

void normalize_u16_neon(
    const std::uint16_t *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t rows, std::int32_t columns,
    const IntegerConversion &conversion) {
    normalize_2d_input(
        input, input_row_stride, output, output_row_stride,
        rows, columns, conversion);
}

void convert_to_u8_neon(
    const float *input, std::ptrdiff_t input_row_stride,
    std::uint8_t *output, std::ptrdiff_t output_row_stride,
    std::int32_t rows, std::int32_t columns,
    const IntegerConversion &conversion) {
    convert_2d_output(
        input, input_row_stride, output, output_row_stride,
        rows, columns, conversion);
}

void convert_to_u16_neon(
    const float *input, std::ptrdiff_t input_row_stride,
    std::uint16_t *output, std::ptrdiff_t output_row_stride,
    std::int32_t rows, std::int32_t columns,
    const IntegerConversion &conversion) {
    convert_2d_output(
        input, input_row_stride, output, output_row_stride,
        rows, columns, conversion);
}

} // namespace dsmvc

#undef DSMVC_FORCE_INLINE
