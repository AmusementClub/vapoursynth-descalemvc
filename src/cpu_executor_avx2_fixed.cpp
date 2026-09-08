#include <dsmvc/engine.hpp>

#include "cpu_packed.hpp"
#include "cpu_horizontal_window.hpp"
#include "cpu_float_workspace.hpp"
#include "cpu_rhs_span.hpp"

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

template <int Bandwidth>
[[nodiscard]] DSMVC_FORCE_INLINE __m256 multiply_transpose(
    const detail::PackedCpuPlan &packed, const float *scratch,
    std::int32_t row) noexcept {
    __m256 sum = _mm256_setzero_ps();
    const auto left = packed.weights_left[static_cast<std::size_t>(row)];
    const auto right = packed.weights_right[static_cast<std::size_t>(row)];
    const auto base = static_cast<std::size_t>(row)
        * static_cast<std::size_t>(packed.weights_columns);
    const auto *weights = packed.weights.data() + base;
    const auto *source_base = scratch + static_cast<std::size_t>(left) * 8U;
    if (detail::try_accumulate_rhs_span<Bandwidth>(
            right - left, [&]<int Tap>() noexcept {
                sum = _mm256_fmadd_ps(_mm256_set1_ps(weights[Tap]),
                    _mm256_load_ps(source_base + Tap * 8U), sum);
            })) {
        return sum;
    }
    for (std::int32_t source = left; source < right; ++source) {
        sum = _mm256_fmadd_ps(
            _mm256_set1_ps(packed.weights[
                base + static_cast<std::size_t>(source - left)]),
            _mm256_load_ps(
                scratch + static_cast<std::size_t>(source) * 8U), sum);
    }
    return sum;
}

struct HorizontalOps {
    using Vector = __m256;
    static constexpr std::size_t lanes = 8;
    [[nodiscard]] static DSMVC_FORCE_INLINE Vector zero() noexcept {
        return _mm256_setzero_ps();
    }
    [[nodiscard]] static DSMVC_FORCE_INLINE Vector load(const float *p) noexcept {
        return _mm256_loadu_ps(p);
    }
    static DSMVC_FORCE_INLINE void store(float *p, Vector v) noexcept {
        _mm256_storeu_ps(p, v);
    }
    [[nodiscard]] static DSMVC_FORCE_INLINE Vector subtract_product(
        Vector v, float c, Vector previous) noexcept {
        return _mm256_fnmadd_ps(_mm256_set1_ps(c), previous, v);
    }
    [[nodiscard]] static DSMVC_FORCE_INLINE Vector multiply(
        Vector v, float c) noexcept {
        return _mm256_mul_ps(v, _mm256_set1_ps(c));
    }
};

void unpack_work(const AxisPlan &plan, const float *work,
                 float *output, std::ptrdiff_t stride) noexcept {
    const auto full_destination = plan.destination_size & ~7;
    for (std::int32_t j = 0; j < full_destination; j += 8) {
        __m256 x0 = _mm256_load_ps(work + static_cast<std::size_t>(j + 0) * 8U);
        __m256 x1 = _mm256_load_ps(work + static_cast<std::size_t>(j + 1) * 8U);
        __m256 x2 = _mm256_load_ps(work + static_cast<std::size_t>(j + 2) * 8U);
        __m256 x3 = _mm256_load_ps(work + static_cast<std::size_t>(j + 3) * 8U);
        __m256 x4 = _mm256_load_ps(work + static_cast<std::size_t>(j + 4) * 8U);
        __m256 x5 = _mm256_load_ps(work + static_cast<std::size_t>(j + 5) * 8U);
        __m256 x6 = _mm256_load_ps(work + static_cast<std::size_t>(j + 6) * 8U);
        __m256 x7 = _mm256_load_ps(work + static_cast<std::size_t>(j + 7) * 8U);
        transpose8(x0, x1, x2, x3, x4, x5, x6, x7);
        _mm256_storeu_ps(output + 0 * stride + j, x0);
        _mm256_storeu_ps(output + 1 * stride + j, x1);
        _mm256_storeu_ps(output + 2 * stride + j, x2);
        _mm256_storeu_ps(output + 3 * stride + j, x3);
        _mm256_storeu_ps(output + 4 * stride + j, x4);
        _mm256_storeu_ps(output + 5 * stride + j, x5);
        _mm256_storeu_ps(output + 6 * stride + j, x6);
        _mm256_storeu_ps(output + 7 * stride + j, x7);
    }
    if (full_destination == plan.destination_size) return;
    const auto j = full_destination;
    alignas(32) float tail[64];
    __m256 x0 = _mm256_load_ps(work + static_cast<std::size_t>(j + 0) * 8U);
    __m256 x1 = _mm256_load_ps(work + static_cast<std::size_t>(j + 1) * 8U);
    __m256 x2 = _mm256_load_ps(work + static_cast<std::size_t>(j + 2) * 8U);
    __m256 x3 = _mm256_load_ps(work + static_cast<std::size_t>(j + 3) * 8U);
    __m256 x4 = _mm256_load_ps(work + static_cast<std::size_t>(j + 4) * 8U);
    __m256 x5 = _mm256_load_ps(work + static_cast<std::size_t>(j + 5) * 8U);
    __m256 x6 = _mm256_load_ps(work + static_cast<std::size_t>(j + 6) * 8U);
    __m256 x7 = _mm256_load_ps(work + static_cast<std::size_t>(j + 7) * 8U);
    transpose8(x0, x1, x2, x3, x4, x5, x6, x7);
    _mm256_store_ps(tail + 0U * 8U, x0);
    _mm256_store_ps(tail + 1U * 8U, x1);
    _mm256_store_ps(tail + 2U * 8U, x2);
    _mm256_store_ps(tail + 3U * 8U, x3);
    _mm256_store_ps(tail + 4U * 8U, x4);
    _mm256_store_ps(tail + 5U * 8U, x5);
    _mm256_store_ps(tail + 6U * 8U, x6);
    _mm256_store_ps(tail + 7U * 8U, x7);
    const auto remaining = plan.destination_size - full_destination;
    for (std::int32_t row = 0; row < 8; ++row) {
        std::copy_n(tail + static_cast<std::size_t>(row) * 8U, remaining,
                    output + static_cast<std::ptrdiff_t>(row) * stride + j);
    }
}

template <std::int32_t Bandwidth>
void solve_fixed(const AxisPlan &plan, const detail::PackedCpuPlan &packed,
                 const float *scratch, float *work, float *output,
                 std::ptrdiff_t stride) noexcept {
    detail::solve_horizontal_window<Bandwidth, HorizontalOps>(
        packed, plan.destination_size, work,
        [&](std::int32_t i) noexcept {
            return multiply_transpose<Bandwidth>(packed, scratch, i);
        });
    unpack_work(plan, work, output, stride);
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

    thread_local std::vector<float> scratch;
    const auto scratch_vectors = static_cast<std::size_t>(packed.padded_source_size)
        + static_cast<std::size_t>(packed.padded_destination_size);
    auto *scratch_data = detail::aligned_float_workspace<32>(
        scratch, detail::checked_size_product(scratch_vectors, 8U,
            "AVX2 fixed row scratch"), "AVX2 fixed row scratch");
    auto *work = scratch_data
        + static_cast<std::size_t>(packed.padded_source_size) * 8U;
    const auto solve_block = [&](std::int32_t row) {
        auto *block_output =
            output + static_cast<std::ptrdiff_t>(row) * output_row_stride;
        transpose_source(
            input + static_cast<std::ptrdiff_t>(row) * input_row_stride,
            input_row_stride, plan.source_size,
            packed.padded_source_size, scratch_data);
        if (plan.half_bandwidth == 5) {
            solve_fixed<5>(plan, packed, scratch_data, work,
                            block_output, output_row_stride);
        } else {
            solve_fixed<7>(plan, packed, scratch_data, work,
                            block_output, output_row_stride);
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
