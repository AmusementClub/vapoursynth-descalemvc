#pragma once

#include <dsmvc/engine.hpp>

#include <cstddef>
#include <cstdint>

namespace dsmvc::detail {

[[nodiscard]] std::int32_t padding_index(
    std::int64_t index, std::int32_t size, BorderMode mode);

void inverse_axis_f32_ordered(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_stride,
    float *output, std::ptrdiff_t output_stride) noexcept;

void inverse_axis_f64_ordered(
    const AxisPlan &plan,
    const double *input, std::ptrdiff_t input_stride,
    double *output, std::ptrdiff_t output_stride) noexcept;

void inverse_axis_f64(
    const AxisPlan &plan,
    const double *input, std::ptrdiff_t input_stride,
    double *output, std::ptrdiff_t output_stride) noexcept;

} // namespace dsmvc::detail
