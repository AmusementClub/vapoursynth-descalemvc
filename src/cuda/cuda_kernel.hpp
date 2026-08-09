#pragma once

#include <cstdint>
#include <type_traits>

namespace dsmvc::cuda_kernel {

enum class PlanPrecision : std::uint8_t {
    float32,
    float64,
};

struct AxisPlanDescriptor {
    std::uint32_t source_size = 0U;
    std::uint32_t destination_size = 0U;
    std::uint32_t half_bandwidth = 0U;
};

struct IntegerConversionDescriptor {
    float input_offset = 0.0F;
    float input_scale = 1.0F;
    float output_scale = 1.0F;
    float output_offset = 0.0F;
    std::uint32_t output_maximum = 255U;
};

static_assert(std::is_trivially_copyable_v<AxisPlanDescriptor>);
static_assert(std::is_trivially_copyable_v<IntegerConversionDescriptor>);
static_assert(sizeof(AxisPlanDescriptor) == 12U);
static_assert(sizeof(IntegerConversionDescriptor) == 20U);

} // namespace dsmvc::cuda_kernel
