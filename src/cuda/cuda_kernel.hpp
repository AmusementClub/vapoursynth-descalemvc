#pragma once

#include <cstdint>
#include <type_traits>

namespace dsmvc::cuda_kernel {

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

// One independent axis operation in a heterogeneous CUDA batch. Device
// addresses use fixed-width integers so the fatbin ABI is identical on every
// supported 64-bit host platform.
struct AxisBatchJobDescriptor {
    AxisPlanDescriptor plan{};
    std::uint32_t vector_count = 0U;
    std::uint64_t input = 0U;
    std::uint64_t output = 0U;
    std::uint64_t transpose_offsets = 0U;
    std::uint64_t transpose_indices = 0U;
    std::uint64_t transpose_weights = 0U;
    std::uint64_t lower_ld = 0U;
    std::uint64_t upper_l = 0U;
    std::uint64_t inverse_diagonal = 0U;
};

static_assert(std::is_trivially_copyable_v<AxisPlanDescriptor>);
static_assert(std::is_trivially_copyable_v<IntegerConversionDescriptor>);
static_assert(std::is_trivially_copyable_v<AxisBatchJobDescriptor>);
static_assert(sizeof(AxisPlanDescriptor) == 12U);
static_assert(sizeof(IntegerConversionDescriptor) == 20U);
static_assert(sizeof(AxisBatchJobDescriptor) == 80U);

} // namespace dsmvc::cuda_kernel
