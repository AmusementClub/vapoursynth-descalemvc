#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <dsmvc/engine.hpp>

namespace dsmvc::detail {

struct PackedCpuPlan {
    std::shared_ptr<const AxisPlan> axis;
    const AxisPlan *identity = nullptr;
    std::int32_t padded_source_size = 0;
    std::int32_t padded_destination_size = 0;
    std::int32_t weights_columns = 0;
    std::vector<std::int32_t> weights_left;
    std::vector<std::int32_t> weights_right;
    std::vector<float> weights;
    std::vector<float> lower_ld;
    std::vector<float> upper_l;
    std::vector<float> inverse_diagonal;
};

[[nodiscard]] PackedCpuPlan pack_cpu_plan(
    std::shared_ptr<const AxisPlan> plan, const AxisPlan *identity = nullptr);

} // namespace dsmvc::detail
