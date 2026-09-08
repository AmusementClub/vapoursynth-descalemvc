#include <dsmvc/engine.hpp>
#ifdef DSMVC_TEST_METAL
#include "metal_float_executor_apple.hpp"
#else
#include "vulkan/vulkan_executor.hpp"
#endif

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
constexpr int sources = 64;
constexpr int destinations = 18;
constexpr int vectors = 33;
constexpr int padding = 3;
constexpr float sentinel = -777.0F;

std::shared_ptr<const dsmvc::AxisPlan> sparse_plan(int bandwidth) {
    auto plan = std::make_shared<dsmvc::AxisPlan>();
    plan->source_size = sources;
    plan->destination_size = destinations;
    plan->support = 18;
    plan->half_bandwidth = bandwidth;
    plan->active_length = destinations;
    plan->transpose_offsets.push_back(0);
    // Isolate RHS arithmetic with an identity solve. Every sparse count 0..17
    // appears, including all selected counts and their immediate neighbours.
    for (int count = 0; count < destinations; ++count) {
        for (int tap = 0; tap < count; ++tap) {
            plan->transpose_indices.push_back(tap * 3 + count % 13);
            const auto bits = 0x3e800000U
                | (((tap + 1U) * 2246822519U + count) & 0x007fffffU)
                | ((tap & 1U) << 31U);
            plan->transpose_weights.push_back(std::bit_cast<float>(bits));
        }
        plan->transpose_offsets.push_back(plan->transpose_indices.size());
    }
    plan->lower_ld.assign(bandwidth * destinations, 0.0F);
    plan->upper_l.assign(bandwidth * destinations, 0.0F);
    plan->inverse_diagonal.assign(destinations, 1.0F);
    if (!plan->valid()) throw std::runtime_error("invalid sparse fixture");
    return plan;
}

void test(int bandwidth, bool vertical) {
    const auto plan = sparse_plan(bandwidth);
    const int input_stride = (vertical ? vectors : sources) + padding;
    const int output_stride = (vertical ? vectors : destinations) + padding;
    const int input_rows = vertical ? sources : vectors;
    const int output_rows = vertical ? destinations : vectors;
    constexpr int guard = 16;
    std::vector<float> input(input_rows * input_stride);
    for (std::uint32_t i = 0; i < input.size(); ++i) {
        input[i] = std::bit_cast<float>(0x3f000000U | ((i * 2654435761U) & 0x007fffffU));
    }
    std::vector<float> expected(2 * guard + output_rows * output_stride, sentinel);
    for (int vector = 0; vector < vectors; ++vector) {
        for (int index = 0; index < destinations; ++index) {
            float sum = 0.0F;
            for (auto entry = plan->transpose_offsets[index];
                 entry < plan->transpose_offsets[index + 1]; ++entry) {
                const auto source = plan->transpose_indices[entry];
                sum = std::fma(plan->transpose_weights[entry],
                    input[vertical ? source * input_stride + vector
                                   : vector * input_stride + source], sum);
            }
            expected[guard + (vertical ? index * output_stride + vector
                                      : vector * output_stride + index)] = sum;
        }
    }
    std::vector<float> output(expected.size(), sentinel);
#ifdef DSMVC_TEST_METAL
    dsmvc::AxisRequest request;
    request.source_size = request.destination_size = vectors;
    request.active_length = vectors;
    request.kernel.kind = dsmvc::KernelKind::bilinear;
    request.f64_mode = dsmvc::F64Mode::float32_only;
    const auto identity = std::make_shared<const dsmvc::AxisPlan>(dsmvc::build_axis_plan(request));
    dsmvc::experimental::MetalFloatExecutor executor(
        vertical ? identity : plan, vertical ? plan : identity, 2U);
    std::vector<float> second(expected.size(), sentinel);
    std::array frames{
        dsmvc::experimental::MetalFloatFrame{input.data(), input_stride * 4,
                                             output.data() + guard, output_stride * 4},
        dsmvc::experimental::MetalFloatFrame{input.data(), input_stride * 4,
                                             second.data() + guard, output_stride * 4},
    };
    executor.execute(frames);
    if (output != second) throw std::runtime_error("Metal batch output mismatch");
#else
    dsmvc::vulkan_detail::VulkanExecutor executor;
    executor.prepare(plan);
    executor.seal();
    if (vertical) {
        executor.inverse_columns(*plan, input.data(), input_stride,
                                 output.data() + guard, output_stride, vectors, {});
    } else {
        executor.inverse_rows(*plan, input.data(), input_stride,
                              output.data() + guard, output_stride, vectors, {});
    }
#endif
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(output[i]) != std::bit_cast<std::uint32_t>(expected[i])) {
            throw std::runtime_error("RHS mismatch or guard write: bandwidth="
                + std::to_string(bandwidth) + " vertical=" + std::to_string(vertical)
                + " index=" + std::to_string(i));
        }
    }
}
} // namespace

int main() {
    try {
        for (const int band : {0, 1, 3, 5, 7, 9, 11}) {
            test(band, false);
            test(band, true);
        }
        std::cout << "RHS tests passed: 14 routes/bands, counts 0..17, sparse indices, 33-vector tails, padded output and guards\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
