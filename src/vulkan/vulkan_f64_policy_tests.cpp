#include "vulkan_f64.hpp"

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

using dsmvc::vulkan_detail::VulkanF64WordLayout;
using dsmvc::vulkan_detail::VulkanFloat64Capabilities;
using dsmvc::vulkan_detail::VulkanPlanWordLayout;

int main() {
    VulkanFloat64Capabilities supported{
        true, true, true, true, 256U, 1U << 27U};
    assert(supported.strict_supported());
    assert(supported.missing_requirements().empty());
    assert(supported.requirement_error().empty());

    VulkanFloat64Capabilities missing{};
    assert(!missing.strict_supported());
    assert(missing.requirement_error()
           == "Vulkan Float64 capability contract failed: missing shaderFloat64, "
              "shaderRoundingModeRTEFloat64, "
              "shaderSignedZeroInfNanPreserveFloat64");
    missing = supported;
    missing.denorm_preserve_float64 = false;
    assert(missing.strict_supported());
    assert(missing.missing_requirements().empty());
    assert(missing.requirement_error().empty());

    const auto check_single_missing = [&](auto clear, const char *name) {
        auto capabilities = supported;
        clear(capabilities);
        assert(!capabilities.strict_supported());
        assert(capabilities.missing_requirements() == name);
        assert(capabilities.requirement_error()
               == std::string{"Vulkan Float64 capability contract failed: missing "}
                    + name);
    };
    check_single_missing(
        [](auto &value) { value.shader_float64 = false; }, "shaderFloat64");
    check_single_missing(
        [](auto &value) { value.rounding_mode_rte_float64 = false; },
        "shaderRoundingModeRTEFloat64");
    check_single_missing(
        [](auto &value) {
            value.signed_zero_inf_nan_preserve_float64 = false;
        },
        "shaderSignedZeroInfNanPreserveFloat64");
    VulkanF64WordLayout layout;
    assert(layout.add_words(3U, "offsets") == 0U);
    assert(layout.add_doubles(2U, "weights") == 4U);
    assert(layout.add_words(1U, "tail") == 8U);
    assert(layout.words() == 9U);
    assert(layout.bytes() == 36U);

    const auto retained = VulkanPlanWordLayout::make(
        4U, 5U, 5U, 6U, 3U, 9U, true, true);
    assert(retained.offsets == 0U);
    assert(retained.indices == 4U);
    assert(retained.weights_f32 == 9U);
    assert(retained.diagonal_f32 == 26U);
    assert((retained.weights_f64 & 1U) == 0U);
    assert((retained.lower_f64 & 1U) == 0U);
    assert(retained.lower_f64 == retained.upper_f64);
    assert((retained.diagonal_f64 & 1U) == 0U);
    assert(retained.storage_bytes()
           == static_cast<std::size_t>(retained.storage_words) * 4U);

    const auto promoted = VulkanPlanWordLayout::make(
        4U, 5U, 5U, 6U, 3U, 0U, false, true);
    assert(promoted.lower_f64 != promoted.upper_f64);
    assert((promoted.lower_f64 & 1U) == 0U);
    assert((promoted.upper_f64 & 1U) == 0U);

    const auto float32_only = VulkanPlanWordLayout::make(
        4U, 5U, 5U, 6U, 3U, 0U, false, false);
    assert(float32_only.storage_words == 29U);
    assert(float32_only.weights_f64 == 0U);

    bool overflow_rejected = false;
    try {
        VulkanF64WordLayout overflow;
        (void)overflow.add_doubles(
            std::numeric_limits<std::size_t>::max(), "overflow");
    } catch (const std::length_error &) {
        overflow_rejected = true;
    }
    assert(overflow_rejected);
}
