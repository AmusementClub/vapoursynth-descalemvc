#include "vulkan_f64.hpp"

#include <cstdint>
#include <limits>
#include <source_location>
#include <stdexcept>
#include <string>

using dsmvc::vulkan_detail::VulkanF64WordLayout;
using dsmvc::vulkan_detail::VulkanFloat64Capabilities;
using dsmvc::vulkan_detail::VulkanPlanWordLayout;

namespace {

void require(bool condition, const std::source_location location =
                 std::source_location::current()) {
    if (!condition) {
        throw std::runtime_error(
            "Vulkan Float64 policy check failed at line "
            + std::to_string(location.line()));
    }
}

} // namespace

int main() {
    VulkanFloat64Capabilities supported{
        true, true, true, true, 256U, 1U << 27U};
    require(supported.strict_supported());
    require(supported.missing_requirements().empty());
    require(supported.requirement_error().empty());

    VulkanFloat64Capabilities missing{};
    require(!missing.strict_supported());
    require(missing.requirement_error()
           == "Vulkan Float64 capability contract failed: missing shaderFloat64, "
              "shaderRoundingModeRTEFloat64, "
              "shaderSignedZeroInfNanPreserveFloat64");
    missing = supported;
    missing.denorm_preserve_float64 = false;
    require(missing.strict_supported());
    require(missing.missing_requirements().empty());
    require(missing.requirement_error().empty());

    const auto check_single_missing = [&](auto clear, const char *name) {
        auto capabilities = supported;
        clear(capabilities);
        require(!capabilities.strict_supported());
        require(capabilities.missing_requirements() == name);
        require(capabilities.requirement_error()
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
    require(layout.add_words(3U, "offsets") == 0U);
    require(layout.add_doubles(2U, "weights") == 4U);
    require(layout.add_words(1U, "tail") == 8U);
    require(layout.words() == 9U);
    require(layout.bytes() == 36U);

    const auto retained = VulkanPlanWordLayout::make(
        4U, 5U, 5U, 6U, 3U, 9U, true, true);
    require(retained.offsets == 0U);
    require(retained.indices == 4U);
    require(retained.weights_f32 == 9U);
    require(retained.diagonal_f32 == 26U);
    require((retained.weights_f64 & 1U) == 0U);
    require((retained.lower_f64 & 1U) == 0U);
    require(retained.lower_f64 == retained.upper_f64);
    require((retained.diagonal_f64 & 1U) == 0U);
    require(retained.storage_bytes()
           == static_cast<std::size_t>(retained.storage_words) * 4U);

    const auto promoted = VulkanPlanWordLayout::make(
        4U, 5U, 5U, 6U, 3U, 0U, false, true);
    require(promoted.lower_f64 != promoted.upper_f64);
    require((promoted.lower_f64 & 1U) == 0U);
    require((promoted.upper_f64 & 1U) == 0U);

    const auto float32_only = VulkanPlanWordLayout::make(
        4U, 5U, 5U, 6U, 3U, 0U, false, false);
    require(float32_only.storage_words == 29U);
    require(float32_only.weights_f64 == 0U);

    bool overflow_rejected = false;
    try {
        VulkanF64WordLayout overflow;
        (void)overflow.add_doubles(
            std::numeric_limits<std::size_t>::max(), "overflow");
    } catch (const std::length_error &) {
        overflow_rejected = true;
    }
    require(overflow_rejected);
}
