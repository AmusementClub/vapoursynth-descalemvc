#include "vulkan_f64.hpp"

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

using dsmvc::vulkan_detail::VulkanF64WordLayout;
using dsmvc::vulkan_detail::VulkanFloat64Capabilities;

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
              "shaderSignedZeroInfNanPreserveFloat64, "
              "shaderDenormPreserveFloat64");
    missing = supported;
    missing.denorm_preserve_float64 = false;
    assert(missing.requirement_error()
           == "Vulkan Float64 capability contract failed: missing "
              "shaderDenormPreserveFloat64");

    VulkanF64WordLayout layout;
    assert(layout.add_words(3U, "offsets") == 0U);
    assert(layout.add_doubles(2U, "weights") == 4U);
    assert(layout.add_words(1U, "tail") == 8U);
    assert(layout.words() == 9U);
    assert(layout.bytes() == 36U);

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
