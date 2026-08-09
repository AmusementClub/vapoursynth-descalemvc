#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace dsmvc::vulkan_detail {

struct VulkanFloat64Capabilities {
    bool shader_float64 = false;
    bool rounding_mode_rte_float64 = false;
    bool signed_zero_inf_nan_preserve_float64 = false;
    bool denorm_preserve_float64 = false;
    std::uint64_t min_storage_buffer_offset_alignment = 0U;
    std::uint64_t max_storage_buffer_range = 0U;

    [[nodiscard]] bool strict_supported() const noexcept {
        return shader_float64 && rounding_mode_rte_float64
            && signed_zero_inf_nan_preserve_float64
            && denorm_preserve_float64;
    }

    [[nodiscard]] std::string missing_requirements() const {
        std::string result;
        const auto add = [&](std::string_view name) {
            if (!result.empty()) result += ", ";
            result += name;
        };
        if (!shader_float64) add("shaderFloat64");
        if (!rounding_mode_rte_float64) {
            add("shaderRoundingModeRTEFloat64");
        }
        if (!signed_zero_inf_nan_preserve_float64) {
            add("shaderSignedZeroInfNanPreserveFloat64");
        }
        if (!denorm_preserve_float64) add("shaderDenormPreserveFloat64");
        return result;
    }

    [[nodiscard]] std::string requirement_error() const {
        return strict_supported() ? std::string{}
            : "Vulkan Float64 capability contract failed: missing "
                + missing_requirements();
    }
};

class VulkanF64WordLayout {
public:
    [[nodiscard]] std::uint32_t add_words(
        std::size_t count, std::string_view label) {
        return add(count, 1U, label);
    }

    [[nodiscard]] std::uint32_t add_doubles(
        std::size_t count, std::string_view label) {
        words_ = align_words(words_, 2U, label);
        if (count > std::numeric_limits<std::size_t>::max() / 2U) {
            throw std::length_error(std::string{label} + " exceeds addressable memory");
        }
        return add(count * 2U, 2U, label);
    }

    [[nodiscard]] std::size_t words() const noexcept { return words_; }

    [[nodiscard]] std::size_t bytes() const {
        if (words_ > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
            throw std::length_error("Vulkan Float64 layout exceeds addressable memory");
        }
        return words_ * sizeof(std::uint32_t);
    }

private:
    [[nodiscard]] static std::size_t align_words(
        std::size_t value, std::size_t alignment, std::string_view label) {
        const std::size_t remainder = value % alignment;
        if (remainder == 0U) return value;
        const std::size_t padding = alignment - remainder;
        if (padding > std::numeric_limits<std::size_t>::max() - value) {
            throw std::length_error(std::string{label} + " exceeds addressable memory");
        }
        return value + padding;
    }

    [[nodiscard]] std::uint32_t add(
        std::size_t count, std::size_t required_alignment,
        std::string_view label) {
        if (words_ % required_alignment != 0U) {
            throw std::logic_error(
                std::string{label} + " has an invalid Vulkan word alignment");
        }
        if (count > std::numeric_limits<std::size_t>::max() - words_) {
            throw std::length_error(std::string{label} + " exceeds addressable memory");
        }
        const std::size_t offset = words_;
        words_ += count;
        if (offset > std::numeric_limits<std::uint32_t>::max()
            || words_ > std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error(
                std::string{label} + " exceeds Vulkan shader ABI limits");
        }
        return static_cast<std::uint32_t>(offset);
    }

    std::size_t words_ = 0U;
};

struct VulkanPlanWordLayout {
    std::uint32_t offsets = 0U;
    std::uint32_t indices = 0U;
    std::uint32_t weights_f32 = 0U;
    std::uint32_t lower_f32 = 0U;
    std::uint32_t upper_f32 = 0U;
    std::uint32_t diagonal_f32 = 0U;
    std::uint32_t weights_f64 = 0U;
    std::uint32_t lower_f64 = 0U;
    std::uint32_t upper_f64 = 0U;
    std::uint32_t diagonal_f64 = 0U;
    std::uint32_t storage_words = 0U;

    [[nodiscard]] static VulkanPlanWordLayout make(
        std::size_t offset_count, std::size_t index_count,
        std::size_t weight_count, std::size_t factor_count,
        std::size_t diagonal_count, std::size_t raw_factor_count,
        bool retained_float64, bool include_float64) {
        VulkanF64WordLayout builder;
        VulkanPlanWordLayout result;
        result.offsets = builder.add_words(offset_count, "transpose offsets");
        result.indices = builder.add_words(index_count, "transpose indices");
        result.weights_f32 = builder.add_words(weight_count, "Float32 weights");
        result.lower_f32 = builder.add_words(factor_count, "Float32 lower factors");
        result.upper_f32 = builder.add_words(factor_count, "Float32 upper factors");
        result.diagonal_f32 = builder.add_words(
            diagonal_count, "Float32 inverse diagonal");
        if (include_float64) {
            result.weights_f64 = builder.add_doubles(
                weight_count, "Float64 weights");
            result.lower_f64 = builder.add_doubles(
                retained_float64 ? raw_factor_count : factor_count,
                "Float64 lower factors");
            if (retained_float64) {
                result.upper_f64 = result.lower_f64;
            } else {
                result.upper_f64 = builder.add_doubles(
                    factor_count, "Float64 upper factors");
            }
            result.diagonal_f64 = builder.add_doubles(
                diagonal_count, "Float64 inverse diagonal");
        }
        if (builder.words() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error(
                "Vulkan plan layout exceeds Vulkan shader ABI limits");
        }
        result.storage_words = static_cast<std::uint32_t>(builder.words());
        return result;
    }

    [[nodiscard]] std::size_t storage_bytes() const noexcept {
        return static_cast<std::size_t>(storage_words) * sizeof(std::uint32_t);
    }
};

} // namespace dsmvc::vulkan_detail
