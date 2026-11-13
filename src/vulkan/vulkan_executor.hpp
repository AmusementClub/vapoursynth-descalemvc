#pragma once

#include <dsmvc/engine.hpp>

#include "vulkan_f64.hpp"

#include <memory>
#include <span>
#include <string>

namespace dsmvc::vulkan_detail {

[[nodiscard]] bool backend_available() noexcept;
void require_backend_available();
[[nodiscard]] VulkanFloat64Capabilities selected_float64_capabilities();
[[nodiscard]] std::string selected_float64_capability_report();

class VulkanExecutor {
public:
    VulkanExecutor();
    // Runs on a core's Vulkan device (GPU frames); see VulkanCoreDevice.
    explicit VulkanExecutor(const VulkanCoreDevice &device);
    ~VulkanExecutor();
    VulkanExecutor(const VulkanExecutor &) noexcept = default;
    VulkanExecutor &operator=(const VulkanExecutor &) noexcept = default;
    VulkanExecutor(VulkanExecutor &&) noexcept = default;
    VulkanExecutor &operator=(VulkanExecutor &&) noexcept = default;

    [[nodiscard]] const char *name() const noexcept;
    [[nodiscard]] bool input_cache_enabled() const noexcept;

    void prepare(std::shared_ptr<const AxisPlan> plan) const;
    void seal() const;

    void inverse_rows(const AxisPlan &plan,
                      const float *input, std::ptrdiff_t input_row_stride,
                      float *output, std::ptrdiff_t output_row_stride,
                      std::int32_t row_count,
                      std::shared_ptr<const void> input_lifetime) const;
    void inverse_columns(const AxisPlan &plan,
                         const float *input, std::ptrdiff_t input_row_stride,
                         float *output, std::ptrdiff_t output_row_stride,
                         std::int32_t column_count,
                         std::shared_ptr<const void> input_lifetime) const;
    void inverse_2d(const AxisPlan &horizontal, const AxisPlan &vertical,
                    const float *input, std::ptrdiff_t input_row_stride,
                    float *output, std::ptrdiff_t output_row_stride,
                    std::shared_ptr<const void> input_lifetime) const;
    void inverse_2d_u8(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const std::uint8_t *input, std::ptrdiff_t input_row_stride,
        std::uint8_t *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion,
        std::shared_ptr<const void> input_lifetime) const;
    void inverse_2d_u16(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const std::uint16_t *input, std::ptrdiff_t input_row_stride,
        std::uint16_t *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion,
        std::shared_ptr<const void> input_lifetime) const;

    [[nodiscard]] bool inverse_2d_frame(
        SampleKind kind, std::span<const FramePlane2D> planes, bool gpu_frame,
        const std::shared_ptr<const void> &input_lifetime) const;

    void inverse_rows_resident(
        const AxisPlan &plan, const GpuPlane &input, const GpuPlane &output,
        std::int32_t row_count) const;
    void inverse_columns_resident(
        const AxisPlan &plan, const GpuPlane &input, const GpuPlane &output,
        std::int32_t column_count) const;
    void inverse_2d_resident(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const GpuPlane &input, const GpuPlane &output) const;
    void inverse_2d_u8_resident(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const GpuPlane &input, const GpuPlane &output,
        const IntegerConversion &conversion) const;
    void inverse_2d_u16_resident(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const GpuPlane &input, const GpuPlane &output,
        const IntegerConversion &conversion) const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace dsmvc::vulkan_detail
