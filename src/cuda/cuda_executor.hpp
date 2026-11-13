#pragma once

#include <dsmvc/engine.hpp>

#include <memory>

namespace dsmvc::cuda_detail {

[[nodiscard]] bool backend_available() noexcept;

class CudaExecutor {
public:
    CudaExecutor();
    // For GPU frames: the CUDA device in use must be the GPU with this UUID.
    explicit CudaExecutor(const std::uint8_t device_uuid[16]);
    ~CudaExecutor();
    CudaExecutor(const CudaExecutor &) noexcept = default;
    CudaExecutor &operator=(const CudaExecutor &) noexcept = default;
    CudaExecutor(CudaExecutor &&) noexcept = default;
    CudaExecutor &operator=(CudaExecutor &&) noexcept = default;

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

    // GPU frames exported by the core (see ExportedPlane).
    [[nodiscard]] bool has_exported_semaphore(std::uint64_t semaphore) const;
    [[nodiscard]] bool import_exported_semaphore(
        std::uint64_t semaphore, int handle_type, std::intptr_t handle) const;
    void inverse_rows_exported(
        const AxisPlan &plan, const ExportedPlane &input,
        const ExportedPlane &output, std::int32_t row_count) const;
    void inverse_columns_exported(
        const AxisPlan &plan, const ExportedPlane &input,
        const ExportedPlane &output, std::int32_t column_count) const;
    void inverse_2d_exported(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const ExportedPlane &input, const ExportedPlane &output) const;
    void inverse_2d_u8_exported(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const ExportedPlane &input, const ExportedPlane &output,
        const IntegerConversion &conversion) const;
    void inverse_2d_u16_exported(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const ExportedPlane &input, const ExportedPlane &output,
        const IntegerConversion &conversion) const;

    // Device pointers of the planes of a GPU frame pair.
    struct ResidentDevice {
        const std::byte *source = nullptr;
        std::byte *result = nullptr;
        // The source's producer pair, waited for on the stream; null when the
        // caller waited on the host.
        void *ready_semaphore = nullptr; // cudaExternalSemaphore_t
        std::uint64_t ready_value = 0;
    };

private:
    [[nodiscard]] ResidentDevice resident_device(
        const ExportedPlane &input, const std::byte *source,
        std::byte *result) const;
    template <class Sample>
    void inverse_2d_integer_exported(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const ExportedPlane &input, const ExportedPlane &output,
        const IntegerConversion &conversion) const;
    void inverse_rows_impl(
        const AxisPlan &plan, const float *input,
        std::ptrdiff_t input_row_stride, float *output,
        std::ptrdiff_t output_row_stride, std::int32_t row_count,
        std::shared_ptr<const void> input_lifetime,
        const ResidentDevice *resident) const;
    void inverse_columns_impl(
        const AxisPlan &plan, const float *input,
        std::ptrdiff_t input_row_stride, float *output,
        std::ptrdiff_t output_row_stride, std::int32_t column_count,
        std::shared_ptr<const void> input_lifetime,
        const ResidentDevice *resident) const;

    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace dsmvc::cuda_detail
