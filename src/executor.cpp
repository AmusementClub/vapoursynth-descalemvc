#include <dsmvc/engine.hpp>

#if defined(DSMVC_HAS_CUDA)
#include "cuda/cuda_executor.hpp"
#endif
#if defined(DSMVC_HAS_VULKAN)
#include "vulkan/vulkan_executor.hpp"
#endif

#include <memory>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace dsmvc {

void close_exported_plane(const ExportedPlane &plane) noexcept {
#if defined(_WIN32)
    // A KMT handle (0x4) is a global name, not a reference.
    if (plane.handle != 0 && plane.handle != -1 && plane.handle_type != 0x4) {
        CloseHandle(reinterpret_cast<HANDLE>(plane.handle));
    }
#else
    if (plane.handle >= 0) ::close(static_cast<int>(plane.handle));
#endif
}

struct Executor::Impl {
    Impl(BackendKind requested, CpuPath cpu_path)
        : backend(resolve_backend(requested)) {
        if (backend == BackendKind::cpu) {
            cpu = std::make_unique<CpuExecutor>(cpu_path);
            return;
        }
#if defined(DSMVC_HAS_CUDA)
        if (backend == BackendKind::cuda) {
            cuda = std::make_unique<cuda_detail::CudaExecutor>();
            return;
        }
#endif
#if defined(DSMVC_HAS_VULKAN)
        if (backend == BackendKind::vulkan) {
            vulkan = std::make_unique<vulkan_detail::VulkanExecutor>();
            return;
        }
#endif
        throw std::logic_error("resolved backend has no executor");
    }

#if defined(DSMVC_HAS_VULKAN)
    struct ResidentTag {};
    Impl(ResidentTag, const VulkanCoreDevice &device)
        : backend(BackendKind::vulkan), resident(ResidentKind::vulkan),
          vulkan(std::make_unique<vulkan_detail::VulkanExecutor>(device)) {}
#endif
#if defined(DSMVC_HAS_CUDA)
    struct CudaResidentTag {};
    Impl(CudaResidentTag, const std::uint8_t *device_uuid)
        : backend(BackendKind::cuda), resident(ResidentKind::cuda),
          cuda(std::make_unique<cuda_detail::CudaExecutor>(device_uuid)) {}
#endif

    BackendKind backend = BackendKind::cpu;
    ResidentKind resident = ResidentKind::none;
    std::unique_ptr<CpuExecutor> cpu;
#if defined(DSMVC_HAS_CUDA)
    std::unique_ptr<cuda_detail::CudaExecutor> cuda;
#endif
#if defined(DSMVC_HAS_VULKAN)
    std::unique_ptr<vulkan_detail::VulkanExecutor> vulkan;
#endif
};

Executor::Executor(BackendKind requested, CpuPath cpu_path)
    : impl_(std::make_shared<Impl>(requested, cpu_path)) {}

Executor::Executor(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

Executor::~Executor() = default;

Executor Executor::on_core_vulkan([[maybe_unused]] const VulkanCoreDevice &device) {
#if defined(DSMVC_HAS_VULKAN)
    return Executor(std::make_shared<Impl>(Impl::ResidentTag{}, device));
#else
    throw std::runtime_error("this build of dsmvc has no Vulkan support");
#endif
}

bool Executor::resident() const noexcept {
    return impl_->resident != ResidentKind::none;
}

ResidentKind Executor::resident_kind() const noexcept { return impl_->resident; }

Executor Executor::on_core_cuda([[maybe_unused]] const std::uint8_t device_uuid[16]) {
#if defined(DSMVC_HAS_CUDA)
    return Executor(std::make_shared<Impl>(Impl::CudaResidentTag{}, device_uuid));
#else
    throw std::runtime_error("this build of dsmvc has no CUDA support");
#endif
}

void Executor::inverse_rows_resident(
    [[maybe_unused]] const AxisPlan &plan, [[maybe_unused]] const GpuPlane &input,
    [[maybe_unused]] const GpuPlane &output,
    [[maybe_unused]] std::int32_t row_count) const {
#if defined(DSMVC_HAS_VULKAN)
    if (impl_->resident == ResidentKind::vulkan) {
        impl_->vulkan->inverse_rows_resident(plan, input, output, row_count);
        return;
    }
#endif
    throw std::logic_error("executor does not run on GPU frames");
}

void Executor::inverse_columns_resident(
    [[maybe_unused]] const AxisPlan &plan, [[maybe_unused]] const GpuPlane &input,
    [[maybe_unused]] const GpuPlane &output,
    [[maybe_unused]] std::int32_t column_count) const {
#if defined(DSMVC_HAS_VULKAN)
    if (impl_->resident == ResidentKind::vulkan) {
        impl_->vulkan->inverse_columns_resident(plan, input, output, column_count);
        return;
    }
#endif
    throw std::logic_error("executor does not run on GPU frames");
}


bool Executor::has_exported_semaphore(
    [[maybe_unused]] std::uint64_t semaphore) const {
#if defined(DSMVC_HAS_CUDA)
    if (impl_->resident == ResidentKind::cuda) {
        return impl_->cuda->has_exported_semaphore(semaphore);
    }
#endif
    return false;
}

bool Executor::import_exported_semaphore(
    [[maybe_unused]] std::uint64_t semaphore,
    int handle_type, std::intptr_t handle) const {
#if defined(DSMVC_HAS_CUDA)
    if (impl_->resident == ResidentKind::cuda) {
        return impl_->cuda->import_exported_semaphore(
            semaphore, handle_type, handle);
    }
#endif
    ExportedPlane unused;
    unused.handle = handle;
    unused.handle_type = handle_type;
    close_exported_plane(unused);
    return false;
}

void Executor::inverse_rows_exported(
    [[maybe_unused]] const AxisPlan &plan, [[maybe_unused]] const ExportedPlane &input,
    [[maybe_unused]] const ExportedPlane &output,
    [[maybe_unused]] std::int32_t row_count) const {
#if defined(DSMVC_HAS_CUDA)
    if (impl_->resident == ResidentKind::cuda) {
        impl_->cuda->inverse_rows_exported(plan, input, output, row_count);
        return;
    }
#endif
    throw std::logic_error("executor does not run on exported GPU frames");
}

void Executor::inverse_columns_exported(
    [[maybe_unused]] const AxisPlan &plan, [[maybe_unused]] const ExportedPlane &input,
    [[maybe_unused]] const ExportedPlane &output,
    [[maybe_unused]] std::int32_t column_count) const {
#if defined(DSMVC_HAS_CUDA)
    if (impl_->resident == ResidentKind::cuda) {
        impl_->cuda->inverse_columns_exported(plan, input, output, column_count);
        return;
    }
#endif
    throw std::logic_error("executor does not run on exported GPU frames");
}

void Executor::inverse_2d_exported(
    [[maybe_unused]] const AxisPlan &horizontal,
    [[maybe_unused]] const AxisPlan &vertical,
    [[maybe_unused]] const ExportedPlane &input,
    [[maybe_unused]] const ExportedPlane &output) const {
#if defined(DSMVC_HAS_CUDA)
    if (impl_->resident == ResidentKind::cuda) {
        impl_->cuda->inverse_2d_exported(horizontal, vertical, input, output);
        return;
    }
#endif
    throw std::logic_error("executor does not run on exported GPU frames");
}

void Executor::inverse_2d_u8_resident(
    [[maybe_unused]] const AxisPlan &horizontal,
    [[maybe_unused]] const AxisPlan &vertical,
    [[maybe_unused]] const GpuPlane &input,
    [[maybe_unused]] const GpuPlane &output,
    [[maybe_unused]] const IntegerConversion &conversion) const {
#if defined(DSMVC_HAS_VULKAN)
    if (impl_->resident == ResidentKind::vulkan) {
        impl_->vulkan->inverse_2d_u8_resident(
            horizontal, vertical, input, output, conversion);
        return;
    }
#endif
    throw std::logic_error("executor does not run on GPU frames");
}

void Executor::inverse_2d_u8_exported(
    [[maybe_unused]] const AxisPlan &horizontal,
    [[maybe_unused]] const AxisPlan &vertical,
    [[maybe_unused]] const ExportedPlane &input,
    [[maybe_unused]] const ExportedPlane &output,
    [[maybe_unused]] const IntegerConversion &conversion) const {
#if defined(DSMVC_HAS_CUDA)
    if (impl_->resident == ResidentKind::cuda) {
        impl_->cuda->inverse_2d_u8_exported(
            horizontal, vertical, input, output, conversion);
        return;
    }
#endif
    throw std::logic_error("executor does not run on exported GPU frames");
}

void Executor::inverse_2d_u16_resident(
    [[maybe_unused]] const AxisPlan &horizontal,
    [[maybe_unused]] const AxisPlan &vertical,
    [[maybe_unused]] const GpuPlane &input,
    [[maybe_unused]] const GpuPlane &output,
    [[maybe_unused]] const IntegerConversion &conversion) const {
#if defined(DSMVC_HAS_VULKAN)
    if (impl_->resident == ResidentKind::vulkan) {
        impl_->vulkan->inverse_2d_u16_resident(
            horizontal, vertical, input, output, conversion);
        return;
    }
#endif
    throw std::logic_error("executor does not run on GPU frames");
}

void Executor::inverse_2d_u16_exported(
    [[maybe_unused]] const AxisPlan &horizontal,
    [[maybe_unused]] const AxisPlan &vertical,
    [[maybe_unused]] const ExportedPlane &input,
    [[maybe_unused]] const ExportedPlane &output,
    [[maybe_unused]] const IntegerConversion &conversion) const {
#if defined(DSMVC_HAS_CUDA)
    if (impl_->resident == ResidentKind::cuda) {
        impl_->cuda->inverse_2d_u16_exported(
            horizontal, vertical, input, output, conversion);
        return;
    }
#endif
    throw std::logic_error("executor does not run on exported GPU frames");
}

void Executor::inverse_2d_resident(
    [[maybe_unused]] const AxisPlan &horizontal,
    [[maybe_unused]] const AxisPlan &vertical,
    [[maybe_unused]] const GpuPlane &input,
    [[maybe_unused]] const GpuPlane &output) const {
#if defined(DSMVC_HAS_VULKAN)
    if (impl_->resident == ResidentKind::vulkan) {
        impl_->vulkan->inverse_2d_resident(horizontal, vertical, input, output);
        return;
    }
#endif
    throw std::logic_error("executor does not run on GPU frames");
}

BackendKind Executor::backend() const noexcept { return impl_->backend; }

const char *Executor::name() const noexcept {
    if (impl_->cpu) return impl_->cpu->name();
#if defined(DSMVC_HAS_CUDA)
    if (impl_->cuda) return impl_->cuda->name();
#endif
#if defined(DSMVC_HAS_VULKAN)
    if (impl_->vulkan) return impl_->vulkan->name();
#endif
    return "unknown";
}

bool Executor::input_cache_enabled() const noexcept {
#if defined(DSMVC_HAS_VULKAN)
    if (impl_->vulkan) return impl_->vulkan->input_cache_enabled();
#endif
#if defined(DSMVC_HAS_CUDA)
    return impl_->cuda && impl_->cuda->input_cache_enabled();
#else
    return false;
#endif
}

CpuPlanPackingStats Executor::cpu_plan_packing_stats() const noexcept {
    return impl_->cpu ? impl_->cpu->packing_stats() : CpuPlanPackingStats{};
}

void Executor::prepare(std::shared_ptr<const AxisPlan> plan) const {
    if (impl_->cpu) {
        impl_->cpu->prepare(std::move(plan));
        return;
    }
#if defined(DSMVC_HAS_CUDA)
    if (impl_->cuda) {
        impl_->cuda->prepare(std::move(plan));
    }
#endif
#if defined(DSMVC_HAS_VULKAN)
    if (impl_->vulkan) {
        impl_->vulkan->prepare(std::move(plan));
    }
#endif
}

void Executor::defer(std::shared_ptr<const AxisPlan> plan) const {
    if (impl_->cpu) {
        impl_->cpu->defer(std::move(plan));
        return;
    }
    prepare(std::move(plan));
}

void Executor::seal() const {
    if (impl_->cpu) {
        impl_->cpu->seal();
        return;
    }
#if defined(DSMVC_HAS_CUDA)
    if (impl_->cuda) impl_->cuda->seal();
#endif
#if defined(DSMVC_HAS_VULKAN)
    if (impl_->vulkan) impl_->vulkan->seal();
#endif
}

void Executor::inverse_rows(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t row_count,
    [[maybe_unused]] std::shared_ptr<const void> input_lifetime) const {
    if (impl_->cpu) {
        impl_->cpu->inverse_rows(
            plan, input, input_row_stride, output, output_row_stride, row_count);
        return;
    }
#if defined(DSMVC_HAS_CUDA)
    if (impl_->cuda) {
        impl_->cuda->inverse_rows(
            plan, input, input_row_stride, output, output_row_stride, row_count,
            std::move(input_lifetime));
    }
#endif
#if defined(DSMVC_HAS_VULKAN)
    if (impl_->vulkan) {
        impl_->vulkan->inverse_rows(
            plan, input, input_row_stride, output, output_row_stride, row_count,
            std::move(input_lifetime));
    }
#endif
}

void Executor::inverse_columns(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t column_count,
    [[maybe_unused]] std::shared_ptr<const void> input_lifetime) const {
    if (impl_->cpu) {
        impl_->cpu->inverse_columns(
            plan, input, input_row_stride, output, output_row_stride,
            column_count);
        return;
    }
#if defined(DSMVC_HAS_CUDA)
    if (impl_->cuda) {
        impl_->cuda->inverse_columns(
            plan, input, input_row_stride, output, output_row_stride, column_count,
            std::move(input_lifetime));
    }
#endif
#if defined(DSMVC_HAS_VULKAN)
    if (impl_->vulkan) {
        impl_->vulkan->inverse_columns(
            plan, input, input_row_stride, output, output_row_stride,
            column_count, std::move(input_lifetime));
    }
#endif
}

bool Executor::inverse_2d_frame(
    [[maybe_unused]] SampleKind kind,
    [[maybe_unused]] std::span<const FramePlane2D> planes,
    [[maybe_unused]] const std::shared_ptr<const void> &input_lifetime) const {
#if defined(DSMVC_HAS_VULKAN)
    if (impl_->vulkan) {
        return impl_->vulkan->inverse_2d_frame(
            kind, planes, impl_->resident == ResidentKind::vulkan,
            input_lifetime);
    }
#endif
    return false;
}

void Executor::inverse_2d(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    [[maybe_unused]] std::shared_ptr<const void> input_lifetime) const {
    if (impl_->cpu) {
        impl_->cpu->inverse_2d(
            horizontal, vertical, input, input_row_stride,
            output, output_row_stride);
        return;
    }
#if defined(DSMVC_HAS_CUDA)
    if (impl_->cuda) {
        impl_->cuda->inverse_2d(
            horizontal, vertical, input, input_row_stride,
            output, output_row_stride, std::move(input_lifetime));
    }
#endif
#if defined(DSMVC_HAS_VULKAN)
    if (impl_->vulkan) {
        impl_->vulkan->inverse_2d(
            horizontal, vertical, input, input_row_stride, output,
            output_row_stride, std::move(input_lifetime));
    }
#endif
}

void Executor::inverse_2d_u8(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::uint8_t *input, std::ptrdiff_t input_row_stride,
    std::uint8_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion,
    [[maybe_unused]] std::shared_ptr<const void> input_lifetime) const {
    if (impl_->cpu) {
        impl_->cpu->inverse_2d_u8(
            horizontal, vertical, input, input_row_stride,
            output, output_row_stride, conversion);
        return;
    }
#if defined(DSMVC_HAS_CUDA)
    if (impl_->cuda) {
        impl_->cuda->inverse_2d_u8(
            horizontal, vertical, input, input_row_stride,
            output, output_row_stride, conversion, std::move(input_lifetime));
    }
#endif
#if defined(DSMVC_HAS_VULKAN)
    if (impl_->vulkan) {
        impl_->vulkan->inverse_2d_u8(
            horizontal, vertical, input, input_row_stride, output,
            output_row_stride, conversion, std::move(input_lifetime));
    }
#endif
}

void Executor::inverse_2d_u16(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::uint16_t *input, std::ptrdiff_t input_row_stride,
    std::uint16_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion,
    [[maybe_unused]] std::shared_ptr<const void> input_lifetime) const {
    if (impl_->cpu) {
        impl_->cpu->inverse_2d_u16(
            horizontal, vertical, input, input_row_stride,
            output, output_row_stride, conversion);
        return;
    }
#if defined(DSMVC_HAS_CUDA)
    if (impl_->cuda) {
        impl_->cuda->inverse_2d_u16(
            horizontal, vertical, input, input_row_stride,
            output, output_row_stride, conversion, std::move(input_lifetime));
    }
#endif
#if defined(DSMVC_HAS_VULKAN)
    if (impl_->vulkan) {
        impl_->vulkan->inverse_2d_u16(
            horizontal, vertical, input, input_row_stride, output,
            output_row_stride, conversion, std::move(input_lifetime));
    }
#endif
}

void Executor::inverse_2d_u8_streamed(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::uint8_t *input, std::ptrdiff_t input_row_stride,
    std::uint8_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion,
    [[maybe_unused]] std::shared_ptr<const void> input_lifetime) const {
    if (impl_->cpu) {
        impl_->cpu->inverse_2d_u8_streamed(
            horizontal, vertical, input, input_row_stride,
            output, output_row_stride, conversion);
        return;
    }
#if defined(DSMVC_HAS_CUDA)
    if (impl_->cuda) {
        impl_->cuda->inverse_2d_u8(
            horizontal, vertical, input, input_row_stride,
            output, output_row_stride, conversion, std::move(input_lifetime));
    }
#endif
#if defined(DSMVC_HAS_VULKAN)
    if (impl_->vulkan) {
        impl_->vulkan->inverse_2d_u8(
            horizontal, vertical, input, input_row_stride, output,
            output_row_stride, conversion, std::move(input_lifetime));
    }
#endif
}

void Executor::inverse_2d_u16_streamed(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::uint16_t *input, std::ptrdiff_t input_row_stride,
    std::uint16_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion,
    [[maybe_unused]] std::shared_ptr<const void> input_lifetime) const {
    if (impl_->cpu) {
        impl_->cpu->inverse_2d_u16_streamed(
            horizontal, vertical, input, input_row_stride,
            output, output_row_stride, conversion);
        return;
    }
#if defined(DSMVC_HAS_CUDA)
    if (impl_->cuda) {
        impl_->cuda->inverse_2d_u16(
            horizontal, vertical, input, input_row_stride,
            output, output_row_stride, conversion, std::move(input_lifetime));
    }
#endif
#if defined(DSMVC_HAS_VULKAN)
    if (impl_->vulkan) {
        impl_->vulkan->inverse_2d_u16(
            horizontal, vertical, input, input_row_stride, output,
            output_row_stride, conversion, std::move(input_lifetime));
    }
#endif
}

} // namespace dsmvc
