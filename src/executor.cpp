#include <dsmvc/engine.hpp>

#if defined(DSMVC_HAS_CUDA)
#include "cuda/cuda_executor.hpp"
#endif

#include <memory>
#include <stdexcept>

namespace dsmvc {

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
        throw std::logic_error("resolved backend has no executor");
    }

    BackendKind backend = BackendKind::cpu;
    std::unique_ptr<CpuExecutor> cpu;
#if defined(DSMVC_HAS_CUDA)
    std::unique_ptr<cuda_detail::CudaExecutor> cuda;
#endif
};

Executor::Executor(BackendKind requested, CpuPath cpu_path)
    : impl_(std::make_shared<Impl>(requested, cpu_path)) {}

Executor::~Executor() = default;

BackendKind Executor::backend() const noexcept { return impl_->backend; }

const char *Executor::name() const noexcept {
    if (impl_->cpu) return impl_->cpu->name();
#if defined(DSMVC_HAS_CUDA)
    if (impl_->cuda) return impl_->cuda->name();
#endif
    return "unknown";
}

bool Executor::input_cache_enabled() const noexcept {
#if defined(DSMVC_HAS_CUDA)
    return impl_->cuda && impl_->cuda->input_cache_enabled();
#else
    return false;
#endif
}

void Executor::prepare(std::shared_ptr<const AxisPlan> plan) const {
    if (impl_->cpu) {
        impl_->cpu->prepare(std::move(plan));
        return;
    }
#if defined(DSMVC_HAS_CUDA)
    impl_->cuda->prepare(std::move(plan));
#endif
}

void Executor::seal() const {
    if (impl_->cpu) {
        impl_->cpu->seal();
        return;
    }
#if defined(DSMVC_HAS_CUDA)
    impl_->cuda->seal();
#endif
}

void Executor::inverse_rows(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t row_count,
    std::shared_ptr<const void> input_lifetime) const {
    if (impl_->cpu) {
        impl_->cpu->inverse_rows(
            plan, input, input_row_stride, output, output_row_stride, row_count);
        return;
    }
#if defined(DSMVC_HAS_CUDA)
    impl_->cuda->inverse_rows(
        plan, input, input_row_stride, output, output_row_stride, row_count,
        std::move(input_lifetime));
#endif
}

void Executor::inverse_columns(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t column_count,
    std::shared_ptr<const void> input_lifetime) const {
    if (impl_->cpu) {
        impl_->cpu->inverse_columns(
            plan, input, input_row_stride, output, output_row_stride,
            column_count);
        return;
    }
#if defined(DSMVC_HAS_CUDA)
    impl_->cuda->inverse_columns(
        plan, input, input_row_stride, output, output_row_stride, column_count,
        std::move(input_lifetime));
#endif
}

void Executor::inverse_2d(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::shared_ptr<const void> input_lifetime) const {
    if (impl_->cpu) {
        impl_->cpu->inverse_2d(
            horizontal, vertical, input, input_row_stride,
            output, output_row_stride);
        return;
    }
#if defined(DSMVC_HAS_CUDA)
    impl_->cuda->inverse_2d(
        horizontal, vertical, input, input_row_stride,
        output, output_row_stride, std::move(input_lifetime));
#endif
}

void Executor::inverse_2d_u8(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::uint8_t *input, std::ptrdiff_t input_row_stride,
    std::uint8_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion,
    std::shared_ptr<const void> input_lifetime) const {
    if (impl_->cpu) {
        impl_->cpu->inverse_2d_u8(
            horizontal, vertical, input, input_row_stride,
            output, output_row_stride, conversion);
        return;
    }
#if defined(DSMVC_HAS_CUDA)
    impl_->cuda->inverse_2d_u8(
        horizontal, vertical, input, input_row_stride,
        output, output_row_stride, conversion, std::move(input_lifetime));
#endif
}

void Executor::inverse_2d_u16(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::uint16_t *input, std::ptrdiff_t input_row_stride,
    std::uint16_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion,
    std::shared_ptr<const void> input_lifetime) const {
    if (impl_->cpu) {
        impl_->cpu->inverse_2d_u16(
            horizontal, vertical, input, input_row_stride,
            output, output_row_stride, conversion);
        return;
    }
#if defined(DSMVC_HAS_CUDA)
    impl_->cuda->inverse_2d_u16(
        horizontal, vertical, input, input_row_stride,
        output, output_row_stride, conversion, std::move(input_lifetime));
#endif
}

void Executor::inverse_2d_u8_streamed(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::uint8_t *input, std::ptrdiff_t input_row_stride,
    std::uint8_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion,
    std::shared_ptr<const void> input_lifetime) const {
    if (impl_->cpu) {
        impl_->cpu->inverse_2d_u8_streamed(
            horizontal, vertical, input, input_row_stride,
            output, output_row_stride, conversion);
        return;
    }
#if defined(DSMVC_HAS_CUDA)
    impl_->cuda->inverse_2d_u8(
        horizontal, vertical, input, input_row_stride,
        output, output_row_stride, conversion, std::move(input_lifetime));
#endif
}

void Executor::inverse_2d_u16_streamed(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::uint16_t *input, std::ptrdiff_t input_row_stride,
    std::uint16_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion,
    std::shared_ptr<const void> input_lifetime) const {
    if (impl_->cpu) {
        impl_->cpu->inverse_2d_u16_streamed(
            horizontal, vertical, input, input_row_stride,
            output, output_row_stride, conversion);
        return;
    }
#if defined(DSMVC_HAS_CUDA)
    impl_->cuda->inverse_2d_u16(
        horizontal, vertical, input, input_row_stride,
        output, output_row_stride, conversion, std::move(input_lifetime));
#endif
}

} // namespace dsmvc
