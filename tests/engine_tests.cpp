#include <dsmvc/engine.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void test_backend_selection() {
    require(dsmvc::parse_backend("AUTO") == dsmvc::BackendKind::automatic,
            "backend parsing failed");
    require(dsmvc::resolve_backend(dsmvc::BackendKind::automatic)
                == dsmvc::BackendKind::cpu,
            "automatic backend did not select CPU");
    bool rejected = false;
    try {
        (void)dsmvc::resolve_backend(dsmvc::BackendKind::cuda);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "uncompiled CUDA backend was not rejected");
}

void test_identity_bilinear() {
    dsmvc::AxisRequest request;
    request.source_size = 32;
    request.destination_size = 32;
    request.active_length = 32.0;
    request.kernel.kind = dsmvc::KernelKind::bilinear;
    const auto plan = dsmvc::build_axis_plan(request);
    std::vector<float> input(32);
    std::vector<float> output(32);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i) / 31.0F;
    getnative::inverse_axis_f32(plan, input, output);
    for (std::size_t i = 0; i < input.size(); ++i) {
        require(std::abs(input[i] - output[i]) < 1.0e-5F,
                "identity bilinear solve drifted");
    }
}

void test_custom_plan() {
    dsmvc::AxisRequest request;
    request.source_size = 48;
    request.destination_size = 32;
    request.active_length = 32.0;
    request.kernel.kind = dsmvc::KernelKind::custom;
    request.kernel.taps = 1;
    const auto plan = dsmvc::build_axis_plan(
        request, [](double x) { return std::max(1.0 - x, 0.0); });
    require(plan.valid(), "custom axis plan is invalid");
    require(plan.half_bandwidth == 1, "custom plan bandwidth is incorrect");
}

void test_executor_agreement() {
    dsmvc::AxisRequest request;
    request.source_size = 96;
    request.destination_size = 64;
    request.active_length = 64.0;
    request.kernel.kind = dsmvc::KernelKind::bicubic;
    request.kernel.b = 0.0;
    request.kernel.c = 0.5;
    const auto plan = dsmvc::build_axis_plan(request);
    constexpr int rows = 16;
    std::vector<float> input(static_cast<std::size_t>(rows) * 96U);
    std::vector<float> scalar(static_cast<std::size_t>(rows) * 64U);
    std::vector<float> optimized(static_cast<std::size_t>(rows) * 64U);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((i * 29U) % 257U) / 256.0F;
    }
    dsmvc::CpuExecutor scalar_executor(dsmvc::CpuPath::scalar);
    scalar_executor.inverse_rows(plan, input.data(), 96, scalar.data(), 64, rows);
    dsmvc::CpuExecutor automatic_executor(dsmvc::CpuPath::automatic);
    automatic_executor.inverse_rows(plan, input.data(), 96, optimized.data(), 64, rows);
    float maximum = 0.0F;
    for (std::size_t i = 0; i < scalar.size(); ++i) {
        maximum = std::max(maximum, std::abs(scalar[i] - optimized[i]));
    }
    require(maximum < 2.0e-5F, "optimized executor differs from scalar executor");
}

void test_executor_padded_geometry() {
    dsmvc::AxisRequest request;
    request.source_size = 1920;
    request.destination_size = 1692;
    request.active_length = 1691.5555555555557;
    request.shift = 0.2222222222221717;
    request.kernel.kind = dsmvc::KernelKind::bicubic;
    request.kernel.b = 0.0;
    request.kernel.c = 1.0;
    const auto plan = dsmvc::build_axis_plan(request);
    constexpr int rows = 1080;
    constexpr int input_stride = 1920;
    constexpr int output_stride = 1696;
    std::vector<float> input(static_cast<std::size_t>(rows) * input_stride);
    std::vector<float> scalar(static_cast<std::size_t>(rows) * output_stride);
    std::vector<float> optimized(static_cast<std::size_t>(rows) * output_stride);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((i * 29U) % 257U) / 256.0F;
    }
    dsmvc::CpuExecutor scalar_executor(dsmvc::CpuPath::scalar);
    scalar_executor.inverse_rows(
        plan, input.data(), input_stride, scalar.data(), output_stride, rows);
    dsmvc::CpuExecutor optimized_executor(dsmvc::CpuPath::automatic);
    for (int iteration = 0; iteration < 16; ++iteration) {
        optimized_executor.inverse_rows(
            plan, input.data(), input_stride, optimized.data(), output_stride, rows);
    }
    float maximum = 0.0F;
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < request.destination_size; ++column) {
            const auto index = static_cast<std::size_t>(row) * output_stride
                + static_cast<std::size_t>(column);
            maximum = std::max(maximum, std::abs(scalar[index] - optimized[index]));
        }
    }
    require(maximum < 2.0e-5F, "padded optimized executor differs from scalar");
}

void test_column_executor_padded_geometry() {
    dsmvc::AxisRequest request;
    request.source_size = 1080;
    request.destination_size = 952;
    request.active_length = 951.5;
    request.shift = 0.25;
    request.kernel.kind = dsmvc::KernelKind::bicubic;
    request.kernel.b = 0.0;
    request.kernel.c = 1.0;
    const auto plan = dsmvc::build_axis_plan(request);
    constexpr int columns = 1692;
    constexpr int stride = 1696;
    std::vector<float> input(static_cast<std::size_t>(request.source_size) * stride);
    std::vector<float> scalar(static_cast<std::size_t>(request.destination_size) * stride);
    std::vector<float> optimized(static_cast<std::size_t>(request.destination_size) * stride);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((i * 17U) % 263U) / 262.0F;
    }
    dsmvc::CpuExecutor scalar_executor(dsmvc::CpuPath::scalar);
    scalar_executor.inverse_columns(
        plan, input.data(), stride, scalar.data(), stride, columns);
    dsmvc::CpuExecutor optimized_executor(dsmvc::CpuPath::automatic);
    for (int iteration = 0; iteration < 16; ++iteration) {
        optimized_executor.inverse_columns(
            plan, input.data(), stride, optimized.data(), stride, columns);
    }
    float maximum = 0.0F;
    for (int row = 0; row < request.destination_size; ++row) {
        for (int column = 0; column < columns; ++column) {
            const auto index = static_cast<std::size_t>(row) * stride
                + static_cast<std::size_t>(column);
            maximum = std::max(maximum, std::abs(scalar[index] - optimized[index]));
        }
    }
    require(maximum < 2.0e-5F, "column optimized executor differs from scalar");
}

} // namespace

int main() {
    try {
        test_backend_selection();
        test_identity_bilinear();
        test_custom_plan();
        test_executor_agreement();
        test_executor_padded_geometry();
        test_column_executor_padded_geometry();
        std::cout << "dsmvc engine tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "dsmvc engine test failure: " << error.what() << '\n';
        return 1;
    }
}
