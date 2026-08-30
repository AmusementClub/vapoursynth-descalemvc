#include <dsmvc/engine.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

struct Case {
    const char *filter;
    const char *axis;
    dsmvc::KernelKind kind;
    int taps;
    bool columns;
};

std::vector<float> make_input(std::size_t size) {
    std::vector<float> input(size);
    for (std::size_t i = 0; i < size; ++i) {
        const auto value = static_cast<int>((i * 17U + 13U) % 257U) - 128;
        input[i] = static_cast<float>(value) / 128.0F;
    }
    return input;
}

std::shared_ptr<const dsmvc::AxisPlan> plan_for(const Case &c) {
    dsmvc::AxisRequest request;
    request.source_size = c.columns ? 951 : 1920;
    request.destination_size = c.columns ? 952 : 1692;
    request.active_length = c.columns ? 951.5 : 1691.5555555555557;
    request.shift = c.columns ? 0.25 : 0.2222222222221717;
    request.border = dsmvc::BorderMode::mirror;
    request.kernel.kind = c.kind;
    request.kernel.taps = c.taps;
    return std::make_shared<const dsmvc::AxisPlan>(dsmvc::build_axis_plan(request));
}

double run(const dsmvc::CpuExecutor &executor, const dsmvc::AxisPlan &plan,
           bool columns, std::size_t iterations) {
    constexpr int stride = 1920;
    const int input_rows = columns ? plan.source_size : 256;
    const int output_rows = columns ? plan.destination_size : input_rows;
    auto input = make_input(static_cast<std::size_t>(stride) * input_rows);
    std::vector<float> output(static_cast<std::size_t>(stride) * output_rows, 0.0F);
    const auto start = Clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        if (columns) {
            executor.inverse_columns(plan, input.data(), stride, output.data(),
                                      stride, 1692);
        } else {
            executor.inverse_rows(plan, input.data(), stride, output.data(),
                                  stride, input_rows);
        }
    }
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}
}

int main(int argc, char **argv) {
    const std::size_t samples = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 5U;
    const std::size_t iterations = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 3U;
    if (samples == 0U || iterations == 0U || !dsmvc::cpu_avx2_available()) {
        std::cerr << "AVX2 unavailable or invalid arguments\n";
        return EXIT_FAILURE;
    }
    const dsmvc::CpuExecutor avx2(dsmvc::CpuPath::avx2);
    const Case cases[] = {
        {"Debilinear", "width", dsmvc::KernelKind::bilinear, 0, false},
        {"Debicubic", "width", dsmvc::KernelKind::bicubic, 0, false},
        {"Delanczos3", "width", dsmvc::KernelKind::lanczos, 3, false},
        {"Despline64", "width", dsmvc::KernelKind::spline64, 0, false},
        {"Debilinear", "height", dsmvc::KernelKind::bilinear, 0, true},
        {"Debicubic", "height", dsmvc::KernelKind::bicubic, 0, true},
        {"Delanczos3", "height", dsmvc::KernelKind::lanczos, 3, true},
        {"Despline64", "height", dsmvc::KernelKind::spline64, 0, true},
    };
    std::cout << std::fixed << std::setprecision(3);
    for (const auto &c : cases) {
        const auto plan = plan_for(c);
        (void)run(avx2, *plan, c.columns, 1U);
        std::vector<double> samples_ms;
        for (std::size_t i = 0; i < samples; ++i) {
            samples_ms.push_back(run(avx2, *plan, c.columns, iterations));
        }
        std::cout << c.filter << ' ' << c.axis
                  << " avx2_ms=" << median(std::move(samples_ms)) << '\n';
    }
}
