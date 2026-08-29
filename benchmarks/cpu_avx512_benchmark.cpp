#include <dsmvc/engine.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
struct Case { const char *name; dsmvc::KernelKind kind; int taps; bool columns; };

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

double run(const dsmvc::CpuExecutor &executor,
           const dsmvc::AxisPlan &plan, bool columns, std::size_t iterations) {
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

void execute(const dsmvc::CpuExecutor &executor,
             const dsmvc::AxisPlan &plan, bool columns,
             std::vector<float> &input, std::vector<float> &output) {
    constexpr int stride = 1920;
    const int input_rows = columns ? plan.source_size : 256;
    if (columns) {
        executor.inverse_columns(plan, input.data(), stride, output.data(),
                                 stride, 1692);
    } else {
        executor.inverse_rows(plan, input.data(), stride, output.data(),
                              stride, input_rows);
    }
}

struct ErrorStats { double max_abs = 0.0; std::size_t nonfinite = 0U; };

ErrorStats compare_outputs(const std::vector<float> &a,
                           const std::vector<float> &b,
                           const dsmvc::AxisPlan &plan, bool columns) {
    constexpr int stride = 1920;
    const int rows = columns ? plan.destination_size : 256;
    const int cols = columns ? 1692 : plan.destination_size;
    ErrorStats result;
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < cols; ++column) {
            const auto index = static_cast<std::size_t>(row) * stride + column;
            const float avx2 = a[index];
            const float avx512 = b[index];
            if (!std::isfinite(avx2) || !std::isfinite(avx512)) {
                ++result.nonfinite;
            } else {
                result.max_abs = std::max(
                    result.max_abs,
                    static_cast<double>(std::abs(avx2 - avx512)));
            }
        }
    }
    return result;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}
}

int main(int argc, char **argv) {
    const std::size_t samples = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 3U;
    const std::size_t iterations = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 1U;
    const std::string_view filter = argc > 3 ? std::string_view(argv[3]) : std::string_view{};
    if (samples == 0U || iterations == 0U || !dsmvc::cpu_avx512_available()) {
        std::cerr << "AVX-512 unavailable or invalid arguments\n";
        return EXIT_FAILURE;
    }
    const dsmvc::CpuExecutor avx2(dsmvc::CpuPath::avx2);
    const dsmvc::CpuExecutor avx512(dsmvc::CpuPath::avx512);
    const Case cases[] = {
        {"b1-horizontal", dsmvc::KernelKind::bilinear, 0, false},
        {"b3-horizontal", dsmvc::KernelKind::bicubic, 0, false},
        {"b5-horizontal", dsmvc::KernelKind::lanczos, 3, false},
        {"b7-horizontal", dsmvc::KernelKind::spline64, 0, false},
        {"b5-vertical", dsmvc::KernelKind::lanczos, 3, true},
        {"b7-vertical", dsmvc::KernelKind::spline64, 0, true},
    };
    std::cout << std::fixed << std::setprecision(3);
    for (const auto &c : cases) {
        if (!filter.empty() && filter != c.name) continue;
        const auto plan = plan_for(c);
        (void)run(avx2, *plan, c.columns, 1U);
        (void)run(avx512, *plan, c.columns, 1U);
        std::vector<double> a, b;
        for (std::size_t i = 0; i < samples; ++i) {
            double avx2_ms;
            double avx512_ms;
            if ((i & 1U) == 0U) {
                avx2_ms = run(avx2, *plan, c.columns, iterations);
                avx512_ms = run(avx512, *plan, c.columns, iterations);
            } else {
                avx512_ms = run(avx512, *plan, c.columns, iterations);
                avx2_ms = run(avx2, *plan, c.columns, iterations);
            }
            a.push_back(avx2_ms); b.push_back(avx512_ms);
        }
        const double old_ms = median(a), new_ms = median(b);
        constexpr int stride = 1920;
        const int input_rows = c.columns ? plan->source_size : 256;
        const int output_rows = c.columns ? plan->destination_size : input_rows;
        auto input = make_input(static_cast<std::size_t>(stride) * input_rows);
        std::vector<float> avx2_output(static_cast<std::size_t>(stride)
                                       * output_rows, 0.0F);
        std::vector<float> avx512_output(avx2_output.size(), 0.0F);
        execute(avx2, *plan, c.columns, input, avx2_output);
        execute(avx512, *plan, c.columns, input, avx512_output);
        const auto error = compare_outputs(
            avx2_output, avx512_output, *plan, c.columns);
        std::cout << c.name << " avx2_ms=" << old_ms
                  << " avx512_ms=" << new_ms
                  << " speedup=" << old_ms / new_ms << "x"
                  << " max_abs_error=" << std::scientific
                  << std::setprecision(6) << error.max_abs << std::fixed
                  << std::setprecision(3)
                  << " nonfinite=" << error.nonfinite << '\n';
    }
}
