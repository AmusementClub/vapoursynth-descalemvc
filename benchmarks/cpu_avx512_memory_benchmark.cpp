#include <dsmvc/engine.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::int32_t source_width = 1920;
constexpr std::int32_t destination_width = 1692;
constexpr std::size_t eviction_bytes = 128U * 1024U * 1024U;
volatile std::uint64_t eviction_sink = 0U;

dsmvc::AxisPlan make_plan(std::string_view kernel) {
    dsmvc::AxisRequest request;
    request.source_size = source_width;
    request.destination_size = destination_width;
    request.active_length = 1691.5555555555557;
    request.shift = 0.2222222222221717;
    request.border = dsmvc::BorderMode::mirror;
    if (kernel == "b1") {
        request.kernel.kind = dsmvc::KernelKind::bilinear;
    } else if (kernel == "b3") {
        request.kernel.kind = dsmvc::KernelKind::bicubic;
    } else if (kernel == "b5") {
        request.kernel.kind = dsmvc::KernelKind::lanczos;
        request.kernel.taps = 3;
    } else {
        request.kernel.kind = dsmvc::KernelKind::spline64;
    }
    return dsmvc::build_axis_plan(request);
}

std::vector<float> make_input(std::size_t size) {
    std::vector<float> input(size);
    for (std::size_t i = 0; i < size; ++i) {
        const auto value = static_cast<int>((i * 17U + 13U) % 257U) - 128;
        input[i] = static_cast<float>(value) / 128.0F;
    }
    return input;
}

void evict_cache(std::vector<std::uint8_t> &eviction) {
    std::uint64_t sum = 0U;
    for (std::size_t i = 0; i < eviction.size(); i += 64U) {
        eviction[i] = static_cast<std::uint8_t>(eviction[i] + 1U);
        sum += eviction[i];
    }
    eviction_sink = sum;
}

void execute(const dsmvc::CpuExecutor &executor, const dsmvc::AxisPlan &plan,
             const std::vector<float> &input, std::vector<float> &output,
             std::int32_t rows) {
    executor.inverse_rows(plan, input.data(), source_width, output.data(),
                          destination_width, rows);
}

double sample(const dsmvc::CpuExecutor &executor, const dsmvc::AxisPlan &plan,
              const std::vector<float> &input, std::vector<float> &output,
              std::vector<std::uint8_t> &eviction, std::int32_t rows,
              std::size_t iterations, bool cold) {
    double elapsed_ms = 0.0;
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        if (cold) evict_cache(eviction);
        const auto start = Clock::now();
        execute(executor, plan, input, output, rows);
        elapsed_ms += std::chrono::duration<double, std::milli>(
            Clock::now() - start).count();
    }
    return elapsed_ms / static_cast<double>(iterations);
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

double max_error(const std::vector<float> &a, const std::vector<float> &b) {
    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result = std::max(result,
            static_cast<double>(std::abs(a[i] - b[i])));
    }
    return result;
}
} // namespace

int main(int argc, char **argv) {
    const std::string_view mode = argc > 1 ? argv[1] : "pair";
    const auto rows = static_cast<std::int32_t>(
        argc > 2 ? std::strtol(argv[2], nullptr, 10) : 256L);
    const std::size_t samples = argc > 3
        ? std::strtoul(argv[3], nullptr, 10) : 11U;
    const std::size_t iterations = argc > 4
        ? std::strtoul(argv[4], nullptr, 10) : 5U;
    const bool cold = argc > 5 && std::string_view(argv[5]) == "cold";
    const std::string_view kernel = argc > 6 ? argv[6] : "b7";
    if ((mode != "pair" && mode != "avx2" && mode != "avx512")
        || (kernel != "b1" && kernel != "b3"
            && kernel != "b5" && kernel != "b7")
        || rows <= 0 || samples == 0U || iterations == 0U
        || !dsmvc::cpu_avx512_available()) {
        std::cerr << "usage: benchmark [pair|avx2|avx512] rows samples "
                     "iterations [hot|cold] [b1|b3|b5|b7]\n";
        return EXIT_FAILURE;
    }

    const auto plan = make_plan(kernel);
    const dsmvc::CpuExecutor avx2(dsmvc::CpuPath::avx2);
    const dsmvc::CpuExecutor avx512(dsmvc::CpuPath::avx512);
    auto input = make_input(static_cast<std::size_t>(rows) * source_width);
    std::vector<float> avx2_output(
        static_cast<std::size_t>(rows) * destination_width);
    std::vector<float> avx512_output(avx2_output.size());
    std::vector<std::uint8_t> eviction(cold ? eviction_bytes : 0U);

    execute(avx2, plan, input, avx2_output, rows);
    execute(avx512, plan, input, avx512_output, rows);
    std::vector<double> avx2_samples;
    std::vector<double> avx512_samples;
    for (std::size_t index = 0; index < samples; ++index) {
        if (mode == "pair" && (index & 1U) != 0U) {
            avx512_samples.push_back(sample(
                avx512, plan, input, avx512_output, eviction,
                rows, iterations, cold));
            avx2_samples.push_back(sample(
                avx2, plan, input, avx2_output, eviction,
                rows, iterations, cold));
        } else {
            if (mode != "avx512") {
                avx2_samples.push_back(sample(
                    avx2, plan, input, avx2_output, eviction,
                    rows, iterations, cold));
            }
            if (mode != "avx2") {
                avx512_samples.push_back(sample(
                    avx512, plan, input, avx512_output, eviction,
                    rows, iterations, cold));
            }
        }
    }

    const double working_set_mib = static_cast<double>(rows)
        * static_cast<double>(source_width + destination_width) * sizeof(float)
        / (1024.0 * 1024.0);
    std::cout << std::fixed << std::setprecision(3)
              << "mode=" << mode << " kernel=" << kernel
              << " cache=" << (cold ? "cold" : "hot")
              << " rows=" << rows << " working_set_mib=" << working_set_mib;
    if (!avx2_samples.empty()) {
        std::cout << " avx2_ms=" << median(avx2_samples);
    }
    if (!avx512_samples.empty()) {
        std::cout << " avx512_ms=" << median(avx512_samples);
    }
    if (mode == "pair") {
        const auto avx2_ms = median(avx2_samples);
        const auto avx512_ms = median(avx512_samples);
        std::cout << " speedup=" << avx2_ms / avx512_ms << "x"
                  << " max_abs_error=" << std::scientific
                  << std::setprecision(6)
                  << max_error(avx2_output, avx512_output);
    }
    std::cout << '\n';
}
