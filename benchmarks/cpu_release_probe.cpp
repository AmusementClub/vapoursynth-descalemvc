#include <dsmvc/engine.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef DSMVC_BENCH_HAS_AVX512_API
#define DSMVC_BENCH_HAS_AVX512_API 1
#endif

namespace {
using Clock = std::chrono::steady_clock;

struct Buffer {
    explicit Buffer(std::size_t count) : storage(count + 32U) {
        void *pointer = storage.data();
        auto bytes = storage.size() * sizeof(float);
        if (!std::align(64U, (count + 4U) * sizeof(float), pointer, bytes)) {
            throw std::runtime_error("could not align benchmark buffer");
        }
        // Match a common std::vector allocation offset, identically for A/B.
        data = static_cast<float *>(pointer) + 4U;
    }
    std::vector<float> storage;
    float *data = nullptr;
};

struct CpuTimes { std::uint64_t total = 0U, idle = 0U; };
CpuTimes cpu1_times() {
    std::ifstream stream("/proc/stat");
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.starts_with("cpu1 ")) continue;
        std::istringstream fields(line.substr(5));
        CpuTimes result;
        for (int index = 0; index < 8; ++index) {
            std::uint64_t value = 0U;
            fields >> value;
            result.total += value;
            if (index == 3) result.idle = value;
        }
        return result;
    }
    throw std::runtime_error("Linux CPU1 telemetry is required");
}

std::uint64_t hash(const float *data, std::size_t count) {
    const auto *bytes = reinterpret_cast<const unsigned char *>(data);
    std::uint64_t result = 1469598103934665603ULL;
    for (std::size_t index = 0; index < count * sizeof(float); ++index) {
        result = (result ^ bytes[index]) * 1099511628211ULL;
    }
    return result;
}
}

int main(int argc, char **argv) try {
    if (argc == 2 && std::string(argv[1]) == "--capabilities") {
        bool avx512 = false;
#if DSMVC_BENCH_HAS_AVX512_API
        avx512 = dsmvc::cpu_avx512_available();
#endif
        std::cout << "{\"avx2\":" << (dsmvc::cpu_avx2_available() ? "true" : "false")
                  << ",\"avx512\":" << (avx512 ? "true" : "false") << "}\n";
        return 0;
    }
    if (argc != 7) throw std::runtime_error(
        "usage: probe kernel width|height avx2|avx512 samples iterations output.bin");
    const std::string kernel = argv[1], axis = argv[2], isa = argv[3];
    const auto samples = std::stoull(argv[4]);
    const auto iterations = std::stoull(argv[5]);
    if (samples == 0 || samples > 1000 || iterations == 0) {
        throw std::runtime_error("invalid sample count or iteration count");
    }
    if (axis != "width" && axis != "height") {
        throw std::runtime_error("invalid axis");
    }
    auto path = dsmvc::CpuPath::avx2;
    if (isa == "avx512") {
#if DSMVC_BENCH_HAS_AVX512_API
        path = dsmvc::CpuPath::avx512;
#else
        throw std::runtime_error("this baseline predates the AVX-512 API");
#endif
    } else if (isa != "avx2") {
        throw std::runtime_error("invalid ISA");
    }
    dsmvc::AxisRequest request;
    const bool columns = axis == "height";
    request.source_size = columns ? 951 : 1920;
    request.destination_size = columns ? 952 : 1692;
    request.active_length = columns ? 951.5 : 1691.5555555555557;
    request.shift = columns ? 0.25 : 0.2222222222221717;
    request.border = dsmvc::BorderMode::mirror;
    request.kernel.taps = 0;
    if (kernel == "bilinear") request.kernel.kind = dsmvc::KernelKind::bilinear;
    else if (kernel == "bicubic") request.kernel.kind = dsmvc::KernelKind::bicubic;
    else if (kernel == "lanczos3") {
        request.kernel.kind = dsmvc::KernelKind::lanczos;
        request.kernel.taps = 3;
    } else if (kernel == "spline64") {
        request.kernel.kind = dsmvc::KernelKind::spline64;
    } else throw std::runtime_error("unknown kernel");
    // Preserve the release benchmark's automatic precision selection.
    const auto plan = dsmvc::build_axis_plan(request);
    const dsmvc::CpuExecutor executor(path);
    constexpr int stride = 1920;
    const int input_rows = columns ? plan.source_size : 256;
    const int output_rows = columns ? plan.destination_size : input_rows;
    const int output_columns = columns ? 1692 : plan.destination_size;
    const auto input_count = static_cast<std::size_t>(stride) * input_rows;
    const auto output_count = static_cast<std::size_t>(stride) * output_rows;
    Buffer input(input_count), output(output_count), reference(output_count);
    for (std::size_t index = 0; index < input_count; ++index) {
        const auto value = static_cast<int>((index * 17U + 13U) % 257U) - 128;
        input.data[index] = static_cast<float>(value) / 128.0F;
    }
    const auto input_hash = hash(input.data, input_count);
    const auto execute = [&](const dsmvc::CpuExecutor &selected, float *destination) {
        if (columns) selected.inverse_columns(plan, input.data, stride,
            destination, stride, output_columns);
        else selected.inverse_rows(plan, input.data, stride,
            destination, stride, input_rows);
    };
    execute(executor, output.data);
    std::vector<double> elapsed;
    std::vector<CpuTimes> deltas;
    for (std::size_t sample = 0; sample < samples; ++sample) {
        const auto before = cpu1_times();
        const auto start = Clock::now();
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            execute(executor, output.data);
        }
        elapsed.push_back(std::chrono::duration<double, std::milli>(
            Clock::now() - start).count());
        const auto after = cpu1_times();
        deltas.push_back({after.total - before.total, after.idle - before.idle});
    }
    const dsmvc::CpuExecutor avx2(dsmvc::CpuPath::avx2);
    execute(avx2, reference.data);
    double max_error = 0.0;
    bool padding_unchanged = true;
    std::ofstream file(argv[6], std::ios::binary);
    if (!file) throw std::runtime_error("output proof file could not be opened");
    for (int row = 0; row < output_rows; ++row) {
        const auto offset = static_cast<std::size_t>(row) * stride;
        for (int column = 0; column < output_columns; ++column) {
            const auto index = offset + column;
            if (!std::isfinite(output.data[index]) || !std::isfinite(reference.data[index])) {
                throw std::runtime_error("non-finite output");
            }
            max_error = std::max(max_error,
                std::abs(static_cast<double>(output.data[index]) - reference.data[index]));
        }
        for (int column = output_columns; column < stride; ++column) {
            padding_unchanged &= output.data[offset + column] == 0.0F;
        }
        file.write(reinterpret_cast<const char *>(output.data + offset),
                   output_columns * sizeof(float));
    }
    if (!file || hash(input.data, input_count) != input_hash || max_error > 3.0e-5) {
        throw std::runtime_error("numeric/input/proof gate failed");
    }
    file.close();
    auto ordered = elapsed;
    std::sort(ordered.begin(), ordered.end());
    std::cout << std::setprecision(17)
        << "{\"kernel\":\"" << kernel << "\",\"axis\":\"" << axis
        << "\",\"isa\":\"" << isa << "\",\"actual_path\":\"" << executor.name()
        << "\",\"source_size\":" << plan.source_size
        << ",\"destination_size\":" << plan.destination_size
        << ",\"input_rows\":" << input_rows << ",\"stride\":" << stride
        << ",\"bandwidth\":" << plan.half_bandwidth
        << ",\"actual_f64\":" << (plan.requires_float64() ? "true" : "false")
        << ",\"iterations\":" << iterations << ",\"samples\":" << samples
        << ",\"median_ms\":" << ordered[ordered.size() / 2U]
        << ",\"input_hash\":" << input_hash
        << ",\"input_mod64\":" << reinterpret_cast<std::uintptr_t>(input.data) % 64U
        << ",\"output_mod64\":" << reinterpret_cast<std::uintptr_t>(output.data) % 64U
        << ",\"max_abs_vs_avx2\":" << max_error
        << ",\"padding_unchanged\":" << (padding_unchanged ? "true" : "false")
        << ",\"timed_source_fills\":0,\"samples_ms\":[";
    for (std::size_t index = 0; index < samples; ++index) {
        if (index != 0) std::cout << ',';
        std::cout << elapsed[index];
    }
    std::cout << "],\"cpu1_samples\":[";
    for (std::size_t index = 0; index < samples; ++index) {
        if (index != 0) std::cout << ',';
        std::cout << "{\"total\":" << deltas[index].total
                  << ",\"idle\":" << deltas[index].idle << '}';
    }
    std::cout << "]}\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
