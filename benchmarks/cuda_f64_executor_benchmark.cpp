#include <dsmvc/engine.hpp>

#include "cuda/cuda_executor.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

enum class Operation {
    rows,
    columns,
    image_2d,
};

enum class Precision {
    float32,
    float64,
};

struct Options {
    enum class Mode {
        float32,
        float64,
        all,
    };

    Mode mode = Mode::all;
    int pairs = 30;
    int warmups = 3;
    std::string label;
    std::string source_sha;
    std::string binary_sha;
    std::optional<std::string> output_path;
    std::string command;
};

struct BenchmarkCase {
    std::string name;
    Operation operation{};
    std::shared_ptr<const dsmvc::AxisPlan> horizontal;
    std::shared_ptr<const dsmvc::AxisPlan> vertical;
    std::int32_t vector_count = 0;
    std::int32_t input_rows = 0;
    std::int32_t input_columns = 0;
    std::int32_t output_rows = 0;
    std::int32_t output_columns = 0;
    std::int32_t input_stride = 0;
    std::int32_t output_stride = 0;
    std::vector<float> input;
    std::vector<float> cpu_output;
    std::vector<float> cuda_output;
};

[[noreturn]] void usage_error(std::string message) {
    throw std::invalid_argument(
        std::move(message)
        + "\nusage: dsmvc_cuda_f64_executor_benchmark"
          " --label LABEL --source-sha SHA --binary-sha SHA"
          " [--mode f64|f32|all] [--pairs 30..200]"
          " [--warmups 0..20] [--output FILE]");
}

std::string json_escape(std::string_view value) {
    std::string result;
    result.reserve(value.size() + 8U);
    for (const unsigned char byte : value) {
        switch (byte) {
        case '\\': result += "\\\\"; break;
        case '"': result += "\\\""; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:
            if (byte < 0x20U) {
                constexpr char digits[] = "0123456789abcdef";
                result += "\\u00";
                result += digits[byte >> 4U];
                result += digits[byte & 0x0fU];
            } else {
                result += static_cast<char>(byte);
            }
        }
    }
    return result;
}

std::string quoted(std::string_view value) {
    return '"' + json_escape(value) + '"';
}

int parse_integer(std::string_view text, std::string_view option) {
    std::size_t consumed = 0U;
    int result = 0;
    try {
        result = std::stoi(std::string(text), &consumed);
    } catch (...) {
        usage_error(std::string(option) + " requires an integer");
    }
    if (consumed != text.size()) {
        usage_error(std::string(option) + " requires an integer");
    }
    return result;
}

Options parse_options(int argc, char **argv) {
    Options result;
    for (int index = 0; index < argc; ++index) {
        if (index != 0) result.command += ' ';
        result.command += argv[index];
    }
    const auto next = [&](int &index, std::string_view option) {
        if (++index >= argc) usage_error(std::string(option) + " needs a value");
        return std::string_view(argv[index]);
    };
    for (int index = 1; index < argc; ++index) {
        const std::string_view option(argv[index]);
        if (option == "--label") {
            result.label = next(index, option);
        } else if (option == "--source-sha") {
            result.source_sha = next(index, option);
        } else if (option == "--binary-sha") {
            result.binary_sha = next(index, option);
        } else if (option == "--pairs") {
            result.pairs = parse_integer(next(index, option), option);
        } else if (option == "--warmups") {
            result.warmups = parse_integer(next(index, option), option);
        } else if (option == "--output") {
            result.output_path = std::string(next(index, option));
        } else if (option == "--mode") {
            const auto value = next(index, option);
            if (value == "f32") {
                result.mode = Options::Mode::float32;
            } else if (value == "f64") {
                result.mode = Options::Mode::float64;
            } else if (value == "all") {
                result.mode = Options::Mode::all;
            } else {
                usage_error("--mode must be f64, f32, or all");
            }
        } else {
            usage_error("unknown option: " + std::string(option));
        }
    }
    if (result.label.empty() || result.source_sha.empty()
        || result.binary_sha.empty()) {
        usage_error("--label, --source-sha, and --binary-sha are required");
    }
    if (result.pairs < 30 || result.pairs > 200) {
        usage_error("--pairs must be between 30 and 200");
    }
    if (result.warmups < 0 || result.warmups > 20) {
        usage_error("--warmups must be between 0 and 20");
    }
    return result;
}

const char *precision_name(Precision precision) {
    return precision == Precision::float64 ? "f64" : "f32";
}

const char *operation_name(Operation operation) {
    switch (operation) {
    case Operation::rows: return "rows";
    case Operation::columns: return "columns";
    case Operation::image_2d: return "2d";
    }
    return "unknown";
}

dsmvc::AxisRequest axis_request(
    std::int32_t source, std::int32_t destination,
    std::int32_t bandwidth, Precision precision) {
    dsmvc::AxisRequest request;
    request.source_size = source;
    request.destination_size = destination;
    request.active_length = static_cast<double>(destination) - 0.25;
    request.shift = 0.125;
    request.border = dsmvc::BorderMode::symmetric;
    request.f64_mode = precision == Precision::float64
        ? dsmvc::F64Mode::float64_only
        : dsmvc::F64Mode::float32_only;
    switch (bandwidth) {
    case 1:
        request.kernel.kind = dsmvc::KernelKind::bilinear;
        request.kernel.taps = 1;
        break;
    case 3:
        request.kernel.kind = dsmvc::KernelKind::bicubic;
        request.kernel.taps = 3;
        request.kernel.b = 0.0;
        request.kernel.c = 0.5;
        break;
    case 5:
        request.kernel.kind = dsmvc::KernelKind::lanczos;
        request.kernel.taps = 3;
        break;
    case 7:
        request.kernel.kind = dsmvc::KernelKind::spline64;
        request.kernel.taps = 3;
        break;
    case 9:
        request.kernel.kind = dsmvc::KernelKind::custom;
        request.kernel.taps = 5;
        break;
    default:
        throw std::logic_error("unsupported benchmark bandwidth");
    }
    return request;
}

std::shared_ptr<const dsmvc::AxisPlan> make_plan(
    std::int32_t source, std::int32_t destination,
    std::int32_t bandwidth, Precision precision) {
    auto request = axis_request(source, destination, bandwidth, precision);
    dsmvc::AxisPlan plan;
    if (bandwidth == 9) {
        plan = dsmvc::build_axis_plan(request, [](double x) {
            return std::max(1.0 - std::abs(x), 0.0);
        });
    } else {
        plan = dsmvc::build_axis_plan(request);
    }
    if (!plan.valid() || plan.half_bandwidth != bandwidth
        || plan.requires_float64() != (precision == Precision::float64)) {
        throw std::runtime_error("benchmark plan selected unexpected precision");
    }
    return std::make_shared<const dsmvc::AxisPlan>(std::move(plan));
}

std::uint32_t next_random(std::uint32_t &state) {
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    return state;
}

void initialize_case(BenchmarkCase &item, std::uint32_t seed) {
    item.input_stride = item.input_columns + 5;
    item.output_stride = item.output_columns + 7;
    item.input.assign(
        static_cast<std::size_t>(item.input_rows)
            * static_cast<std::size_t>(item.input_stride),
        17.0F);
    item.cpu_output.assign(
        static_cast<std::size_t>(item.output_rows)
            * static_cast<std::size_t>(item.output_stride),
        -41.0F);
    item.cuda_output = item.cpu_output;
    std::uint32_t state = seed;
    for (std::int32_t row = 0; row < item.input_rows; ++row) {
        for (std::int32_t column = 0; column < item.input_columns; ++column) {
            const std::uint32_t bits = 0x3f000000U
                | (next_random(state) & 0x007fffffU);
            item.input[static_cast<std::size_t>(row)
                           * static_cast<std::size_t>(item.input_stride)
                       + static_cast<std::size_t>(column)] =
                std::bit_cast<float>(bits) - 0.75F;
        }
    }
}

BenchmarkCase rows_case(
    std::string name, std::int32_t bandwidth,
    std::int32_t source, std::int32_t destination,
    std::int32_t rows, Precision precision, std::uint32_t seed) {
    BenchmarkCase result;
    result.name = std::move(name);
    result.operation = Operation::rows;
    result.horizontal = make_plan(source, destination, bandwidth, precision);
    result.vector_count = rows;
    result.input_rows = rows;
    result.input_columns = source;
    result.output_rows = rows;
    result.output_columns = destination;
    initialize_case(result, seed);
    return result;
}

BenchmarkCase columns_case(
    std::string name, std::int32_t bandwidth,
    std::int32_t source, std::int32_t destination,
    std::int32_t columns, Precision precision, std::uint32_t seed) {
    BenchmarkCase result;
    result.name = std::move(name);
    result.operation = Operation::columns;
    result.vertical = make_plan(source, destination, bandwidth, precision);
    result.vector_count = columns;
    result.input_rows = source;
    result.input_columns = columns;
    result.output_rows = destination;
    result.output_columns = columns;
    initialize_case(result, seed);
    return result;
}

BenchmarkCase image_case(
    std::string name,
    std::int32_t horizontal_bandwidth,
    std::int32_t horizontal_source,
    std::int32_t horizontal_destination,
    std::int32_t vertical_bandwidth,
    std::int32_t vertical_source,
    std::int32_t vertical_destination,
    Precision precision, std::uint32_t seed) {
    BenchmarkCase result;
    result.name = std::move(name);
    result.operation = Operation::image_2d;
    result.horizontal = make_plan(
        horizontal_source, horizontal_destination,
        horizontal_bandwidth, precision);
    result.vertical = make_plan(
        vertical_source, vertical_destination,
        vertical_bandwidth, precision);
    result.input_rows = vertical_source;
    result.input_columns = horizontal_source;
    result.output_rows = vertical_destination;
    result.output_columns = horizontal_destination;
    initialize_case(result, seed);
    return result;
}

std::vector<BenchmarkCase> make_cases(Precision precision) {
    std::vector<BenchmarkCase> result;
    result.reserve(7U);
    result.push_back(rows_case(
        "rows_b1_769x576x257", 1, 769, 576, 257,
        precision, 0xf6400101U));
    result.push_back(columns_case(
        "columns_b3_433x324x577", 3, 433, 324, 577,
        precision, 0xf6400302U));
    result.push_back(rows_case(
        "rows_b5_769x576x257", 5, 769, 576, 257,
        precision, 0xf6400503U));
    result.push_back(columns_case(
        "columns_b7_433x324x577", 7, 433, 324, 577,
        precision, 0xf6400704U));
    result.push_back(rows_case(
        "rows_generic_b9_385x288x193", 9, 385, 288, 193,
        precision, 0xf6400905U));
    result.push_back(image_case(
        "2d_b3_b5_513x289_to_384x216",
        3, 513, 384, 5, 289, 216,
        precision, 0xf6423506U));
    result.push_back(image_case(
        "2d_b7_b9_385x217_to_288x162",
        7, 385, 288, 9, 217, 162,
        precision, 0xf6427907U));
    return result;
}

void prepare_cases(
    dsmvc::cuda_detail::CudaExecutor &cuda,
    const std::vector<BenchmarkCase> &cases) {
    for (const auto &item : cases) {
        if (item.horizontal) cuda.prepare(item.horizontal);
        if (item.vertical) cuda.prepare(item.vertical);
    }
    cuda.seal();
}

void execute_cpu(BenchmarkCase &item, const dsmvc::CpuExecutor &cpu) {
    switch (item.operation) {
    case Operation::rows:
        cpu.inverse_rows(
            *item.horizontal, item.input.data(), item.input_stride,
            item.cpu_output.data(), item.output_stride, item.vector_count);
        break;
    case Operation::columns:
        cpu.inverse_columns(
            *item.vertical, item.input.data(), item.input_stride,
            item.cpu_output.data(), item.output_stride, item.vector_count);
        break;
    case Operation::image_2d:
        cpu.inverse_2d(
            *item.horizontal, *item.vertical,
            item.input.data(), item.input_stride,
            item.cpu_output.data(), item.output_stride);
        break;
    }
}

void execute_cuda(
    BenchmarkCase &item, const dsmvc::cuda_detail::CudaExecutor &cuda) {
    switch (item.operation) {
    case Operation::rows:
        cuda.inverse_rows(
            *item.horizontal, item.input.data(), item.input_stride,
            item.cuda_output.data(), item.output_stride,
            item.vector_count, {});
        break;
    case Operation::columns:
        cuda.inverse_columns(
            *item.vertical, item.input.data(), item.input_stride,
            item.cuda_output.data(), item.output_stride,
            item.vector_count, {});
        break;
    case Operation::image_2d:
        cuda.inverse_2d(
            *item.horizontal, *item.vertical,
            item.input.data(), item.input_stride,
            item.cuda_output.data(), item.output_stride, {});
        break;
    }
}

std::int64_t time_cpu(
    BenchmarkCase &item, const dsmvc::CpuExecutor &cpu) {
    const auto started = Clock::now();
    execute_cpu(item, cpu);
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - started).count();
}

std::int64_t time_cuda(
    BenchmarkCase &item, const dsmvc::cuda_detail::CudaExecutor &cuda) {
    const auto started = Clock::now();
    execute_cuda(item, cuda);
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - started).count();
}

std::uint32_t ordered_float_bits(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return (bits & 0x80000000U) != 0U ? ~bits : bits | 0x80000000U;
}

std::uint32_t float_ulp_distance(float left, float right) {
    const auto lhs = ordered_float_bits(left);
    const auto rhs = ordered_float_bits(right);
    return lhs > rhs ? lhs - rhs : rhs - lhs;
}

struct Correctness {
    std::uint32_t maximum_ulp = 0U;
    double maximum_absolute = 0.0;
    double checksum = 0.0;
};

Correctness check_outputs(const BenchmarkCase &item, Precision precision) {
    Correctness result;
    for (std::int32_t row = 0; row < item.output_rows; ++row) {
        for (std::int32_t column = 0; column < item.output_columns; ++column) {
            const auto index = static_cast<std::size_t>(row)
                    * static_cast<std::size_t>(item.output_stride)
                + static_cast<std::size_t>(column);
            const float expected = item.cpu_output[index];
            const float actual = item.cuda_output[index];
            if (!std::isfinite(expected) || !std::isfinite(actual)) {
                throw std::runtime_error(item.name + " produced nonfinite output");
            }
            result.maximum_ulp = std::max(
                result.maximum_ulp, float_ulp_distance(expected, actual));
            result.maximum_absolute = std::max(
                result.maximum_absolute,
                std::abs(static_cast<double>(expected)
                         - static_cast<double>(actual)));
            result.checksum += static_cast<double>(actual)
                * static_cast<double>((column & 15) + 1);
        }
        for (std::int32_t column = item.output_columns;
             column < item.output_stride; ++column) {
            const auto index = static_cast<std::size_t>(row)
                    * static_cast<std::size_t>(item.output_stride)
                + static_cast<std::size_t>(column);
            if (item.cuda_output[index] != -41.0F) {
                throw std::runtime_error(item.name + " overwrote output padding");
            }
        }
    }
    if (precision == Precision::float64 && result.maximum_ulp > 1U) {
        throw std::runtime_error(item.name + " exceeded one output ULP");
    }
    if (precision == Precision::float32 && result.maximum_absolute > 1.0e-3) {
        throw std::runtime_error(item.name + " failed the F32 control bound");
    }
    return result;
}

void write_case_metadata(
    std::ostream &output, const BenchmarkCase &item, Precision precision) {
    output << "{\"event\":\"case\",\"precision\":"
           << quoted(precision_name(precision))
           << ",\"case\":" << quoted(item.name)
           << ",\"operation\":" << quoted(operation_name(item.operation))
           << ",\"input_rows\":" << item.input_rows
           << ",\"input_columns\":" << item.input_columns
           << ",\"output_rows\":" << item.output_rows
           << ",\"output_columns\":" << item.output_columns
           << ",\"input_stride\":" << item.input_stride
           << ",\"output_stride\":" << item.output_stride;
    if (item.horizontal) {
        output << ",\"horizontal_bandwidth\":"
               << item.horizontal->half_bandwidth;
    }
    if (item.vertical) {
        output << ",\"vertical_bandwidth\":"
               << item.vertical->half_bandwidth;
    }
    output << "}\n";
}

void write_sample(
    std::ostream &output, std::string_view event,
    const BenchmarkCase &item, Precision precision,
    int sample, std::string_view order,
    std::int64_t cpu_ns, std::int64_t cuda_ns) {
    output << "{\"event\":" << quoted(event)
           << ",\"precision\":" << quoted(precision_name(precision))
           << ",\"case\":" << quoted(item.name)
           << ",\"sample\":" << sample
           << ",\"order\":" << quoted(order)
           << ",\"cpu_ns\":" << cpu_ns
           << ",\"cuda_ns\":" << cuda_ns
           << ",\"cpu_over_cuda\":" << std::setprecision(17)
           << static_cast<double>(cpu_ns) / static_cast<double>(cuda_ns)
           << "}\n";
    output.flush();
}

void run_precision(
    std::ostream &output, const Options &options, Precision precision,
    std::vector<BenchmarkCase> cases) {
    const dsmvc::CpuExecutor cpu(dsmvc::CpuPath::scalar);
    dsmvc::cuda_detail::CudaExecutor cuda;
    prepare_cases(cuda, cases);

    for (auto &item : cases) {
        write_case_metadata(output, item, precision);
        execute_cpu(item, cpu);
        execute_cuda(item, cuda);
        const auto correctness = check_outputs(item, precision);
        output << "{\"event\":\"correctness\",\"precision\":"
               << quoted(precision_name(precision))
               << ",\"case\":" << quoted(item.name)
               << ",\"maximum_ulp\":" << correctness.maximum_ulp
               << ",\"maximum_absolute\":" << std::setprecision(17)
               << correctness.maximum_absolute
               << ",\"checksum\":" << correctness.checksum << "}\n";

        for (int warmup = 0; warmup < options.warmups; ++warmup) {
            std::int64_t cpu_ns = 0;
            std::int64_t cuda_ns = 0;
            const bool cpu_first = (warmup & 1) == 0;
            if (cpu_first) {
                cpu_ns = time_cpu(item, cpu);
                cuda_ns = time_cuda(item, cuda);
            } else {
                cuda_ns = time_cuda(item, cuda);
                cpu_ns = time_cpu(item, cpu);
            }
            write_sample(
                output, "warmup", item, precision, warmup,
                cpu_first ? "cpu-cuda" : "cuda-cpu", cpu_ns, cuda_ns);
        }
        for (int pair = 0; pair < options.pairs; ++pair) {
            std::int64_t cpu_ns = 0;
            std::int64_t cuda_ns = 0;
            const bool cpu_first = (pair & 1) == 0;
            if (cpu_first) {
                cpu_ns = time_cpu(item, cpu);
                cuda_ns = time_cuda(item, cuda);
            } else {
                cuda_ns = time_cuda(item, cuda);
                cpu_ns = time_cpu(item, cpu);
            }
            write_sample(
                output, "sample", item, precision, pair,
                cpu_first ? "cpu-cuda" : "cuda-cpu", cpu_ns, cuda_ns);
        }
    }
}

void write_metadata(std::ostream &output, const Options &options) {
    int runtime_version = 0;
    int driver_version = 0;
    int device = 0;
    int clock_rate = 0;
    int memory_clock_rate = 0;
    int memory_bus_width = 0;
    cudaDeviceProp properties{};
    if (cudaRuntimeGetVersion(&runtime_version) != cudaSuccess
        || cudaDriverGetVersion(&driver_version) != cudaSuccess
        || cudaGetDevice(&device) != cudaSuccess
        || cudaGetDeviceProperties(&properties, device) != cudaSuccess
        || cudaDeviceGetAttribute(
               &clock_rate, cudaDevAttrClockRate, device) != cudaSuccess
        || cudaDeviceGetAttribute(
               &memory_clock_rate, cudaDevAttrMemoryClockRate,
               device) != cudaSuccess
        || cudaDeviceGetAttribute(
               &memory_bus_width, cudaDevAttrGlobalMemoryBusWidth,
               device) != cudaSuccess) {
        throw std::runtime_error("failed to query CUDA benchmark provenance");
    }
    output << "{\"event\":\"metadata\","
           << "\"schema\":\"dsmvc-cuda-f64-executor-paired-v1\","
           << "\"label\":" << quoted(options.label) << ','
           << "\"source_sha\":" << quoted(options.source_sha) << ','
           << "\"binary_sha256\":" << quoted(options.binary_sha) << ','
           << "\"command\":" << quoted(options.command) << ','
           << "\"pairs\":" << options.pairs << ','
           << "\"warmups\":" << options.warmups << ','
           << "\"clock\":\"steady_clock_nanoseconds\","
           << "\"cpu_path\":\"scalar\","
           << "\"cxx_compiler\":" << quoted(__VERSION__) << ','
           << "\"logical_cpu_count\":"
           << std::thread::hardware_concurrency() << ','
           << "\"gpu_name\":" << quoted(properties.name) << ','
           << "\"compute_capability\":"
           << quoted(std::to_string(properties.major) + "."
                     + std::to_string(properties.minor)) << ','
           << "\"global_memory_bytes\":" << properties.totalGlobalMem << ','
           << "\"clock_rate_khz\":" << clock_rate << ','
           << "\"memory_clock_rate_khz\":"
           << memory_clock_rate << ','
           << "\"memory_bus_width_bits\":"
           << memory_bus_width << ','
           << "\"cuda_runtime_version\":" << runtime_version << ','
           << "\"cuda_driver_version\":" << driver_version << ','
           << "\"cuda_environment\":{";
    constexpr std::array environment_names{
        "DSMVC_CUDA_STREAMS",
        "DSMVC_CUDA_SPLIT_RHS",
        "DSMVC_CUDA_HOST_TRANSFER",
        "DSMVC_CUDA_HORIZONTAL_THREADS",
        "DSMVC_CUDA_VERTICAL_THREADS",
        "DSMVC_CUDA_SPLIT_HORIZONTAL_THREADS",
        "DSMVC_CUDA_SPLIT_VERTICAL_THREADS",
        "DSMVC_CUDA_HORIZONTAL_GLOBAL_TRANSPOSE",
        "DSMVC_CUDA_INPUT_CACHE_MB",
        "DSMVC_CUDA_PLAN_CACHE_MB",
    };
    for (std::size_t index = 0U; index < environment_names.size(); ++index) {
        if (index != 0U) output << ',';
        const char *value = std::getenv(environment_names[index]);
        output << quoted(environment_names[index]) << ':';
        if (value) {
            output << quoted(value);
        } else {
            output << "null";
        }
    }
    output << "}}\n";
    output.flush();
}

int run(const Options &options) {
    std::ofstream file;
    std::ostream *output = &std::cout;
    if (options.output_path) {
        file.open(*options.output_path, std::ios::out | std::ios::trunc);
        if (!file) {
            throw std::runtime_error(
                "cannot open benchmark output: " + *options.output_path);
        }
        output = &file;
    }

    std::vector<BenchmarkCase> f64_cases;
    std::vector<BenchmarkCase> f32_cases;
    if (options.mode == Options::Mode::float64
        || options.mode == Options::Mode::all) {
        f64_cases = make_cases(Precision::float64);
    }
    if (options.mode == Options::Mode::float32
        || options.mode == Options::Mode::all) {
        f32_cases = make_cases(Precision::float32);
    }
    const auto validated_cases = f64_cases.size() + f32_cases.size();

    if (!dsmvc::cuda_detail::backend_available()) {
        *output << "{\"event\":\"skip\",\"reason\":"
                << quoted("no CUDA device is available")
                << ",\"label\":" << quoted(options.label)
                << ",\"source_sha\":" << quoted(options.source_sha)
                << ",\"binary_sha256\":" << quoted(options.binary_sha)
                << ",\"validated_case_count\":" << validated_cases
                << "}\n";
        output->flush();
        std::cerr << "CUDA benchmark skipped: no CUDA device is available\n";
        return 77;
    }

    write_metadata(*output, options);
    if (options.mode == Options::Mode::float64
        || options.mode == Options::Mode::all) {
        run_precision(
            *output, options, Precision::float64, std::move(f64_cases));
    }
    if (options.mode == Options::Mode::float32
        || options.mode == Options::Mode::all) {
        run_precision(
            *output, options, Precision::float32, std::move(f32_cases));
    }
    *output << "{\"event\":\"complete\",\"status\":\"passed\"}\n";
    output->flush();
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception &error) {
        std::cerr << "CUDA F64 executor benchmark failure: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
