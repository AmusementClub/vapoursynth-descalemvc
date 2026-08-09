#include "vulkan_executor.hpp"

#include <dsmvc/engine.hpp>

#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifndef DSMVC_VULKAN_F64_BENCHMARK_F32_ONLY
#define DSMVC_VULKAN_F64_BENCHMARK_F32_ONLY 0
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::int32_t source_width = 1920;
constexpr std::int32_t source_height = 1080;
constexpr std::int32_t destination_width = 1692;
constexpr std::int32_t destination_height = 952;
constexpr std::int32_t row_count = 256;
constexpr std::int32_t column_count = 512;
constexpr std::size_t minimum_samples = 30U;

std::atomic<std::uint64_t> benchmark_sink{};

enum class Precision { float32, float64 };
enum class Operation { rows, columns, two_dimensional };

struct Configuration {
    Precision precision = DSMVC_VULKAN_F64_BENCHMARK_F32_ONLY
        ? Precision::float32 : Precision::float64;
    std::size_t samples = minimum_samples;
    std::size_t warmups = 3U;
    bool server = false;
    std::string label = "candidate";
    std::string source_sha = "unknown";
    std::string binary_sha = "unknown";
    std::string command;
};

struct DeviceProbe {
    bool hardware = false;
    std::string reason;
    VkPhysicalDeviceProperties properties{};
    std::uint32_t index = 0U;
};

struct InstanceHandle {
    VkInstance value = VK_NULL_HANDLE;

    ~InstanceHandle() {
        if (value != VK_NULL_HANDLE) vkDestroyInstance(value, nullptr);
    }
};

struct PlanSpec {
    std::string_view name;
    dsmvc::KernelKind kernel;
    std::int32_t taps;
    std::int32_t expected_half_bandwidth;
};

constexpr PlanSpec plan_specs[] = {
    {"b1", dsmvc::KernelKind::bilinear, 1, 1},
    {"b3", dsmvc::KernelKind::bicubic, 3, 3},
    {"b5", dsmvc::KernelKind::lanczos, 3, 5},
    {"b7", dsmvc::KernelKind::spline64, 4, 7},
    {"generic-b9", dsmvc::KernelKind::lanczos, 5, 9},
};

[[nodiscard]] std::size_t parse_size(
    std::string_view text, std::string_view option, bool allow_zero = false) {
    std::size_t value = 0U;
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), value, 10);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()
        || (!allow_zero && value == 0U)) {
        throw std::invalid_argument(
            std::string{option} + " requires "
            + (allow_zero ? "a nonnegative" : "a positive") + " integer");
    }
    return value;
}

[[nodiscard]] Configuration parse_arguments(int argc, char **argv) {
    Configuration result;
    for (int index = 0; index < argc; ++index) {
        if (index != 0) result.command.push_back(' ');
        result.command += argv[index];
    }
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        const auto next = [&](std::string_view option) {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string{option} + " requires a value");
            }
            return std::string_view{argv[++index]};
        };
        if (argument == "--precision") {
            const auto value = next(argument);
            if (value == "f32") {
                result.precision = Precision::float32;
            } else if (value == "f64") {
#if DSMVC_VULKAN_F64_BENCHMARK_F32_ONLY
                throw std::invalid_argument(
                    "this baseline-compatible binary supports only --precision f32");
#else
                result.precision = Precision::float64;
#endif
            } else {
                throw std::invalid_argument("--precision must be f32 or f64");
            }
        } else if (argument == "--samples") {
            result.samples = parse_size(next(argument), argument);
        } else if (argument == "--warmups") {
            result.warmups = parse_size(next(argument), argument, true);
        } else if (argument == "--label") {
            result.label = std::string{next(argument)};
        } else if (argument == "--source-sha") {
            result.source_sha = std::string{next(argument)};
        } else if (argument == "--binary-sha") {
            result.binary_sha = std::string{next(argument)};
        } else if (argument == "--server-f32") {
            result.server = true;
            result.precision = Precision::float32;
        } else {
            throw std::invalid_argument(
                "usage: dsmvc_vulkan_f64_benchmark "
                "[--precision f32|f64] [--samples N] [--warmups N] "
                "[--label NAME] [--source-sha SHA] [--binary-sha SHA] "
                "[--server-f32]");
        }
    }
    if (result.samples < minimum_samples) {
        throw std::invalid_argument("--samples must be at least 30");
    }
    if (result.samples > 200U) {
        throw std::invalid_argument("--samples must be at most 200");
    }
    if (result.warmups > 20U) {
        throw std::invalid_argument("--warmups must be at most 20");
    }
    return result;
}

[[nodiscard]] bool parse_u32(
    std::string_view text, int base, std::uint32_t &value) noexcept {
    if (base == 16 && text.size() > 2U && text[0] == '0'
        && (text[1] == 'x' || text[1] == 'X')) {
        text.remove_prefix(2U);
    }
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), value, base);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

[[nodiscard]] DeviceProbe probe_device() {
    const VkApplicationInfo application{
        VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr,
        "dsmvc Vulkan Float64 benchmark", 1U,
        "dsmvc", 1U, VK_API_VERSION_1_2,
    };
    const VkInstanceCreateInfo create_info{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0U,
        &application, 0U, nullptr, 0U, nullptr,
    };
    InstanceHandle instance;
    const VkResult created = vkCreateInstance(&create_info, nullptr, &instance.value);
    if (created != VK_SUCCESS) {
        return {false, "vkCreateInstance failed with VkResult "
                           + std::to_string(static_cast<int>(created))};
    }

    std::uint32_t count = 0U;
    VkResult result = vkEnumeratePhysicalDevices(instance.value, &count, nullptr);
    if (result != VK_SUCCESS || count == 0U) {
        return {false, "no Vulkan physical device was enumerated"};
    }
    std::vector<VkPhysicalDevice> devices(count);
    result = vkEnumeratePhysicalDevices(instance.value, &count, devices.data());
    if (result != VK_SUCCESS || count == 0U) {
        return {false, "Vulkan physical-device enumeration failed"};
    }
    devices.resize(count);

    std::optional<std::uint32_t> selected;
    if (const char *requested_text = std::getenv("DSMVC_VULKAN_DEVICE");
        requested_text && *requested_text) {
        const std::string_view requested{requested_text};
        const auto separator = requested.find(':');
        if (separator == std::string_view::npos) {
            std::uint32_t index = 0U;
            if (!parse_u32(requested, 10, index) || index >= count) {
                return {false, "DSMVC_VULKAN_DEVICE does not name an enumerated index"};
            }
            selected = index;
        } else {
            std::uint32_t vendor = 0U;
            std::uint32_t device = 0U;
            if (!parse_u32(requested.substr(0U, separator), 16, vendor)
                || !parse_u32(requested.substr(separator + 1U), 16, device)) {
                return {false, "DSMVC_VULKAN_DEVICE vendor:device is invalid"};
            }
            for (std::uint32_t index = 0U; index < count; ++index) {
                VkPhysicalDeviceProperties properties{};
                vkGetPhysicalDeviceProperties(devices[index], &properties);
                if (properties.vendorID == vendor && properties.deviceID == device) {
                    selected = index;
                    break;
                }
            }
            if (!selected) {
                return {false, "DSMVC_VULKAN_DEVICE vendor:device was not enumerated"};
            }
        }
    } else {
        for (std::uint32_t index = 0U; index < count; ++index) {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(devices[index], &properties);
            if (properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) {
                selected = index;
                break;
            }
        }
        if (!selected) selected = 0U;
    }

    DeviceProbe probe;
    probe.index = *selected;
    vkGetPhysicalDeviceProperties(devices[*selected], &probe.properties);
    if (probe.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
        probe.reason = std::string{"selected Vulkan device is software/CPU: "}
            + probe.properties.deviceName;
        return probe;
    }

    std::uint32_t queue_count = 0U;
    vkGetPhysicalDeviceQueueFamilyProperties(
        devices[*selected], &queue_count, nullptr);
    std::vector<VkQueueFamilyProperties> queues(queue_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        devices[*selected], &queue_count, queues.data());
    if (std::none_of(queues.begin(), queues.end(), [](const auto &queue) {
            return queue.queueCount != 0U
                && (queue.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0U;
        })) {
        probe.reason = "selected Vulkan device has no compute queue";
        return probe;
    }
    probe.hardware = true;
    return probe;
}

[[nodiscard]] dsmvc::AxisRequest make_request(
    std::int32_t source, std::int32_t destination, double active_length,
    double shift, const PlanSpec &spec, Precision precision) {
    dsmvc::AxisRequest request;
    request.source_size = source;
    request.destination_size = destination;
    request.active_length = active_length;
    request.shift = shift;
    request.kernel.kind = spec.kernel;
    request.kernel.taps = spec.taps;
    request.kernel.b = 0.0;
    request.kernel.c = 0.5;
    request.border = dsmvc::BorderMode::symmetric;
    request.f64_mode = precision == Precision::float64
        ? dsmvc::F64Mode::float64_only : dsmvc::F64Mode::float32_only;
    return request;
}

void fill_input(std::vector<float> &values, std::uint32_t seed) {
    std::uint32_t state = seed;
    for (float &value : values) {
        state ^= state << 13U;
        state ^= state >> 17U;
        state ^= state << 5U;
        value = static_cast<float>(state & 0xffffU) / 65535.0F;
    }
}

[[nodiscard]] std::string_view operation_name(Operation operation) noexcept {
    switch (operation) {
    case Operation::rows: return "rows";
    case Operation::columns: return "columns";
    case Operation::two_dimensional: return "2d";
    }
    return "unknown";
}

struct Fixture {
    std::string name;
    Operation operation{};
    std::shared_ptr<const dsmvc::AxisPlan> horizontal;
    std::shared_ptr<const dsmvc::AxisPlan> vertical;
    std::shared_ptr<std::vector<float>> input;
    std::vector<float> cpu_output;
    std::vector<float> vulkan_output;
    std::ptrdiff_t input_stride = 0;
    std::ptrdiff_t output_stride = 0;
    std::int32_t work_count = 0;

    [[nodiscard]] std::int32_t half_bandwidth() const noexcept {
        return operation == Operation::columns
            ? vertical->half_bandwidth : horizontal->half_bandwidth;
    }

    void execute_cpu(const dsmvc::CpuExecutor &cpu) {
        switch (operation) {
        case Operation::rows:
            cpu.inverse_rows(
                *horizontal, input->data(), input_stride,
                cpu_output.data(), output_stride, work_count);
            break;
        case Operation::columns:
            cpu.inverse_columns(
                *vertical, input->data(), input_stride,
                cpu_output.data(), output_stride, work_count);
            break;
        case Operation::two_dimensional:
            cpu.inverse_2d(
                *horizontal, *vertical, input->data(), input_stride,
                cpu_output.data(), output_stride);
            break;
        }
    }

    void execute_vulkan(const dsmvc::vulkan_detail::VulkanExecutor &vulkan) {
        std::shared_ptr<const void> lifetime = input;
        switch (operation) {
        case Operation::rows:
            vulkan.inverse_rows(
                *horizontal, input->data(), input_stride,
                vulkan_output.data(), output_stride, work_count, lifetime);
            break;
        case Operation::columns:
            vulkan.inverse_columns(
                *vertical, input->data(), input_stride,
                vulkan_output.data(), output_stride, work_count, lifetime);
            break;
        case Operation::two_dimensional:
            vulkan.inverse_2d(
                *horizontal, *vertical, input->data(), input_stride,
                vulkan_output.data(), output_stride, lifetime);
            break;
        }
    }
};

[[nodiscard]] std::vector<Fixture> make_fixtures(Precision precision) {
    std::vector<Fixture> fixtures;
    fixtures.reserve(std::size(plan_specs) * 3U);
    std::uint32_t seed = 0x64000000U;
    for (const auto &spec : plan_specs) {
        auto horizontal = std::make_shared<const dsmvc::AxisPlan>(
            dsmvc::build_axis_plan(make_request(
                source_width, destination_width, 1691.5555555555557,
                0.2222222222221717, spec, precision)));
        auto vertical = std::make_shared<const dsmvc::AxisPlan>(
            dsmvc::build_axis_plan(make_request(
                source_height, destination_height, 951.5, 0.25,
                spec, precision)));
        if (horizontal->half_bandwidth != spec.expected_half_bandwidth
            || vertical->half_bandwidth != spec.expected_half_bandwidth) {
            throw std::runtime_error(
                std::string{"benchmark fixture bandwidth drifted: "}
                + std::string{spec.name});
        }

        Fixture rows_fixture;
        rows_fixture.name = std::string{"rows-"} + std::string{spec.name};
        rows_fixture.operation = Operation::rows;
        rows_fixture.horizontal = horizontal;
        rows_fixture.vertical = vertical;
        rows_fixture.input_stride = source_width;
        rows_fixture.output_stride = destination_width;
        rows_fixture.work_count = row_count;
        rows_fixture.input = std::make_shared<std::vector<float>>(
            static_cast<std::size_t>(row_count) * source_width);
        rows_fixture.cpu_output.resize(
            static_cast<std::size_t>(row_count) * destination_width);
        rows_fixture.vulkan_output.resize(rows_fixture.cpu_output.size());
        fill_input(*rows_fixture.input, seed++);
        fixtures.push_back(std::move(rows_fixture));

        Fixture columns_fixture;
        columns_fixture.name = std::string{"columns-"} + std::string{spec.name};
        columns_fixture.operation = Operation::columns;
        columns_fixture.horizontal = horizontal;
        columns_fixture.vertical = vertical;
        columns_fixture.input_stride = column_count;
        columns_fixture.output_stride = column_count;
        columns_fixture.work_count = column_count;
        columns_fixture.input = std::make_shared<std::vector<float>>(
            static_cast<std::size_t>(source_height) * column_count);
        columns_fixture.cpu_output.resize(
            static_cast<std::size_t>(destination_height) * column_count);
        columns_fixture.vulkan_output.resize(columns_fixture.cpu_output.size());
        fill_input(*columns_fixture.input, seed++);
        fixtures.push_back(std::move(columns_fixture));

        Fixture two_dimensional_fixture;
        two_dimensional_fixture.name =
            std::string{"2d-"} + std::string{spec.name};
        two_dimensional_fixture.operation = Operation::two_dimensional;
        two_dimensional_fixture.horizontal = horizontal;
        two_dimensional_fixture.vertical = vertical;
        two_dimensional_fixture.input_stride = source_width;
        two_dimensional_fixture.output_stride = destination_width;
        two_dimensional_fixture.input = std::make_shared<std::vector<float>>(
            static_cast<std::size_t>(source_height) * source_width);
        two_dimensional_fixture.cpu_output.resize(
            static_cast<std::size_t>(destination_height) * destination_width);
        two_dimensional_fixture.vulkan_output.resize(
            two_dimensional_fixture.cpu_output.size());
        fill_input(*two_dimensional_fixture.input, seed++);
        fixtures.push_back(std::move(two_dimensional_fixture));
    }
    return fixtures;
}

void consume(const std::vector<float> &values) noexcept {
    if (values.empty()) return;
    const std::uint64_t bits = std::bit_cast<std::uint32_t>(values.front())
        ^ (static_cast<std::uint64_t>(
               std::bit_cast<std::uint32_t>(values[values.size() / 2U])) << 16U)
        ^ (static_cast<std::uint64_t>(
               std::bit_cast<std::uint32_t>(values.back())) << 32U);
    benchmark_sink.fetch_xor(bits, std::memory_order_relaxed);
}

[[nodiscard]] std::uint32_t ordered_float_bits(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    return (bits & 0x80000000U) != 0U ? ~bits : bits | 0x80000000U;
}

[[nodiscard]] std::uint32_t ulp_distance(float left, float right) noexcept {
    const std::uint32_t lhs = ordered_float_bits(left);
    const std::uint32_t rhs = ordered_float_bits(right);
    return lhs > rhs ? lhs - rhs : rhs - lhs;
}

void verify_fixture(
    Fixture &fixture, Precision precision, const dsmvc::CpuExecutor &cpu,
    const dsmvc::vulkan_detail::VulkanExecutor &vulkan) {
    fixture.execute_cpu(cpu);
    fixture.execute_vulkan(vulkan);
    std::uint32_t maximum_ulp = 0U;
    double maximum_absolute = 0.0;
    double checksum = 0.0;
    for (std::size_t index = 0U; index < fixture.cpu_output.size(); ++index) {
        const float expected = fixture.cpu_output[index];
        const float actual = fixture.vulkan_output[index];
        if (!std::isfinite(expected) || !std::isfinite(actual)) {
            throw std::runtime_error(
                fixture.name + " produced non-finite benchmark output");
        }
        maximum_ulp = std::max(maximum_ulp, ulp_distance(expected, actual));
        maximum_absolute = std::max(
            maximum_absolute,
            std::abs(static_cast<double>(expected)
                     - static_cast<double>(actual)));
        checksum += static_cast<double>(actual)
            * static_cast<double>((index & 15U) + 1U);
    }
    if (precision == Precision::float64 && maximum_ulp > 1U) {
        throw std::runtime_error(
            fixture.name + " exceeded one output ULP before timing");
    }
    if (precision == Precision::float32 && maximum_absolute > 1.0e-3) {
        throw std::runtime_error(
            fixture.name + " failed the F32 control bound before timing");
    }
    std::cerr << "correctness fixture=" << fixture.name
              << " maximum_ulp=" << maximum_ulp
              << " maximum_absolute=" << std::setprecision(17)
              << maximum_absolute << " checksum=" << checksum << '\n';
}

template <class Function>
[[nodiscard]] std::uint64_t measure(Function function) {
    const auto started = Clock::now();
    function();
    const auto stopped = Clock::now();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            stopped - started).count());
}

[[nodiscard]] double median(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("no benchmark samples");
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2U;
    return (values.size() & 1U) != 0U
        ? values[middle]
        : (values[middle - 1U] + values[middle]) / 2.0;
}

[[nodiscard]] std::string csv_field(std::string_view value) {
    if (value.find_first_of(",\"\r\n") == std::string_view::npos) {
        return std::string{value};
    }
    std::string result{"\""};
    for (const char character : value) {
        if (character == '\"') result.push_back('\"');
        result.push_back(character);
    }
    result.push_back('\"');
    return result;
}

[[nodiscard]] std::string compiler_identity() {
#if defined(__clang__)
    return std::string{"Clang "} + __clang_version__;
#elif defined(__GNUC__)
    return std::string{"GCC "} + __VERSION__;
#elif defined(_MSC_VER)
    return std::string{"MSVC "} + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}

[[nodiscard]] std::string environment_value(const char *name) {
    const char *value = std::getenv(name);
    return value ? std::string{value} : std::string{"<unset>"};
}

void prepare_executors(
    std::vector<Fixture> &fixtures, dsmvc::CpuExecutor &cpu,
    dsmvc::vulkan_detail::VulkanExecutor &vulkan) {
    for (const auto &fixture : fixtures) {
        cpu.prepare(fixture.horizontal);
        cpu.prepare(fixture.vertical);
        vulkan.prepare(fixture.horizontal);
        vulkan.prepare(fixture.vertical);
    }
    cpu.seal();
    vulkan.seal();
}

void warm_fixture(
    Fixture &fixture, const dsmvc::CpuExecutor &cpu,
    const dsmvc::vulkan_detail::VulkanExecutor &vulkan,
    std::size_t warmups, bool vulkan_only) {
    for (std::size_t iteration = 0U; iteration < warmups; ++iteration) {
        if (!vulkan_only) {
            fixture.execute_cpu(cpu);
            consume(fixture.cpu_output);
        }
        fixture.execute_vulkan(vulkan);
        consume(fixture.vulkan_output);
    }
}

int run_server(
    const Configuration &configuration, const DeviceProbe &device,
    std::vector<Fixture> &fixtures, const dsmvc::CpuExecutor &cpu,
    const dsmvc::vulkan_detail::VulkanExecutor &vulkan) {
    for (auto &fixture : fixtures) {
        verify_fixture(fixture, Precision::float32, cpu, vulkan);
        warm_fixture(fixture, cpu, vulkan, configuration.warmups, true);
    }
    for (std::size_t index = 0U; index < fixtures.size(); ++index) {
        const auto &fixture = fixtures[index];
        std::cout << "FIXTURE," << index << ',' << csv_field(fixture.name) << ','
                  << operation_name(fixture.operation) << ','
                  << fixture.half_bandwidth() << '\n';
    }
    std::cout << "READY," << fixtures.size() << ','
              << csv_field(device.properties.deviceName) << '\n' << std::flush;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "QUIT") return 0;
        constexpr std::string_view prefix{"RUN,"};
        if (!line.starts_with(prefix)) {
            throw std::runtime_error("benchmark server received an invalid command");
        }
        const std::size_t index = parse_size(
            std::string_view{line}.substr(prefix.size()), "RUN index", true);
        if (index >= fixtures.size()) {
            throw std::runtime_error("benchmark server fixture index is out of range");
        }
        auto &fixture = fixtures[index];
        const std::uint64_t elapsed = measure([&] {
            fixture.execute_vulkan(vulkan);
        });
        consume(fixture.vulkan_output);
        std::cout << "RESULT," << index << ',' << elapsed << '\n' << std::flush;
    }
    return 0;
}

int run_paired(
    const Configuration &configuration, const DeviceProbe &device,
    std::vector<Fixture> &fixtures, const dsmvc::CpuExecutor &cpu,
    const dsmvc::vulkan_detail::VulkanExecutor &vulkan) {
    const std::string_view precision = configuration.precision == Precision::float64
        ? "f64" : "f32";
    std::cout
        << "schema,source_sha,binary_sha256,run_label,precision,command,"
           "cxx_compiler,logical_cpu_count,device_index,device_name,"
           "vendor_id,device_id,api_version,driver_version,fixture,operation,"
           "half_bandwidth,pair,pair_order,backend,duration_ns,"
           "vulkan_device_env,timeline_env,force_non_coherent_env,"
           "split_rhs_env,workgroup_env,input_cache_mb_env,plan_cache_mb_env\n";

    std::vector<std::uint64_t> cpu_totals(configuration.samples, 0U);
    std::vector<std::uint64_t> vulkan_totals(configuration.samples, 0U);
    for (auto &fixture : fixtures) {
        verify_fixture(fixture, configuration.precision, cpu, vulkan);
        warm_fixture(fixture, cpu, vulkan, configuration.warmups, false);
        std::vector<double> ratios;
        ratios.reserve(configuration.samples);
        for (std::size_t pair = 0U; pair < configuration.samples; ++pair) {
            const bool cpu_first = (pair & 1U) == 0U;
            const std::string_view order = cpu_first ? "cpu-vulkan" : "vulkan-cpu";
            std::uint64_t cpu_ns = 0U;
            std::uint64_t vulkan_ns = 0U;
            const auto run_cpu = [&] {
                cpu_ns = measure([&] { fixture.execute_cpu(cpu); });
                consume(fixture.cpu_output);
            };
            const auto run_vulkan = [&] {
                vulkan_ns = measure([&] { fixture.execute_vulkan(vulkan); });
                consume(fixture.vulkan_output);
            };
            if (cpu_first) {
                run_cpu();
                run_vulkan();
            } else {
                run_vulkan();
                run_cpu();
            }
            cpu_totals[pair] += cpu_ns;
            vulkan_totals[pair] += vulkan_ns;
            ratios.push_back(static_cast<double>(cpu_ns)
                             / static_cast<double>(vulkan_ns));

            const auto emit = [&](std::string_view backend, std::uint64_t elapsed) {
                std::cout << "dsmvc-vulkan-executor-paired-v1,"
                          << csv_field(configuration.source_sha) << ','
                          << csv_field(configuration.binary_sha) << ','
                          << csv_field(configuration.label) << ',' << precision << ','
                          << csv_field(configuration.command) << ','
                          << csv_field(compiler_identity()) << ','
                          << std::thread::hardware_concurrency() << ','
                          << device.index << ','
                          << csv_field(device.properties.deviceName) << ','
                          << device.properties.vendorID << ','
                          << device.properties.deviceID << ','
                          << device.properties.apiVersion << ','
                          << device.properties.driverVersion << ','
                          << csv_field(fixture.name) << ','
                          << operation_name(fixture.operation) << ','
                          << fixture.half_bandwidth() << ',' << pair << ','
                          << order << ',' << backend << ',' << elapsed << ','
                          << csv_field(environment_value("DSMVC_VULKAN_DEVICE")) << ','
                          << csv_field(environment_value("DSMVC_VULKAN_TIMELINE")) << ','
                          << csv_field(environment_value(
                                 "DSMVC_VULKAN_FORCE_NON_COHERENT")) << ','
                          << csv_field(environment_value("DSMVC_VULKAN_SPLIT_RHS")) << ','
                          << csv_field(environment_value("DSMVC_VULKAN_WORKGROUP")) << ','
                          << csv_field(environment_value(
                                 "DSMVC_VULKAN_INPUT_CACHE_MB")) << ','
                          << csv_field(environment_value(
                                 "DSMVC_VULKAN_PLAN_CACHE_MB")) << '\n';
            };
            if (cpu_first) {
                emit("cpu-scalar", cpu_ns);
                emit("vulkan", vulkan_ns);
            } else {
                emit("vulkan", vulkan_ns);
                emit("cpu-scalar", cpu_ns);
            }
        }
        std::cerr << "summary fixture=" << fixture.name
                  << " paired_median_cpu_over_vulkan=" << std::setprecision(8)
                  << median(std::move(ratios)) << '\n';
    }

    std::vector<double> total_ratios;
    total_ratios.reserve(configuration.samples);
    for (std::size_t pair = 0U; pair < configuration.samples; ++pair) {
        total_ratios.push_back(static_cast<double>(cpu_totals[pair])
                               / static_cast<double>(vulkan_totals[pair]));
    }
    std::cerr << "summary suite_paired_median_cpu_over_vulkan="
              << std::setprecision(8) << median(std::move(total_ratios)) << '\n';
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Configuration configuration = parse_arguments(argc, argv);
        const DeviceProbe device = probe_device();
        if (!device.hardware) {
            std::cerr << "SKIP: " << device.reason << '\n';
            return 77;
        }
        if (!configuration.server
            && (configuration.source_sha == "unknown"
                || configuration.binary_sha == "unknown")) {
            throw std::invalid_argument(
                "hardware measurements require --source-sha and --binary-sha");
        }

#if !DSMVC_VULKAN_F64_BENCHMARK_F32_ONLY
        if (configuration.precision == Precision::float64) {
            std::cerr
                << dsmvc::vulkan_detail::selected_float64_capability_report()
                << '\n';
            const auto capabilities =
                dsmvc::vulkan_detail::selected_float64_capabilities();
            if (!capabilities.strict_supported()) {
                std::cerr << "SKIP: " << capabilities.requirement_error() << '\n';
                return 77;
            }
        }
#endif

        auto fixtures = make_fixtures(configuration.precision);
        dsmvc::CpuExecutor cpu{dsmvc::CpuPath::scalar};
        dsmvc::vulkan_detail::VulkanExecutor vulkan;
        prepare_executors(fixtures, cpu, vulkan);
        if (configuration.server) {
            return run_server(
                configuration, device, fixtures, cpu, vulkan);
        }
        return run_paired(
            configuration, device, fixtures, cpu, vulkan);
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
