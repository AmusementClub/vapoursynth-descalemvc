#include "numerical_conformance.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Configuration {
    std::size_t samples = 5U;
    std::size_t iterations = 32U;
    std::string source_id = "unknown";
    std::filesystem::path json_output;
};

struct BenchmarkCase {
    std::string name;
    std::string route;
    std::string precision;
    std::shared_ptr<const dsmvc::AxisPlan> plan;
    std::string f32_plan_hash;
    std::string complete_plan_hash;
    std::size_t axes_per_iteration = 1U;
    std::function<void()> execute;
    std::function<std::string()> result_hash;
    std::function<bool()> finite;
    std::vector<double> raw_nanoseconds_per_axis;
};

[[nodiscard]] std::size_t parse_positive(
    std::string_view value, std::string_view option) {
    std::size_t result = 0U;
    if (value.empty()) {
        throw std::invalid_argument(std::string(option) + " requires a value");
    }
    for (const char character : value) {
        if (character < '0' || character > '9') {
            throw std::invalid_argument(
                std::string(option) + " must be a positive integer");
        }
        const auto digit = static_cast<std::size_t>(character - '0');
        if (result > (std::numeric_limits<std::size_t>::max() - digit) / 10U) {
            throw std::invalid_argument(std::string(option) + " is too large");
        }
        result = result * 10U + digit;
    }
    if (result == 0U) {
        throw std::invalid_argument(
            std::string(option) + " must be a positive integer");
    }
    return result;
}

[[nodiscard]] Configuration parse_arguments(int argc, char **argv) {
    Configuration result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        const auto next = [&](std::string_view option) {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    std::string(option) + " requires a value");
            }
            return std::string_view{argv[++index]};
        };
        if (argument == "--samples") {
            result.samples = parse_positive(next(argument), argument);
        } else if (argument == "--iterations") {
            result.iterations = parse_positive(next(argument), argument);
        } else if (argument == "--source-id") {
            result.source_id = std::string(next(argument));
        } else if (argument == "--json-out") {
            result.json_output = std::filesystem::path(next(argument));
        } else {
            throw std::invalid_argument(
                "usage: dsmvc_numerical_contract_benchmark "
                "[--samples N] [--iterations N] [--source-id ID] "
                "[--json-out PATH]");
        }
    }
    return result;
}

[[nodiscard]] std::string json_string(std::string_view value) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char character : value) {
        switch (character) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (character < 0x20U) {
                output << "\\u00" << std::hex << std::setfill('0')
                       << std::setw(2) << static_cast<unsigned>(character)
                       << std::dec;
            } else {
                output << static_cast<char>(character);
            }
        }
    }
    output << '"';
    return output.str();
}

[[nodiscard]] std::string compiler_identity() {
#if defined(__clang__)
    return std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("GCC ") + __VERSION__;
#elif defined(_MSC_VER)
    return std::string("MSVC ") + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}

[[nodiscard]] std::string architecture_identity() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#else
    return "unknown";
#endif
}

[[nodiscard]] std::string executable_hash(
    const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("failed to open benchmark executable");
    std::uint64_t hash = 1469598103934665603ULL;
    std::vector<char> buffer(64U * 1024U);
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        for (std::streamsize index = 0; index < count; ++index) {
            hash ^= static_cast<unsigned char>(buffer[static_cast<std::size_t>(index)]);
            hash *= 1099511628211ULL;
        }
    }
    if (!input.eof()) throw std::runtime_error("failed to hash benchmark executable");
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << hash;
    return output.str();
}

[[nodiscard]] double median(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("no benchmark samples");
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2U;
    return (values.size() & 1U) != 0U
        ? values[middle]
        : (values[middle - 1U] + values[middle]) / 2.0;
}

template <class T>
[[nodiscard]] bool all_finite(std::span<const T> values) {
    return std::all_of(values.begin(), values.end(), [](T value) {
        return std::isfinite(static_cast<double>(value));
    });
}

[[nodiscard]] std::vector<float> make_batched_input(
    const dsmvc::AxisPlan &plan, std::int32_t rows,
    std::int32_t stride, std::uint32_t seed) {
    std::vector<float> result(
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(stride),
        0.0F);
    for (std::int32_t row = 0; row < rows; ++row) {
        const auto input = dsmvc::numerical::make_normal_input(
            static_cast<std::size_t>(plan.source_size),
            seed + static_cast<std::uint32_t>(row));
        std::copy(input.begin(), input.end(),
                  result.begin() + static_cast<std::ptrdiff_t>(row) * stride);
    }
    return result;
}

[[nodiscard]] std::vector<float> pack_batched_output(
    const std::vector<float> &output, const dsmvc::AxisPlan &plan,
    std::int32_t rows, std::int32_t stride) {
    std::vector<float> packed;
    packed.reserve(static_cast<std::size_t>(rows)
                   * static_cast<std::size_t>(plan.destination_size));
    for (std::int32_t row = 0; row < rows; ++row) {
        const auto *begin = output.data()
            + static_cast<std::ptrdiff_t>(row) * stride;
        packed.insert(packed.end(), begin, begin + plan.destination_size);
    }
    return packed;
}

void append_axis_cases(
    std::vector<BenchmarkCase> &cases,
    const dsmvc::AxisRequest &request, std::uint32_t seed,
    std::string precision, std::string expected_f32_plan_hash,
    dsmvc::CpuExecutor &native) {
    auto plan = std::make_shared<const dsmvc::AxisPlan>(
        dsmvc::build_axis_plan(request));
    if (!plan->valid()) throw std::runtime_error("benchmark plan is invalid");
    const std::string actual_f32_plan_hash =
        dsmvc::numerical::f32_plan_hash(request, *plan);
    const std::string actual_complete_plan_hash =
        dsmvc::numerical::complete_plan_hash(request, *plan);
    if (actual_f32_plan_hash != expected_f32_plan_hash) {
        throw std::runtime_error(
            precision + " benchmark F32 plan hash drifted: "
                + actual_f32_plan_hash);
    }

    auto input_f32 = std::make_shared<std::vector<float>>(
        dsmvc::numerical::make_normal_input(
            static_cast<std::size_t>(plan->source_size), seed));
    auto output_f32 = std::make_shared<std::vector<float>>(
        static_cast<std::size_t>(plan->destination_size));

    if (plan->requires_float64()) {
        auto input_f64 = std::make_shared<std::vector<double>>(
            input_f32->begin(), input_f32->end());
        auto ordered_f64 = std::make_shared<std::vector<double>>(
            static_cast<std::size_t>(plan->destination_size));
        auto scalar_f64 = std::make_shared<std::vector<double>>(
            static_cast<std::size_t>(plan->destination_size));
        cases.push_back({
            precision + "-ordered-reference", "ordered-reference", precision,
            plan, actual_f32_plan_hash, actual_complete_plan_hash, 1U,
            [plan, input_f64, ordered_f64] {
                dsmvc::detail::inverse_axis_f64_ordered(
                    *plan, input_f64->data(), 1, ordered_f64->data(), 1);
            },
            [ordered_f64] {
                return dsmvc::numerical::output_hash<double>(*ordered_f64);
            },
            [ordered_f64] { return all_finite<double>(*ordered_f64); },
            {},
        });
        cases.push_back({
            precision + "-current-scalar", "current-scalar", precision,
            plan, actual_f32_plan_hash, actual_complete_plan_hash, 1U,
            [plan, input_f64, scalar_f64] {
                dsmvc::detail::inverse_axis_f64(
                    *plan, input_f64->data(), 1, scalar_f64->data(), 1);
            },
            [scalar_f64] {
                return dsmvc::numerical::output_hash<double>(*scalar_f64);
            },
            [scalar_f64] { return all_finite<double>(*scalar_f64); },
            {},
        });
    } else {
        auto ordered_f32 = std::make_shared<std::vector<float>>(
            static_cast<std::size_t>(plan->destination_size));
        cases.push_back({
            precision + "-ordered-reference", "ordered-reference", precision,
            plan, actual_f32_plan_hash, actual_complete_plan_hash, 1U,
            [plan, input_f32, ordered_f32] {
                dsmvc::detail::inverse_axis_f32_ordered(
                    *plan, input_f32->data(), 1, ordered_f32->data(), 1);
            },
            [ordered_f32] {
                return dsmvc::numerical::output_hash<float>(*ordered_f32);
            },
            [ordered_f32] { return all_finite<float>(*ordered_f32); },
            {},
        });
        cases.push_back({
            precision + "-current-scalar", "current-scalar", precision,
            plan, actual_f32_plan_hash, actual_complete_plan_hash, 1U,
            [plan, input_f32, output_f32] {
                dsmvc::inverse_axis_f32(
                    *plan, input_f32->data(), 1, output_f32->data(), 1);
            },
            [output_f32] {
                return dsmvc::numerical::output_hash<float>(*output_f32);
            },
            [output_f32] { return all_finite<float>(*output_f32); },
            {},
        });
    }

    constexpr std::int32_t rows = 8;
    const auto input_stride = ((plan->source_size + 7) & ~7) + 3;
    const auto output_stride = ((plan->destination_size + 7) & ~7) + 3;
    auto batched_input = std::make_shared<std::vector<float>>(
        make_batched_input(*plan, rows, input_stride, seed ^ 0x5a17c3e9U));
    auto batched_output = std::make_shared<std::vector<float>>(
        static_cast<std::size_t>(rows)
            * static_cast<std::size_t>(output_stride),
        0.0F);
    native.prepare(plan);
    cases.push_back({
        precision + "-native-cpu", "native-cpu", precision,
        plan, actual_f32_plan_hash, actual_complete_plan_hash,
        static_cast<std::size_t>(rows),
        [plan, batched_input, batched_output, input_stride, output_stride,
         &native] {
            native.inverse_rows(
                *plan, batched_input->data(), input_stride,
                batched_output->data(), output_stride, rows);
        },
        [plan, batched_output, output_stride] {
            const auto packed = pack_batched_output(
                *batched_output, *plan, rows, output_stride);
            return dsmvc::numerical::output_hash<float>(packed);
        },
        [plan, batched_output, output_stride] {
            const auto packed = pack_batched_output(
                *batched_output, *plan, rows, output_stride);
            return all_finite<float>(packed);
        },
        {},
    });
}

[[nodiscard]] std::string make_json(
    const Configuration &configuration, const std::vector<BenchmarkCase> &cases,
    std::string_view executable_path, std::string_view executable_identity,
    std::string_view cpu_path) {
    std::ostringstream output;
    output << std::setprecision(17);
    output << "{\"schema_version\":1"
           << ",\"benchmark\":\"dsmvc_numerical_contract\""
           << ",\"source_identity\":" << json_string(configuration.source_id)
           << ",\"samples\":" << configuration.samples
           << ",\"iterations\":" << configuration.iterations
           << ",\"host\":{\"architecture\":"
           << json_string(architecture_identity()) << "}"
           << ",\"build\":{\"compiler\":"
           << json_string(compiler_identity())
           << ",\"cxx_standard\":23}"
           << ",\"executable\":{\"path\":"
           << json_string(executable_path)
           << ",\"fnv1a64\":" << json_string(executable_identity) << "}"
           << ",\"native_cpu_path\":" << json_string(cpu_path)
           << ",\"cases\":[";
    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
        if (case_index != 0U) output << ',';
        const auto &item = cases[case_index];
        output << "{\"name\":" << json_string(item.name)
               << ",\"route\":" << json_string(item.route)
               << ",\"precision\":" << json_string(item.precision)
               << ",\"axes_per_iteration\":" << item.axes_per_iteration
               << ",\"plan\":{\"f32_contract_identity\":"
               << json_string(item.f32_plan_hash)
               << ",\"complete_identity\":"
               << json_string(item.complete_plan_hash)
               << ",\"source_size\":" << item.plan->source_size
               << ",\"destination_size\":" << item.plan->destination_size
               << ",\"half_bandwidth\":" << item.plan->half_bandwidth
               << ",\"normal_rcond\":" << item.plan->normal_rcond
               << ",\"normal_inf_norm\":" << item.plan->normal_inf_norm
               << ",\"requires_float64\":"
               << (item.plan->requires_float64() ? "true" : "false") << "}"
               << ",\"result_identity\":" << json_string(item.result_hash())
               << ",\"nanoseconds_per_axis\":{\"raw\":[";
        for (std::size_t sample = 0; sample < item.raw_nanoseconds_per_axis.size();
             ++sample) {
            if (sample != 0U) output << ',';
            output << item.raw_nanoseconds_per_axis[sample];
        }
        output << "],\"median\":"
               << median(item.raw_nanoseconds_per_axis) << "}}";
    }
    output << "]}";
    return output.str();
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto configuration = parse_arguments(argc, argv);
        const auto fixtures = dsmvc::numerical::axis_fixtures();
        dsmvc::CpuExecutor native(dsmvc::CpuPath::automatic);
        std::vector<BenchmarkCase> cases;
        cases.reserve(9U);

        append_axis_cases(
            cases, fixtures[1].request, fixtures[1].input_seed, "f32",
            fixtures[1].f32_plan_hash, native);
        append_axis_cases(
            cases, fixtures[6].request, fixtures[6].input_seed,
            "automatic-risk-f64", fixtures[6].f32_plan_hash, native);
        append_axis_cases(
            cases, fixtures[5].request, fixtures[5].input_seed,
            "forced-f64", fixtures[5].f32_plan_hash, native);
        native.seal();

        for (auto &item : cases) {
            item.execute();
            if (!item.finite()) {
                throw std::runtime_error(item.name + " warmup produced nonfinite output");
            }
        }

        for (std::size_t sample = 0; sample < configuration.samples; ++sample) {
            for (std::size_t order = 0; order < cases.size(); ++order) {
                const auto case_index = (sample & 1U) == 0U
                    ? order : cases.size() - order - 1U;
                auto &item = cases[case_index];
                const auto start = Clock::now();
                for (std::size_t iteration = 0;
                     iteration < configuration.iterations; ++iteration) {
                    item.execute();
                }
                const double nanoseconds =
                    std::chrono::duration<double, std::nano>(
                        Clock::now() - start).count();
                item.raw_nanoseconds_per_axis.push_back(
                    nanoseconds
                    / static_cast<double>(
                        configuration.iterations * item.axes_per_iteration));
                if (!item.finite()) {
                    throw std::runtime_error(
                        item.name + " benchmark produced nonfinite output");
                }
            }
        }

        const auto executable_path = std::filesystem::absolute(argv[0]);
        const auto identity = executable_hash(executable_path);
        const auto json = make_json(
            configuration, cases, executable_path.string(), identity,
            native.name());
        if (configuration.json_output.empty()) {
            std::cout << json << '\n';
        } else {
            std::ofstream output(configuration.json_output);
            if (!output) throw std::runtime_error("failed to open JSON output");
            output << json << '\n';
            if (!output) throw std::runtime_error("failed to write JSON output");
            for (const auto &item : cases) {
                std::cout << item.name << " median_ns_per_axis="
                          << median(item.raw_nanoseconds_per_axis)
                          << " result=" << item.result_hash() << '\n';
            }
        }
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "dsmvc numerical contract benchmark failure: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
