#include <dsmvc/engine.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::int32_t source_width = 1920;
constexpr std::int32_t source_height = 1080;
constexpr std::int32_t destination_width = 1692;
constexpr std::int32_t destination_height = 952;
constexpr std::int32_t horizontal_input_stride = source_width;
constexpr std::int32_t horizontal_output_stride = 1696;
constexpr std::int32_t vertical_stride = 1696;
constexpr double horizontal_active_length = 1691.5555555555557;
constexpr double horizontal_shift = 0.2222222222221717;
constexpr double vertical_active_length = 951.5;
constexpr double vertical_shift = 0.25;
constexpr double absolute_error_limit = 2.0e-5;
constexpr double relative_error_limit = 1.0e-3;
constexpr double relative_mad_limit = 0.10;

volatile std::uint64_t benchmark_sink = 0U;

enum class Mode {
    compare,
    scalar,
    neon,
};

enum class Orientation {
    horizontal,
    vertical,
};

struct CaseSpec {
    std::string name;
    dsmvc::KernelKind kernel = dsmvc::KernelKind::bilinear;
    std::int32_t taps = 0;
    Orientation orientation = Orientation::horizontal;
    std::int32_t work_count = 0;
};

struct Configuration {
    Mode mode = Mode::compare;
    std::size_t samples = 21U;
    std::size_t iterations = 1U;
    bool assert_gates = false;
    std::string source_id = "unknown";
    std::string build_type = "unknown";
    std::filesystem::path json_output;
};

struct Summary {
    std::vector<double> raw;
    double median = 0.0;
    double mad = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

struct ComparisonState {
    bool finite = true;
    bool identity_stable = true;
    double maximum_absolute_error = 0.0;
    double maximum_relative_error = 0.0;
    std::string scalar_identity;
    std::string neon_identity;
};

struct Fixture {
    CaseSpec spec;
    std::shared_ptr<const dsmvc::AxisPlan> plan;
    std::ptrdiff_t input_stride = 0;
    std::ptrdiff_t output_stride = 0;
    std::int32_t row_count = 0;
    std::int32_t column_count = 0;
    std::vector<float> input;
    std::vector<float> scalar_output;
    std::vector<float> neon_output;
    std::string plan_identity;
};

struct CaseResult {
    std::string name;
    std::string orientation;
    std::int32_t work_count = 0;
    std::int32_t half_bandwidth = 0;
    std::string plan_identity;
    Summary scalar;
    Summary neon;
    ComparisonState comparison;
};

struct CompareResult {
    std::vector<CaseResult> cases;
    std::vector<double> scalar_totals;
    std::vector<double> neon_totals;
    ComparisonState comparison;
};

struct Hasher {
    std::uint64_t value = 1469598103934665603ULL;

    void add_bytes(const void *data, std::size_t size) noexcept {
        const auto *bytes = static_cast<const unsigned char *>(data);
        for (std::size_t index = 0; index < size; ++index) {
            value ^= bytes[index];
            value *= 1099511628211ULL;
        }
    }

    template <class T>
    void add(const T &item) noexcept {
        add_bytes(&item, sizeof(item));
    }

    [[nodiscard]] std::string hex() const {
        std::ostringstream output;
        output << std::hex << std::setfill('0') << std::setw(16) << value;
        return output.str();
    }
};

[[nodiscard]] std::string json_string(std::string_view value) {
    std::string result;
    result.reserve(value.size() + 2U);
    result.push_back('"');
    for (const char raw : value) {
        const auto character = static_cast<unsigned char>(raw);
        switch (character) {
        case '"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:
            if (character < 0x20U) {
                result += "\\u00";
                constexpr char hex[] = "0123456789abcdef";
                result.push_back(hex[character >> 4U]);
                result.push_back(hex[character & 0x0FU]);
            } else {
                result.push_back(static_cast<char>(character));
            }
        }
    }
    result.push_back('"');
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

[[nodiscard]] std::string architecture_identity() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#else
    return "unknown";
#endif
}

[[nodiscard]] std::string mode_name(Mode mode) {
    switch (mode) {
    case Mode::compare: return "compare";
    case Mode::scalar: return "scalar";
    case Mode::neon: return "neon";
    }
    return "unknown";
}

[[nodiscard]] std::string orientation_name(Orientation orientation) {
    return orientation == Orientation::horizontal ? "horizontal" : "vertical";
}

[[nodiscard]] std::size_t parse_positive(std::string_view text,
                                          std::string_view option) {
    std::size_t value = 0U;
    for (const char character : text) {
        if (character < '0' || character > '9') {
            throw std::invalid_argument(
                std::string{option} + " must be a positive integer");
        }
        const auto digit = static_cast<std::size_t>(character - '0');
        if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10U) {
            throw std::invalid_argument(std::string{option} + " is too large");
        }
        value = value * 10U + digit;
    }
    if (value == 0U) {
        throw std::invalid_argument(
            std::string{option} + " must be a positive integer");
    }
    return value;
}

[[nodiscard]] Mode parse_mode(std::string_view text) {
    if (text == "compare") return Mode::compare;
    if (text == "scalar") return Mode::scalar;
    if (text == "neon") return Mode::neon;
    throw std::invalid_argument("--mode must be compare, scalar, or neon");
}

[[nodiscard]] Configuration parse_arguments(int argc, char **argv) {
    Configuration result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        const auto next_value = [&](std::string_view option) {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    std::string{option} + " requires a value");
            }
            return std::string_view{argv[++index]};
        };
        if (argument == "--mode") {
            result.mode = parse_mode(next_value("--mode"));
        } else if (argument == "--samples") {
            result.samples = parse_positive(next_value("--samples"), "--samples");
        } else if (argument == "--iterations") {
            result.iterations = parse_positive(
                next_value("--iterations"), "--iterations");
        } else if (argument == "--source-id") {
            result.source_id = std::string{next_value("--source-id")};
        } else if (argument == "--build-type") {
            result.build_type = std::string{next_value("--build-type")};
        } else if (argument == "--json-out") {
            result.json_output = std::filesystem::path{next_value("--json-out")};
        } else if (argument == "--assert") {
            result.assert_gates = true;
        } else {
            throw std::invalid_argument(
                "usage: dsmvc_cpu_profile_benchmark [--mode compare|scalar|neon] "
                "[--samples N] [--iterations N] [--source-id ID] "
                "[--build-type TYPE] [--json-out PATH] [--assert]");
        }
    }
    if (result.mode == Mode::compare && result.samples < 2U) {
        throw std::invalid_argument("compare mode requires at least two samples");
    }
    if (result.mode != Mode::compare && result.assert_gates) {
        throw std::invalid_argument("--assert is only valid in compare mode");
    }
    return result;
}

[[nodiscard]] std::vector<CaseSpec> make_case_specs() {
    return {
        {"b1-horizontal-small", dsmvc::KernelKind::bilinear, 0,
         Orientation::horizontal, 64},
        {"b1-horizontal-full", dsmvc::KernelKind::bilinear, 0,
         Orientation::horizontal, source_height},
        {"b1-vertical-small", dsmvc::KernelKind::bilinear, 0,
         Orientation::vertical, 128},
        {"b1-vertical-full", dsmvc::KernelKind::bilinear, 0,
         Orientation::vertical, destination_width},
        {"b3-horizontal-small", dsmvc::KernelKind::bicubic, 0,
         Orientation::horizontal, 64},
        {"b3-horizontal-full", dsmvc::KernelKind::bicubic, 0,
         Orientation::horizontal, source_height},
        {"b3-vertical-small", dsmvc::KernelKind::bicubic, 0,
         Orientation::vertical, 128},
        {"b3-vertical-full", dsmvc::KernelKind::bicubic, 0,
         Orientation::vertical, destination_width},
        {"b5-horizontal-small", dsmvc::KernelKind::lanczos, 3,
         Orientation::horizontal, 64},
        {"b5-horizontal-full", dsmvc::KernelKind::lanczos, 3,
         Orientation::horizontal, source_height},
        {"b5-vertical-small", dsmvc::KernelKind::lanczos, 3,
         Orientation::vertical, 128},
        {"b5-vertical-full", dsmvc::KernelKind::lanczos, 3,
         Orientation::vertical, destination_width},
        {"b7-horizontal-small", dsmvc::KernelKind::spline64, 0,
         Orientation::horizontal, 64},
        {"b7-horizontal-full", dsmvc::KernelKind::spline64, 0,
         Orientation::horizontal, source_height},
        {"b7-vertical-small", dsmvc::KernelKind::spline64, 0,
         Orientation::vertical, 128},
        {"b7-vertical-full", dsmvc::KernelKind::spline64, 0,
         Orientation::vertical, destination_width},
    };
}

[[nodiscard]] dsmvc::AxisRequest make_request(const CaseSpec &spec) {
    dsmvc::AxisRequest request;
    if (spec.orientation == Orientation::horizontal) {
        request.source_size = source_width;
        request.destination_size = destination_width;
        request.active_length = horizontal_active_length;
        request.shift = horizontal_shift;
    } else {
        request.source_size = source_height;
        request.destination_size = destination_height;
        request.active_length = vertical_active_length;
        request.shift = vertical_shift;
    }
    request.kernel.kind = spec.kernel;
    request.kernel.taps = spec.taps;
    request.kernel.b = 0.0;
    request.kernel.c = 0.5;
    request.border = dsmvc::BorderMode::mirror;
    return request;
}

void hash_plan(Hasher &hasher, const dsmvc::AxisPlan &plan) {
    hasher.add(plan.source_size);
    hasher.add(plan.destination_size);
    hasher.add(plan.support);
    hasher.add(plan.half_bandwidth);
    hasher.add(plan.active_length);
    hasher.add(plan.shift);
    hasher.add_bytes(plan.transpose_offsets.data(),
                     plan.transpose_offsets.size() * sizeof(std::uint32_t));
    hasher.add_bytes(plan.transpose_indices.data(),
                     plan.transpose_indices.size() * sizeof(std::int32_t));
    hasher.add_bytes(plan.transpose_weights.data(),
                     plan.transpose_weights.size() * sizeof(float));
    hasher.add_bytes(plan.lower_ld.data(), plan.lower_ld.size() * sizeof(float));
    hasher.add_bytes(plan.upper_l.data(), plan.upper_l.size() * sizeof(float));
    hasher.add_bytes(plan.inverse_diagonal.data(),
                     plan.inverse_diagonal.size() * sizeof(float));
}

[[nodiscard]] std::string plan_identity(const dsmvc::AxisPlan &plan) {
    Hasher hasher;
    hash_plan(hasher, plan);
    return hasher.hex();
}

void fill_input(Fixture &fixture, std::size_t case_index) {
    const float phase = static_cast<float>(case_index + 1U) * 0.071F;
    if (fixture.spec.orientation == Orientation::horizontal) {
        for (std::int32_t row = 0; row < fixture.row_count; ++row) {
            for (std::int32_t column = 0; column < source_width; ++column) {
                const auto value = std::sin(
                        static_cast<float>(column) * 0.0031F + phase)
                    + 0.25F * std::cos(
                        static_cast<float>(row) * 0.017F - phase)
                    + 0.03125F * static_cast<float>((row + column) & 31);
                fixture.input[static_cast<std::size_t>(row)
                              * static_cast<std::size_t>(fixture.input_stride)
                              + static_cast<std::size_t>(column)] = value;
            }
        }
    } else {
        for (std::int32_t row = 0; row < source_height; ++row) {
            for (std::int32_t column = 0;
                 column < fixture.input_stride; ++column) {
                const auto value = std::sin(
                        static_cast<float>(row) * 0.0097F + phase)
                    + 0.2F * std::cos(
                        static_cast<float>(column) * 0.0027F - phase)
                    + 0.015625F * static_cast<float>((row * 3 + column) & 63);
                fixture.input[static_cast<std::size_t>(row)
                              * static_cast<std::size_t>(fixture.input_stride)
                              + static_cast<std::size_t>(column)] = value;
            }
        }
    }
}

[[nodiscard]] Fixture make_fixture(const CaseSpec &spec, std::size_t case_index) {
    Fixture fixture;
    fixture.spec = spec;
    fixture.plan = std::make_shared<const dsmvc::AxisPlan>(
        dsmvc::build_axis_plan(make_request(spec)));
    if (!fixture.plan->valid()) {
        throw std::runtime_error("fixed recipe produced an invalid axis plan");
    }
    fixture.plan_identity = plan_identity(*fixture.plan);

    if (spec.orientation == Orientation::horizontal) {
        fixture.input_stride = horizontal_input_stride;
        fixture.output_stride = horizontal_output_stride;
        fixture.row_count = spec.work_count;
        fixture.input.resize(static_cast<std::size_t>(fixture.row_count)
                             * static_cast<std::size_t>(fixture.input_stride));
        fixture.scalar_output.resize(static_cast<std::size_t>(fixture.row_count)
                                     * static_cast<std::size_t>(fixture.output_stride));
        fixture.neon_output.resize(fixture.scalar_output.size());
    } else {
        fixture.input_stride = vertical_stride;
        fixture.output_stride = vertical_stride;
        fixture.column_count = spec.work_count;
        fixture.input.resize(static_cast<std::size_t>(source_height)
                             * static_cast<std::size_t>(fixture.input_stride));
        fixture.scalar_output.resize(static_cast<std::size_t>(destination_height)
                                     * static_cast<std::size_t>(fixture.output_stride));
        fixture.neon_output.resize(fixture.scalar_output.size());
    }
    fill_input(fixture, case_index);
    return fixture;
}

[[nodiscard]] std::uint64_t fnv1a_file(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("failed to open benchmark executable");
    Hasher hasher;
    std::vector<char> buffer(64U * 1024U);
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0) {
            hasher.add_bytes(buffer.data(), static_cast<std::size_t>(count));
        }
    }
    if (!input.eof()) throw std::runtime_error("failed to hash benchmark executable");
    return hasher.value;
}

[[nodiscard]] Summary summarize(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("cannot summarize no samples");
    for (const double value : values) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::runtime_error("benchmark sample is not finite and positive");
        }
    }
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const auto middle = sorted.size() / 2U;
    const double median = (sorted.size() & 1U) != 0U
        ? sorted[middle] : (sorted[middle - 1U] + sorted[middle]) / 2.0;
    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (const double value : values) deviations.push_back(std::abs(value - median));
    std::sort(deviations.begin(), deviations.end());
    const double mad = (deviations.size() & 1U) != 0U
        ? deviations[middle]
        : (deviations[middle - 1U] + deviations[middle]) / 2.0;
    const auto [minimum, maximum] = std::minmax_element(values.begin(), values.end());
    return {std::move(values), median, mad, *minimum, *maximum};
}

template <class Function>
[[nodiscard]] double timed_ms(Function &&function) {
    const auto start = Clock::now();
    std::forward<Function>(function)();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

[[nodiscard]] std::string output_identity(const Fixture &fixture,
                                           const std::vector<float> &output) {
    Hasher hasher;
    if (fixture.spec.orientation == Orientation::horizontal) {
        for (std::int32_t row = 0; row < fixture.row_count; ++row) {
            const auto *begin = output.data()
                + static_cast<std::ptrdiff_t>(row) * fixture.output_stride;
            hasher.add_bytes(begin,
                             static_cast<std::size_t>(destination_width)
                                 * sizeof(float));
        }
    } else {
        for (std::int32_t row = 0; row < destination_height; ++row) {
            const auto *begin = output.data()
                + static_cast<std::ptrdiff_t>(row) * fixture.output_stride;
            hasher.add_bytes(begin,
                             static_cast<std::size_t>(fixture.column_count)
                                 * sizeof(float));
        }
    }
    return hasher.hex();
}

[[nodiscard]] ComparisonState compare_outputs(
    const Fixture &fixture, const std::vector<float> &scalar,
    const std::vector<float> &neon) {
    ComparisonState result;
    result.scalar_identity = output_identity(fixture, scalar);
    result.neon_identity = output_identity(fixture, neon);
    if (fixture.spec.orientation == Orientation::horizontal) {
        for (std::int32_t row = 0; row < fixture.row_count; ++row) {
            const auto *scalar_row = scalar.data()
                + static_cast<std::ptrdiff_t>(row) * fixture.output_stride;
            const auto *neon_row = neon.data()
                + static_cast<std::ptrdiff_t>(row) * fixture.output_stride;
            for (std::int32_t column = 0; column < destination_width; ++column) {
                const float lhs = scalar_row[column];
                const float rhs = neon_row[column];
                if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
                    result.finite = false;
                    continue;
                }
                const double absolute = std::abs(
                    static_cast<double>(lhs) - static_cast<double>(rhs));
                const double relative = absolute / std::max(
                    1.0e-3, std::abs(static_cast<double>(lhs)));
                result.maximum_absolute_error = std::max(
                    result.maximum_absolute_error, absolute);
                result.maximum_relative_error = std::max(
                    result.maximum_relative_error, relative);
            }
        }
    } else {
        for (std::int32_t row = 0; row < destination_height; ++row) {
            const auto *scalar_row = scalar.data()
                + static_cast<std::ptrdiff_t>(row) * fixture.output_stride;
            const auto *neon_row = neon.data()
                + static_cast<std::ptrdiff_t>(row) * fixture.output_stride;
            for (std::int32_t column = 0;
                 column < fixture.column_count; ++column) {
                const float lhs = scalar_row[column];
                const float rhs = neon_row[column];
                if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
                    result.finite = false;
                    continue;
                }
                const double absolute = std::abs(
                    static_cast<double>(lhs) - static_cast<double>(rhs));
                const double relative = absolute / std::max(
                    1.0e-3, std::abs(static_cast<double>(lhs)));
                result.maximum_absolute_error = std::max(
                    result.maximum_absolute_error, absolute);
                result.maximum_relative_error = std::max(
                    result.maximum_relative_error, relative);
            }
        }
    }
    return result;
}

void merge_comparison(ComparisonState &aggregate,
                      const ComparisonState &current) {
    aggregate.finite = aggregate.finite && current.finite;
    aggregate.maximum_absolute_error = std::max(
        aggregate.maximum_absolute_error, current.maximum_absolute_error);
    aggregate.maximum_relative_error = std::max(
        aggregate.maximum_relative_error, current.maximum_relative_error);
}

void observe_pair(Fixture &fixture, CaseResult &case_result,
                  CompareResult &aggregate) {
    const ComparisonState current = compare_outputs(
        fixture, fixture.scalar_output, fixture.neon_output);
    if (!case_result.comparison.scalar_identity.empty()) {
        case_result.comparison.identity_stable =
            case_result.comparison.identity_stable
            && case_result.comparison.scalar_identity == current.scalar_identity
            && case_result.comparison.neon_identity == current.neon_identity;
    }
    if (case_result.comparison.scalar_identity.empty()) {
        case_result.comparison.scalar_identity = current.scalar_identity;
        case_result.comparison.neon_identity = current.neon_identity;
    }
    case_result.comparison.finite =
        case_result.comparison.finite && current.finite;
    case_result.comparison.maximum_absolute_error = std::max(
        case_result.comparison.maximum_absolute_error,
        current.maximum_absolute_error);
    case_result.comparison.maximum_relative_error = std::max(
        case_result.comparison.maximum_relative_error,
        current.maximum_relative_error);
    merge_comparison(aggregate.comparison, current);
}

void run_executor(const dsmvc::CpuExecutor &executor, Fixture &fixture,
                  std::vector<float> &output) {
    if (fixture.spec.orientation == Orientation::horizontal) {
        executor.inverse_rows(
            *fixture.plan, fixture.input.data(), fixture.input_stride,
            output.data(), fixture.output_stride, fixture.row_count);
    } else {
        executor.inverse_columns(
            *fixture.plan, fixture.input.data(), fixture.input_stride,
            output.data(), fixture.output_stride, fixture.column_count);
    }
}

[[nodiscard]] CompareResult run_compare(
    const Configuration &configuration,
    std::vector<Fixture> &fixtures,
    const dsmvc::CpuExecutor &scalar,
    const dsmvc::CpuExecutor &neon) {
    CompareResult result;
    result.scalar_totals.assign(configuration.samples, 0.0);
    result.neon_totals.assign(configuration.samples, 0.0);
    result.cases.reserve(fixtures.size());

    for (Fixture &fixture : fixtures) {
        CaseResult case_result;
        case_result.name = fixture.spec.name;
        case_result.orientation = orientation_name(fixture.spec.orientation);
        case_result.work_count = fixture.spec.work_count;
        case_result.half_bandwidth = fixture.plan->half_bandwidth;
        case_result.plan_identity = fixture.plan_identity;

        run_executor(scalar, fixture, fixture.scalar_output);
        run_executor(neon, fixture, fixture.neon_output);
        observe_pair(fixture, case_result, result);

        std::vector<double> scalar_samples;
        std::vector<double> neon_samples;
        scalar_samples.reserve(configuration.samples);
        neon_samples.reserve(configuration.samples);
        for (std::size_t sample = 0; sample < configuration.samples; ++sample) {
            double scalar_ms = 0.0;
            double neon_ms = 0.0;
            if ((sample & 1U) == 0U) {
                scalar_ms = timed_ms([&] {
                    run_executor(scalar, fixture, fixture.scalar_output);
                });
                neon_ms = timed_ms([&] {
                    run_executor(neon, fixture, fixture.neon_output);
                });
            } else {
                neon_ms = timed_ms([&] {
                    run_executor(neon, fixture, fixture.neon_output);
                });
                scalar_ms = timed_ms([&] {
                    run_executor(scalar, fixture, fixture.scalar_output);
                });
            }
            scalar_samples.push_back(scalar_ms);
            neon_samples.push_back(neon_ms);
            result.scalar_totals[sample] += scalar_ms;
            result.neon_totals[sample] += neon_ms;
            observe_pair(fixture, case_result, result);
            benchmark_sink ^= static_cast<std::uint64_t>(
                std::llround((scalar_ms + neon_ms) * 1000.0));
        }
        case_result.scalar = summarize(std::move(scalar_samples));
        case_result.neon = summarize(std::move(neon_samples));
        result.comparison.identity_stable = result.comparison.identity_stable
            && case_result.comparison.identity_stable;
        result.cases.push_back(std::move(case_result));
    }
    return result;
}

void append_summary(std::ostream &output, const Summary &summary) {
    output << "{\"raw\":[" << std::setprecision(17);
    for (std::size_t index = 0; index < summary.raw.size(); ++index) {
        if (index != 0U) output << ',';
        output << summary.raw[index];
    }
    output << "],\"median\":" << summary.median
           << ",\"mad\":" << summary.mad
           << ",\"minimum\":" << summary.minimum
           << ",\"maximum\":" << summary.maximum << '}';
}

void append_comparison(std::ostream &output, const ComparisonState &comparison) {
    const bool pass = comparison.finite && comparison.identity_stable
        && comparison.maximum_absolute_error <= absolute_error_limit
        && comparison.maximum_relative_error <= relative_error_limit;
    output << "{\"pass\":" << (pass ? "true" : "false")
           << ",\"finite\":" << (comparison.finite ? "true" : "false")
           << ",\"identity_stable\":"
           << (comparison.identity_stable ? "true" : "false")
           << ",\"maximum_absolute_error\":"
           << comparison.maximum_absolute_error
           << ",\"maximum_relative_error\":"
           << comparison.maximum_relative_error
           << ",\"scalar_result_identity\":"
           << json_string(comparison.scalar_identity)
           << ",\"neon_result_identity\":"
           << json_string(comparison.neon_identity) << '}';
}

[[nodiscard]] std::string make_compare_json(
    const Configuration &configuration, const CompareResult &result,
    const std::filesystem::path &executable, int argc, char **argv) {
    const Summary scalar_total = summarize(result.scalar_totals);
    const Summary neon_total = summarize(result.neon_totals);
    const double relative_mad = neon_total.median > 0.0
        ? neon_total.mad / neon_total.median : std::numeric_limits<double>::infinity();
    const bool correctness_pass = result.comparison.finite
        && result.comparison.identity_stable
        && result.comparison.maximum_absolute_error <= absolute_error_limit
        && result.comparison.maximum_relative_error <= relative_error_limit;
    const bool all_gates_pass = correctness_pass && relative_mad <= relative_mad_limit;

    std::ostringstream output;
    output << std::setprecision(17)
           << "{\"schema_version\":1"
           << ",\"benchmark\":\"dsmvc_cpu_fixed_recipe\""
           << ",\"mode\":\"compare\""
           << ",\"source_identity\":" << json_string(configuration.source_id)
           << ",\"arguments\":[";
    for (int index = 0; index < argc; ++index) {
        if (index != 0) output << ',';
        output << json_string(argv[index]);
    }
    output << "]"
           << ",\"executable\":{\"path\":"
           << json_string(std::filesystem::absolute(executable).string())
           << ",\"fnv1a64\":\"" << std::hex << std::setfill('0')
           << std::setw(16) << fnv1a_file(executable) << std::dec << "\"}"
           << ",\"host\":{\"system\":\"Darwin\",\"architecture\":"
           << json_string(architecture_identity())
           << ",\"logical_cpus\":" << std::thread::hardware_concurrency() << '}'
           << ",\"build\":{\"type\":"
           << json_string(configuration.build_type)
           << ",\"compiler\":" << json_string(compiler_identity())
           << ",\"cxx_standard\":23}"
           << ",\"fixed_recipe\":{\"source\":\"1920x1080\""
           << ",\"destination\":\"1692x952\""
           << ",\"plan_prepared_outside_timing\":true"
           << ",\"alternating_order\":\"scalar-neon/neon-scalar\""
           << ",\"samples\":" << configuration.samples << "}"
           << ",\"aggregate\":{\"scalar_ms\":";
    append_summary(output, scalar_total);
    output << ",\"neon_ms\":";
    append_summary(output, neon_total);
    output << ",\"neon_relative_mad\":" << relative_mad
           << ",\"all_gates_pass\":" << (all_gates_pass ? "true" : "false")
           << "}"
           << ",\"correctness\":";
    append_comparison(output, result.comparison);
    output << ",\"cases\":[";
    for (std::size_t index = 0; index < result.cases.size(); ++index) {
        if (index != 0U) output << ',';
        const auto &item = result.cases[index];
        output << "{\"name\":" << json_string(item.name)
               << ",\"orientation\":" << json_string(item.orientation)
               << ",\"work_count\":" << item.work_count
               << ",\"half_bandwidth\":" << item.half_bandwidth
               << ",\"plan_identity\":" << json_string(item.plan_identity)
               << ",\"scalar_ms\":";
        append_summary(output, item.scalar);
        output << ",\"neon_ms\":";
        append_summary(output, item.neon);
        output << ",\"correctness\":";
        append_comparison(output, item.comparison);
        output << '}';
    }
    output << "],\"all_gates_pass\":" << (all_gates_pass ? "true" : "false")
           << "}\n";
    return output.str();
}

void write_json(const std::filesystem::path &path, std::string_view contents) {
    if (path.empty()) return;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("failed to open JSON output");
    output << contents;
    if (!output) throw std::runtime_error("failed to write JSON output");
}

void run_profile(const Configuration &configuration,
                 std::vector<Fixture> &fixtures,
                 const dsmvc::CpuExecutor &neon) {
    Hasher output_hash;
    for (std::size_t iteration = 0; iteration < configuration.iterations; ++iteration) {
        for (Fixture &fixture : fixtures) {
            run_executor(neon, fixture, fixture.neon_output);
        }
    }
    for (Fixture &fixture : fixtures) {
        const auto identity = output_identity(fixture, fixture.neon_output);
        output_hash.add_bytes(identity.data(), identity.size());
    }
    benchmark_sink ^= output_hash.value;
    std::cout << "profile mode=neon iterations=" << configuration.iterations
              << " cases=" << fixtures.size()
              << " result_identity=" << output_hash.hex()
              << " sink=" << benchmark_sink << '\n';
}

[[nodiscard]] std::string make_profile_json(
    const Configuration &configuration, const std::vector<Fixture> &fixtures,
    const std::filesystem::path &executable) {
    std::ostringstream output;
    output << "{\"schema_version\":1"
           << ",\"benchmark\":\"dsmvc_cpu_fixed_recipe\""
           << ",\"mode\":\"" << mode_name(configuration.mode) << "\""
           << ",\"source_identity\":" << json_string(configuration.source_id)
           << ",\"executable\":{\"path\":"
           << json_string(std::filesystem::absolute(executable).string())
           << ",\"fnv1a64\":\"" << std::hex << std::setfill('0')
           << std::setw(16) << fnv1a_file(executable) << std::dec << "\"}"
           << ",\"build\":{\"type\":"
           << json_string(configuration.build_type)
           << ",\"compiler\":" << json_string(compiler_identity()) << "}"
           << ",\"fixed_recipe\":{\"source\":\"1920x1080\""
           << ",\"destination\":\"1692x952\",\"plan_prepared_outside_timing\":true}"
           << ",\"iterations\":" << configuration.iterations
           << ",\"case_count\":" << fixtures.size() << "}\n";
    return output.str();
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Configuration configuration = parse_arguments(argc, argv);
        if (configuration.mode == Mode::neon && !dsmvc::cpu_neon_available()) {
            throw std::runtime_error("NEON is not available on this host");
        }
        if (configuration.mode == Mode::compare
            && (!dsmvc::cpu_neon_available() || !dsmvc::cpu_neon_compiled())) {
            throw std::runtime_error("compare mode requires a compiled native NEON path");
        }

        std::vector<Fixture> fixtures;
        const auto specs = make_case_specs();
        fixtures.reserve(specs.size());
        for (std::size_t index = 0; index < specs.size(); ++index) {
            fixtures.push_back(make_fixture(specs[index], index));
        }

        dsmvc::CpuExecutor scalar(dsmvc::CpuPath::scalar);
        if (configuration.mode == Mode::compare) {
            dsmvc::CpuExecutor neon(dsmvc::CpuPath::neon);
            for (const auto &fixture : fixtures) neon.prepare(fixture.plan);
            neon.seal();
            const auto result = run_compare(configuration, fixtures, scalar, neon);
            const auto json = make_compare_json(
                configuration, result, argv[0], argc, argv);
            write_json(configuration.json_output, json);
            const Summary neon_total = summarize(result.neon_totals);
            const double relative_mad = neon_total.median > 0.0
                ? neon_total.mad / neon_total.median
                : std::numeric_limits<double>::infinity();
            const bool correctness_pass = result.comparison.finite
                && result.comparison.identity_stable
                && result.comparison.maximum_absolute_error <= absolute_error_limit
                && result.comparison.maximum_relative_error <= relative_error_limit;
            const bool gates_pass = correctness_pass && relative_mad <= relative_mad_limit;
            std::cout << std::fixed << std::setprecision(3)
                      << "fixed-recipe source=1920x1080 destination=1692x952"
                      << " samples=" << configuration.samples
                      << " scalar_ms=" << summarize(result.scalar_totals).median
                      << " neon_ms=" << neon_total.median
                      << " neon_relative_mad=" << relative_mad
                      << " max_abs_error="
                      << result.comparison.maximum_absolute_error
                      << " correctness=" << (correctness_pass ? "pass" : "fail")
                      << " gates=" << (gates_pass ? "pass" : "fail") << '\n';
            if (configuration.assert_gates && !gates_pass) return EXIT_FAILURE;
        } else if (configuration.mode == Mode::scalar) {
            for (Fixture &fixture : fixtures) {
                for (std::size_t iteration = 0;
                     iteration < configuration.iterations; ++iteration) {
                    run_executor(scalar, fixture, fixture.scalar_output);
                }
            }
        } else {
            dsmvc::CpuExecutor neon(dsmvc::CpuPath::neon);
            for (const auto &fixture : fixtures) neon.prepare(fixture.plan);
            neon.seal();
            run_profile(configuration, fixtures, neon);
            write_json(configuration.json_output,
                       make_profile_json(configuration, fixtures, argv[0]));
        }
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "fixed-recipe benchmark failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
