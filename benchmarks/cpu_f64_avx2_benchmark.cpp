#include <dsmvc/engine.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/utsname.h>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr float float_guard = -12345.0F;
constexpr std::uint8_t u8_guard = 0xa5U;
constexpr std::uint16_t u16_guard = 0xa55aU;

volatile std::uint64_t benchmark_sink = 0U;

enum class Operation {
    rows,
    columns,
    two_d,
};

enum class SampleKind {
    f32,
    u8,
    u10,
    u16,
};

struct AxisRecipe {
    std::int32_t source = 0;
    std::int32_t destination = 0;
    double active_length = 0.0;
    double shift = 0.0;
    dsmvc::KernelKind kernel = dsmvc::KernelKind::bilinear;
    std::int32_t taps = 0;
    dsmvc::F64Mode mode = dsmvc::F64Mode::float64_only;
    bool custom = false;
};

struct CaseSpec {
    std::string name;
    std::string precision;
    std::string scale;
    std::string bandwidth;
    Operation operation = Operation::rows;
    SampleKind sample = SampleKind::f32;
    AxisRecipe horizontal;
    AxisRecipe vertical;
    std::int32_t independent_rhs = 0;
    bool representative = false;
};

struct Configuration {
    std::size_t samples = 7U;
    std::size_t iterations = 1U;
    bool check_only = false;
    std::string source_id = "unknown";
    std::string binary_sha256 = "unknown";
    std::string case_filter;
    std::filesystem::path json_output;
};

struct Summary {
    std::vector<double> raw;
    double median = 0.0;
    double mad = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

struct Comparison {
    bool finite = true;
    bool guards = true;
    bool bit_exact = true;
    std::uint32_t maximum_ulp = 0U;
    std::uint64_t scalar_hash = 0U;
    std::uint64_t native_hash = 0U;
};

struct CaseResult {
    CaseSpec spec;
    std::int32_t horizontal_bandwidth = 0;
    std::int32_t vertical_bandwidth = 0;
    double horizontal_rcond = 0.0;
    double vertical_rcond = 0.0;
    bool horizontal_f64 = false;
    bool vertical_f64 = false;
    std::uint64_t horizontal_f32_hash = 0U;
    std::uint64_t vertical_f32_hash = 0U;
    Comparison comparison;
    Summary scalar;
    Summary native;
    Summary paired_speedup;
};

class Hasher {
public:
    void add_bytes(const void *data, std::size_t size) noexcept {
        const auto *bytes = static_cast<const unsigned char *>(data);
        for (std::size_t index = 0; index < size; ++index) {
            value_ ^= bytes[index];
            value_ *= 1099511628211ULL;
        }
    }

    template <class T>
    void add(const T &value) noexcept {
        add_bytes(&value, sizeof(value));
    }

    [[nodiscard]] std::uint64_t value() const noexcept { return value_; }

private:
    std::uint64_t value_ = 1469598103934665603ULL;
};

[[nodiscard]] std::string hex64(std::uint64_t value) {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << value;
    return output.str();
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

[[nodiscard]] std::string operation_name(Operation operation) {
    switch (operation) {
    case Operation::rows: return "rows";
    case Operation::columns: return "columns";
    case Operation::two_d: return "2d";
    }
    return "unknown";
}

[[nodiscard]] std::string sample_name(SampleKind sample) {
    switch (sample) {
    case SampleKind::f32: return "float";
    case SampleKind::u8: return "u8-limited-luma";
    case SampleKind::u10: return "u10-limited-chroma";
    case SampleKind::u16: return "u16-full";
    }
    return "unknown";
}

[[nodiscard]] std::size_t parse_positive(
    std::string_view text, std::string_view option) {
    std::size_t value = 0U;
    if (text.empty()) {
        throw std::invalid_argument(std::string(option) + " requires a value");
    }
    for (const char character : text) {
        if (character < '0' || character > '9') {
            throw std::invalid_argument(
                std::string(option) + " must be a positive integer");
        }
        const auto digit = static_cast<std::size_t>(character - '0');
        if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10U) {
            throw std::invalid_argument(std::string(option) + " is too large");
        }
        value = value * 10U + digit;
    }
    if (value == 0U) {
        throw std::invalid_argument(
            std::string(option) + " must be a positive integer");
    }
    return value;
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
        } else if (argument == "--binary-sha256") {
            result.binary_sha256 = std::string(next(argument));
        } else if (argument == "--case-filter") {
            result.case_filter = std::string(next(argument));
        } else if (argument == "--json-out") {
            result.json_output = std::filesystem::path(next(argument));
        } else if (argument == "--check-only") {
            result.check_only = true;
        } else {
            throw std::invalid_argument(
                "usage: dsmvc_cpu_f64_avx2_benchmark [--samples N] "
                "[--iterations N] [--source-id SHA] [--binary-sha256 SHA] "
                "[--case-filter TEXT] [--json-out PATH] [--check-only]");
        }
    }
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
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#else
    return "unknown";
#endif
}

[[nodiscard]] std::string system_identity() {
    struct utsname information {};
    if (uname(&information) != 0) return "unknown";
    return std::string(information.sysname) + " " + information.release;
}

[[nodiscard]] std::string cpu_identity() {
    std::ifstream input("/proc/cpuinfo");
    std::string line;
    while (std::getline(input, line)) {
        constexpr std::string_view prefix = "model name";
        if (!line.starts_with(prefix)) continue;
        const auto separator = line.find(':');
        if (separator == std::string::npos) break;
        const auto first = line.find_first_not_of(" \t", separator + 1U);
        return first == std::string::npos ? "unknown" : line.substr(first);
    }
    return "unknown";
}

[[nodiscard]] dsmvc::CustomKernel generic_kernel() {
    return [](double x) {
        return std::max(1.0 - x, 0.0);
    };
}

[[nodiscard]] std::shared_ptr<const dsmvc::AxisPlan> build_plan(
    const AxisRecipe &recipe) {
    dsmvc::AxisRequest request;
    request.source_size = recipe.source;
    request.destination_size = recipe.destination;
    request.active_length = recipe.active_length;
    request.shift = recipe.shift;
    request.kernel.kind = recipe.kernel;
    request.kernel.taps = recipe.taps;
    request.kernel.b = 0.0;
    request.kernel.c = 0.5;
    request.border = dsmvc::BorderMode::symmetric;
    request.f64_mode = recipe.mode;
    auto plan = std::make_shared<const dsmvc::AxisPlan>(
        dsmvc::build_axis_plan(
            request, recipe.custom ? generic_kernel() : dsmvc::CustomKernel{}));
    if (!plan->valid()) throw std::runtime_error("benchmark plan is invalid");
    return plan;
}

[[nodiscard]] std::uint64_t f32_plan_hash(const dsmvc::AxisPlan &plan) {
    Hasher hash;
    hash.add(plan.source_size);
    hash.add(plan.destination_size);
    hash.add(plan.support);
    hash.add(plan.half_bandwidth);
    hash.add(plan.active_length);
    hash.add(plan.shift);
    hash.add(plan.normal_rcond);
    hash.add_bytes(plan.transpose_offsets.data(),
                   plan.transpose_offsets.size() * sizeof(std::uint32_t));
    hash.add_bytes(plan.transpose_indices.data(),
                   plan.transpose_indices.size() * sizeof(std::int32_t));
    hash.add_bytes(plan.transpose_weights.data(),
                   plan.transpose_weights.size() * sizeof(float));
    hash.add_bytes(plan.lower_ld.data(),
                   plan.lower_ld.size() * sizeof(float));
    hash.add_bytes(plan.upper_l.data(),
                   plan.upper_l.size() * sizeof(float));
    hash.add_bytes(plan.inverse_diagonal.data(),
                   plan.inverse_diagonal.size() * sizeof(float));
    return hash.value();
}

[[nodiscard]] AxisRecipe kernel_recipe(
    std::size_t index, bool horizontal, dsmvc::F64Mode mode) {
    static constexpr dsmvc::KernelKind kernels[] = {
        dsmvc::KernelKind::bilinear,
        dsmvc::KernelKind::bicubic,
        dsmvc::KernelKind::lanczos,
        dsmvc::KernelKind::spline64,
        dsmvc::KernelKind::custom,
    };
    AxisRecipe result;
    result.source = horizontal ? 257 : 193;
    result.destination = horizontal ? 229 : 171;
    result.active_length = horizontal ? 228.5 : 170.5;
    result.shift = horizontal ? 0.25 : -0.125;
    result.kernel = kernels[index];
    result.taps = index == 2U ? 3 : (index == 4U ? 5 : 0);
    result.mode = mode;
    result.custom = index == 4U;
    return result;
}

[[nodiscard]] AxisRecipe full_axis_recipe(std::size_t index, bool horizontal) {
    auto result = kernel_recipe(index, horizontal, dsmvc::F64Mode::float64_only);
    result.source = horizontal ? 1920 : 1080;
    result.destination = horizontal ? 1692 : 952;
    result.active_length = horizontal ? 1691.5555555555557 : 951.5;
    result.shift = horizontal ? 0.2222222222221717 : 0.25;
    return result;
}

[[nodiscard]] AxisRecipe risky_recipe() {
    AxisRecipe result;
    result.source = 1080;
    result.destination = 980;
    result.active_length = 978.1;
    result.shift = 0.95;
    result.kernel = dsmvc::KernelKind::lanczos;
    result.taps = 2;
    result.mode = dsmvc::F64Mode::automatic;
    return result;
}

[[nodiscard]] AxisRecipe identity_recipe(std::int32_t size) {
    AxisRecipe result;
    result.source = size;
    result.destination = size;
    result.active_length = static_cast<double>(size);
    result.shift = 0.0;
    result.kernel = dsmvc::KernelKind::bilinear;
    result.mode = dsmvc::F64Mode::float32_only;
    return result;
}

[[nodiscard]] std::vector<CaseSpec> make_case_specs() {
    static constexpr std::string_view bands[] = {
        "b1", "b3", "b5", "b7", "generic-b9",
    };
    static constexpr SampleKind samples[] = {
        SampleKind::f32, SampleKind::u8, SampleKind::u10, SampleKind::u16,
    };
    std::vector<CaseSpec> result;

    for (std::size_t kernel = 0; kernel < std::size(bands); ++kernel) {
        auto horizontal = full_axis_recipe(kernel, true);
        auto vertical = full_axis_recipe(kernel, false);
        result.push_back({
            std::string(bands[kernel]) + "-rows-full", "forced-f64-safe",
            "representative", std::string(bands[kernel]), Operation::rows,
            SampleKind::f32, horizontal, {}, 1080, true,
        });
        result.push_back({
            std::string(bands[kernel]) + "-columns-full", "forced-f64-safe",
            "representative", std::string(bands[kernel]), Operation::columns,
            SampleKind::f32, vertical, {}, 1692, true,
        });

        const auto safe_horizontal = kernel_recipe(
            kernel, true, dsmvc::F64Mode::float32_only);
        const auto safe_vertical = kernel_recipe(
            kernel, false, dsmvc::F64Mode::float32_only);
        result.push_back({
            std::string(bands[kernel]) + "-horizontal-only-2d",
            "forced-f64-safe/safe", "small", std::string(bands[kernel]),
            Operation::two_d, SampleKind::f32,
            kernel_recipe(kernel, true, dsmvc::F64Mode::float64_only),
            identity_recipe(7), 0, false,
        });
        result.push_back({
            std::string(bands[kernel]) + "-vertical-only-2d",
            "safe/forced-f64-safe", "small", std::string(bands[kernel]),
            Operation::two_d, SampleKind::f32, identity_recipe(7),
            kernel_recipe(kernel, false, dsmvc::F64Mode::float64_only),
            0, false,
        });

        for (const auto sample : samples) {
            result.push_back({
                std::string(bands[kernel]) + "-risky-safe-2d-"
                    + sample_name(sample),
                "risky/safe", "small", std::string(bands[kernel]),
                Operation::two_d, sample, risky_recipe(), safe_vertical,
                0, false,
            });
            result.push_back({
                std::string(bands[kernel]) + "-safe-risky-2d-"
                    + sample_name(sample),
                "safe/risky", "threshold-adjacent", std::string(bands[kernel]),
                Operation::two_d, sample, safe_horizontal, risky_recipe(),
                0, kernel == 1U,
            });
        }
    }

    for (const auto sample : samples) {
        result.push_back({
            "b3-risky-risky-2d-" + sample_name(sample), "risky/risky",
            "representative", "b3", Operation::two_d, sample,
            risky_recipe(), risky_recipe(), 0, true,
        });
    }

    for (const auto rhs : {1, 2, 3, 4, 5, 7}) {
        result.push_back({
            "b3-risky-rows-rhs" + std::to_string(rhs), "automatic-risk-f64",
            rhs < 4 ? "scalar-tail" : "simd-plus-tail", "b3",
            Operation::rows, SampleKind::f32, risky_recipe(), {}, rhs, false,
        });
        result.push_back({
            "b3-risky-columns-rhs" + std::to_string(rhs),
            "automatic-risk-f64", rhs < 4 ? "scalar-tail" : "simd-plus-tail",
            "b3", Operation::columns, SampleKind::f32, risky_recipe(), {}, rhs,
            false,
        });
    }
    return result;
}

[[nodiscard]] dsmvc::IntegerConversion conversion_for(SampleKind sample) {
    switch (sample) {
    case SampleKind::u8:
        return {16.0F, 1.0F / 219.0F, 219.0F, 16.0F, 255U};
    case SampleKind::u10:
        return {512.0F, 1.0F / 896.0F, 896.0F, 512.0F, 1023U};
    case SampleKind::u16:
        return {0.0F, 1.0F / 65535.0F, 65535.0F, 0.0F, 65535U};
    case SampleKind::f32:
        break;
    }
    throw std::invalid_argument("float samples do not use integer conversion");
}

[[nodiscard]] std::uint32_t ordered_float_bits(float value) noexcept {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return (bits & 0x80000000U) != 0U ? ~bits : bits | 0x80000000U;
}

[[nodiscard]] std::uint32_t ulp_distance(float lhs, float rhs) noexcept {
    const auto left = ordered_float_bits(lhs);
    const auto right = ordered_float_bits(rhs);
    return left > right ? left - right : right - left;
}

template <class T>
void fill_input(std::vector<T> &values, std::uint32_t seed,
                std::uint32_t maximum = 0U) {
    std::uint32_t state = seed;
    for (T &value : values) {
        state ^= state << 13U;
        state ^= state >> 17U;
        state ^= state << 5U;
        if constexpr (std::is_same_v<T, float>) {
            value = static_cast<float>(state & 0xffffU) / 32767.5F - 0.5F;
        } else {
            value = static_cast<T>(state % (maximum + 1U));
        }
    }
}

template <class T>
[[nodiscard]] std::uint64_t logical_hash(
    const std::vector<T> &values, std::int32_t rows, std::int32_t columns,
    std::ptrdiff_t stride) {
    Hasher hash;
    for (std::int32_t row = 0; row < rows; ++row) {
        hash.add_bytes(
            values.data() + static_cast<std::ptrdiff_t>(row) * stride,
            static_cast<std::size_t>(columns) * sizeof(T));
    }
    return hash.value();
}

class Fixture {
public:
    explicit Fixture(CaseSpec requested) : spec_(std::move(requested)) {
        horizontal_ = build_plan(spec_.horizontal);
        if (spec_.operation == Operation::two_d) vertical_ = build_plan(spec_.vertical);
        validate_precision_route();

        if (spec_.operation == Operation::rows) {
            input_stride_ = horizontal_->source_size + 3;
            output_stride_ = horizontal_->destination_size + 3;
            input_rows_ = spec_.independent_rhs;
            output_rows_ = spec_.independent_rhs;
            logical_columns_ = horizontal_->destination_size;
        } else if (spec_.operation == Operation::columns) {
            input_stride_ = spec_.independent_rhs + 3;
            output_stride_ = spec_.independent_rhs + 3;
            input_rows_ = horizontal_->source_size;
            output_rows_ = horizontal_->destination_size;
            logical_columns_ = spec_.independent_rhs;
        } else {
            input_stride_ = horizontal_->source_size + 3;
            output_stride_ = horizontal_->destination_size + 3;
            input_rows_ = vertical_->source_size;
            output_rows_ = vertical_->destination_size;
            logical_columns_ = horizontal_->destination_size;
        }

        const auto input_count = static_cast<std::size_t>(input_rows_)
            * static_cast<std::size_t>(input_stride_);
        const auto output_count = static_cast<std::size_t>(output_rows_)
            * static_cast<std::size_t>(output_stride_);
        if (spec_.sample == SampleKind::f32) {
            float_input_.resize(input_count);
            fill_input(float_input_, seed());
            float_scalar_.assign(output_count, float_guard);
            float_native_.assign(output_count, float_guard);
        } else if (spec_.sample == SampleKind::u8) {
            u8_input_.resize(input_count);
            fill_input(u8_input_, seed(), 255U);
            u8_scalar_.assign(output_count, u8_guard);
            u8_native_.assign(output_count, u8_guard);
        } else {
            u16_input_.resize(input_count);
            const auto maximum = spec_.sample == SampleKind::u10 ? 1023U : 65535U;
            fill_input(u16_input_, seed(), maximum);
            u16_scalar_.assign(output_count, u16_guard);
            u16_native_.assign(output_count, u16_guard);
        }
    }

    void prepare(dsmvc::CpuExecutor &executor) const {
        executor.prepare(horizontal_);
        if (vertical_) executor.prepare(vertical_);
        executor.seal();
    }

    void execute(const dsmvc::CpuExecutor &executor, bool native_output) {
        if (spec_.sample == SampleKind::f32) {
            auto &output = native_output ? float_native_ : float_scalar_;
            if (spec_.operation == Operation::rows) {
                executor.inverse_rows(
                    *horizontal_, float_input_.data(), input_stride_, output.data(),
                    output_stride_, spec_.independent_rhs);
            } else if (spec_.operation == Operation::columns) {
                executor.inverse_columns(
                    *horizontal_, float_input_.data(), input_stride_, output.data(),
                    output_stride_, spec_.independent_rhs);
            } else {
                executor.inverse_2d(
                    *horizontal_, *vertical_, float_input_.data(), input_stride_,
                    output.data(), output_stride_);
            }
        } else if (spec_.sample == SampleKind::u8) {
            auto &output = native_output ? u8_native_ : u8_scalar_;
            executor.inverse_2d_u8(
                *horizontal_, *vertical_, u8_input_.data(), input_stride_,
                output.data(), output_stride_, conversion_for(spec_.sample));
        } else {
            auto &output = native_output ? u16_native_ : u16_scalar_;
            executor.inverse_2d_u16(
                *horizontal_, *vertical_, u16_input_.data(), input_stride_,
                output.data(), output_stride_, conversion_for(spec_.sample));
        }
    }

    [[nodiscard]] Comparison compare() const {
        Comparison result;
        if (spec_.sample == SampleKind::f32) {
            result.scalar_hash = logical_hash(
                float_scalar_, output_rows_, logical_columns_, output_stride_);
            result.native_hash = logical_hash(
                float_native_, output_rows_, logical_columns_, output_stride_);
            for (std::int32_t row = 0; row < output_rows_; ++row) {
                for (std::int32_t column = 0; column < logical_columns_; ++column) {
                    const auto index = static_cast<std::size_t>(row)
                            * static_cast<std::size_t>(output_stride_)
                        + static_cast<std::size_t>(column);
                    const float scalar = float_scalar_[index];
                    const float native = float_native_[index];
                    if (!std::isfinite(scalar) || !std::isfinite(native)) {
                        result.finite = false;
                    } else {
                        result.maximum_ulp = std::max(
                            result.maximum_ulp, ulp_distance(scalar, native));
                    }
                }
            }
            result.bit_exact = result.scalar_hash == result.native_hash;
            result.guards = guards_equal(float_scalar_, float_guard)
                && guards_equal(float_native_, float_guard);
        } else if (spec_.sample == SampleKind::u8) {
            result.scalar_hash = logical_hash(
                u8_scalar_, output_rows_, logical_columns_, output_stride_);
            result.native_hash = logical_hash(
                u8_native_, output_rows_, logical_columns_, output_stride_);
            result.bit_exact = logical_equal(u8_scalar_, u8_native_);
            result.guards = guards_equal(u8_scalar_, u8_guard)
                && guards_equal(u8_native_, u8_guard);
        } else {
            result.scalar_hash = logical_hash(
                u16_scalar_, output_rows_, logical_columns_, output_stride_);
            result.native_hash = logical_hash(
                u16_native_, output_rows_, logical_columns_, output_stride_);
            result.bit_exact = logical_equal(u16_scalar_, u16_native_);
            result.guards = guards_equal(u16_scalar_, u16_guard)
                && guards_equal(u16_native_, u16_guard);
        }
        return result;
    }

    [[nodiscard]] const CaseSpec &spec() const noexcept { return spec_; }
    [[nodiscard]] const dsmvc::AxisPlan &horizontal() const noexcept {
        return *horizontal_;
    }
    [[nodiscard]] const dsmvc::AxisPlan *vertical() const noexcept {
        return vertical_.get();
    }

private:
    void validate_precision_route() const {
        constexpr double risky_threshold = 1.0e-4;
        const auto risky = [&](const dsmvc::AxisPlan &plan) {
            return plan.normal_rcond < risky_threshold && plan.requires_float64();
        };
        const auto safe = [&](const dsmvc::AxisPlan &plan) {
            return plan.normal_rcond >= risky_threshold;
        };
        const auto retained_safe = [&](const dsmvc::AxisPlan &plan) {
            return safe(plan) && plan.requires_float64();
        };
        const auto direct_safe = [&](const dsmvc::AxisPlan &plan) {
            return safe(plan) && !plan.requires_float64();
        };

        bool valid = true;
        if (spec_.precision == "automatic-risk-f64") {
            valid = risky(*horizontal_);
        } else if (spec_.precision == "forced-f64-safe") {
            valid = retained_safe(*horizontal_);
        } else if (spec_.precision == "forced-f64-safe/safe") {
            valid = retained_safe(*horizontal_) && direct_safe(*vertical_);
        } else if (spec_.precision == "safe/forced-f64-safe") {
            valid = direct_safe(*horizontal_) && retained_safe(*vertical_);
        } else if (spec_.precision == "risky/risky") {
            valid = risky(*horizontal_) && risky(*vertical_);
        } else if (spec_.precision == "risky/safe") {
            valid = risky(*horizontal_) && direct_safe(*vertical_);
        } else if (spec_.precision == "safe/risky") {
            valid = direct_safe(*horizontal_) && risky(*vertical_);
        }
        if (!valid) {
            throw std::runtime_error(
                spec_.name + " does not match its precision-route label");
        }
    }

    template <class T>
    [[nodiscard]] bool guards_equal(
        const std::vector<T> &values, T guard) const noexcept {
        for (std::int32_t row = 0; row < output_rows_; ++row) {
            for (std::int32_t column = logical_columns_;
                 column < output_stride_; ++column) {
                if (values[static_cast<std::size_t>(row)
                               * static_cast<std::size_t>(output_stride_)
                           + static_cast<std::size_t>(column)] != guard) {
                    return false;
                }
            }
        }
        return true;
    }

    template <class T>
    [[nodiscard]] bool logical_equal(
        const std::vector<T> &lhs, const std::vector<T> &rhs) const noexcept {
        for (std::int32_t row = 0; row < output_rows_; ++row) {
            for (std::int32_t column = 0; column < logical_columns_; ++column) {
                const auto index = static_cast<std::size_t>(row)
                        * static_cast<std::size_t>(output_stride_)
                    + static_cast<std::size_t>(column);
                if (lhs[index] != rhs[index]) return false;
            }
        }
        return true;
    }

    [[nodiscard]] std::uint32_t seed() const noexcept {
        Hasher hash;
        hash.add_bytes(spec_.name.data(), spec_.name.size());
        return static_cast<std::uint32_t>(hash.value());
    }

    CaseSpec spec_;
    std::shared_ptr<const dsmvc::AxisPlan> horizontal_;
    std::shared_ptr<const dsmvc::AxisPlan> vertical_;
    std::ptrdiff_t input_stride_ = 0;
    std::ptrdiff_t output_stride_ = 0;
    std::int32_t input_rows_ = 0;
    std::int32_t output_rows_ = 0;
    std::int32_t logical_columns_ = 0;
    std::vector<float> float_input_;
    std::vector<float> float_scalar_;
    std::vector<float> float_native_;
    std::vector<std::uint8_t> u8_input_;
    std::vector<std::uint8_t> u8_scalar_;
    std::vector<std::uint8_t> u8_native_;
    std::vector<std::uint16_t> u16_input_;
    std::vector<std::uint16_t> u16_scalar_;
    std::vector<std::uint16_t> u16_native_;
};

[[nodiscard]] Summary summarize(std::vector<double> values) {
    if (values.empty()) return {};
    for (const double value : values) {
        if (!std::isfinite(value) || !(value > 0.0)) {
            throw std::runtime_error("timing sample is not finite and positive");
        }
    }
    Summary result;
    result.raw = std::move(values);
    auto sorted = result.raw;
    std::sort(sorted.begin(), sorted.end());
    const auto middle = sorted.size() / 2U;
    result.median = (sorted.size() & 1U) != 0U
        ? sorted[middle] : (sorted[middle - 1U] + sorted[middle]) / 2.0;
    result.minimum = sorted.front();
    result.maximum = sorted.back();
    std::vector<double> deviations;
    deviations.reserve(sorted.size());
    for (const double value : sorted) {
        deviations.push_back(std::abs(value - result.median));
    }
    std::sort(deviations.begin(), deviations.end());
    result.mad = (deviations.size() & 1U) != 0U
        ? deviations[middle]
        : (deviations[middle - 1U] + deviations[middle]) / 2.0;
    return result;
}

template <class Function>
[[nodiscard]] double timed_nanoseconds(std::size_t iterations, Function &&function) {
    const auto start = Clock::now();
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        function();
    }
    const auto elapsed = std::chrono::duration<double, std::nano>(
        Clock::now() - start).count();
    return elapsed / static_cast<double>(iterations);
}

[[nodiscard]] bool comparison_passes(
    const CaseSpec &spec, const Comparison &comparison) noexcept {
    if (!comparison.finite || !comparison.guards) return false;
    if (spec.sample == SampleKind::f32) return comparison.maximum_ulp <= 1U;
    return comparison.bit_exact;
}

[[nodiscard]] CaseResult run_case(
    const Configuration &configuration, const CaseSpec &spec,
    std::vector<double> &scalar_total, std::vector<double> &native_total) {
    Fixture fixture(spec);
    dsmvc::CpuExecutor scalar(dsmvc::CpuPath::scalar);
    dsmvc::CpuExecutor native(dsmvc::CpuPath::automatic);
    fixture.prepare(scalar);
    fixture.prepare(native);
    fixture.execute(scalar, false);
    fixture.execute(native, true);

    CaseResult result;
    result.spec = fixture.spec();
    result.horizontal_bandwidth = fixture.horizontal().half_bandwidth;
    result.horizontal_rcond = fixture.horizontal().normal_rcond;
    result.horizontal_f64 = fixture.horizontal().requires_float64();
    result.horizontal_f32_hash = f32_plan_hash(fixture.horizontal());
    if (const auto *vertical = fixture.vertical()) {
        result.vertical_bandwidth = vertical->half_bandwidth;
        result.vertical_rcond = vertical->normal_rcond;
        result.vertical_f64 = vertical->requires_float64();
        result.vertical_f32_hash = f32_plan_hash(*vertical);
    }
    result.comparison = fixture.compare();
    if (!comparison_passes(spec, result.comparison)) {
        throw std::runtime_error(spec.name + " correctness gate failed");
    }
    if (configuration.check_only) return result;

    std::vector<double> scalar_raw;
    std::vector<double> native_raw;
    std::vector<double> ratios;
    scalar_raw.reserve(configuration.samples);
    native_raw.reserve(configuration.samples);
    ratios.reserve(configuration.samples);
    for (std::size_t sample = 0; sample < configuration.samples; ++sample) {
        double scalar_ns = 0.0;
        double native_ns = 0.0;
        if ((sample & 1U) == 0U) {
            scalar_ns = timed_nanoseconds(configuration.iterations, [&] {
                fixture.execute(scalar, false);
            });
            native_ns = timed_nanoseconds(configuration.iterations, [&] {
                fixture.execute(native, true);
            });
        } else {
            native_ns = timed_nanoseconds(configuration.iterations, [&] {
                fixture.execute(native, true);
            });
            scalar_ns = timed_nanoseconds(configuration.iterations, [&] {
                fixture.execute(scalar, false);
            });
        }
        const auto comparison = fixture.compare();
        if (!comparison_passes(spec, comparison)) {
            throw std::runtime_error(spec.name + " timed correctness gate failed");
        }
        scalar_raw.push_back(scalar_ns);
        native_raw.push_back(native_ns);
        ratios.push_back(scalar_ns / native_ns);
        if (spec.representative) {
            scalar_total[sample] += scalar_ns;
            native_total[sample] += native_ns;
        }
        benchmark_sink ^= comparison.scalar_hash + comparison.native_hash;
    }
    result.scalar = summarize(std::move(scalar_raw));
    result.native = summarize(std::move(native_raw));
    result.paired_speedup = summarize(std::move(ratios));
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

[[nodiscard]] std::string make_json(
    const Configuration &configuration, const std::vector<CaseResult> &results,
    const std::vector<double> &scalar_total,
    const std::vector<double> &native_total,
    int argc, char **argv, std::string_view native_path) {
    std::ostringstream output;
    output << std::setprecision(17)
           << "{\"schema_version\":1"
           << ",\"benchmark\":\"dsmvc_cpu_f64_avx2\""
           << ",\"source_sha\":" << json_string(configuration.source_id)
           << ",\"binary_sha256\":" << json_string(configuration.binary_sha256)
           << ",\"command\":[";
    for (int index = 0; index < argc; ++index) {
        if (index != 0) output << ',';
        output << json_string(argv[index]);
    }
    output << "]"
           << ",\"host\":{\"system\":" << json_string(system_identity())
           << ",\"architecture\":" << json_string(architecture_identity())
           << ",\"cpu\":" << json_string(cpu_identity()) << '}'
           << ",\"build\":{\"compiler\":" << json_string(compiler_identity())
           << ",\"cxx_standard\":23}"
           << ",\"native_cpu_path\":" << json_string(native_path)
           << ",\"protocol\":{\"samples\":" << configuration.samples
           << ",\"iterations\":" << configuration.iterations
           << ",\"alternating_order\":\"scalar-native/native-scalar\""
           << ",\"float_maximum_ulp\":1,\"integer_bit_exact\":true}"
           << ",\"check_only\":" << (configuration.check_only ? "true" : "false")
           << ",\"case_count\":" << results.size();
    if (!configuration.check_only) {
        std::vector<double> aggregate_ratios(configuration.samples);
        for (std::size_t index = 0; index < aggregate_ratios.size(); ++index) {
            aggregate_ratios[index] = scalar_total[index] / native_total[index];
        }
        output << ",\"representative_aggregate\":{\"scalar_ns\":";
        append_summary(output, summarize(scalar_total));
        output << ",\"native_ns\":";
        append_summary(output, summarize(native_total));
        output << ",\"paired_speedup\":";
        append_summary(output, summarize(std::move(aggregate_ratios)));
        output << '}';
    }
    output << ",\"cases\":[";
    for (std::size_t index = 0; index < results.size(); ++index) {
        if (index != 0U) output << ',';
        const auto &item = results[index];
        output << "{\"name\":" << json_string(item.spec.name)
               << ",\"operation\":" << json_string(operation_name(item.spec.operation))
               << ",\"sample\":" << json_string(sample_name(item.spec.sample))
               << ",\"precision\":" << json_string(item.spec.precision)
               << ",\"scale\":" << json_string(item.spec.scale)
               << ",\"bandwidth_class\":" << json_string(item.spec.bandwidth)
               << ",\"independent_rhs\":" << item.spec.independent_rhs
               << ",\"representative\":"
               << (item.spec.representative ? "true" : "false")
               << ",\"horizontal\":{\"half_bandwidth\":"
               << item.horizontal_bandwidth
               << ",\"normal_rcond\":" << item.horizontal_rcond
               << ",\"requires_float64\":"
               << (item.horizontal_f64 ? "true" : "false")
               << ",\"f32_plan_hash\":"
               << json_string(hex64(item.horizontal_f32_hash)) << '}'
               << ",\"vertical\":{\"half_bandwidth\":"
               << item.vertical_bandwidth
               << ",\"normal_rcond\":" << item.vertical_rcond
               << ",\"requires_float64\":"
               << (item.vertical_f64 ? "true" : "false")
               << ",\"f32_plan_hash\":"
               << json_string(hex64(item.vertical_f32_hash)) << '}'
               << ",\"correctness\":{\"pass\":true,\"finite\":"
               << (item.comparison.finite ? "true" : "false")
               << ",\"guards\":"
               << (item.comparison.guards ? "true" : "false")
               << ",\"bit_exact\":"
               << (item.comparison.bit_exact ? "true" : "false")
               << ",\"maximum_ulp\":" << item.comparison.maximum_ulp
               << ",\"scalar_hash\":"
               << json_string(hex64(item.comparison.scalar_hash))
               << ",\"native_hash\":"
               << json_string(hex64(item.comparison.native_hash)) << '}';
        if (!configuration.check_only) {
            output << ",\"scalar_ns\":";
            append_summary(output, item.scalar);
            output << ",\"native_ns\":";
            append_summary(output, item.native);
            output << ",\"paired_speedup\":";
            append_summary(output, item.paired_speedup);
        }
        output << '}';
    }
    output << "]}\n";
    return output.str();
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto configuration = parse_arguments(argc, argv);
        if (!dsmvc::cpu_avx2_compiled() || !dsmvc::cpu_avx2_available()) {
            throw std::runtime_error("a compiled and available AVX2 CPU path is required");
        }

        auto specs = make_case_specs();
        if (!configuration.case_filter.empty()) {
            std::erase_if(specs, [&](const CaseSpec &spec) {
                return spec.name.find(configuration.case_filter) == std::string::npos;
            });
        }
        if (specs.empty()) throw std::runtime_error("case filter matched no cases");

        std::vector<double> scalar_total(configuration.samples, 0.0);
        std::vector<double> native_total(configuration.samples, 0.0);
        std::vector<CaseResult> results;
        results.reserve(specs.size());
        const dsmvc::CpuExecutor path_probe(dsmvc::CpuPath::automatic);
        for (std::size_t index = 0; index < specs.size(); ++index) {
            auto result = run_case(
                configuration, specs[index], scalar_total, native_total);
            std::cout << '[' << (index + 1U) << '/' << specs.size() << "] "
                      << result.spec.name << " correctness=pass";
            if (!configuration.check_only) {
                std::cout << " scalar_ns=" << result.scalar.median
                          << " native_ns=" << result.native.median
                          << " paired_speedup=" << result.paired_speedup.median;
            }
            std::cout << '\n';
            results.push_back(std::move(result));
        }

        const auto json = make_json(
            configuration, results, scalar_total, native_total,
            argc, argv, path_probe.name());
        if (configuration.json_output.empty()) {
            std::cout << json;
        } else {
            std::ofstream output(configuration.json_output,
                                 std::ios::binary | std::ios::trunc);
            if (!output) throw std::runtime_error("failed to open JSON output");
            output << json;
            if (!output) throw std::runtime_error("failed to write JSON output");
        }
        std::cout << "cpu f64 AVX2 benchmark passed cases=" << results.size()
                  << " sink=" << benchmark_sink << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "cpu f64 AVX2 benchmark failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
