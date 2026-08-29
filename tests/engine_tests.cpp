#include <dsmvc/engine.hpp>

#include "axis_plan_internal.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

class JoiningThread {
public:
    template <class Function>
    explicit JoiningThread(Function &&function)
        : thread_(std::forward<Function>(function)) {}

    JoiningThread(const JoiningThread &) = delete;
    JoiningThread &operator=(const JoiningThread &) = delete;
    JoiningThread(JoiningThread &&) noexcept = default;
    JoiningThread &operator=(JoiningThread &&) = delete;

    ~JoiningThread() {
        if (thread_.joinable()) thread_.join();
    }

private:
    std::thread thread_;
};

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

constexpr std::size_t guard_elements = 32U;
constexpr float guard_value = -123456.75F;
constexpr float output_fill = 65432.25F;

class GuardedFloats {
public:
    explicit GuardedFloats(std::size_t size, float fill)
        : size_(size), storage_(size + 2U * guard_elements, guard_value) {
        std::fill_n(data(), size_, fill);
    }

    [[nodiscard]] float *data() noexcept {
        return storage_.data() + static_cast<std::ptrdiff_t>(guard_elements);
    }

    [[nodiscard]] const float *data() const noexcept {
        return storage_.data() + static_cast<std::ptrdiff_t>(guard_elements);
    }

    [[nodiscard]] bool guards_intact() const noexcept {
        return std::all_of(
                   storage_.begin(), storage_.begin()
                       + static_cast<std::ptrdiff_t>(guard_elements),
                   [](float value) { return value == guard_value; })
            && std::all_of(
                storage_.begin()
                    + static_cast<std::ptrdiff_t>(guard_elements + size_),
                storage_.end(),
                [](float value) { return value == guard_value; });
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
    std::size_t size_;
    std::vector<float> storage_;
};

void fill_deterministic(float *values, std::size_t size, std::uint32_t seed) {
    for (std::size_t index = 0; index < size; ++index) {
        auto bits = static_cast<std::uint32_t>(index) + seed;
        bits ^= bits >> 16U;
        bits *= 0x7feb352dU;
        bits ^= bits >> 15U;
        bits *= 0x846ca68bU;
        bits ^= bits >> 16U;
        values[index] = static_cast<float>(bits & 0xffffU) / 32767.5F - 1.0F;
    }
}

struct ErrorStats {
    float maximum = 0.0F;
    double mean = 0.0;
    std::size_t non_finite = 0U;
};

ErrorStats compare_matrix(const float *reference, const float *candidate,
                          std::int32_t rows, std::int32_t columns,
                          std::ptrdiff_t stride) {
    ErrorStats stats;
    double sum = 0.0;
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t column = 0; column < columns; ++column) {
            const auto index = static_cast<std::ptrdiff_t>(row) * stride + column;
            if (!std::isfinite(candidate[index])) {
                ++stats.non_finite;
                continue;
            }
            const auto error = std::abs(reference[index] - candidate[index]);
            stats.maximum = std::max(stats.maximum, error);
            sum += static_cast<double>(error);
        }
    }
    const auto count = static_cast<double>(rows) * static_cast<double>(columns);
    stats.mean = count == 0.0 ? 0.0 : sum / count;
    return stats;
}

void require_agreement(const ErrorStats &stats, std::string_view label,
                       bool report = true) {
    if (report) {
        std::cout << label << ": max_error=" << stats.maximum
                  << " mean_error=" << stats.mean
                  << " non_finite=" << stats.non_finite << '\n';
    }
    require(stats.non_finite == 0U,
            std::string(label) + " produced non-finite output");
    require(stats.maximum < 2.0e-5F,
            std::string(label) + " differs from the scalar executor");
}

[[nodiscard]] dsmvc::AxisPlan make_plan(
    dsmvc::KernelKind kind, std::int32_t source_size,
    std::int32_t destination_size, double active_length, double shift = 0.0,
    double blur = 1.0) {
    dsmvc::AxisRequest request;
    request.source_size = source_size;
    request.destination_size = destination_size;
    request.active_length = active_length;
    request.shift = shift;
    request.kernel.kind = kind;
    if (kind == dsmvc::KernelKind::lanczos) request.kernel.taps = 3;
    request.kernel.blur = blur;
    return dsmvc::build_axis_plan(request);
}

[[nodiscard]] bool native_simd_available() noexcept {
    return dsmvc::cpu_avx512_available() || dsmvc::cpu_avx2_available()
        || dsmvc::cpu_neon_available();
}

[[nodiscard]] dsmvc::CpuPath native_simd_path() noexcept {
    return dsmvc::cpu_avx512_available()
        ? dsmvc::CpuPath::avx512
        : dsmvc::cpu_avx2_available()
            ? dsmvc::CpuPath::avx2 : dsmvc::CpuPath::neon;
}

ErrorStats compare_rows(const dsmvc::CpuExecutor &optimized_executor,
                        const dsmvc::AxisPlan &plan, std::int32_t rows,
                        std::string_view label, bool report = true) {
    const auto input_stride = (plan.source_size + 7) & ~7;
    const auto output_stride = (plan.destination_size + 7) & ~7;
    GuardedFloats input(
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(input_stride),
        0.0F);
    GuardedFloats scalar(
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(output_stride),
        output_fill);
    GuardedFloats optimized(
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(output_stride),
        output_fill);
    fill_deterministic(input.data(), input.size(), 0x10203040U
        + static_cast<std::uint32_t>(rows));

    const dsmvc::CpuExecutor scalar_executor(dsmvc::CpuPath::scalar);
    scalar_executor.inverse_rows(
        plan, input.data(), input_stride, scalar.data(), output_stride, rows);
    optimized_executor.inverse_rows(
        plan, input.data(), input_stride, optimized.data(), output_stride, rows);

    require(input.guards_intact(), std::string(label) + " changed input guards");
    require(scalar.guards_intact(), std::string(label) + " changed scalar guards");
    require(optimized.guards_intact(),
            std::string(label) + " changed optimized guards");
    require(std::all_of(optimized.data(), optimized.data() + optimized.size(),
                        [](float value) { return std::isfinite(value); }),
            std::string(label) + " left a non-finite padded output");

    const auto stats = compare_matrix(
        scalar.data(), optimized.data(), rows, plan.destination_size,
        output_stride);
    require_agreement(stats, label, report);
    return stats;
}

ErrorStats compare_columns(const dsmvc::CpuExecutor &optimized_executor,
                           const dsmvc::AxisPlan &plan,
                           std::int32_t columns, std::int32_t stride,
                           std::string_view label) {
    GuardedFloats input(
        static_cast<std::size_t>(plan.source_size)
            * static_cast<std::size_t>(stride),
        0.0F);
    GuardedFloats scalar(
        static_cast<std::size_t>(plan.destination_size)
            * static_cast<std::size_t>(stride),
        output_fill);
    GuardedFloats optimized(
        static_cast<std::size_t>(plan.destination_size)
            * static_cast<std::size_t>(stride),
        output_fill);
    fill_deterministic(input.data(), input.size(), 0x50607080U
        + static_cast<std::uint32_t>(columns));

    const dsmvc::CpuExecutor scalar_executor(dsmvc::CpuPath::scalar);
    scalar_executor.inverse_columns(
        plan, input.data(), stride, scalar.data(), stride, columns);
    optimized_executor.inverse_columns(
        plan, input.data(), stride, optimized.data(), stride, columns);

    require(input.guards_intact(), std::string(label) + " changed input guards");
    require(scalar.guards_intact(), std::string(label) + " changed scalar guards");
    require(optimized.guards_intact(),
            std::string(label) + " changed optimized guards");
    require(std::all_of(optimized.data(), optimized.data() + optimized.size(),
                        [](float value) { return std::isfinite(value); }),
            std::string(label) + " left a non-finite padded output");

    const auto stats = compare_matrix(
        scalar.data(), optimized.data(), plan.destination_size, columns, stride);
    require_agreement(stats, label);
    return stats;
}

ErrorStats compare_2d(dsmvc::CpuPath path, dsmvc::KernelKind kind,
                      std::int32_t source_width,
                      std::int32_t destination_width,
                      std::int32_t source_height,
                      std::int32_t destination_height,
                      bool padded_output, std::string_view label) {
    auto horizontal = std::make_shared<const dsmvc::AxisPlan>(make_plan(
        kind, source_width, destination_width,
        static_cast<double>(destination_width) - 0.25, 0.125));
    auto vertical = std::make_shared<const dsmvc::AxisPlan>(make_plan(
        kind, source_height, destination_height,
        static_cast<double>(destination_height) - 0.5, 0.25));
    const auto input_stride = (source_width + 7) & ~7;
    const auto intermediate_stride = (destination_width + 7) & ~7;
    const auto output_stride = padded_output
        ? intermediate_stride : destination_width;

    GuardedFloats input(
        static_cast<std::size_t>(source_height)
            * static_cast<std::size_t>(input_stride),
        0.0F);
    GuardedFloats intermediate(
        static_cast<std::size_t>(source_height)
            * static_cast<std::size_t>(intermediate_stride),
        output_fill);
    GuardedFloats legacy(
        static_cast<std::size_t>(destination_height)
            * static_cast<std::size_t>(output_stride),
        output_fill);
    GuardedFloats streamed(
        static_cast<std::size_t>(destination_height)
            * static_cast<std::size_t>(output_stride),
        output_fill);
    fill_deterministic(input.data(), input.size(), 0xa5a50000U
        + static_cast<std::uint32_t>(kind)
        + static_cast<std::uint32_t>(source_height));

    dsmvc::CpuExecutor executor(path);
    executor.prepare(horizontal);
    executor.prepare(vertical);
    executor.seal();
    executor.inverse_rows(
        *horizontal, input.data(), input_stride,
        intermediate.data(), intermediate_stride, source_height);
    executor.inverse_columns(
        *vertical, intermediate.data(), intermediate_stride,
        legacy.data(), output_stride, destination_width);
    executor.inverse_2d(
        *horizontal, *vertical, input.data(), input_stride,
        streamed.data(), output_stride);

    require(input.guards_intact(), std::string(label) + " changed input guards");
    require(intermediate.guards_intact(),
            std::string(label) + " changed intermediate guards");
    require(legacy.guards_intact(), std::string(label) + " changed legacy guards");
    require(streamed.guards_intact(),
            std::string(label) + " changed streamed guards");
    const auto stats = compare_matrix(
        legacy.data(), streamed.data(), destination_height,
        destination_width, output_stride);
    require_agreement(stats, label);

    std::size_t different = 0U;
    for (std::int32_t row = 0; row < destination_height; ++row) {
        for (std::int32_t column = 0; column < destination_width; ++column) {
            const auto index = static_cast<std::ptrdiff_t>(row)
                * output_stride + column;
            different += legacy.data()[index] != streamed.data()[index];
        }
    }
    std::cout << label << ": bit_differences=" << different << '\n';
    // Only scalar is the canonical bit-exact reference. SIMD routes may use
    // different FMA widths and evaluation schedules, covered by the numeric
    // tolerance above.
    if (path == dsmvc::CpuPath::scalar) {
        require(different == 0U,
                std::string(label)
                    + " is not bit-exact with the legacy two-pass path");
    }
    return stats;
}

template <class Sample>
void compare_integer_2d(
    dsmvc::CpuPath path, dsmvc::KernelKind kind,
    const dsmvc::IntegerConversion &conversion,
    std::string_view label, std::int32_t source_height = 73,
    std::int32_t destination_height = 51) {
    constexpr std::int32_t source_width = 97;
    constexpr std::int32_t destination_width = 67;
    constexpr std::int32_t input_stride = 104;
    constexpr std::int32_t float_output_stride = 72;
    constexpr std::int32_t integer_output_stride = 73;
    auto horizontal = std::make_shared<const dsmvc::AxisPlan>(make_plan(
        kind, source_width, destination_width, 66.75, 0.125));
    auto vertical = std::make_shared<const dsmvc::AxisPlan>(make_plan(
        kind, source_height, destination_height,
        static_cast<double>(destination_height) - 0.5, 0.25));

    std::vector<Sample> integer_input(
        static_cast<std::size_t>(source_height)
            * static_cast<std::size_t>(input_stride));
    for (std::size_t index = 0; index < integer_input.size(); ++index) {
        auto bits = static_cast<std::uint32_t>(index) + 0x31415926U;
        bits ^= bits >> 16U;
        bits *= 0x7feb352dU;
        bits ^= bits >> 15U;
        integer_input[index] = static_cast<Sample>(
            bits % (conversion.output_maximum + 1U));
    }
    std::vector<float> float_input(integer_input.size());
    for (std::size_t index = 0; index < integer_input.size(); ++index) {
        float_input[index] =
            (static_cast<float>(integer_input[index]) - conversion.input_offset)
            * conversion.input_scale;
    }
    std::vector<float> float_output(
        static_cast<std::size_t>(destination_height)
            * static_cast<std::size_t>(float_output_stride),
        output_fill);
    std::vector<Sample> reference(
        static_cast<std::size_t>(destination_height)
            * static_cast<std::size_t>(integer_output_stride));
    std::vector<Sample> candidate(reference.size());
    std::vector<Sample> streamed(reference.size());

    dsmvc::CpuExecutor executor(path);
    executor.prepare(horizontal);
    executor.prepare(vertical);
    executor.seal();
    executor.inverse_2d(
        *horizontal, *vertical, float_input.data(), input_stride,
        float_output.data(), float_output_stride);
    for (std::int32_t row = 0; row < destination_height; ++row) {
        for (std::int32_t column = 0; column < destination_width; ++column) {
            const auto float_index = static_cast<std::ptrdiff_t>(row)
                * float_output_stride + column;
            const auto integer_index = static_cast<std::ptrdiff_t>(row)
                * integer_output_stride + column;
            const auto scaled = std::clamp(
                float_output[float_index] * conversion.output_scale
                    + conversion.output_offset,
                0.0F, static_cast<float>(conversion.output_maximum));
            reference[integer_index] =
                static_cast<Sample>(std::nearbyint(scaled));
        }
    }
    if constexpr (std::is_same_v<Sample, std::uint8_t>) {
        executor.inverse_2d_u8(
            *horizontal, *vertical, integer_input.data(), input_stride,
            candidate.data(), integer_output_stride, conversion);
        executor.inverse_2d_u8_streamed(
            *horizontal, *vertical, integer_input.data(), input_stride,
            streamed.data(), integer_output_stride, conversion);
    } else {
        executor.inverse_2d_u16(
            *horizontal, *vertical, integer_input.data(), input_stride,
            candidate.data(), integer_output_stride, conversion);
        executor.inverse_2d_u16_streamed(
            *horizontal, *vertical, integer_input.data(), input_stride,
            streamed.data(), integer_output_stride, conversion);
    }

    std::size_t different = 0U;
    std::size_t streamed_different = 0U;
    std::uint32_t maximum_error = 0U;
    std::uint32_t streamed_maximum_error = 0U;
    for (std::int32_t row = 0; row < destination_height; ++row) {
        for (std::int32_t column = 0; column < destination_width; ++column) {
            const auto index = static_cast<std::ptrdiff_t>(row)
                * integer_output_stride + column;
            const auto left = static_cast<std::uint32_t>(reference[index]);
            const auto right = static_cast<std::uint32_t>(candidate[index]);
            const auto streamed_right = static_cast<std::uint32_t>(streamed[index]);
            different += left != right;
            streamed_different += left != streamed_right;
            maximum_error = std::max(
                maximum_error, left > right ? left - right : right - left);
            streamed_maximum_error = std::max(
                streamed_maximum_error,
                left > streamed_right ? left - streamed_right
                                      : streamed_right - left);
        }
    }
    std::cout << label << ": integer_differences=" << different
              << " max_error=" << maximum_error
              << " streamed_differences=" << streamed_different
              << " streamed_max_error=" << streamed_maximum_error << '\n';
    require(different == 0U,
            std::string(label) + " differs from the Float32 Point reference");
    require(streamed_different == 0U,
            std::string(label)
                + " streamed path differs from the Float32 Point reference");
}

void test_backend_selection() {
    require(dsmvc::parse_backend("AUTO") == dsmvc::BackendKind::automatic,
            "backend parsing failed");
    require(dsmvc::resolve_backend(dsmvc::BackendKind::automatic)
                == dsmvc::BackendKind::cpu,
            "automatic backend did not select CPU");
    bool rejected = false;
    try {
        require(dsmvc::resolve_backend(dsmvc::BackendKind::cuda)
                    == dsmvc::BackendKind::cuda,
                "available CUDA backend did not resolve to CUDA");
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected != dsmvc::cuda_available(),
            "CUDA resolution disagrees with runtime availability");
    const auto capabilities = dsmvc::backend_capabilities();
    const auto metal = std::find_if(
        capabilities.begin(), capabilities.end(), [](const auto &capability) {
            return capability.kind == dsmvc::BackendKind::metal;
        });
    require(metal != capabilities.end(), "Metal capability is missing");
    require(metal->compiled == dsmvc::metal_compiled()
                && metal->device_available == dsmvc::metal_available(),
            "Metal capability reporting is inconsistent");
    const auto cuda = std::find_if(
        capabilities.begin(), capabilities.end(), [](const auto &capability) {
            return capability.kind == dsmvc::BackendKind::cuda;
        });
    require(cuda != capabilities.end(), "CUDA capability is missing");
    require(cuda->compiled == dsmvc::cuda_compiled()
                && cuda->device_available == dsmvc::cuda_available(),
            "CUDA capability reporting is inconsistent");
    rejected = false;
    try {
        require(dsmvc::resolve_backend(dsmvc::BackendKind::vulkan)
                    == dsmvc::BackendKind::vulkan,
                "available Vulkan backend did not resolve to Vulkan");
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected != dsmvc::vulkan_available(),
            "Vulkan resolution disagrees with runtime availability");
    const auto vulkan = std::find_if(
        capabilities.begin(), capabilities.end(), [](const auto &capability) {
            return capability.kind == dsmvc::BackendKind::vulkan;
        });
    require(vulkan != capabilities.end(), "Vulkan capability is missing");
    require(vulkan->compiled == dsmvc::vulkan_compiled()
                && vulkan->device_available == dsmvc::vulkan_available(),
            "Vulkan capability reporting is inconsistent");
}

void test_accelerator_executor_agreement(
    dsmvc::BackendKind backend, bool available, const char *backend_label) {
    if (!available) {
        std::cout << backend_label
                  << " executor tests skipped: no compatible device\n";
        return;
    }

    constexpr std::int32_t source_width = 97;
    constexpr std::int32_t destination_width = 67;
    constexpr std::int32_t source_height = 73;
    constexpr std::int32_t destination_height = 51;
    constexpr std::int32_t input_stride = 104;
    constexpr std::int32_t output_stride = 72;

    struct PlanPair {
        const char *name;
        std::shared_ptr<const dsmvc::AxisPlan> horizontal;
        std::shared_ptr<const dsmvc::AxisPlan> vertical;
    };
    std::vector<PlanPair> pairs;
    for (const auto &[kind, name] : {
             std::pair{dsmvc::KernelKind::bilinear, "b1"},
             std::pair{dsmvc::KernelKind::bicubic, "b3"},
             std::pair{dsmvc::KernelKind::lanczos, "b5"},
             std::pair{dsmvc::KernelKind::spline64, "b7"},
         }) {
        pairs.push_back({
            name,
            std::make_shared<const dsmvc::AxisPlan>(make_plan(
                kind, source_width, destination_width, 66.75, 0.125)),
            std::make_shared<const dsmvc::AxisPlan>(make_plan(
                kind, source_height, destination_height, 50.5, 0.25)),
        });
    }

    dsmvc::AxisRequest generic_horizontal_request;
    generic_horizontal_request.source_size = source_width;
    generic_horizontal_request.destination_size = destination_width;
    generic_horizontal_request.active_length = 66.75;
    generic_horizontal_request.shift = 0.125;
    generic_horizontal_request.kernel.kind = dsmvc::KernelKind::spline64;
    generic_horizontal_request.kernel.blur = 1.25;
    auto generic_vertical_request = generic_horizontal_request;
    generic_vertical_request.source_size = source_height;
    generic_vertical_request.destination_size = destination_height;
    generic_vertical_request.active_length = 50.5;
    generic_vertical_request.shift = 0.25;
    pairs.push_back({
        "blur-b9",
        std::make_shared<const dsmvc::AxisPlan>(
            dsmvc::build_axis_plan(generic_horizontal_request)),
        std::make_shared<const dsmvc::AxisPlan>(
            dsmvc::build_axis_plan(generic_vertical_request)),
    });
    require(pairs.back().horizontal->half_bandwidth > 7,
            "accelerator blur fixture did not exceed the specialized bandwidths");
    generic_horizontal_request.kernel.blur = 1.251;
    generic_vertical_request.kernel.blur = 1.251;
    pairs.push_back({
        "blur-b11",
        std::make_shared<const dsmvc::AxisPlan>(
            dsmvc::build_axis_plan(generic_horizontal_request)),
        std::make_shared<const dsmvc::AxisPlan>(
            dsmvc::build_axis_plan(generic_vertical_request)),
    });
    require(pairs.back().horizontal->half_bandwidth == 11
                && pairs.back().vertical->half_bandwidth == 11,
            "accelerator blur H11 fixture did not use half-bandwidth 11");

    dsmvc::Executor accelerator(backend);
    require(accelerator.backend() == backend,
            "accelerator executor resolved to the wrong backend");
    for (const auto &pair : pairs) {
        accelerator.prepare(pair.horizontal);
        accelerator.prepare(pair.vertical);
    }
    accelerator.seal();
    accelerator.seal();

    std::vector<float> input(
        static_cast<std::size_t>(source_height)
            * static_cast<std::size_t>(input_stride));
    fill_deterministic(input.data(), input.size(), 0xc001d00dU);
    const dsmvc::CpuExecutor scalar(dsmvc::CpuPath::scalar);
    std::vector<float> reference(
        static_cast<std::size_t>(destination_height)
            * static_cast<std::size_t>(output_stride),
        output_fill);
    std::vector<float> candidate(reference.size(), output_fill);

    for (const auto &pair : pairs) {
        std::fill(reference.begin(), reference.end(), output_fill);
        std::fill(candidate.begin(), candidate.end(), output_fill);
        scalar.inverse_2d(
            *pair.horizontal, *pair.vertical, input.data(), input_stride,
            reference.data(), output_stride);
        accelerator.inverse_2d(
            *pair.horizontal, *pair.vertical, input.data(), input_stride,
            candidate.data(), output_stride);
        const auto stats = compare_matrix(
            reference.data(), candidate.data(), destination_height,
            destination_width, output_stride);
        std::cout << backend_label << ' ' << pair.name << " 2D: max_error="
                  << stats.maximum << " mean_error=" << stats.mean
                  << " non_finite=" << stats.non_finite << '\n';
        require(stats.non_finite == 0U && stats.maximum < 1.0e-4F,
                std::string(backend_label) + ' ' + pair.name
                    + " 2D differs from the scalar executor");
    }

    const auto &separate = pairs[3];
    constexpr std::int32_t intermediate_stride = 72;
    std::vector<float> cpu_intermediate(
        static_cast<std::size_t>(source_height)
            * static_cast<std::size_t>(intermediate_stride),
        output_fill);
    std::vector<float> accelerator_intermediate(
        cpu_intermediate.size(), output_fill);
    scalar.inverse_rows(
        *separate.horizontal, input.data(), input_stride,
        cpu_intermediate.data(), intermediate_stride, source_height);
    accelerator.inverse_rows(
        *separate.horizontal, input.data(), input_stride,
        accelerator_intermediate.data(), intermediate_stride, source_height);
    auto stats = compare_matrix(
        cpu_intermediate.data(), accelerator_intermediate.data(), source_height,
        destination_width, intermediate_stride);
    require(stats.non_finite == 0U && stats.maximum < 1.0e-4F,
            "accelerator row executor differs from scalar");

    const dsmvc::AxisPlan unprepared_horizontal = *separate.horizontal;
    std::fill(
        accelerator_intermediate.begin(), accelerator_intermediate.end(), output_fill);
    accelerator.inverse_rows(
        unprepared_horizontal, input.data(), input_stride,
        accelerator_intermediate.data(), intermediate_stride, source_height);
    stats = compare_matrix(
        cpu_intermediate.data(), accelerator_intermediate.data(), source_height,
        destination_width, intermediate_stride);
    require(stats.non_finite == 0U && stats.maximum < 1.0e-4F,
            "accelerator direct unprepared plan differs from scalar");

    std::fill(reference.begin(), reference.end(), output_fill);
    std::fill(candidate.begin(), candidate.end(), output_fill);
    scalar.inverse_columns(
        *separate.vertical, cpu_intermediate.data(), intermediate_stride,
        reference.data(), output_stride, destination_width);
    accelerator.inverse_columns(
        *separate.vertical, cpu_intermediate.data(), intermediate_stride,
        candidate.data(), output_stride, destination_width);
    stats = compare_matrix(
        reference.data(), candidate.data(), destination_height,
        destination_width, output_stride);
    require(stats.non_finite == 0U && stats.maximum < 1.0e-4F,
            "accelerator column executor differs from scalar");
    const std::vector<float> concurrency_reference = reference;

    auto cached_column_input =
        std::make_shared<std::vector<float>>(cpu_intermediate);
    const std::shared_ptr<const void> cached_column_lifetime(
        cached_column_input,
        static_cast<const void *>(cached_column_input->data()));
    for (int repeat = 0; repeat < 2; ++repeat) {
        std::fill(candidate.begin(), candidate.end(), output_fill);
        accelerator.inverse_columns(
            *separate.vertical, cached_column_input->data(),
            intermediate_stride, candidate.data(), output_stride,
            destination_width, cached_column_lifetime);
        stats = compare_matrix(
            concurrency_reference.data(), candidate.data(), destination_height,
            destination_width, output_stride);
        require(stats.non_finite == 0U && stats.maximum < 1.0e-4F,
                "cached accelerator column executor differs from scalar");
    }

    const dsmvc::IntegerConversion u8_conversion{
        16.0F, 1.0F / 219.0F, 219.0F, 16.0F, 255U};
    std::vector<std::uint8_t> integer_input(input.size());
    std::vector<float> normalized(input.size());
    for (std::size_t index = 0; index < integer_input.size(); ++index) {
        integer_input[index] = static_cast<std::uint8_t>(
            16U + (index * 37U + 11U) % 220U);
        normalized[index] =
            (static_cast<float>(integer_input[index])
                 - u8_conversion.input_offset)
            * u8_conversion.input_scale;
    }
    scalar.inverse_2d(
        *separate.horizontal, *separate.vertical,
        normalized.data(), input_stride, reference.data(), output_stride);
    std::vector<std::uint8_t> integer_reference(reference.size());
    std::vector<std::uint8_t> integer_candidate(reference.size(), 0U);
    const auto maximum_integer_difference = [=](
        const auto &left, const auto &right) {
        std::uint32_t maximum = 0U;
        for (std::int32_t row = 0; row < destination_height; ++row) {
            for (std::int32_t column = 0; column < destination_width; ++column) {
                const auto index = static_cast<std::ptrdiff_t>(row)
                    * output_stride + column;
                const auto lhs = static_cast<std::uint32_t>(left[index]);
                const auto rhs = static_cast<std::uint32_t>(right[index]);
                maximum = std::max(
                    maximum, lhs > rhs ? lhs - rhs : rhs - lhs);
            }
        }
        return maximum;
    };
    for (std::int32_t row = 0; row < destination_height; ++row) {
        for (std::int32_t column = 0; column < destination_width; ++column) {
            const auto index = static_cast<std::ptrdiff_t>(row)
                * output_stride + column;
            const float converted = std::clamp(
                reference[index] * u8_conversion.output_scale
                    + u8_conversion.output_offset,
                0.0F, 255.0F);
            integer_reference[index] =
                static_cast<std::uint8_t>(std::nearbyint(converted));
        }
    }
    accelerator.inverse_2d_u8(
        *separate.horizontal, *separate.vertical,
        integer_input.data(), input_stride,
        integer_candidate.data(), output_stride, u8_conversion);
    std::uint32_t maximum_integer_error = maximum_integer_difference(
        integer_reference, integer_candidate);
    require(maximum_integer_error <= 1U,
            "accelerator u8 conversion differs from the Float32 reference by more than one");

    auto cached_u8_input =
        std::make_shared<std::vector<std::uint8_t>>(integer_input);
    const std::shared_ptr<const void> cached_u8_lifetime(
        cached_u8_input, static_cast<const void *>(cached_u8_input->data()));
    std::fill(integer_candidate.begin(), integer_candidate.end(), 0U);
    accelerator.inverse_2d_u8(
        *separate.horizontal, *separate.vertical,
        cached_u8_input->data(), input_stride,
        integer_candidate.data(), output_stride, u8_conversion,
        cached_u8_lifetime);
    require(maximum_integer_difference(
                integer_reference, integer_candidate) <= 1U,
            "cached accelerator u8 conversion differs from the Float32 reference");

    const dsmvc::IntegerConversion full_range_u8_conversion{
        0.0F, 1.0F / 255.0F, 255.0F, 0.0F, 255U};
    for (std::size_t index = 0; index < cached_u8_input->size(); ++index) {
        normalized[index] = static_cast<float>((*cached_u8_input)[index])
            * full_range_u8_conversion.input_scale;
    }
    scalar.inverse_2d(
        *separate.horizontal, *separate.vertical,
        normalized.data(), input_stride, reference.data(), output_stride);
    std::vector<std::uint8_t> full_range_reference(reference.size());
    std::vector<std::uint8_t> full_range_candidate(reference.size(), 0U);
    for (std::int32_t row = 0; row < destination_height; ++row) {
        for (std::int32_t column = 0; column < destination_width; ++column) {
            const auto index = static_cast<std::ptrdiff_t>(row)
                * output_stride + column;
            full_range_reference[index] = static_cast<std::uint8_t>(
                std::nearbyint(std::clamp(
                    reference[index] * full_range_u8_conversion.output_scale,
                    0.0F, 255.0F)));
        }
    }
    accelerator.inverse_2d_u8(
        *separate.horizontal, *separate.vertical,
        cached_u8_input->data(), input_stride,
        full_range_candidate.data(), output_stride, full_range_u8_conversion,
        cached_u8_lifetime);
    require(maximum_integer_difference(
                full_range_reference, full_range_candidate) <= 1U,
            "accelerator input cache reused the wrong u8 conversion");

    const dsmvc::IntegerConversion u16_conversion{
        512.0F, 1.0F / 896.0F, 896.0F, 512.0F, 1023U};
    std::vector<std::uint16_t> u16_input(input.size());
    for (std::size_t index = 0; index < u16_input.size(); ++index) {
        u16_input[index] = static_cast<std::uint16_t>(
            (index * 73U + 19U) % 1024U);
        normalized[index] =
            (static_cast<float>(u16_input[index])
                 - u16_conversion.input_offset)
            * u16_conversion.input_scale;
    }
    scalar.inverse_2d(
        *separate.horizontal, *separate.vertical,
        normalized.data(), input_stride, reference.data(), output_stride);
    std::vector<std::uint16_t> u16_reference(reference.size());
    std::vector<std::uint16_t> u16_candidate(reference.size(), 0U);
    for (std::int32_t row = 0; row < destination_height; ++row) {
        for (std::int32_t column = 0; column < destination_width; ++column) {
            const auto index = static_cast<std::ptrdiff_t>(row)
                * output_stride + column;
            const float converted = std::clamp(
                reference[index] * u16_conversion.output_scale
                    + u16_conversion.output_offset,
                0.0F, 1023.0F);
            u16_reference[index] =
                static_cast<std::uint16_t>(std::nearbyint(converted));
        }
    }
    accelerator.inverse_2d_u16(
        *separate.horizontal, *separate.vertical,
        u16_input.data(), input_stride,
        u16_candidate.data(), output_stride, u16_conversion);
    maximum_integer_error = maximum_integer_difference(
        u16_reference, u16_candidate);
    require(maximum_integer_error <= 1U,
            "accelerator u16 conversion differs from the Float32 reference by more than one");

    auto cached_u16_input =
        std::make_shared<std::vector<std::uint16_t>>(u16_input);
    const std::shared_ptr<const void> cached_u16_lifetime(
        cached_u16_input, static_cast<const void *>(cached_u16_input->data()));
    std::fill(u16_candidate.begin(), u16_candidate.end(), 0U);
    accelerator.inverse_2d_u16(
        *separate.horizontal, *separate.vertical,
        cached_u16_input->data(), input_stride,
        u16_candidate.data(), output_stride, u16_conversion,
        cached_u16_lifetime);
    require(maximum_integer_difference(u16_reference, u16_candidate) <= 1U,
            "cached accelerator u16 conversion differs from the Float32 reference");

    auto cached_input = std::make_shared<std::vector<float>>(input);
    const std::shared_ptr<const void> cached_input_lifetime(
        cached_input, static_cast<const void *>(cached_input->data()));
    const auto check_concurrent = [&](
        const PlanPair &pair, const std::vector<float> &expected,
        const std::shared_ptr<std::vector<float>> &shared_input,
        std::size_t caller_count, const char *label) {
        const std::shared_ptr<const void> shared_lifetime(
            shared_input, static_cast<const void *>(shared_input->data()));
        std::vector<std::exception_ptr> errors(caller_count);
        std::barrier start(static_cast<std::ptrdiff_t>(errors.size()));
        std::vector<JoiningThread> callers;
        callers.reserve(errors.size());
        for (std::size_t index = 0; index < errors.size(); ++index) {
            callers.emplace_back([&, index] {
                try {
                    std::vector<float> concurrent(candidate.size(), output_fill);
                    start.arrive_and_wait();
                    accelerator.inverse_2d(
                        *pair.horizontal, *pair.vertical,
                        shared_input->data(), input_stride,
                        concurrent.data(), output_stride, shared_lifetime);
                    const auto concurrent_stats = compare_matrix(
                        expected.data(), concurrent.data(), destination_height,
                        destination_width, output_stride);
                    require(
                        concurrent_stats.non_finite == 0U
                            && concurrent_stats.maximum < 1.0e-4F,
                        std::string("concurrent ") + backend_label + ' ' + label
                            + " execution differs from scalar");
                } catch (...) {
                    errors[index] = std::current_exception();
                }
            });
        }
        callers.clear();
        for (const auto &error : errors) {
            if (error) std::rethrow_exception(error);
        }
    };
    check_concurrent(
        separate, concurrency_reference, cached_input, 6U, "b7");

    const auto &limited = pairs[1];
    std::vector<float> limited_reference(reference.size(), output_fill);
    scalar.inverse_2d(
        *limited.horizontal, *limited.vertical,
        cached_input->data(), input_stride,
        limited_reference.data(), output_stride);
    auto fresh_limited_input = std::make_shared<std::vector<float>>(input);
    check_concurrent(
        limited, limited_reference, fresh_limited_input, 16U,
        "fresh-cache b3");

    std::fill(
        accelerator_intermediate.begin(), accelerator_intermediate.end(), output_fill);
    accelerator.inverse_rows(
        *separate.horizontal, cached_input->data(), input_stride,
        accelerator_intermediate.data(), intermediate_stride, source_height,
        cached_input_lifetime);
    stats = compare_matrix(
        cpu_intermediate.data(), accelerator_intermediate.data(), source_height,
        destination_width, intermediate_stride);
    require(stats.non_finite == 0U && stats.maximum < 1.0e-4F,
            "cached accelerator row executor differs from scalar");

    const char *eviction = std::getenv("DSMVC_TEST_VULKAN_CACHE_EVICTION");
    if (backend == dsmvc::BackendKind::vulkan && eviction
        && std::string_view{eviction} == "1") {
        std::vector<std::shared_ptr<std::vector<float>>> cache_inputs;
        cache_inputs.reserve(40U);
        for (std::size_t index = 0U; index < 40U; ++index) {
            auto cache_input = std::make_shared<std::vector<float>>(input);
            const std::shared_ptr<const void> cache_lifetime(
                cache_input, static_cast<const void *>(cache_input->data()));
            std::fill(candidate.begin(), candidate.end(), output_fill);
            accelerator.inverse_2d(
                *limited.horizontal, *limited.vertical,
                cache_input->data(), input_stride, candidate.data(), output_stride,
                cache_lifetime);
            const auto eviction_stats = compare_matrix(
                limited_reference.data(), candidate.data(), destination_height,
                destination_width, output_stride);
            require(
                eviction_stats.non_finite == 0U
                    && eviction_stats.maximum < 1.0e-4F,
                "Vulkan input-cache eviction execution differs from scalar");
            cache_inputs.push_back(std::move(cache_input));
        }

        const std::shared_ptr<const void> first_lifetime(
            cache_inputs.front(),
            static_cast<const void *>(cache_inputs.front()->data()));
        std::fill(candidate.begin(), candidate.end(), output_fill);
        accelerator.inverse_2d(
            *limited.horizontal, *limited.vertical,
            cache_inputs.front()->data(), input_stride,
            candidate.data(), output_stride, first_lifetime);
        const auto reupload_stats = compare_matrix(
            limited_reference.data(), candidate.data(), destination_height,
            destination_width, output_stride);
        require(
            reupload_stats.non_finite == 0U
                && reupload_stats.maximum < 1.0e-4F,
            "Vulkan evicted input re-upload differs from scalar");
    }
}

void test_cpu_path_selection() {
    require(!dsmvc::cpu_avx2_available() || dsmvc::cpu_avx2_compiled(),
            "available AVX2 path was not compiled");
    require(!dsmvc::cpu_avx512_available() || dsmvc::cpu_avx512_compiled(),
            "available AVX-512 path was not compiled");
    require(!dsmvc::cpu_neon_available() || dsmvc::cpu_neon_compiled(),
            "available NEON path was not compiled");

    const dsmvc::CpuExecutor automatic(dsmvc::CpuPath::automatic);
    if (native_simd_available()) {
        const auto expected = native_simd_path();
        require(automatic.path() == expected,
                "automatic dispatch did not select native SIMD");
        const dsmvc::CpuExecutor explicit_native(expected);
        require(explicit_native.path() == expected,
                "explicit native SIMD dispatch selected the wrong path");
        require(std::string_view(explicit_native.name())
                    == (expected == dsmvc::CpuPath::avx512
                            ? "avx512-fma"
                            : expected == dsmvc::CpuPath::avx2
                                ? "avx2-fma" : "neon-fma"),
                "native SIMD executor reported the wrong name");
    } else {
        require(automatic.path() == dsmvc::CpuPath::scalar,
                "automatic dispatch did not fall back to scalar");
    }
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
    dsmvc::inverse_axis_f32(plan, input, output);
    for (std::size_t i = 0; i < input.size(); ++i) {
        require(std::abs(input[i] - output[i]) < 1.0e-5F,
                "identity bilinear solve drifted");
    }
}

[[nodiscard]] dsmvc::AxisPlan make_conditioned_lanczos2_plan() {
    dsmvc::AxisRequest request;
    request.source_size = 1080;
    request.destination_size = 980;
    request.active_length = 978.1;
    request.shift = 0.95;
    request.kernel.kind = dsmvc::KernelKind::lanczos;
    request.kernel.taps = 2;
    return dsmvc::build_axis_plan(request);
}

[[nodiscard]] dsmvc::AxisPlan make_conditioned_bilinear_plan() {
    dsmvc::AxisRequest request;
    request.source_size = 1920;
    request.destination_size = 1734;
    request.active_length = 1732.0888888888887;
    request.shift = 0.9555555555556339;
    request.kernel.kind = dsmvc::KernelKind::bilinear;
    return dsmvc::build_axis_plan(request);
}

void test_conditioned_bilinear_reference() {
    const auto conditioned = make_conditioned_bilinear_plan();
    require(conditioned.requires_float64(),
            "conditioned bilinear geometry did not select Float64");
    require(conditioned.normal_rcond > 2.0e-6
                && conditioned.normal_rcond < 2.5e-6,
            "conditioned bilinear reciprocal condition estimate drifted");

    std::vector<float> input(
        static_cast<std::size_t>(conditioned.source_size));
    std::uint32_t state = 0x9743b111U;
    for (float &value : input) {
        state ^= state << 13U;
        state ^= state >> 17U;
        state ^= state << 5U;
        value = static_cast<float>(state & 0xffffU) / 65535.0F;
    }
    std::vector<float> robust(
        static_cast<std::size_t>(conditioned.destination_size));
    std::vector<float> legacy(robust.size());
    dsmvc::inverse_axis_f32(conditioned, input, robust);
    auto float32_only = conditioned;
    float32_only.normal_inf_norm = 0.0;
    float32_only.transpose_weights_f64.clear();
    float32_only.normal_bands_f64.clear();
    float32_only.ldlt_bands_f64.clear();
    float32_only.inverse_diagonal_f64.clear();
    dsmvc::inverse_axis_f32(float32_only, input, legacy);

    // Independent 60-digit solve of the retained bilinear operator.
    constexpr double first_reference = 64.704580093370353;
    constexpr double last_reference = 6.4342414526327527;
    const double robust_error = std::max(
        std::abs(static_cast<double>(robust.front()) - first_reference),
        std::abs(static_cast<double>(robust.back()) - last_reference));
    const double legacy_error = std::max(
        std::abs(static_cast<double>(legacy.front()) - first_reference),
        std::abs(static_cast<double>(legacy.back()) - last_reference));
    require(robust_error < 3.0e-6,
            "Float64 bilinear solve differs from the high-precision reference");
    require(legacy_error > robust_error * 500.0,
            "conditioned bilinear solve did not improve on Float32 factors");

    dsmvc::AxisRequest identity_request;
    identity_request.source_size = 4;
    identity_request.destination_size = 4;
    identity_request.active_length = 4.0;
    identity_request.kernel.kind = dsmvc::KernelKind::bilinear;
    const auto identity = dsmvc::build_axis_plan(identity_request);
    const auto input_stride = conditioned.source_size;
    const auto output_stride = conditioned.destination_size;
    std::vector<float> input_2d(
        static_cast<std::size_t>(identity.source_size)
            * static_cast<std::size_t>(input_stride));
    std::vector<float> reference(
        static_cast<std::size_t>(identity.destination_size)
            * static_cast<std::size_t>(output_stride));
    std::vector<float> output(reference.size(), output_fill);
    fill_deterministic(input_2d.data(), input_2d.size(), 0x97432d64U);
    for (std::int32_t row = 0; row < identity.source_size; ++row) {
        dsmvc::inverse_axis_f32(
            conditioned,
            input_2d.data() + static_cast<std::ptrdiff_t>(row) * input_stride,
            1,
            reference.data() + static_cast<std::ptrdiff_t>(row) * output_stride,
            1);
    }
    const dsmvc::CpuExecutor executor(dsmvc::CpuPath::automatic);
    executor.inverse_2d(
        conditioned, identity, input_2d.data(), input_stride,
        output.data(), output_stride);
    const auto stats = compare_matrix(
        reference.data(), output.data(), identity.destination_size,
        conditioned.destination_size, output_stride);
    require(stats.non_finite == 0U && stats.maximum < 3.0e-6F,
            "horizontal Float64 2D path differs from the axis reference");
    require(executor.packing_stats().pack_executions == 0U,
            "horizontal Float64 2D path packed a Float32 plan");
}

void test_condition_aware_float64_axis() {
    dsmvc::AxisRequest exact_request;
    exact_request.source_size = 1080;
    exact_request.destination_size = 978;
    exact_request.active_length = 978.0;
    exact_request.shift = 0.0;
    exact_request.kernel.kind = dsmvc::KernelKind::lanczos;
    exact_request.kernel.taps = 2;
    const auto exact = dsmvc::build_axis_plan(exact_request);
    require(!exact.requires_float64() && exact.normal_rcond >= 1.0e-4,
            "exact 978.0 geometry did not retain the Float32 fast path");

    auto conditioned = make_conditioned_lanczos2_plan();
    require(conditioned.requires_float64(),
            "conditioned 978.1 geometry did not select Float64");
    require(conditioned.normal_rcond > 3.5e-6
                && conditioned.normal_rcond < 4.5e-6,
            "conditioned 978.1 reciprocal condition estimate drifted");
    require(conditioned.transpose_weights_f64.size()
                == conditioned.transpose_weights.size()
                && conditioned.normal_bands_f64.size()
                    == (static_cast<std::size_t>(conditioned.half_bandwidth) + 1U)
                        * static_cast<std::size_t>(conditioned.destination_size)
                && conditioned.ldlt_bands_f64.size()
                    == (static_cast<std::size_t>(conditioned.half_bandwidth) + 1U)
                        * static_cast<std::size_t>(conditioned.destination_size)
                && conditioned.inverse_diagonal_f64.size()
                    == static_cast<std::size_t>(conditioned.destination_size)
                && std::isfinite(conditioned.normal_inf_norm)
                && conditioned.normal_inf_norm > 0.0,
            "conditioned plan did not retain complete Float64 data");
    require(conditioned.storage_bytes() > exact.storage_bytes(),
            "Float64 plan storage was not included in cache accounting");

    std::vector<float> input(
        static_cast<std::size_t>(conditioned.source_size));
    std::uint32_t state = 0x9781a5c3U;
    for (float &value : input) {
        state ^= state << 13U;
        state ^= state >> 17U;
        state ^= state << 5U;
        value = static_cast<float>(state & 0xffffU) / 65535.0F;
    }
    std::vector<float> robust(
        static_cast<std::size_t>(conditioned.destination_size));
    std::vector<float> legacy(robust.size());
    dsmvc::inverse_axis_f32(conditioned, input, robust);
    auto float32_only = conditioned;
    float32_only.normal_inf_norm = 0.0;
    float32_only.transpose_weights_f64.clear();
    float32_only.normal_bands_f64.clear();
    float32_only.ldlt_bands_f64.clear();
    float32_only.inverse_diagonal_f64.clear();
    require(float32_only.valid() && !float32_only.requires_float64(),
            "Float32 comparison plan is invalid");
    dsmvc::inverse_axis_f32(float32_only, input, legacy);

    // Direct Float64 Householder QR references for the retained operator.
    constexpr double first_qr = 74.22802437585693;
    constexpr double last_qr = -119.91860242539072;
    const double robust_error = std::max(
        std::abs(static_cast<double>(robust.front()) - first_qr),
        std::abs(static_cast<double>(robust.back()) - last_qr));
    const double legacy_error = std::max(
        std::abs(static_cast<double>(legacy.front()) - first_qr),
        std::abs(static_cast<double>(legacy.back()) - last_qr));
    require(robust_error < 4.0e-6,
            "Float64 conditioned solve differs from the QR reference");
    require(legacy_error > robust_error * 20.0,
            "conditioned Float64 solve did not improve on Float32 factors");

    auto retained = std::make_shared<const dsmvc::AxisPlan>(conditioned);
    dsmvc::CpuExecutor prepared(dsmvc::CpuPath::automatic);
    prepared.prepare(retained);
    require(prepared.packing_stats().pack_executions == 0U,
            "Float64 plan was packed for a Float32 SIMD path");
    prepared.seal();
    std::vector<float> one_row(robust.size());
    prepared.inverse_rows(
        *retained, input.data(), conditioned.source_size,
        one_row.data(), conditioned.destination_size, 1);
    require(prepared.packing_stats().pack_executions == 0U,
            "Float64 execution lazily packed a Float32 SIMD plan");
    require(one_row == robust,
            "prepared Float64 row execution differs from the axis solve");

    constexpr std::int32_t batch_count = 269;
    const auto row_input_stride = conditioned.source_size + 3;
    const auto row_output_stride = conditioned.destination_size + 3;
    std::vector<float> row_input(
        static_cast<std::size_t>(batch_count)
            * static_cast<std::size_t>(row_input_stride));
    std::vector<float> row_reference(
        static_cast<std::size_t>(batch_count)
            * static_cast<std::size_t>(row_output_stride),
        output_fill);
    std::vector<float> row_candidate(row_reference.size(), output_fill);
    fill_deterministic(row_input.data(), row_input.size(), 0x9781f64aU);
    for (std::int32_t row = 0; row < batch_count; ++row) {
        dsmvc::inverse_axis_f32(
            conditioned,
            row_input.data() + static_cast<std::ptrdiff_t>(row)
                * row_input_stride,
            1,
            row_reference.data() + static_cast<std::ptrdiff_t>(row)
                * row_output_stride,
            1);
    }
    prepared.inverse_rows(
        conditioned, row_input.data(), row_input_stride,
        row_candidate.data(), row_output_stride, batch_count);
    auto stats = compare_matrix(
        row_reference.data(), row_candidate.data(), batch_count,
        conditioned.destination_size, row_output_stride);
    std::cout << "conditioned Float64 rows: max_error=" << stats.maximum
              << " mean_error=" << stats.mean << '\n';
    require(stats.non_finite == 0U && stats.maximum < 4.0e-6F,
            "batched Float64 rows differ from scalar Double");

    const auto column_stride = batch_count + 3;
    std::vector<float> column_input(
        static_cast<std::size_t>(conditioned.source_size)
            * static_cast<std::size_t>(column_stride));
    std::vector<float> column_reference(
        static_cast<std::size_t>(conditioned.destination_size)
            * static_cast<std::size_t>(column_stride),
        output_fill);
    std::vector<float> column_candidate(column_reference.size(), output_fill);
    fill_deterministic(
        column_input.data(), column_input.size(), 0x9781c01aU);
    for (std::int32_t column = 0; column < batch_count; ++column) {
        dsmvc::inverse_axis_f32(
            conditioned, column_input.data() + column, column_stride,
            column_reference.data() + column, column_stride);
    }
    prepared.inverse_columns(
        conditioned, column_input.data(), column_stride,
        column_candidate.data(), column_stride, batch_count);
    stats = compare_matrix(
        column_reference.data(), column_candidate.data(),
        conditioned.destination_size, batch_count, column_stride);
    std::cout << "conditioned Float64 columns: max_error=" << stats.maximum
              << " mean_error=" << stats.mean << '\n';
    require(stats.non_finite == 0U && stats.maximum < 4.0e-6F,
            "batched Float64 columns differ from scalar Double");
    require(prepared.packing_stats().pack_executions == 0U,
            "batched Float64 execution packed a Float32 plan");
}

void test_conditioned_float64_2d_intermediate() {
    dsmvc::AxisRequest horizontal_request;
    horizontal_request.source_size = 8;
    horizontal_request.destination_size = 8;
    horizontal_request.active_length = 8.0;
    horizontal_request.kernel.kind = dsmvc::KernelKind::bilinear;
    const auto horizontal = dsmvc::build_axis_plan(horizontal_request);
    const auto vertical = make_conditioned_lanczos2_plan();
    require(!horizontal.requires_float64() && vertical.requires_float64(),
            "2D precision-routing fixture is invalid");

    constexpr std::int32_t stride = 8;
    std::vector<float> input(
        static_cast<std::size_t>(vertical.source_size) * stride);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>((index * 73U + 19U) & 4095U)
            / 4095.0F;
    }
    std::vector<float> output(
        static_cast<std::size_t>(vertical.destination_size) * stride);
    std::vector<float> reference(output.size());
    const dsmvc::CpuExecutor executor(dsmvc::CpuPath::automatic);
    executor.inverse_2d(
        horizontal, vertical, input.data(), stride, output.data(), stride);

    std::vector<float> source_column(
        static_cast<std::size_t>(vertical.source_size));
    std::vector<float> destination_column(
        static_cast<std::size_t>(vertical.destination_size));
    for (std::int32_t column = 0; column < stride; ++column) {
        for (std::int32_t row = 0; row < vertical.source_size; ++row) {
            source_column[static_cast<std::size_t>(row)] =
                input[static_cast<std::ptrdiff_t>(row) * stride + column];
        }
        dsmvc::inverse_axis_f32(vertical, source_column, destination_column);
        for (std::int32_t row = 0; row < vertical.destination_size; ++row) {
            reference[static_cast<std::ptrdiff_t>(row) * stride + column] =
                destination_column[static_cast<std::size_t>(row)];
        }
    }
    const auto stats = compare_matrix(
        reference.data(), output.data(), vertical.destination_size,
        stride, stride);
    std::cout << "conditioned Float64 2D: max_error=" << stats.maximum
              << " mean_error=" << stats.mean << '\n';
    require(stats.non_finite == 0U && stats.maximum < 4.0e-6F,
            "conditioned 2D path differs from the scalar Double reference");
    require(executor.packing_stats().pack_executions == 0U,
            "conditioned 2D path packed a Float32 axis plan");

    const dsmvc::IntegerConversion conversion{
        0.0F, 1.0F / 1023.0F, 1023.0F, 0.0F, 1023U};
    const dsmvc::CpuExecutor scalar(dsmvc::CpuPath::scalar);

    const auto nontrivial_horizontal = make_plan(
        dsmvc::KernelKind::bicubic, 17, 13, 12.75, 0.125);
    require(!nontrivial_horizontal.requires_float64(),
            "safe horizontal Double fixture selected Float64");
    constexpr std::int32_t nontrivial_input_stride = 20;
    constexpr std::int32_t nontrivial_output_stride = 16;
    std::vector<float> nontrivial_input(
        static_cast<std::size_t>(vertical.source_size)
            * nontrivial_input_stride);
    std::vector<float> nontrivial_reference(
        static_cast<std::size_t>(vertical.destination_size)
            * nontrivial_output_stride,
        output_fill);
    std::vector<float> nontrivial_output(
        nontrivial_reference.size(), output_fill);
    fill_deterministic(
        nontrivial_input.data(), nontrivial_input.size(), 0xf64b3a5eU);
    scalar.inverse_2d(
        nontrivial_horizontal, vertical,
        nontrivial_input.data(), nontrivial_input_stride,
        nontrivial_reference.data(), nontrivial_output_stride);
    executor.inverse_2d(
        nontrivial_horizontal, vertical,
        nontrivial_input.data(), nontrivial_input_stride,
        nontrivial_output.data(), nontrivial_output_stride);
    const auto nontrivial_stats = compare_matrix(
        nontrivial_reference.data(), nontrivial_output.data(),
        vertical.destination_size, nontrivial_horizontal.destination_size,
        nontrivial_output_stride);
    std::cout << "conditioned Float64 2D nontrivial horizontal: max_error="
              << nontrivial_stats.maximum
              << " mean_error=" << nontrivial_stats.mean << '\n';
    require(nontrivial_stats.non_finite == 0U
                && nontrivial_stats.maximum < 4.0e-6F,
            "safe horizontal axis lost Double intermediate precision");

    std::vector<std::uint16_t> integer_input(input.size());
    for (std::size_t index = 0; index < integer_input.size(); ++index) {
        integer_input[index] = static_cast<std::uint16_t>(
            (index * 73U + 19U) & 1023U);
    }
    std::vector<std::uint16_t> integer_reference(output.size());
    std::vector<std::uint16_t> integer_output(output.size());
    std::vector<std::uint16_t> integer_streamed(output.size());
    scalar.inverse_2d_u16(
        horizontal, vertical, integer_input.data(), stride,
        integer_reference.data(), stride, conversion);
    executor.inverse_2d_u16(
        horizontal, vertical, integer_input.data(), stride,
        integer_output.data(), stride, conversion);
    executor.inverse_2d_u16_streamed(
        horizontal, vertical, integer_input.data(), stride,
        integer_streamed.data(), stride, conversion);
    require(integer_output == integer_reference,
            "conditioned U16 2D path differs from scalar Double");
    require(integer_streamed == integer_reference,
            "conditioned streamed U16 path differs from scalar Double");
}

void test_padding_index_patterns() {
    constexpr std::int32_t size = 4;
    constexpr std::int64_t first = -8;
    const auto check = [](dsmvc::BorderMode mode,
                          const std::array<std::int32_t, 20> &expected,
                          std::string_view label) {
        for (std::size_t offset = 0; offset < expected.size(); ++offset) {
            const auto index = first + static_cast<std::int64_t>(offset);
            const auto actual = dsmvc::detail::padding_index(index, size, mode);
            require(actual == expected[offset],
                    std::string(label) + " padding pattern differs at index "
                        + std::to_string(index));
        }
    };

    check(dsmvc::BorderMode::zero,
          {-1, -1, -1, -1, -1, -1, -1, -1, 0, 1,
           2, 3, -1, -1, -1, -1, -1, -1, -1, -1},
          "zero");
    check(dsmvc::BorderMode::repeat,
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           2, 3, 3, 3, 3, 3, 3, 3, 3, 3},
          "repeat");
    check(dsmvc::BorderMode::reflect101,
          {2, 1, 0, 1, 2, 3, 2, 1, 0, 1,
           2, 3, 2, 1, 0, 1, 2, 3, 2, 1},
          "reflect101");
    check(dsmvc::BorderMode::symmetric,
          {0, 1, 2, 3, 3, 2, 1, 0, 0, 1,
           2, 3, 3, 2, 1, 0, 0, 1, 2, 3},
          "symmetric");
    check(dsmvc::BorderMode::mirror,
          {-1, -1, -1, -1, 3, 2, 1, 0, 0, 1,
           2, 3, 3, 2, 1, 0, -1, -1, -1, -1},
          "legacy mirror");

    for (const auto mode : {
             dsmvc::BorderMode::repeat,
             dsmvc::BorderMode::reflect101,
             dsmvc::BorderMode::symmetric}) {
        require(dsmvc::detail::padding_index(-1000000000000LL, 1, mode) == 0
                    && dsmvc::detail::padding_index(1000000000000LL, 1, mode) == 0,
                "single-pixel periodic padding did not remain at index zero");
    }

    bool rejected = false;
    try {
        (void)dsmvc::detail::padding_index(
            0, size, static_cast<dsmvc::BorderMode>(255));
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "invalid padding mode was accepted");
}

void test_f64_mode_selection_and_cache() {
    dsmvc::clear_planner_caches();
    dsmvc::AxisRequest request;
    request.source_size = 1080;
    request.destination_size = 980;
    request.active_length = 978.1;
    request.shift = 0.95;
    request.kernel.kind = dsmvc::KernelKind::lanczos;
    request.kernel.taps = 2;

    const auto automatic = dsmvc::get_or_build_axis_plan(request);
    request.f64_mode = dsmvc::F64Mode::float32_only;
    const auto float32 = dsmvc::get_or_build_axis_plan(request);
    request.f64_mode = dsmvc::F64Mode::float64_only;
    const auto float64 = dsmvc::get_or_build_axis_plan(request);

    require(automatic->requires_float64(),
            "automatic mode did not promote a conditioned plan");
    require(!float32->requires_float64(),
            "Float32-only mode retained Float64 factors");
    require(float64->requires_float64(),
            "Float64-only mode did not retain Float64 factors");
    require(automatic != float32 && automatic != float64 && float32 != float64,
            "precision modes shared a plan-cache entry");
    require(automatic->normal_rcond == float32->normal_rcond
                && automatic->normal_rcond == float64->normal_rcond,
            "precision mode changed the condition estimate");

    const auto stats = dsmvc::planner_cache_stats();
    require(stats.plan_builds == 3 && stats.plan_entries == 3,
            "precision modes were not cached independently");
    require(stats.geometry_builds == 1 && stats.geometry_hits >= 2,
            "precision modes did not reuse sampling geometry");

    request.source_size = 8;
    request.destination_size = 8;
    request.active_length = 8.0;
    request.shift = 0.0;
    request.kernel.kind = dsmvc::KernelKind::bilinear;
    request.f64_mode = dsmvc::F64Mode::float64_only;
    require(dsmvc::build_axis_plan(request).requires_float64(),
            "Float64-only mode did not promote a well-conditioned plan");

    request.f64_mode = static_cast<dsmvc::F64Mode>(255);
    bool rejected = false;
    try {
        (void)dsmvc::build_axis_plan(request);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "invalid f64 mode was accepted");
}

void test_blur_precision_routing() {
    dsmvc::AxisRequest request;
    request.source_size = 1080;
    request.destination_size = 980;
    request.active_length = 978.1;
    request.shift = 0.95;
    request.kernel.kind = dsmvc::KernelKind::lanczos;
    request.kernel.taps = 2;
    const auto unblurred = dsmvc::build_axis_plan(request);

    request.kernel.blur = 1.25;
    const auto automatic = dsmvc::build_axis_plan(request);
    request.f64_mode = dsmvc::F64Mode::float32_only;
    const auto float32 = dsmvc::build_axis_plan(request);
    request.f64_mode = dsmvc::F64Mode::float64_only;
    const auto float64 = dsmvc::build_axis_plan(request);
    require(automatic.normal_rcond != unblurred.normal_rcond,
            "blur did not affect the normal-matrix condition estimate");
    require(automatic.normal_rcond == float32.normal_rcond
                && automatic.normal_rcond == float64.normal_rcond,
            "f64mode changed the blur plan condition estimate");
    require(automatic.requires_float64()
                == (automatic.normal_rcond < 1.0e-4),
            "automatic blur plan did not follow its rcond");
    require(!float32.requires_float64() && float64.requires_float64(),
            "forced precision modes changed semantics for blur plans");
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

void test_blur_support_and_weights() {
    struct SupportCase {
        dsmvc::KernelKind kind;
        std::int32_t taps;
        std::int32_t base_support;
    };
    constexpr std::array cases{
        SupportCase{dsmvc::KernelKind::bilinear, 0, 1},
        SupportCase{dsmvc::KernelKind::bicubic, 0, 2},
        SupportCase{dsmvc::KernelKind::lanczos, 2, 2},
        SupportCase{dsmvc::KernelKind::lanczos, 3, 3},
        SupportCase{dsmvc::KernelKind::spline16, 0, 2},
        SupportCase{dsmvc::KernelKind::spline36, 0, 3},
        SupportCase{dsmvc::KernelKind::spline64, 0, 4},
    };
    constexpr std::array blurs{0.75, 1.0, 1.01, 1.25, 1.5};
    for (const auto &test_case : cases) {
        for (const double blur : blurs) {
            dsmvc::AxisRequest request;
            request.source_size = 97;
            request.destination_size = 67;
            request.active_length = 66.75;
            request.shift = 0.125;
            request.kernel.kind = test_case.kind;
            request.kernel.taps = test_case.taps;
            request.kernel.blur = blur;
            request.f64_mode = dsmvc::F64Mode::float32_only;
            const auto plan = dsmvc::build_axis_plan(request);
            const auto expected_support = static_cast<std::int32_t>(
                std::ceil(static_cast<double>(test_case.base_support) * blur));
            require(plan.support == expected_support,
                    "blur effective support differs from ceil(base * blur)");
            require(plan.half_bandwidth
                        == std::min(2 * expected_support - 1,
                                    request.destination_size - 1),
                    "blur half-bandwidth is inconsistent with support");
        }
    }

    dsmvc::AxisRequest default_request;
    default_request.source_size = 47;
    default_request.destination_size = 39;
    default_request.active_length = 39.0;
    default_request.shift = -0.3125;
    default_request.kernel.kind = dsmvc::KernelKind::bicubic;
    default_request.kernel.b = 0.0;
    default_request.kernel.c = 0.5;
    default_request.f64_mode = dsmvc::F64Mode::float32_only;
    const auto default_plan = dsmvc::build_axis_plan(default_request);
    auto explicit_unity_request = default_request;
    explicit_unity_request.kernel.blur = 1.0;
    const auto explicit_unity_plan =
        dsmvc::build_axis_plan(explicit_unity_request);
    require(default_plan.transpose_offsets == explicit_unity_plan.transpose_offsets
                && default_plan.transpose_indices
                    == explicit_unity_plan.transpose_indices
                && default_plan.transpose_weights
                    == explicit_unity_plan.transpose_weights
                && default_plan.lower_ld == explicit_unity_plan.lower_ld
                && default_plan.upper_l == explicit_unity_plan.upper_l
                && default_plan.inverse_diagonal
                    == explicit_unity_plan.inverse_diagonal
                && default_plan.normal_rcond == explicit_unity_plan.normal_rcond,
            "default blur and explicit blur=1 produced different plans");

    dsmvc::AxisRequest lanczos_request;
    lanczos_request.source_size = 1;
    lanczos_request.destination_size = 8;
    lanczos_request.active_length = 8.0;
    lanczos_request.kernel.kind = dsmvc::KernelKind::lanczos;
    lanczos_request.kernel.taps = 2;
    lanczos_request.kernel.blur = 1.5;
    lanczos_request.border = dsmvc::BorderMode::zero;
    lanczos_request.f64_mode = dsmvc::F64Mode::float32_only;
    const auto lanczos_plan = dsmvc::build_axis_plan(lanczos_request);
    require(lanczos_plan.support == 3 && lanczos_plan.half_bandwidth == 5,
            "blurred Lanczos2 did not expand to support 3");
    const auto sinc = [](double value) {
        if (value == 0.0) return 1.0;
        const double scaled = std::numbers::pi * value;
        return std::sin(scaled) / scaled;
    };
    std::array<double, 6> expected{};
    double total = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const double distance = std::abs(
            (-2.5 + static_cast<double>(index)) / 1.5);
        expected[index] = std::abs(distance) < 2.0
            ? sinc(distance) * sinc(distance / 2.0) : 0.0;
        total += expected[index];
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const std::size_t destination = index + 1U;
        const auto begin = lanczos_plan.transpose_offsets[destination];
        const auto end = lanczos_plan.transpose_offsets[destination + 1U];
        if (expected[index] == 0.0) {
            require(begin == end,
                    "blurred Lanczos2 retained a zero window weight");
            continue;
        }
        require(end == begin + 1U
                    && lanczos_plan.transpose_indices[begin] == 0
                    && lanczos_plan.transpose_weights[begin]
                        == static_cast<float>(expected[index] / total),
                "blurred Lanczos2 window did not use the original taps=2");
    }

    dsmvc::AxisRequest bilinear_request;
    bilinear_request.source_size = 48;
    bilinear_request.destination_size = 32;
    bilinear_request.active_length = 31.75;
    bilinear_request.shift = 0.125;
    bilinear_request.kernel.kind = dsmvc::KernelKind::bilinear;
    bilinear_request.kernel.blur = 1.5;
    bilinear_request.f64_mode = dsmvc::F64Mode::float32_only;
    const auto bilinear_plan = dsmvc::build_axis_plan(bilinear_request);
    auto custom_request = bilinear_request;
    custom_request.kernel.kind = dsmvc::KernelKind::custom;
    custom_request.kernel.taps = 1;
    const auto custom_plan = dsmvc::build_axis_plan(
        custom_request,
        [](double distance) { return std::max(1.0 - distance, 0.0); });
    require(custom_plan.support == bilinear_plan.support
                && custom_plan.transpose_offsets == bilinear_plan.transpose_offsets
                && custom_plan.transpose_indices == bilinear_plan.transpose_indices
                && custom_plan.transpose_weights == bilinear_plan.transpose_weights,
            "custom kernel did not receive distance / blur");

    for (const double blur : {
             0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::infinity()}) {
        auto invalid = default_request;
        invalid.kernel.blur = blur;
        bool rejected = false;
        try {
            (void)dsmvc::build_axis_plan(invalid);
        } catch (const std::invalid_argument &error) {
            rejected = std::string_view{error.what()}
                == "blur must be finite and greater than zero";
        }
        require(rejected, "invalid blur did not produce the public error");
    }
    auto excessive = default_request;
    excessive.kernel.blur = std::numeric_limits<double>::max();
    bool excessive_rejected = false;
    try {
        (void)dsmvc::build_axis_plan(excessive);
    } catch (const std::length_error &error) {
        excessive_rejected = std::string_view{error.what()}
            == "effective filter support is too large";
    }
    require(excessive_rejected,
            "overflowing effective support did not produce a checked-size error");
}

void test_blur_cache_isolation() {
    dsmvc::clear_planner_caches();
    dsmvc::AxisRequest request;
    request.source_size = 96;
    request.destination_size = 64;
    request.active_length = 63.75;
    request.shift = 0.125;
    request.kernel.kind = dsmvc::KernelKind::bicubic;
    request.kernel.blur = 1.01;

    const auto first = dsmvc::get_or_build_axis_plan(request);
    request.kernel.blur = 1.25;
    const auto second = dsmvc::get_or_build_axis_plan(request);
    require(first != second && first->support == second->support,
            "same-support blur values shared a plan-cache entry");
    auto stats = dsmvc::planner_cache_stats();
    require(stats.plan_builds == 2 && stats.plan_entries == 2,
            "different blur values were not cached independently");
    require(stats.geometry_builds == 1 && stats.geometry_hits == 1,
            "same-support blur values did not reuse sampling geometry");

    request.kernel.blur = 1.5;
    std::vector<std::shared_ptr<const dsmvc::AxisPlan>> plans(8);
    std::vector<JoiningThread> workers;
    for (std::size_t index = 0; index < plans.size(); ++index) {
        workers.emplace_back([&, index] {
            plans[index] = dsmvc::get_or_build_axis_plan(request);
        });
    }
    workers.clear();
    for (const auto &plan : plans) {
        require(plan == plans.front(),
                "blur plan single-flight returned inconsistent entries");
    }
    stats = dsmvc::planner_cache_stats();
    require(stats.plan_builds == 3
                && stats.plan_hits == plans.size() - 1U
                && stats.geometry_builds == 1
                && stats.geometry_hits == 2,
            "blur plan single-flight/cache accounting is inconsistent");
}

void test_inverse_only_cache() {
    dsmvc::clear_planner_caches();
    dsmvc::AxisRequest request;
    request.source_size = 96;
    request.destination_size = 64;
    request.active_length = 63.75;
    request.shift = 0.125;
    request.kernel.kind = dsmvc::KernelKind::bicubic;
    request.kernel.b = 0.0;
    request.kernel.c = 0.5;

    std::vector<std::shared_ptr<const dsmvc::AxisPlan>> plans(8);
    std::vector<JoiningThread> workers;
    for (std::size_t index = 0; index < plans.size(); ++index) {
        workers.emplace_back([&, index] {
            plans[index] = dsmvc::get_or_build_axis_plan(request);
        });
    }
    workers.clear();
    for (const auto &plan : plans) {
        require(plan == plans.front(), "single-flight plan cache did not share a plan");
        require(plan->valid(), "cached inverse-only plan is invalid");
    }
    auto stats = dsmvc::planner_cache_stats();
    require(stats.plan_builds == 1, "single-flight cache duplicated a plan build");
    require(stats.plan_hits == plans.size() - 1U, "plan hit accounting is incorrect");
    require(stats.geometry_builds == 1, "geometry was built more than once");

    request.kernel.b = 0.7;
    request.kernel.c = 0.6;
    const auto second = dsmvc::get_or_build_axis_plan(request);
    require(second != plans.front(), "different bicubic parameters shared a plan");
    stats = dsmvc::planner_cache_stats();
    require(stats.geometry_hits >= 1, "bicubic family did not reuse geometry");
}

void test_large_support_compatibility() {
    dsmvc::AxisRequest request;
    request.source_size = 32;
    request.destination_size = 16;
    request.active_length = 16.0;
    request.kernel.kind = dsmvc::KernelKind::lanczos;
    request.kernel.taps = 16;
    require(dsmvc::build_axis_plan(request).valid(),
            "Lanczos taps=16 was rejected");

    request.kernel.kind = dsmvc::KernelKind::custom;
    request.kernel.taps = 65;
    require(dsmvc::build_axis_plan(
                request, [](double x) { return std::max(1.0 - x, 0.0); }).valid(),
            "custom taps=65 was rejected");
}

void test_axis_plan_validation() {
    const auto valid = make_plan(dsmvc::KernelKind::spline64, 96, 64, 63.75, 0.125);
    require(valid.valid(), "validation fixture is invalid");

    const auto rejects = [](dsmvc::AxisPlan plan, std::string_view label) {
        require(!plan.valid(), std::string("malformed plan accepted: ")
            + std::string(label));
    };

    auto malformed = valid;
    malformed.transpose_offsets.front() = 1U;
    rejects(std::move(malformed), "nonzero first offset");

    malformed = valid;
    malformed.transpose_offsets[1] = malformed.transpose_offsets.back() + 1U;
    rejects(std::move(malformed), "out-of-range offset");

    malformed = valid;
    malformed.transpose_offsets[1] = malformed.transpose_offsets.back();
    malformed.transpose_offsets[2] = 0U;
    rejects(std::move(malformed), "nonmonotonic offsets");

    malformed = valid;
    --malformed.transpose_offsets.back();
    rejects(std::move(malformed), "incorrect final offset");

    malformed = valid;
    malformed.transpose_indices.front() = -1;
    rejects(std::move(malformed), "negative source index");

    malformed = valid;
    malformed.transpose_indices.front() = valid.source_size;
    rejects(std::move(malformed), "out-of-range source index");

    malformed = valid;
    bool swapped = false;
    for (std::size_t row = 0; row + 1U < malformed.transpose_offsets.size(); ++row) {
        const auto begin = malformed.transpose_offsets[row];
        const auto end = malformed.transpose_offsets[row + 1U];
        if (end - begin >= 2U) {
            std::swap(malformed.transpose_indices[begin],
                      malformed.transpose_indices[begin + 1U]);
            swapped = true;
            break;
        }
    }
    require(swapped, "validation fixture has no multi-entry CSR row");
    rejects(std::move(malformed), "unordered source indices");

    malformed = valid;
    malformed.transpose_weights.front() =
        std::numeric_limits<float>::quiet_NaN();
    rejects(std::move(malformed), "non-finite transpose weight");

    malformed = valid;
    malformed.lower_ld.front() = std::numeric_limits<float>::infinity();
    rejects(std::move(malformed), "non-finite lower factor");

    malformed = valid;
    malformed.upper_l.front() = -std::numeric_limits<float>::infinity();
    rejects(std::move(malformed), "non-finite upper factor");

    malformed = valid;
    malformed.inverse_diagonal.front() =
        std::numeric_limits<float>::quiet_NaN();
    rejects(std::move(malformed), "non-finite inverse diagonal");

    malformed = make_conditioned_lanczos2_plan();
    malformed.normal_rcond = std::numeric_limits<double>::quiet_NaN();
    rejects(std::move(malformed), "non-finite reciprocal condition estimate");

    malformed = make_conditioned_lanczos2_plan();
    malformed.transpose_weights_f64.front() =
        std::numeric_limits<double>::quiet_NaN();
    rejects(std::move(malformed), "non-finite Float64 transpose weight");

    malformed = make_conditioned_lanczos2_plan();
    malformed.normal_bands_f64.pop_back();
    rejects(std::move(malformed), "truncated Float64 normal bands");

    malformed = make_conditioned_lanczos2_plan();
    malformed.normal_bands_f64.front() =
        std::numeric_limits<double>::quiet_NaN();
    rejects(std::move(malformed), "non-finite Float64 normal band");

    malformed = make_conditioned_lanczos2_plan();
    malformed.normal_inf_norm = std::numeric_limits<double>::infinity();
    rejects(std::move(malformed), "non-finite normal infinity norm");

    malformed = make_conditioned_lanczos2_plan();
    malformed.normal_inf_norm = 0.0;
    rejects(std::move(malformed), "zero normal infinity norm");

    malformed = make_conditioned_lanczos2_plan();
    malformed.ldlt_bands_f64.pop_back();
    rejects(std::move(malformed), "truncated Float64 factor bands");

    malformed = make_conditioned_lanczos2_plan();
    malformed.inverse_diagonal_f64.pop_back();
    rejects(std::move(malformed), "truncated Float64 inverse diagonal");

    malformed = valid;
    malformed.normal_inf_norm = 1.0;
    rejects(std::move(malformed), "normal infinity norm on Float32 plan");

    malformed = valid;
    malformed.transpose_offsets.front() = 1U;
    std::vector<float> input(static_cast<std::size_t>(valid.source_size), 0.0F);
    std::vector<float> output(static_cast<std::size_t>(valid.destination_size), 0.0F);
    bool threw = false;
    try {
        dsmvc::inverse_axis_f32(
            malformed, input.data(), 1, output.data(), 1);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    require(threw, "inverse executor accepted a malformed plan");
}

void test_b5_b7_executor_agreement() {
    auto horizontal_b5 = std::make_shared<const dsmvc::AxisPlan>(make_plan(
        dsmvc::KernelKind::lanczos, 1920, 1692,
        1691.5555555555557, 0.2222222222221717));
    auto horizontal_b7 = std::make_shared<const dsmvc::AxisPlan>(make_plan(
        dsmvc::KernelKind::spline64, 1920, 1692,
        1691.5555555555557, 0.2222222222221717));
    auto vertical_b5 = std::make_shared<const dsmvc::AxisPlan>(make_plan(
        dsmvc::KernelKind::lanczos, 1080, 952, 951.5, 0.25));
    auto vertical_b7 = std::make_shared<const dsmvc::AxisPlan>(make_plan(
        dsmvc::KernelKind::spline64, 1080, 952, 951.5, 0.25));
    auto horizontal_b9 = std::make_shared<const dsmvc::AxisPlan>(make_plan(
        dsmvc::KernelKind::spline64, 1920, 1692,
        1691.5555555555557, 0.2222222222221717, 1.25));
    auto vertical_b9 = std::make_shared<const dsmvc::AxisPlan>(make_plan(
        dsmvc::KernelKind::spline64, 1080, 952, 951.5, 0.25, 1.25));
    require(horizontal_b5->half_bandwidth == 5,
            "Lanczos3 did not produce half-bandwidth 5");
    require(vertical_b5->half_bandwidth == 5,
            "vertical Lanczos3 did not produce half-bandwidth 5");
    require(horizontal_b7->half_bandwidth == 7,
            "Spline64 did not produce half-bandwidth 7");
    require(vertical_b7->half_bandwidth == 7,
            "vertical Spline64 did not produce half-bandwidth 7");
    require(horizontal_b9->half_bandwidth == 9
                && vertical_b9->half_bandwidth == 9,
            "Spline64 blur=1.25 did not produce half-bandwidth 9");

    dsmvc::CpuExecutor optimized(dsmvc::CpuPath::automatic);
    optimized.prepare(horizontal_b5);
    optimized.prepare(horizontal_b7);
    optimized.prepare(vertical_b5);
    optimized.prepare(vertical_b7);
    optimized.prepare(horizontal_b9);
    optimized.prepare(vertical_b9);
    optimized.seal();

    for (const auto rows : {8, 9, 16, 17}) {
        compare_rows(optimized, *horizontal_b5, rows,
                     "Lanczos3 b5 horizontal rows=" + std::to_string(rows));
        compare_rows(optimized, *horizontal_b7, rows,
                     "Spline64 b7 horizontal rows=" + std::to_string(rows));
        compare_rows(optimized, *horizontal_b9, rows,
                     "Spline64 blur b9 horizontal rows=" + std::to_string(rows));
    }

    if (dsmvc::cpu_neon_available()) {
        const dsmvc::CpuExecutor neon(dsmvc::CpuPath::neon);
        for (const auto rows : {4, 5}) {
            compare_rows(
                neon, *horizontal_b5, rows,
                "NEON Lanczos3 b5 horizontal rows=" + std::to_string(rows));
            compare_rows(
                neon, *horizontal_b7, rows,
                "NEON Spline64 b7 horizontal rows=" + std::to_string(rows));
            compare_rows(
                neon, *horizontal_b9, rows,
                "NEON Spline64 blur b9 horizontal rows=" + std::to_string(rows));
        }
    }

    constexpr std::int32_t stride = 1696;
    compare_columns(
        optimized, *vertical_b5, 1692, stride,
        "Lanczos3 b5 vertical padded vectors");
    compare_columns(
        optimized, *vertical_b7, 1692, stride,
        "Spline64 b7 vertical padded vectors");
    compare_columns(
        optimized, *vertical_b5, 1693, stride,
        "Lanczos3 b5 vertical non-vector tail");
    compare_columns(
        optimized, *vertical_b7, 1693, stride,
        "Spline64 b7 vertical non-vector tail");
    compare_columns(
        optimized, *vertical_b9, 1692, stride,
        "Spline64 blur b9 vertical padded vectors");
    compare_columns(
        optimized, *vertical_b9, 1693, stride,
        "Spline64 blur b9 vertical non-vector tail");

    std::vector<std::exception_ptr> errors(8U);
    std::vector<JoiningThread> callers;
    callers.reserve(errors.size());
    for (std::size_t index = 0; index < errors.size(); ++index) {
        callers.emplace_back([&, index] {
            try {
                const auto &plan = (index & 1U) == 0U
                    ? *horizontal_b5 : *horizontal_b7;
                const auto rows = static_cast<std::int32_t>(
                    std::vector<int>{8, 9, 16, 17}[index % 4U]);
                (void)compare_rows(
                    optimized, plan, rows, "concurrent prepared executor", false);
            } catch (...) {
                errors[index] = std::current_exception();
            }
        });
    }
    callers.clear();
    for (const auto &error : errors) {
        if (error) std::rethrow_exception(error);
    }
}

void test_b3_neon_row_pair_agreement() {
    if (!dsmvc::cpu_neon_available()) return;

    const auto horizontal_b3 = make_plan(
        dsmvc::KernelKind::bicubic, 1920, 1692,
        1691.5555555555557, 0.2222222222221717);
    require(horizontal_b3.half_bandwidth == 3,
            "Bicubic did not produce half-bandwidth 3");
    const dsmvc::CpuExecutor neon(dsmvc::CpuPath::neon);
    for (const auto rows : {8, 9, 12, 13, 16, 17}) {
        compare_rows(
            neon, horizontal_b3, rows,
            "NEON Bicubic b3 paired rows=" + std::to_string(rows));
    }
}

void test_streamed_2d_executor_agreement() {
    const struct {
        dsmvc::KernelKind kind;
        const char *name;
    } kernels[] = {
        {dsmvc::KernelKind::bilinear, "b1"},
        {dsmvc::KernelKind::bicubic, "b3"},
        {dsmvc::KernelKind::lanczos, "b5"},
        {dsmvc::KernelKind::spline64, "b7"},
    };
    for (const auto &kernel : kernels) {
        for (const auto path : {dsmvc::CpuPath::scalar,
                                dsmvc::CpuPath::automatic}) {
            const std::string prefix = std::string("streamed 2D ")
                + kernel.name
                + (path == dsmvc::CpuPath::scalar ? " scalar" : " automatic");
            (void)compare_2d(
                path, kernel.kind, 97, 67, 73, 51, true,
                prefix + " padded vectors and odd rows");
            (void)compare_2d(
                path, kernel.kind, 97, 67, 73, 51, false,
                prefix + " scalar column tail");
        }
    }

    if (dsmvc::cpu_avx2_available()) {
        (void)compare_2d(
            dsmvc::CpuPath::avx2, dsmvc::KernelKind::spline64,
            800, 640, 600, 480, true,
            "streamed 2D b7 parallel destination bands");
    } else if (dsmvc::cpu_neon_available()) {
        (void)compare_2d(
            dsmvc::CpuPath::neon, dsmvc::KernelKind::spline64,
            800, 640, 600, 480, true,
            "NEON streamed 2D b7 parallel destination bands");
    }
}

void test_integer_2d_executor_agreement() {
    const dsmvc::IntegerConversion limited_luma_u8{
        16.0F, 1.0F / 219.0F, 219.0F, 16.0F, 255U};
    const dsmvc::IntegerConversion limited_chroma_u10{
        512.0F, 1.0F / 896.0F, 896.0F, 512.0F, 1023U};
    const dsmvc::IntegerConversion full_luma_u16{
        0.0F, 1.0F / 65535.0F, 65535.0F, 0.0F, 65535U};
    const struct {
        dsmvc::KernelKind kind;
        const char *name;
    } kernels[] = {
        {dsmvc::KernelKind::bilinear, "b1"},
        {dsmvc::KernelKind::bicubic, "b3"},
        {dsmvc::KernelKind::lanczos, "b5"},
        {dsmvc::KernelKind::spline64, "b7"},
    };
    for (const auto &kernel : kernels) {
        compare_integer_2d<std::uint8_t>(
            dsmvc::CpuPath::automatic, kernel.kind, limited_luma_u8,
            std::string("integer u8 limited luma ") + kernel.name);
    }
    compare_integer_2d<std::uint16_t>(
        dsmvc::CpuPath::automatic, dsmvc::KernelKind::spline64,
        limited_chroma_u10, "integer u10 limited chroma b7");
    compare_integer_2d<std::uint16_t>(
        dsmvc::CpuPath::automatic, dsmvc::KernelKind::bicubic,
        full_luma_u16, "integer u16 full luma b3");
    compare_integer_2d<std::uint8_t>(
        dsmvc::CpuPath::scalar, dsmvc::KernelKind::lanczos,
        limited_luma_u8, "integer u8 scalar b5");
    if (dsmvc::cpu_neon_available()) {
        compare_integer_2d<std::uint8_t>(
            dsmvc::CpuPath::neon, dsmvc::KernelKind::bicubic,
            limited_luma_u8, "NEON integer u8 buffered and streamed b3");
        compare_integer_2d<std::uint16_t>(
            dsmvc::CpuPath::neon, dsmvc::KernelKind::spline64,
            full_luma_u16, "NEON integer u16 buffered and streamed b7");
        compare_integer_2d<std::uint8_t>(
            dsmvc::CpuPath::neon, dsmvc::KernelKind::bilinear,
            limited_luma_u8, "NEON integer streamed short-source fallback",
            3, 2);
    }
}

void test_executor_plan_ownership() {
    if (!native_simd_available()) return;

    const auto b7_value = make_plan(
        dsmvc::KernelKind::spline64, 96, 64, 63.75, 0.125);
    const auto b5_value = make_plan(
        dsmvc::KernelKind::lanczos, 96, 64, 63.75, 0.125);
    require(b7_value.half_bandwidth == 7 && b5_value.half_bandwidth == 5,
            "ownership fixtures have unexpected bandwidths");

    dsmvc::CpuExecutor borrowed(native_simd_path());
    auto reassigned = b7_value;
    (void)compare_rows(borrowed, reassigned, 8,
                       "borrowed plan before reassignment");
    reassigned = b5_value;
    (void)compare_rows(borrowed, reassigned, 8,
                       "borrowed plan after reassignment");

    alignas(dsmvc::AxisPlan) std::byte storage[sizeof(dsmvc::AxisPlan)];
    auto *placed = std::construct_at(
        reinterpret_cast<dsmvc::AxisPlan *>(storage), b7_value);
    (void)compare_rows(borrowed, *placed, 8, "borrowed stack address first use");
    std::destroy_at(placed);
    placed = std::construct_at(
        reinterpret_cast<dsmvc::AxisPlan *>(storage), b5_value);
    (void)compare_rows(borrowed, *placed, 8, "borrowed stack address reuse");
    std::destroy_at(placed);

    std::weak_ptr<const dsmvc::AxisPlan> weak;
    {
        dsmvc::CpuExecutor prepared(native_simd_path());
        auto owned = std::make_shared<const dsmvc::AxisPlan>(b7_value);
        weak = owned;
        prepared.prepare(owned);
        owned.reset();
        require(!weak.expired(), "prepared executor did not retain its axis plan");
        prepared.seal();
        const auto retained = weak.lock();
        require(retained != nullptr, "prepared axis expired after sealing");
        (void)compare_rows(prepared, *retained, 8, "retained prepared plan");

        bool rejected = false;
        try {
            prepared.prepare(retained);
        } catch (const std::logic_error &) {
            rejected = true;
        }
        require(rejected, "sealed executor accepted repeated preparation");

        (void)compare_rows(prepared, b5_value, 8,
                           "direct unprepared execution after seal");
    }
    require(weak.expired(), "packed-plan cache retained an expired prepared plan");
}

void test_concurrent_prepare_and_seal() {
    if (!native_simd_available()) return;

    dsmvc::CpuExecutor executor(native_simd_path());
    std::vector<std::shared_ptr<const dsmvc::AxisPlan>> plans;
    for (std::uint32_t index = 0; index < 12U; ++index) {
        plans.push_back(std::make_shared<const dsmvc::AxisPlan>(make_plan(
            (index & 1U) == 0U ? dsmvc::KernelKind::lanczos
                               : dsmvc::KernelKind::spline64,
            96, 64, 63.5 + static_cast<double>(index) / 64.0,
            static_cast<double>(index) / 128.0)));
    }

    std::barrier start(static_cast<std::ptrdiff_t>(plans.size() + 1U));
    std::atomic<std::size_t> prepared{0U};
    std::atomic<std::size_t> rejected{0U};
    std::vector<std::exception_ptr> errors(plans.size());
    std::vector<JoiningThread> workers;
    workers.reserve(plans.size());
    for (std::size_t index = 0; index < plans.size(); ++index) {
        workers.emplace_back([&, index] {
            start.arrive_and_wait();
            try {
                executor.prepare(plans[index]);
                prepared.fetch_add(1U, std::memory_order_relaxed);
            } catch (const std::logic_error &) {
                rejected.fetch_add(1U, std::memory_order_relaxed);
            } catch (...) {
                errors[index] = std::current_exception();
            }
        });
    }
    start.arrive_and_wait();
    executor.seal();
    workers.clear();
    for (const auto &error : errors) {
        if (error) std::rethrow_exception(error);
    }
    require(prepared.load(std::memory_order_relaxed)
                + rejected.load(std::memory_order_relaxed) == plans.size(),
            "concurrent prepare/seal lost an operation");

    bool rejected_after_seal = false;
    try {
        executor.prepare(plans.front());
    } catch (const std::logic_error &) {
        rejected_after_seal = true;
    }
    require(rejected_after_seal, "preparation succeeded after concurrent seal");
    (void)compare_rows(executor, *plans.back(), 8,
                       "sealed lookup after concurrent preparation");
}

void test_lazy_plan_single_flight() {
    if (!native_simd_available()) return;

    auto plan = std::make_shared<const dsmvc::AxisPlan>(make_plan(
        dsmvc::KernelKind::spline64, 2048, 1536, 1535.75, 0.125));
    dsmvc::CpuExecutor executor(native_simd_path());
    executor.defer(plan);
    executor.seal();
    auto stats = executor.packing_stats();
    require(stats.pack_executions == 0U && stats.lazy_requests == 0U,
            "deferred CPU plan was packed before first CPU selection");

    constexpr std::size_t callers = 8U;
    constexpr std::int32_t rows = 8;
    std::vector<float> input(
        static_cast<std::size_t>(plan->source_size) * rows);
    fill_deterministic(input.data(), input.size(), 0x51f17eU);
    std::vector<std::vector<float>> outputs(
        callers, std::vector<float>(
            static_cast<std::size_t>(plan->destination_size) * rows));
    std::barrier start(static_cast<std::ptrdiff_t>(callers));
    std::vector<std::exception_ptr> errors(callers);
    std::vector<JoiningThread> threads;
    threads.reserve(callers);
    for (std::size_t index = 0; index < callers; ++index) {
        threads.emplace_back([&, index] {
            try {
                start.arrive_and_wait();
                executor.inverse_rows(
                    *plan, input.data(), plan->source_size,
                    outputs[index].data(), plan->destination_size, rows);
            } catch (...) {
                errors[index] = std::current_exception();
            }
        });
    }
    threads.clear();
    for (const auto &error : errors) {
        if (error) std::rethrow_exception(error);
    }
    for (std::size_t index = 1; index < outputs.size(); ++index) {
        require(outputs[index] == outputs.front(),
                "single-flight CPU plan produced inconsistent output");
    }
    stats = executor.packing_stats();
    require(stats.pack_executions == 1U,
            "same deferred CPU plan was packed more than once");
    require(stats.lazy_requests == callers && stats.lazy_hits == callers - 1U,
            "lazy CPU plan hit accounting is inconsistent");
    require(stats.maximum_concurrent_packs == 1U,
            "same-plan single-flight allowed concurrent packing");
}

} // namespace

int main() {
    try {
        test_backend_selection();
        test_accelerator_executor_agreement(
            dsmvc::BackendKind::cuda, dsmvc::cuda_available(), "CUDA");
        test_accelerator_executor_agreement(
            dsmvc::BackendKind::vulkan, dsmvc::vulkan_available(), "Vulkan");
        test_cpu_path_selection();
        test_identity_bilinear();
        test_conditioned_bilinear_reference();
        test_condition_aware_float64_axis();
        test_conditioned_float64_2d_intermediate();
        test_padding_index_patterns();
        test_f64_mode_selection_and_cache();
        test_blur_precision_routing();
        test_custom_plan();
        test_blur_support_and_weights();
        test_blur_cache_isolation();
        test_inverse_only_cache();
        test_large_support_compatibility();
        test_axis_plan_validation();
        test_b3_neon_row_pair_agreement();
        test_b5_b7_executor_agreement();
        test_streamed_2d_executor_agreement();
        test_integer_2d_executor_agreement();
        test_executor_plan_ownership();
        test_concurrent_prepare_and_seal();
        test_lazy_plan_single_flight();
        std::cout << "dsmvc engine tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "dsmvc engine test failure: " << error.what() << '\n';
        return 1;
    }
}
