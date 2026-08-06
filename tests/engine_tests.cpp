#include <dsmvc/engine.hpp>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

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
    std::int32_t destination_size, double active_length, double shift = 0.0) {
    dsmvc::AxisRequest request;
    request.source_size = source_size;
    request.destination_size = destination_size;
    request.active_length = active_length;
    request.shift = shift;
    request.kernel.kind = kind;
    if (kind == dsmvc::KernelKind::lanczos) request.kernel.taps = 3;
    return dsmvc::build_axis_plan(request);
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
    require(different == 0U,
            std::string(label) + " is not bit-exact with the legacy two-pass path");
    return stats;
}

template <class Sample>
void compare_integer_2d(
    dsmvc::CpuPath path, dsmvc::KernelKind kind,
    const dsmvc::IntegerConversion &conversion,
    std::string_view label) {
    constexpr std::int32_t source_width = 97;
    constexpr std::int32_t destination_width = 67;
    constexpr std::int32_t source_height = 73;
    constexpr std::int32_t destination_height = 51;
    constexpr std::int32_t input_stride = 104;
    constexpr std::int32_t float_output_stride = 72;
    constexpr std::int32_t integer_output_stride = 73;
    auto horizontal = std::make_shared<const dsmvc::AxisPlan>(make_plan(
        kind, source_width, destination_width, 66.75, 0.125));
    auto vertical = std::make_shared<const dsmvc::AxisPlan>(make_plan(
        kind, source_height, destination_height, 50.5, 0.25));

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
    const auto cuda = std::find_if(
        capabilities.begin(), capabilities.end(), [](const auto &capability) {
            return capability.kind == dsmvc::BackendKind::cuda;
        });
    require(cuda != capabilities.end(), "CUDA capability is missing");
    require(cuda->compiled == dsmvc::cuda_compiled()
                && cuda->device_available == dsmvc::cuda_available(),
            "CUDA capability reporting is inconsistent");
}

void test_cuda_executor_agreement() {
    if (!dsmvc::cuda_available()) {
        std::cout << "CUDA executor tests skipped: no compatible device\n";
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
    generic_horizontal_request.kernel.kind = dsmvc::KernelKind::lanczos;
    generic_horizontal_request.kernel.taps = 5;
    auto generic_vertical_request = generic_horizontal_request;
    generic_vertical_request.source_size = source_height;
    generic_vertical_request.destination_size = destination_height;
    generic_vertical_request.active_length = 50.5;
    generic_vertical_request.shift = 0.25;
    pairs.push_back({
        "generic-b9",
        std::make_shared<const dsmvc::AxisPlan>(
            dsmvc::build_axis_plan(generic_horizontal_request)),
        std::make_shared<const dsmvc::AxisPlan>(
            dsmvc::build_axis_plan(generic_vertical_request)),
    });
    require(pairs.back().horizontal->half_bandwidth > 7,
            "CUDA generic fixture did not exceed the specialized bandwidths");

    dsmvc::Executor cuda(dsmvc::BackendKind::cuda);
    require(cuda.backend() == dsmvc::BackendKind::cuda,
            "CUDA executor resolved to the wrong backend");
    for (const auto &pair : pairs) {
        cuda.prepare(pair.horizontal);
        cuda.prepare(pair.vertical);
    }
    cuda.seal();
    cuda.seal();

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
        cuda.inverse_2d(
            *pair.horizontal, *pair.vertical, input.data(), input_stride,
            candidate.data(), output_stride);
        const auto stats = compare_matrix(
            reference.data(), candidate.data(), destination_height,
            destination_width, output_stride);
        std::cout << "CUDA " << pair.name << " 2D: max_error="
                  << stats.maximum << " mean_error=" << stats.mean
                  << " non_finite=" << stats.non_finite << '\n';
        require(stats.non_finite == 0U && stats.maximum < 1.0e-4F,
                std::string("CUDA ") + pair.name
                    + " 2D differs from the scalar executor");
    }

    const auto &separate = pairs[3];
    constexpr std::int32_t intermediate_stride = 72;
    std::vector<float> cpu_intermediate(
        static_cast<std::size_t>(source_height)
            * static_cast<std::size_t>(intermediate_stride),
        output_fill);
    std::vector<float> cuda_intermediate(cpu_intermediate.size(), output_fill);
    scalar.inverse_rows(
        *separate.horizontal, input.data(), input_stride,
        cpu_intermediate.data(), intermediate_stride, source_height);
    cuda.inverse_rows(
        *separate.horizontal, input.data(), input_stride,
        cuda_intermediate.data(), intermediate_stride, source_height);
    auto stats = compare_matrix(
        cpu_intermediate.data(), cuda_intermediate.data(), source_height,
        destination_width, intermediate_stride);
    require(stats.non_finite == 0U && stats.maximum < 1.0e-4F,
            "CUDA row executor differs from scalar");

    const dsmvc::AxisPlan unprepared_horizontal = *separate.horizontal;
    std::fill(cuda_intermediate.begin(), cuda_intermediate.end(), output_fill);
    cuda.inverse_rows(
        unprepared_horizontal, input.data(), input_stride,
        cuda_intermediate.data(), intermediate_stride, source_height);
    stats = compare_matrix(
        cpu_intermediate.data(), cuda_intermediate.data(), source_height,
        destination_width, intermediate_stride);
    require(stats.non_finite == 0U && stats.maximum < 1.0e-4F,
            "CUDA direct unprepared plan differs from scalar");

    std::fill(reference.begin(), reference.end(), output_fill);
    std::fill(candidate.begin(), candidate.end(), output_fill);
    scalar.inverse_columns(
        *separate.vertical, cpu_intermediate.data(), intermediate_stride,
        reference.data(), output_stride, destination_width);
    cuda.inverse_columns(
        *separate.vertical, cpu_intermediate.data(), intermediate_stride,
        candidate.data(), output_stride, destination_width);
    stats = compare_matrix(
        reference.data(), candidate.data(), destination_height,
        destination_width, output_stride);
    require(stats.non_finite == 0U && stats.maximum < 1.0e-4F,
            "CUDA column executor differs from scalar");
    const std::vector<float> concurrency_reference = reference;

    auto cached_column_input =
        std::make_shared<std::vector<float>>(cpu_intermediate);
    const std::shared_ptr<const void> cached_column_lifetime(
        cached_column_input,
        static_cast<const void *>(cached_column_input->data()));
    for (int repeat = 0; repeat < 2; ++repeat) {
        std::fill(candidate.begin(), candidate.end(), output_fill);
        cuda.inverse_columns(
            *separate.vertical, cached_column_input->data(),
            intermediate_stride, candidate.data(), output_stride,
            destination_width, cached_column_lifetime);
        stats = compare_matrix(
            concurrency_reference.data(), candidate.data(), destination_height,
            destination_width, output_stride);
        require(stats.non_finite == 0U && stats.maximum < 1.0e-4F,
                "cached CUDA column executor differs from scalar");
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
    cuda.inverse_2d_u8(
        *separate.horizontal, *separate.vertical,
        integer_input.data(), input_stride,
        integer_candidate.data(), output_stride, u8_conversion);
    std::uint32_t maximum_integer_error = maximum_integer_difference(
        integer_reference, integer_candidate);
    require(maximum_integer_error <= 1U,
            "CUDA u8 conversion differs from the Float32 reference by more than one");

    auto cached_u8_input =
        std::make_shared<std::vector<std::uint8_t>>(integer_input);
    const std::shared_ptr<const void> cached_u8_lifetime(
        cached_u8_input, static_cast<const void *>(cached_u8_input->data()));
    std::fill(integer_candidate.begin(), integer_candidate.end(), 0U);
    cuda.inverse_2d_u8(
        *separate.horizontal, *separate.vertical,
        cached_u8_input->data(), input_stride,
        integer_candidate.data(), output_stride, u8_conversion,
        cached_u8_lifetime);
    require(maximum_integer_difference(
                integer_reference, integer_candidate) <= 1U,
            "cached CUDA u8 conversion differs from the Float32 reference");

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
    cuda.inverse_2d_u8(
        *separate.horizontal, *separate.vertical,
        cached_u8_input->data(), input_stride,
        full_range_candidate.data(), output_stride, full_range_u8_conversion,
        cached_u8_lifetime);
    require(maximum_integer_difference(
                full_range_reference, full_range_candidate) <= 1U,
            "CUDA input cache reused the wrong u8 conversion");

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
    cuda.inverse_2d_u16(
        *separate.horizontal, *separate.vertical,
        u16_input.data(), input_stride,
        u16_candidate.data(), output_stride, u16_conversion);
    maximum_integer_error = maximum_integer_difference(
        u16_reference, u16_candidate);
    require(maximum_integer_error <= 1U,
            "CUDA u16 conversion differs from the Float32 reference by more than one");

    auto cached_u16_input =
        std::make_shared<std::vector<std::uint16_t>>(u16_input);
    const std::shared_ptr<const void> cached_u16_lifetime(
        cached_u16_input, static_cast<const void *>(cached_u16_input->data()));
    std::fill(u16_candidate.begin(), u16_candidate.end(), 0U);
    cuda.inverse_2d_u16(
        *separate.horizontal, *separate.vertical,
        cached_u16_input->data(), input_stride,
        u16_candidate.data(), output_stride, u16_conversion,
        cached_u16_lifetime);
    require(maximum_integer_difference(u16_reference, u16_candidate) <= 1U,
            "cached CUDA u16 conversion differs from the Float32 reference");

    auto cached_input = std::make_shared<std::vector<float>>(input);
    const std::shared_ptr<const void> cached_input_lifetime(
        cached_input, static_cast<const void *>(cached_input->data()));
    const auto check_concurrent = [&](
        const PlanPair &pair, const std::vector<float> &expected,
        const char *label) {
        std::vector<std::exception_ptr> errors(6U);
        std::barrier start(static_cast<std::ptrdiff_t>(errors.size()));
        std::vector<std::jthread> callers;
        callers.reserve(errors.size());
        for (std::size_t index = 0; index < errors.size(); ++index) {
            callers.emplace_back([&, index] {
                try {
                    std::vector<float> concurrent(candidate.size(), output_fill);
                    start.arrive_and_wait();
                    cuda.inverse_2d(
                        *pair.horizontal, *pair.vertical,
                        cached_input->data(), input_stride,
                        concurrent.data(), output_stride, cached_input_lifetime);
                    const auto concurrent_stats = compare_matrix(
                        expected.data(), concurrent.data(), destination_height,
                        destination_width, output_stride);
                    require(
                        concurrent_stats.non_finite == 0U
                            && concurrent_stats.maximum < 1.0e-4F,
                        std::string("concurrent CUDA ") + label
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
    check_concurrent(separate, concurrency_reference, "b7");

    const auto &limited = pairs[1];
    std::vector<float> limited_reference(reference.size(), output_fill);
    scalar.inverse_2d(
        *limited.horizontal, *limited.vertical,
        cached_input->data(), input_stride,
        limited_reference.data(), output_stride);
    check_concurrent(limited, limited_reference, "b3");

    std::fill(cuda_intermediate.begin(), cuda_intermediate.end(), output_fill);
    cuda.inverse_rows(
        *separate.horizontal, cached_input->data(), input_stride,
        cuda_intermediate.data(), intermediate_stride, source_height,
        cached_input_lifetime);
    stats = compare_matrix(
        cpu_intermediate.data(), cuda_intermediate.data(), source_height,
        destination_width, intermediate_stride);
    require(stats.non_finite == 0U && stats.maximum < 1.0e-4F,
            "cached CUDA row executor differs from scalar");
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
    std::vector<std::jthread> workers;
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
    require(horizontal_b5->half_bandwidth == 5,
            "Lanczos3 did not produce half-bandwidth 5");
    require(vertical_b5->half_bandwidth == 5,
            "vertical Lanczos3 did not produce half-bandwidth 5");
    require(horizontal_b7->half_bandwidth == 7,
            "Spline64 did not produce half-bandwidth 7");
    require(vertical_b7->half_bandwidth == 7,
            "vertical Spline64 did not produce half-bandwidth 7");

    dsmvc::CpuExecutor optimized(dsmvc::CpuPath::automatic);
    optimized.prepare(horizontal_b5);
    optimized.prepare(horizontal_b7);
    optimized.prepare(vertical_b5);
    optimized.prepare(vertical_b7);
    optimized.seal();

    for (const auto rows : {8, 9, 16, 17}) {
        compare_rows(optimized, *horizontal_b5, rows,
                     "Lanczos3 b5 horizontal rows=" + std::to_string(rows));
        compare_rows(optimized, *horizontal_b7, rows,
                     "Spline64 b7 horizontal rows=" + std::to_string(rows));
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

    std::vector<std::exception_ptr> errors(8U);
    std::vector<std::jthread> callers;
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
}

void test_executor_plan_ownership() {
    if (!dsmvc::cpu_avx2_available()) return;

    const auto b7_value = make_plan(
        dsmvc::KernelKind::spline64, 96, 64, 63.75, 0.125);
    const auto b5_value = make_plan(
        dsmvc::KernelKind::lanczos, 96, 64, 63.75, 0.125);
    require(b7_value.half_bandwidth == 7 && b5_value.half_bandwidth == 5,
            "ownership fixtures have unexpected bandwidths");

    dsmvc::CpuExecutor borrowed(dsmvc::CpuPath::avx2);
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
        dsmvc::CpuExecutor prepared(dsmvc::CpuPath::avx2);
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
    if (!dsmvc::cpu_avx2_available()) return;

    dsmvc::CpuExecutor executor(dsmvc::CpuPath::avx2);
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
    std::vector<std::jthread> workers;
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

} // namespace

int main() {
    try {
        test_backend_selection();
        test_cuda_executor_agreement();
        test_identity_bilinear();
        test_custom_plan();
        test_inverse_only_cache();
        test_large_support_compatibility();
        test_axis_plan_validation();
        test_b5_b7_executor_agreement();
        test_streamed_2d_executor_agreement();
        test_integer_2d_executor_agreement();
        test_executor_plan_ownership();
        test_concurrent_prepare_and_seal();
        std::cout << "dsmvc engine tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "dsmvc engine test failure: " << error.what() << '\n';
        return 1;
    }
}
