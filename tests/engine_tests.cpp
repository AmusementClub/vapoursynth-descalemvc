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

void test_backend_selection() {
    require(dsmvc::parse_backend("AUTO") == dsmvc::BackendKind::automatic,
            "backend parsing failed");
    require(dsmvc::resolve_backend(dsmvc::BackendKind::automatic)
                == dsmvc::BackendKind::cpu,
            "automatic backend did not select CPU");
    bool rejected = false;
    try {
        (void)dsmvc::resolve_backend(dsmvc::BackendKind::cuda);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "uncompiled CUDA backend was not rejected");
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
        test_identity_bilinear();
        test_custom_plan();
        test_inverse_only_cache();
        test_large_support_compatibility();
        test_axis_plan_validation();
        test_b5_b7_executor_agreement();
        test_executor_plan_ownership();
        test_concurrent_prepare_and_seal();
        std::cout << "dsmvc engine tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "dsmvc engine test failure: " << error.what() << '\n';
        return 1;
    }
}
