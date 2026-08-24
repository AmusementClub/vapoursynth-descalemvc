#include <dsmvc/engine.hpp>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <os/signpost.h>

#include <dispatch/dispatch.h>

#include "dsmvc_metal_routes_metallib.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::int32_t source_width = 1920;
constexpr std::int32_t source_height = 1080;
constexpr std::int32_t destination_width = 1692;
constexpr std::int32_t destination_height = 952;
constexpr double horizontal_active_length = 1691.5555555555557;
constexpr double horizontal_shift = 0.2222222222221717;
constexpr double vertical_active_length = 951.5;
constexpr double vertical_shift = 0.25;
constexpr std::uint32_t maximum_sample_error = 1U;

enum class Route : std::uint8_t {
    neon,
    metal,
    heterogeneous,
};

constexpr std::array<Route, 3> all_routes{
    Route::neon,
    Route::metal,
    Route::heterogeneous,
};

[[nodiscard]] constexpr std::size_t route_index(Route route) noexcept {
    return static_cast<std::size_t>(route);
}

[[nodiscard]] const char *route_name(Route route) noexcept {
    switch (route) {
    case Route::neon: return "neon";
    case Route::metal: return "metal";
    case Route::heterogeneous: return "neon+metal";
    }
    return "unknown";
}

[[nodiscard]] Route parse_route(std::string_view name) {
    if (name == "neon") return Route::neon;
    if (name == "metal") return Route::metal;
    if (name == "neon+metal") return Route::heterogeneous;
    throw std::invalid_argument("unknown YUV profile route: " + std::string{name});
}

struct Configuration {
    std::size_t samples = 21U;
    std::size_t warmups = 5U;
    std::size_t threads_per_threadgroup = 128U;
    std::size_t batch_size = 16U;
    std::size_t narrow_cpu_frames = 12U;
    std::size_t wide_cpu_frames = 9U;
    std::size_t narrow_cpu_concurrency = 16U;
    std::size_t wide_cpu_concurrency = 8U;
    std::size_t profile_iterations = 0U;
    std::string profile_route = "neon+metal";
    std::string profile_case = "yuv420p10-b7-spline64";
    bool assert_gates = false;
    std::filesystem::path json_output;
};

struct CaseSpec {
    const char *name = "";
    dsmvc::KernelKind kind = dsmvc::KernelKind::bicubic;
    std::int32_t taps = 0;
};

constexpr std::array<CaseSpec, 6> case_specs{{
    {"b1-bilinear", dsmvc::KernelKind::bilinear, 0},
    {"spline16", dsmvc::KernelKind::spline16, 0},
    {"b3-bicubic", dsmvc::KernelKind::bicubic, 0},
    {"spline36", dsmvc::KernelKind::spline36, 0},
    {"b5-lanczos3", dsmvc::KernelKind::lanczos, 3},
    {"b7-spline64", dsmvc::KernelKind::spline64, 0},
}};

struct PlaneGeometry {
    std::uint32_t source_width = 0U;
    std::uint32_t source_height = 0U;
    std::uint32_t destination_width = 0U;
    std::uint32_t destination_height = 0U;
    std::uint32_t input_stride = 0U;
    std::uint32_t intermediate_stride = 0U;
    std::uint32_t result_stride = 0U;
    std::uint32_t output_stride = 0U;
    bool chroma = false;
};

struct AxisJob {
    std::uint32_t source_size = 0U;
    std::uint32_t destination_size = 0U;
    std::uint32_t vector_count = 0U;
    std::uint32_t input_stride = 0U;
    std::uint32_t output_stride = 0U;
    std::uint32_t direction = 0U;
    std::uint32_t half_bandwidth = 0U;
    std::uint32_t reserved = 0U;
    std::uint32_t batch_count = 0U;
    std::uint32_t input_frame_stride = 0U;
    std::uint32_t output_frame_stride = 0U;
    std::uint32_t reserved_2 = 0U;
};

struct ConvertJob {
    std::uint32_t width = 0U;
    std::uint32_t height = 0U;
    std::uint32_t input_stride = 0U;
    std::uint32_t output_stride = 0U;
    std::uint32_t batch_count = 0U;
    std::uint32_t input_frame_stride = 0U;
    std::uint32_t output_frame_stride = 0U;
    std::uint32_t reserved = 0U;
};

static_assert(sizeof(AxisJob) == 48U);
static_assert(sizeof(ConvertJob) == 32U);
static_assert(sizeof(dsmvc::IntegerConversion) == 20U);

struct Measurement {
    double wall_ms = 0.0;
    double upload_ms = 0.0;
    double encode_ms = 0.0;
    double submit_wait_ms = 0.0;
    double gpu_ms = 0.0;
    double download_ms = 0.0;
};

struct Summary {
    std::vector<double> raw;
    double median = 0.0;
    double mad = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

struct ErrorStats {
    std::uint64_t differing_samples = 0U;
    std::uint32_t maximum_error = 0U;
};

struct LogicalAccessBytes {
    std::uint64_t image_reads = 0U;
    std::uint64_t image_writes = 0U;
    std::uint64_t plan_reads = 0U;

    LogicalAccessBytes &operator+=(const LogicalAccessBytes &other) noexcept {
        image_reads += other.image_reads;
        image_writes += other.image_writes;
        plan_reads += other.plan_reads;
        return *this;
    }

    [[nodiscard]] std::uint64_t total() const noexcept {
        return image_reads + image_writes + plan_reads;
    }
};

struct TransposeReuseStats {
    std::uint64_t entries = 0U;
    std::uint64_t adjacent_reused_entries = 0U;
    std::uint64_t one_step_loads = 0U;
    std::size_t unique_indices = 0U;
    std::size_t max_entries_per_destination = 0U;
    std::size_t max_adjacent_union = 0U;
    std::size_t noncontiguous_destinations = 0U;
    std::size_t max_index_span = 0U;
    std::uint32_t max_start_advance = 0U;
    std::uint32_t max_end_advance = 0U;
    bool monotonic_bounds = true;
};

struct PlanBuffers {
    id<MTLBuffer> transpose_offsets = nil;
    id<MTLBuffer> transpose_indices = nil;
    id<MTLBuffer> transpose_weights = nil;
    id<MTLBuffer> lower_ld = nil;
    id<MTLBuffer> upper_l = nil;
    id<MTLBuffer> inverse_diagonal = nil;
    std::uint32_t half_bandwidth = 0U;
    std::size_t bytes = 0U;
};

struct PreparedPlane {
    PlanBuffers horizontal;
    PlanBuffers vertical;
    id<MTLBuffer> input = nil;
    id<MTLBuffer> intermediate = nil;
    id<MTLBuffer> result = nil;
    id<MTLBuffer> output = nil;
    std::size_t working_set_bytes = 0U;
};

template <class Sample>
struct PlaneFixture {
    PlaneGeometry geometry;
    dsmvc::IntegerConversion conversion;
    std::shared_ptr<const dsmvc::AxisPlan> horizontal;
    std::shared_ptr<const dsmvc::AxisPlan> vertical;
    std::vector<Sample> input;
    std::vector<Sample> oracle;
    std::vector<Sample> output;
    PreparedPlane metal;
};

template <class Sample>
struct Fixture {
    CaseSpec spec;
    const char *format_name = "";
    std::uint32_t bits_per_sample = 0U;
    std::array<PlaneFixture<Sample>, 3> planes;
    std::size_t batch_size = 1U;
    std::size_t heterogeneous_cpu_frames = 1U;
};

struct CaseResult {
    std::string name;
    std::string format;
    std::int32_t half_bandwidth = 0;
    std::size_t working_set_bytes = 0U;
    std::size_t plan_buffer_bytes = 0U;
    std::size_t host_copy_bytes = 0U;
    LogicalAccessBytes metal_logical_accesses;
    std::array<TransposeReuseStats, 2> horizontal_transpose_reuse;
    std::size_t heterogeneous_cpu_frames = 0U;
    std::size_t cpu_concurrency = 0U;
    std::array<std::vector<Measurement>, all_routes.size()> measurements;
    std::array<ErrorStats, all_routes.size()> errors;
    std::array<std::uint64_t, all_routes.size()> identities{};
};

class BatchPool {
    struct JobState {
        JobState(std::size_t count, std::function<void(std::size_t)> function)
            : task_count(count), job(std::move(function)) {}

        std::size_t task_count = 0U;
        std::function<void(std::size_t)> job;
        std::atomic<std::size_t> next{0U};
        std::mutex error_mutex;
        std::exception_ptr error;
    };

public:
    explicit BatchPool(std::size_t parallelism)
        : parallelism_(std::max<std::size_t>(parallelism, 1U)),
          start_(static_cast<std::ptrdiff_t>(parallelism_)),
          finish_(static_cast<std::ptrdiff_t>(parallelism_)) {
        workers_.reserve(parallelism_ - 1U);
        for (std::size_t index = 1U; index < parallelism_; ++index) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~BatchPool() {
        stopping_ = true;
        start_.arrive_and_wait();
        for (auto &worker : workers_) worker.join();
    }

    BatchPool(const BatchPool &) = delete;
    BatchPool &operator=(const BatchPool &) = delete;

    template <class Function>
    void run(std::size_t task_count, Function &&function) {
        if (task_count == 0U) return;
        auto state = std::make_shared<JobState>(
            task_count, std::forward<Function>(function));
        state_ = state;
        start_.arrive_and_wait();
        execute(state);
        finish_.arrive_and_wait();
        state_.reset();
        if (state->error) std::rethrow_exception(state->error);
    }

private:
    static void execute(const std::shared_ptr<JobState> &state) {
        for (;;) {
            const std::size_t task = state->next.fetch_add(
                1U, std::memory_order_relaxed);
            if (task >= state->task_count) return;
            try {
                state->job(task);
            } catch (...) {
                const std::scoped_lock lock(state->error_mutex);
                if (!state->error) state->error = std::current_exception();
            }
        }
    }

    void worker_loop() {
        for (;;) {
            start_.arrive_and_wait();
            if (stopping_) return;
            execute(state_);
            finish_.arrive_and_wait();
        }
    }

    std::size_t parallelism_ = 1U;
    std::barrier<> start_;
    std::barrier<> finish_;
    std::vector<std::thread> workers_;
    std::shared_ptr<JobState> state_;
    bool stopping_ = false;
};

class DedicatedWorker {
public:
    DedicatedWorker() : worker_([this] { loop(); }) {}

    ~DedicatedWorker() {
        stopping_ = true;
        start_.arrive_and_wait();
        worker_.join();
    }

    DedicatedWorker(const DedicatedWorker &) = delete;
    DedicatedWorker &operator=(const DedicatedWorker &) = delete;

    template <class Function>
    void start(Function &&function) {
        job_ = std::forward<Function>(function);
        error_ = nullptr;
        start_.arrive_and_wait();
    }

    void wait() {
        finish_.arrive_and_wait();
        if (error_) std::rethrow_exception(error_);
    }

private:
    void loop() {
        for (;;) {
            start_.arrive_and_wait();
            if (stopping_) return;
            try {
                job_();
            } catch (...) {
                error_ = std::current_exception();
            }
            finish_.arrive_and_wait();
        }
    }

    std::barrier<> start_{2};
    std::barrier<> finish_{2};
    std::thread worker_;
    std::function<void()> job_;
    std::exception_ptr error_;
    bool stopping_ = false;
};

[[nodiscard]] std::size_t parse_positive(std::string_view text,
                                          std::string_view option) {
    std::size_t value = 0U;
    if (text.empty()) {
        throw std::invalid_argument(std::string{option} + " requires a value");
    }
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

[[nodiscard]] Configuration parse_arguments(int argc, char **argv) {
    Configuration result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        const auto next = [&](std::string_view option) {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string{option} + " requires a value");
            }
            return std::string_view{argv[++index]};
        };
        if (argument == "--samples") {
            result.samples = parse_positive(next(argument), argument);
        } else if (argument == "--warmups") {
            result.warmups = parse_positive(next(argument), argument);
        } else if (argument == "--threads-per-threadgroup") {
            result.threads_per_threadgroup = parse_positive(next(argument), argument);
        } else if (argument == "--batch-size") {
            result.batch_size = parse_positive(next(argument), argument);
        } else if (argument == "--heterogeneous-cpu-frames") {
            const auto value = parse_positive(next(argument), argument);
            result.narrow_cpu_frames = value;
            result.wide_cpu_frames = value;
        } else if (argument == "--narrow-cpu-frames") {
            result.narrow_cpu_frames = parse_positive(next(argument), argument);
        } else if (argument == "--wide-cpu-frames") {
            result.wide_cpu_frames = parse_positive(next(argument), argument);
        } else if (argument == "--cpu-concurrency") {
            const auto value = parse_positive(next(argument), argument);
            result.narrow_cpu_concurrency = value;
            result.wide_cpu_concurrency = value;
        } else if (argument == "--narrow-cpu-concurrency") {
            result.narrow_cpu_concurrency = parse_positive(next(argument), argument);
        } else if (argument == "--wide-cpu-concurrency") {
            result.wide_cpu_concurrency = parse_positive(next(argument), argument);
        } else if (argument == "--profile-iterations") {
            result.profile_iterations = parse_positive(next(argument), argument);
        } else if (argument == "--profile-route") {
            result.profile_route = std::string{next(argument)};
        } else if (argument == "--profile-case") {
            result.profile_case = std::string{next(argument)};
        } else if (argument == "--json-out") {
            result.json_output = std::filesystem::path{next(argument)};
        } else if (argument == "--assert") {
            result.assert_gates = true;
        } else {
            throw std::invalid_argument(
                "usage: dsmvc_metal_yuv_benchmark [--samples N] [--warmups N] "
                "[--threads-per-threadgroup N] [--batch-size N] "
                "[--heterogeneous-cpu-frames N] [--narrow-cpu-frames N] "
                "[--wide-cpu-frames N] [--cpu-concurrency N] "
                "[--narrow-cpu-concurrency N] [--wide-cpu-concurrency N] "
                "[--profile-iterations N] [--profile-route ROUTE] "
                "[--profile-case CASE] "
                "[--json-out PATH] [--assert]");
        }
    }
    if (result.batch_size > 64U) {
        throw std::invalid_argument("--batch-size exceeds 64");
    }
    if (result.narrow_cpu_frames > result.batch_size
        || result.wide_cpu_frames > result.batch_size) {
        throw std::invalid_argument("CPU frame split exceeds --batch-size");
    }
    if (result.narrow_cpu_concurrency > 64U
        || result.wide_cpu_concurrency > 64U) {
        throw std::invalid_argument("CPU concurrency exceeds 64");
    }
    if (result.threads_per_threadgroup > 1024U) {
        throw std::invalid_argument("--threads-per-threadgroup exceeds 1024");
    }
    return result;
}

[[nodiscard]] std::string ns_error(NSError *error, std::string_view fallback) {
    if (error == nil || error.localizedDescription == nil) {
        return std::string{fallback};
    }
    const char *description = error.localizedDescription.UTF8String;
    return description == nullptr ? std::string{fallback} : std::string{description};
}

template <class Function>
[[nodiscard]] double timed_ms(Function &&function) {
    const auto start = Clock::now();
    std::forward<Function>(function)();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

[[nodiscard]] Summary summarize(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("cannot summarize no samples");
    for (const double value : values) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::runtime_error("timing sample is not finite and nonnegative");
        }
    }
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t middle = sorted.size() / 2U;
    const double median = (sorted.size() & 1U) != 0U
        ? sorted[middle] : (sorted[middle - 1U] + sorted[middle]) / 2.0;
    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (const double value : values) deviations.push_back(std::abs(value - median));
    std::sort(deviations.begin(), deviations.end());
    const double mad = (deviations.size() & 1U) != 0U
        ? deviations[middle]
        : (deviations[middle - 1U] + deviations[middle]) / 2.0;
    return {std::move(values), median, mad, sorted.front(), sorted.back()};
}

template <class Projection>
[[nodiscard]] Summary summarize_measurements(
    const std::vector<Measurement> &measurements, Projection projection) {
    std::vector<double> values;
    values.reserve(measurements.size());
    for (const auto &measurement : measurements) {
        values.push_back(projection(measurement));
    }
    return summarize(std::move(values));
}

[[nodiscard]] constexpr std::uint32_t align_up(
    std::uint32_t value, std::uint32_t alignment) noexcept {
    return (value + alignment - 1U) / alignment * alignment;
}

template <class Sample>
[[nodiscard]] PlaneGeometry make_geometry(bool chroma) {
    const std::uint32_t divisor = chroma ? 2U : 1U;
    PlaneGeometry geometry;
    geometry.source_width = static_cast<std::uint32_t>(source_width) / divisor;
    geometry.source_height = static_cast<std::uint32_t>(source_height) / divisor;
    geometry.destination_width =
        static_cast<std::uint32_t>(destination_width) / divisor;
    geometry.destination_height =
        static_cast<std::uint32_t>(destination_height) / divisor;
    geometry.input_stride = align_up(
        geometry.source_width * sizeof(Sample), 64U) / sizeof(Sample);
    // Observed from VapourSynth R78 for this fixed 1692x952 YUV420 layout.
    // This models row traffic only; it is not a general allocator contract.
    const std::uint32_t output_stride_bytes = sizeof(Sample) == 1U
        ? (chroma ? 864U : 1696U)
        : (chroma ? 1696U : 3392U);
    geometry.output_stride = output_stride_bytes / sizeof(Sample);
    geometry.intermediate_stride = align_up(
        geometry.destination_width * sizeof(float), 64U) / sizeof(float);
    geometry.result_stride = geometry.intermediate_stride;
    geometry.chroma = chroma;
    return geometry;
}

[[nodiscard]] dsmvc::AxisRequest make_request(
    const CaseSpec &spec, const PlaneGeometry &geometry, bool horizontal) {
    dsmvc::AxisRequest request;
    request.source_size = static_cast<std::int32_t>(
        horizontal ? geometry.source_width : geometry.source_height);
    request.destination_size = static_cast<std::int32_t>(
        horizontal ? geometry.destination_width : geometry.destination_height);
    const double source_scale = geometry.chroma
        ? static_cast<double>(request.source_size)
            / static_cast<double>(horizontal ? source_width : source_height)
        : 1.0;
    request.active_length = source_scale * (horizontal
        ? horizontal_active_length : vertical_active_length);
    if (geometry.chroma && horizontal) {
        request.shift = 0.25
            - 0.25 * static_cast<double>(destination_width)
                / static_cast<double>(source_width)
            + horizontal_shift * static_cast<double>(request.source_size)
                / static_cast<double>(source_width);
    } else {
        request.shift = source_scale
            * (horizontal ? horizontal_shift : vertical_shift);
    }
    request.kernel.kind = spec.kind;
    request.kernel.taps = spec.taps;
    request.kernel.b = 0.0;
    request.kernel.c = 0.5;
    request.border = dsmvc::BorderMode::mirror;
    return request;
}

[[nodiscard]] std::size_t uploaded_plan_bytes(
    const dsmvc::AxisPlan &plan) noexcept {
    return plan.transpose_offsets.size() * sizeof(std::uint32_t)
        + plan.transpose_indices.size() * sizeof(std::int32_t)
        + plan.transpose_weights.size() * sizeof(float)
        + plan.lower_ld.size() * sizeof(float)
        + plan.upper_l.size() * sizeof(float)
        + plan.inverse_diagonal.size() * sizeof(float);
}

[[nodiscard]] TransposeReuseStats transpose_reuse_stats(
    const dsmvc::AxisPlan &plan) {
    TransposeReuseStats result;
    std::vector<bool> seen(static_cast<std::size_t>(plan.source_size), false);
    std::size_t previous_begin = 0U;
    std::size_t previous_end = 0U;
    std::int32_t previous_minimum = 0;
    std::int32_t previous_maximum = 0;
    bool previous_nonempty = false;

    for (std::size_t destination = 0U;
         destination < static_cast<std::size_t>(plan.destination_size);
         ++destination) {
        const auto begin = static_cast<std::size_t>(
            plan.transpose_offsets[destination]);
        const auto end = static_cast<std::size_t>(
            plan.transpose_offsets[destination + 1U]);
        const std::size_t count = end - begin;
        result.entries += count;
        result.max_entries_per_destination = std::max(
            result.max_entries_per_destination, count);

        for (std::size_t entry = begin; entry < end; ++entry) {
            const auto source = static_cast<std::size_t>(
                plan.transpose_indices[entry]);
            if (!seen[source]) {
                seen[source] = true;
                ++result.unique_indices;
            }
        }
        bool contiguous = true;
        for (std::size_t entry = begin + (count != 0U ? 1U : 0U);
             entry < end; ++entry) {
            contiguous = contiguous
                && plan.transpose_indices[entry]
                    == plan.transpose_indices[entry - 1U] + 1;
        }
        if (!contiguous) ++result.noncontiguous_destinations;

        std::size_t intersection = 0U;
        if (destination != 0U) {
            std::size_t left = previous_begin;
            std::size_t right = begin;
            while (left < previous_end && right < end) {
                const auto previous = plan.transpose_indices[left];
                const auto current = plan.transpose_indices[right];
                if (previous == current) {
                    ++intersection;
                    ++left;
                    ++right;
                } else if (previous < current) {
                    ++left;
                } else {
                    ++right;
                }
            }
        }
        result.adjacent_reused_entries += intersection;
        result.one_step_loads += count - intersection;
        result.max_adjacent_union = std::max(
            result.max_adjacent_union,
            count + (previous_end - previous_begin) - intersection);

        const bool nonempty = begin != end;
        if (previous_nonempty && nonempty) {
            const auto minimum = plan.transpose_indices[begin];
            const auto maximum = plan.transpose_indices[end - 1U];
            result.max_start_advance = std::max(
                result.max_start_advance,
                static_cast<std::uint32_t>(minimum - previous_minimum));
            result.max_end_advance = std::max(
                result.max_end_advance,
                static_cast<std::uint32_t>(maximum - previous_maximum));
            result.monotonic_bounds = result.monotonic_bounds
                && minimum >= previous_minimum
                && maximum >= previous_maximum;
        }
        if (nonempty) {
            previous_minimum = plan.transpose_indices[begin];
            previous_maximum = plan.transpose_indices[end - 1U];
            result.max_index_span = std::max(
                result.max_index_span,
                static_cast<std::size_t>(
                    previous_maximum - previous_minimum + 1));
        }
        previous_nonempty = nonempty;
        previous_begin = begin;
        previous_end = end;
    }
    return result;
}

[[nodiscard]] std::uint64_t recurrence_link_count(
    const dsmvc::AxisPlan &plan) noexcept {
    const auto destination = static_cast<std::uint64_t>(plan.destination_size);
    const auto half_bandwidth = static_cast<std::uint64_t>(plan.half_bandwidth);
    std::uint64_t result = 0U;
    for (std::uint64_t index = 0U; index < destination; ++index) {
        result += std::min(index, half_bandwidth);
    }
    return result;
}

[[nodiscard]] LogicalAccessBytes logical_axis_accesses(
    const dsmvc::AxisPlan &plan, std::uint64_t vector_count,
    std::uint64_t input_element_bytes) noexcept {
    const auto destination = static_cast<std::uint64_t>(plan.destination_size);
    const auto transpose_entries = static_cast<std::uint64_t>(
        plan.transpose_indices.size());
    const std::uint64_t links = recurrence_link_count(plan);
    const std::uint64_t backward_outputs = destination > 0U
        ? destination - 1U : 0U;
    const bool register_window = plan.half_bandwidth == 5
        || plan.half_bandwidth == 7;
    const std::uint64_t recurrence_image_reads = register_window
        ? backward_outputs : 2U * links + backward_outputs;

    LogicalAccessBytes result;
    result.image_reads = vector_count * (
        transpose_entries * input_element_bytes
        + recurrence_image_reads * sizeof(float));
    result.image_writes = vector_count
        * (destination + backward_outputs) * sizeof(float);
    result.plan_reads = vector_count * (
        2U * destination * sizeof(std::uint32_t)
        + transpose_entries
            * (sizeof(std::int32_t) + sizeof(float))
        + 2U * links * sizeof(float)
        + destination * sizeof(float));
    return result;
}

template <class Sample>
[[nodiscard]] LogicalAccessBytes logical_metal_accesses(
    const Fixture<Sample> &fixture) noexcept {
    LogicalAccessBytes result;
    for (const auto &plane : fixture.planes) {
        const auto &geometry = plane.geometry;
        result += logical_axis_accesses(
            *plane.horizontal, geometry.source_height, sizeof(Sample));
        result += logical_axis_accesses(
            *plane.vertical, geometry.destination_width, sizeof(float));
        const std::uint64_t pixels = static_cast<std::uint64_t>(
            geometry.destination_width) * geometry.destination_height;
        result.image_reads += pixels * sizeof(float);
        result.image_writes += pixels * sizeof(Sample);
    }
    return result;
}

[[nodiscard]] dsmvc::IntegerConversion make_conversion(
    std::uint32_t bits, bool chroma) {
    const std::uint32_t depth_scale = 1U << (bits - 8U);
    const std::uint32_t offset = (chroma ? 128U : 16U) * depth_scale;
    const std::uint32_t scale = (chroma ? 224U : 219U) * depth_scale;
    return {
        static_cast<float>(offset),
        1.0F / static_cast<float>(scale),
        static_cast<float>(scale),
        static_cast<float>(offset),
        (1U << bits) - 1U,
    };
}

template <class Sample>
void fill_input(PlaneFixture<Sample> &plane, std::size_t case_index,
                std::size_t plane_index, std::size_t batch_size) {
    const auto &geometry = plane.geometry;
    const std::size_t frame_elements = static_cast<std::size_t>(
        geometry.source_height) * geometry.input_stride;
    const auto nominal_scale = static_cast<std::uint32_t>(
        std::lround(1.0F / plane.conversion.input_scale));
    const auto nominal_offset = static_cast<std::uint32_t>(
        plane.conversion.input_offset);
    const auto nominal_minimum = geometry.chroma
        ? nominal_offset - nominal_scale / 2U : nominal_offset;
    std::fill(plane.input.begin(), plane.input.end(), Sample{0});
    for (std::size_t frame = 0; frame < batch_size; ++frame) {
        for (std::uint32_t row = 0U; row < geometry.source_height; ++row) {
            for (std::uint32_t column = 0U;
                 column < geometry.source_width; ++column) {
                std::uint32_t bits = static_cast<std::uint32_t>(
                    frame * 0x9e3779b9U + row * 0x85ebca6bU
                    + column * 0xc2b2ae35U + case_index * 0x27d4eb2dU
                    + plane_index * 0x165667b1U);
                bits ^= bits >> 16U;
                bits *= 0x7feb352dU;
                bits ^= bits >> 15U;
                const auto value = nominal_minimum + bits % (nominal_scale + 1U);
                const auto index = frame * frame_elements
                    + static_cast<std::size_t>(row) * geometry.input_stride
                    + column;
                plane.input[index] = static_cast<Sample>(value);
            }
        }
    }
}

class MetalYuvRunner {
public:
    explicit MetalYuvRunner(std::size_t threads_per_threadgroup)
        : threads_per_threadgroup_(threads_per_threadgroup) {
        const auto start = Clock::now();
        device_ = MTLCreateSystemDefaultDevice();
        if (device_ == nil) throw std::runtime_error("no Metal device is available");
        if (!device_.hasUnifiedMemory) {
            throw std::runtime_error("Metal YUV benchmark requires unified memory");
        }
        queue_ = [device_ newCommandQueue];
        if (queue_ == nil) throw std::runtime_error("Metal command queue creation failed");
        dispatch_data_t data = dispatch_data_create(
            dsmvc_metal_routes_metallib, dsmvc_metal_routes_metallib_size,
            dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NSError *error = nil;
        library_ = [device_ newLibraryWithData:data error:&error];
        if (library_ == nil) {
            throw std::runtime_error(ns_error(error, "embedded metallib load failed"));
        }
        const std::array<NSString *, 5> float_names{
            @"inverse_axis_generic", @"inverse_axis_h1", @"inverse_axis_h3",
            @"inverse_axis_h5", @"inverse_axis_h7"};
        const std::array<NSString *, 5> u8_names{
            @"inverse_axis_u8_generic", @"inverse_axis_u8_h1",
            @"inverse_axis_u8_h3", @"inverse_axis_u8_h5",
            @"inverse_axis_u8_h7"};
        const std::array<NSString *, 5> u16_names{
            @"inverse_axis_u16_generic", @"inverse_axis_u16_h1",
            @"inverse_axis_u16_h3", @"inverse_axis_u16_h5",
            @"inverse_axis_u16_h7"};
        for (std::size_t index = 0U; index < float_names.size(); ++index) {
            float_pipelines_[index] = make_pipeline(float_names[index]);
            u8_pipelines_[index] = make_pipeline(u8_names[index]);
            u16_pipelines_[index] = make_pipeline(u16_names[index]);
        }
        convert_u8_ = make_pipeline(@"convert_f32_to_u8");
        convert_u16_ = make_pipeline(@"convert_f32_to_u16");
        setup_ms_ = std::chrono::duration<double, std::milli>(
            Clock::now() - start).count();
    }

    [[nodiscard]] const char *device_name() const noexcept {
        const char *name = device_.name.UTF8String;
        return name == nullptr ? "Metal device" : name;
    }

    [[nodiscard]] double setup_ms() const noexcept { return setup_ms_; }

    template <class Sample>
    [[nodiscard]] PreparedPlane prepare(
        const PlaneFixture<Sample> &plane, std::size_t batch_size,
        const PreparedPlane *scratch_owner = nullptr,
        const PreparedPlane *plan_owner = nullptr) {
        PreparedPlane prepared;
        prepared.horizontal = plan_owner
            ? plan_owner->horizontal : prepare_plan(*plane.horizontal);
        prepared.vertical = plan_owner
            ? plan_owner->vertical : prepare_plan(*plane.vertical);
        const auto &geometry = plane.geometry;
        const std::size_t input_bytes = static_cast<std::size_t>(
            geometry.source_height) * geometry.input_stride
            * sizeof(Sample) * batch_size;
        const std::size_t intermediate_bytes = static_cast<std::size_t>(
            geometry.source_height) * geometry.intermediate_stride
            * sizeof(float) * batch_size;
        const std::size_t result_bytes = static_cast<std::size_t>(
            geometry.destination_height) * geometry.result_stride
            * sizeof(float) * batch_size;
        const std::size_t output_bytes = static_cast<std::size_t>(
            geometry.destination_height) * geometry.output_stride
            * sizeof(Sample) * batch_size;
        prepared.input = make_empty_buffer(input_bytes, @"dsmvc YUV input");
        if (scratch_owner) {
            if (scratch_owner->intermediate.length < intermediate_bytes
                || scratch_owner->result.length < result_bytes) {
                throw std::length_error("shared Metal YUV scratch is too small");
            }
            prepared.intermediate = scratch_owner->intermediate;
            prepared.result = scratch_owner->result;
        } else {
            prepared.intermediate = make_empty_buffer(
                intermediate_bytes, @"dsmvc YUV intermediate");
            prepared.result = make_empty_buffer(
                result_bytes, @"dsmvc YUV result");
        }
        prepared.output = make_empty_buffer(output_bytes, @"dsmvc YUV output");
        prepared.working_set_bytes = input_bytes + output_bytes;
        if (!scratch_owner) {
            prepared.working_set_bytes += intermediate_bytes + result_bytes;
        }
        return prepared;
    }

    template <class Sample>
    [[nodiscard]] Measurement run(
        Fixture<Sample> &fixture, std::size_t first_frame,
        std::size_t frame_count) {
        if (frame_count == 0U
            || first_frame + frame_count > fixture.batch_size) {
            throw std::invalid_argument("invalid Metal YUV frame range");
        }
        @autoreleasepool {
            Measurement measurement;
            const auto wall_start = Clock::now();
            measurement.upload_ms = timed_ms([&] {
                for (auto &plane : fixture.planes) {
                    const std::size_t frame_elements = static_cast<std::size_t>(
                        plane.geometry.source_height)
                        * plane.geometry.input_stride;
                    std::memcpy(
                        plane.metal.input.contents,
                        plane.input.data() + first_frame * frame_elements,
                        frame_count * frame_elements * sizeof(Sample));
                }
            });
            id<MTLCommandBuffer> command = [queue_ commandBuffer];
            if (command == nil) throw std::runtime_error("Metal command buffer failed");
            command.label = @"dsmvc YUV Metal route";
            measurement.encode_ms = timed_ms([&] {
                for (auto &plane : fixture.planes) {
                    encode_plane(command, plane, frame_count);
                }
            });
            measurement.submit_wait_ms = timed_ms([&] {
                [command commit];
                [command waitUntilCompleted];
            });
            if (command.status == MTLCommandBufferStatusError) {
                throw std::runtime_error(ns_error(command.error, "Metal execution failed"));
            }
            const CFTimeInterval gpu_start = command.GPUStartTime;
            const CFTimeInterval gpu_end = command.GPUEndTime;
            if (gpu_end >= gpu_start && gpu_start > 0.0) {
                measurement.gpu_ms = (gpu_end - gpu_start) * 1000.0;
            }
            measurement.download_ms = timed_ms([&] {
                for (auto &plane : fixture.planes) {
                    const std::size_t frame_elements = static_cast<std::size_t>(
                        plane.geometry.destination_height)
                        * plane.geometry.output_stride;
                    std::memcpy(
                        plane.output.data() + first_frame * frame_elements,
                        plane.metal.output.contents,
                        frame_count * frame_elements * sizeof(Sample));
                }
            });
            measurement.wall_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - wall_start).count();
            return measurement;
        }
    }

private:
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLLibrary> library_ = nil;
    std::array<id<MTLComputePipelineState>, 5> float_pipelines_{};
    std::array<id<MTLComputePipelineState>, 5> u8_pipelines_{};
    std::array<id<MTLComputePipelineState>, 5> u16_pipelines_{};
    id<MTLComputePipelineState> convert_u8_ = nil;
    id<MTLComputePipelineState> convert_u16_ = nil;
    std::size_t threads_per_threadgroup_ = 32U;
    double setup_ms_ = 0.0;

    [[nodiscard]] id<MTLComputePipelineState> make_pipeline(NSString *name) {
        id<MTLFunction> function = [library_ newFunctionWithName:name];
        if (function == nil) {
            throw std::runtime_error(
                "embedded metallib is missing " + std::string{name.UTF8String});
        }
        NSError *error = nil;
        id<MTLComputePipelineState> pipeline =
            [device_ newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            throw std::runtime_error(ns_error(error, "Metal pipeline creation failed"));
        }
        return pipeline;
    }

    [[nodiscard]] id<MTLBuffer> make_empty_buffer(
        std::size_t bytes, NSString *label) {
        if (bytes == 0U || bytes > device_.maxBufferLength) {
            throw std::length_error("invalid Metal YUV buffer size");
        }
        id<MTLBuffer> buffer = [device_ newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
        if (buffer == nil) throw std::runtime_error("Metal buffer allocation failed");
        buffer.label = label;
        return buffer;
    }

    template <class Value>
    [[nodiscard]] id<MTLBuffer> make_plan_buffer(
        const std::vector<Value> &values, NSString *label) {
        if (values.empty()) throw std::invalid_argument("Metal plan buffer is empty");
        id<MTLBuffer> buffer = [device_ newBufferWithBytes:values.data()
                                                   length:values.size() * sizeof(Value)
                                                  options:MTLResourceStorageModeShared];
        if (buffer == nil) throw std::runtime_error("Metal plan upload failed");
        buffer.label = label;
        return buffer;
    }

    [[nodiscard]] PlanBuffers prepare_plan(const dsmvc::AxisPlan &plan) {
        if (!plan.valid()) throw std::invalid_argument("cannot upload invalid plan");
        PlanBuffers prepared;
        prepared.transpose_offsets = make_plan_buffer(
            plan.transpose_offsets, @"dsmvc YUV transpose offsets");
        prepared.transpose_indices = make_plan_buffer(
            plan.transpose_indices, @"dsmvc YUV transpose indices");
        prepared.transpose_weights = make_plan_buffer(
            plan.transpose_weights, @"dsmvc YUV transpose weights");
        prepared.lower_ld = make_plan_buffer(plan.lower_ld, @"dsmvc YUV lower LD");
        prepared.upper_l = make_plan_buffer(plan.upper_l, @"dsmvc YUV upper L");
        prepared.inverse_diagonal = make_plan_buffer(
            plan.inverse_diagonal, @"dsmvc YUV inverse diagonal");
        prepared.half_bandwidth = static_cast<std::uint32_t>(plan.half_bandwidth);
        prepared.bytes = uploaded_plan_bytes(plan);
        return prepared;
    }

    [[nodiscard]] static std::size_t pipeline_index(
        std::uint32_t half_bandwidth) noexcept {
        switch (half_bandwidth) {
        case 1U: return 1U;
        case 3U: return 2U;
        case 5U: return 3U;
        case 7U: return 4U;
        default: return 0U;
        }
    }

    void bind_plan(id<MTLComputeCommandEncoder> encoder,
                   const PlanBuffers &plan) {
        [encoder setBuffer:plan.transpose_offsets offset:0 atIndex:2];
        [encoder setBuffer:plan.transpose_indices offset:0 atIndex:3];
        [encoder setBuffer:plan.transpose_weights offset:0 atIndex:4];
        [encoder setBuffer:plan.lower_ld offset:0 atIndex:5];
        [encoder setBuffer:plan.upper_l offset:0 atIndex:6];
        [encoder setBuffer:plan.inverse_diagonal offset:0 atIndex:7];
    }

    void dispatch(id<MTLComputeCommandEncoder> encoder,
                  id<MTLComputePipelineState> pipeline,
                  std::size_t count) const {
        const NSUInteger threads = std::min<NSUInteger>(
            static_cast<NSUInteger>(threads_per_threadgroup_),
            pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(count, 1U, 1U)
             threadsPerThreadgroup:MTLSizeMake(threads, 1U, 1U)];
    }

    template <class Sample>
    void encode_plane(id<MTLCommandBuffer> command,
                      PlaneFixture<Sample> &plane,
                      std::size_t frame_count) {
        const auto &geometry = plane.geometry;
        std::array<dsmvc::IntegerConversion, 64> conversions{};
        std::fill_n(conversions.begin(), frame_count, plane.conversion);
        const AxisJob horizontal_job{
            geometry.source_width,
            geometry.destination_width,
            geometry.source_height,
            geometry.input_stride,
            geometry.intermediate_stride,
            0U,
            plane.metal.horizontal.half_bandwidth,
            0U,
            static_cast<std::uint32_t>(frame_count),
            geometry.source_height * geometry.input_stride,
            geometry.source_height * geometry.intermediate_stride,
            0U,
        };
        id<MTLComputeCommandEncoder> horizontal = [command computeCommandEncoder];
        if (horizontal == nil) throw std::runtime_error("Metal encoder failed");
        horizontal.label = @"dsmvc YUV horizontal inverse";
        id<MTLComputePipelineState> horizontal_pipeline =
            std::is_same_v<Sample, std::uint8_t>
            ? u8_pipelines_[pipeline_index(horizontal_job.half_bandwidth)]
            : u16_pipelines_[pipeline_index(horizontal_job.half_bandwidth)];
        [horizontal setComputePipelineState:horizontal_pipeline];
        [horizontal setBuffer:plane.metal.input offset:0 atIndex:0];
        [horizontal setBytes:&horizontal_job length:sizeof(horizontal_job) atIndex:1];
        bind_plan(horizontal, plane.metal.horizontal);
        [horizontal setBuffer:plane.metal.intermediate offset:0 atIndex:8];
        [horizontal setBytes:conversions.data()
                      length:frame_count * sizeof(conversions.front())
                     atIndex:9];
        dispatch(horizontal, horizontal_pipeline,
                 static_cast<std::size_t>(geometry.source_height) * frame_count);
        [horizontal endEncoding];

        const AxisJob vertical_job{
            geometry.source_height,
            geometry.destination_height,
            geometry.destination_width,
            geometry.intermediate_stride,
            geometry.result_stride,
            1U,
            plane.metal.vertical.half_bandwidth,
            0U,
            static_cast<std::uint32_t>(frame_count),
            geometry.source_height * geometry.intermediate_stride,
            geometry.destination_height * geometry.result_stride,
            0U,
        };
        id<MTLComputeCommandEncoder> vertical = [command computeCommandEncoder];
        if (vertical == nil) throw std::runtime_error("Metal encoder failed");
        vertical.label = @"dsmvc YUV vertical inverse";
        id<MTLComputePipelineState> vertical_pipeline =
            float_pipelines_[pipeline_index(vertical_job.half_bandwidth)];
        [vertical setComputePipelineState:vertical_pipeline];
        [vertical setBuffer:plane.metal.intermediate offset:0 atIndex:0];
        [vertical setBytes:&vertical_job length:sizeof(vertical_job) atIndex:1];
        bind_plan(vertical, plane.metal.vertical);
        [vertical setBuffer:plane.metal.result offset:0 atIndex:8];
        dispatch(vertical, vertical_pipeline,
                 static_cast<std::size_t>(geometry.destination_width) * frame_count);
        [vertical endEncoding];

        const ConvertJob convert_job{
            geometry.destination_width,
            geometry.destination_height,
            geometry.result_stride,
            geometry.output_stride,
            static_cast<std::uint32_t>(frame_count),
            geometry.destination_height * geometry.result_stride,
            geometry.destination_height * geometry.output_stride,
            0U,
        };
        id<MTLComputeCommandEncoder> convert = [command computeCommandEncoder];
        if (convert == nil) throw std::runtime_error("Metal encoder failed");
        convert.label = @"dsmvc YUV integer conversion";
        id<MTLComputePipelineState> convert_pipeline =
            std::is_same_v<Sample, std::uint8_t> ? convert_u8_ : convert_u16_;
        [convert setComputePipelineState:convert_pipeline];
        [convert setBuffer:plane.metal.result offset:0 atIndex:0];
        [convert setBytes:&convert_job length:sizeof(convert_job) atIndex:1];
        [convert setBytes:conversions.data()
                   length:frame_count * sizeof(conversions.front())
                  atIndex:2];
        [convert setBuffer:plane.metal.output offset:0 atIndex:3];
        dispatch(convert, convert_pipeline,
                 static_cast<std::size_t>(geometry.destination_width)
                     * geometry.destination_height * frame_count);
        [convert endEncoding];
    }
};

template <class Sample>
void run_cpu_frames(Fixture<Sample> &fixture,
                    const dsmvc::CpuExecutor &neon,
                    BatchPool &pool, std::size_t frame_count) {
    pool.run(frame_count, [&](std::size_t frame) {
        for (auto &plane : fixture.planes) {
            const auto &geometry = plane.geometry;
            const std::size_t input_frame = static_cast<std::size_t>(
                geometry.source_height) * geometry.input_stride;
            const std::size_t output_frame = static_cast<std::size_t>(
                geometry.destination_height) * geometry.output_stride;
            const auto *input = plane.input.data() + frame * input_frame;
            auto *output = plane.output.data() + frame * output_frame;
            const bool buffered = frame == 0U;
            if constexpr (std::is_same_v<Sample, std::uint8_t>) {
                if (buffered) {
                    neon.inverse_2d_u8(
                        *plane.horizontal, *plane.vertical,
                        input, geometry.input_stride,
                        output, geometry.output_stride, plane.conversion);
                } else {
                    neon.inverse_2d_u8_streamed(
                        *plane.horizontal, *plane.vertical,
                        input, geometry.input_stride,
                        output, geometry.output_stride, plane.conversion);
                }
            } else {
                if (buffered) {
                    neon.inverse_2d_u16(
                        *plane.horizontal, *plane.vertical,
                        input, geometry.input_stride,
                        output, geometry.output_stride, plane.conversion);
                } else {
                    neon.inverse_2d_u16_streamed(
                        *plane.horizontal, *plane.vertical,
                        input, geometry.input_stride,
                        output, geometry.output_stride, plane.conversion);
                }
            }
        }
    });
}

template <class Sample>
[[nodiscard]] Measurement run_neon(
    Fixture<Sample> &fixture, const dsmvc::CpuExecutor &neon,
    BatchPool &pool) {
    Measurement measurement;
    measurement.wall_ms = timed_ms([&] {
        run_cpu_frames(fixture, neon, pool, fixture.batch_size);
    });
    return measurement;
}

template <class Sample>
[[nodiscard]] Measurement run_heterogeneous(
    Fixture<Sample> &fixture, MetalYuvRunner &metal,
    const dsmvc::CpuExecutor &neon, BatchPool &pool,
    DedicatedWorker &gpu_worker) {
    const std::size_t cpu_frames = fixture.heterogeneous_cpu_frames;
    const std::size_t gpu_frames = fixture.batch_size - cpu_frames;
    if (gpu_frames == 0U) return run_neon(fixture, neon, pool);

    Measurement gpu_measurement;
    std::exception_ptr cpu_error;
    std::exception_ptr gpu_error;
    const auto start = Clock::now();
    gpu_worker.start([&] {
        gpu_measurement = metal.run(fixture, cpu_frames, gpu_frames);
    });
    try {
        run_cpu_frames(fixture, neon, pool, cpu_frames);
    } catch (...) {
        cpu_error = std::current_exception();
    }
    try {
        gpu_worker.wait();
    } catch (...) {
        gpu_error = std::current_exception();
    }
    if (cpu_error) std::rethrow_exception(cpu_error);
    if (gpu_error) std::rethrow_exception(gpu_error);
    gpu_measurement.wall_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
    return gpu_measurement;
}

template <class Sample>
[[nodiscard]] Measurement run_route(
    Route route, Fixture<Sample> &fixture, MetalYuvRunner &metal,
    const dsmvc::CpuExecutor &neon, BatchPool &pool,
    DedicatedWorker &gpu_worker) {
    Measurement measurement;
    switch (route) {
    case Route::neon:
        measurement = run_neon(fixture, neon, pool);
        break;
    case Route::metal:
        measurement = metal.run(fixture, 0U, fixture.batch_size);
        break;
    case Route::heterogeneous:
        measurement = run_heterogeneous(
            fixture, metal, neon, pool, gpu_worker);
        break;
    }
    const double divisor = static_cast<double>(fixture.batch_size);
    measurement.wall_ms /= divisor;
    measurement.upload_ms /= divisor;
    measurement.encode_ms /= divisor;
    measurement.submit_wait_ms /= divisor;
    measurement.gpu_ms /= divisor;
    measurement.download_ms /= divisor;
    return measurement;
}

template <class Sample>
[[nodiscard]] ErrorStats compare_output(const Fixture<Sample> &fixture) {
    ErrorStats result;
    for (const auto &plane : fixture.planes) {
        const auto &geometry = plane.geometry;
        const std::size_t frame_elements = static_cast<std::size_t>(
            geometry.destination_height) * geometry.output_stride;
        for (std::size_t frame = 0U; frame < fixture.batch_size; ++frame) {
            for (std::uint32_t row = 0U; row < geometry.destination_height; ++row) {
                for (std::uint32_t column = 0U;
                     column < geometry.destination_width; ++column) {
                    const std::size_t index = frame * frame_elements
                        + static_cast<std::size_t>(row) * geometry.output_stride
                        + column;
                    const auto expected = static_cast<std::uint32_t>(
                        plane.oracle[index]);
                    const auto actual = static_cast<std::uint32_t>(
                        plane.output[index]);
                    const auto difference = expected > actual
                        ? expected - actual : actual - expected;
                    result.differing_samples += expected != actual;
                    result.maximum_error = std::max(
                        result.maximum_error, difference);
                }
            }
        }
    }
    return result;
}

template <class Sample>
[[nodiscard]] std::uint64_t output_identity(const Fixture<Sample> &fixture) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto &plane : fixture.planes) {
        const auto &geometry = plane.geometry;
        const std::size_t frame_elements = static_cast<std::size_t>(
            geometry.destination_height) * geometry.output_stride;
        for (std::size_t frame = 0U; frame < fixture.batch_size; ++frame) {
            for (std::uint32_t row = 0U; row < geometry.destination_height; ++row) {
                const auto *bytes = reinterpret_cast<const unsigned char *>(
                    plane.output.data() + frame * frame_elements
                    + static_cast<std::size_t>(row) * geometry.output_stride);
                const std::size_t count = static_cast<std::size_t>(
                    geometry.destination_width) * sizeof(Sample);
                for (std::size_t index = 0U; index < count; ++index) {
                    hash ^= bytes[index];
                    hash *= 1099511628211ULL;
                }
            }
        }
    }
    return hash;
}

template <class Sample>
[[nodiscard]] Fixture<Sample> make_fixture(
    const CaseSpec &spec, std::size_t case_index,
    const char *format_name, std::uint32_t bits,
    const Configuration &configuration, MetalYuvRunner &metal,
    dsmvc::CpuExecutor &neon) {
    Fixture<Sample> fixture;
    fixture.spec = spec;
    fixture.format_name = format_name;
    fixture.bits_per_sample = bits;
    fixture.batch_size = configuration.batch_size;
    for (std::size_t plane_index = 0U;
         plane_index < fixture.planes.size(); ++plane_index) {
        auto &plane = fixture.planes[plane_index];
        plane.geometry = make_geometry<Sample>(plane_index != 0U);
        plane.conversion = make_conversion(bits, plane_index != 0U);
        plane.horizontal = std::make_shared<const dsmvc::AxisPlan>(
            dsmvc::build_axis_plan(make_request(spec, plane.geometry, true)));
        plane.vertical = std::make_shared<const dsmvc::AxisPlan>(
            dsmvc::build_axis_plan(make_request(spec, plane.geometry, false)));
        const std::size_t input_elements = static_cast<std::size_t>(
            plane.geometry.source_height) * plane.geometry.input_stride
            * configuration.batch_size;
        const std::size_t output_elements = static_cast<std::size_t>(
            plane.geometry.destination_height) * plane.geometry.output_stride
            * configuration.batch_size;
        plane.input.resize(input_elements);
        plane.oracle.resize(output_elements);
        plane.output.resize(output_elements);
        fill_input(plane, case_index, plane_index, configuration.batch_size);
        neon.prepare(plane.horizontal);
        neon.prepare(plane.vertical);
        const PreparedPlane *scratch_owner = plane_index == 0U
            ? nullptr : &fixture.planes[0].metal;
        const PreparedPlane *plan_owner = plane_index == 2U
            ? &fixture.planes[1].metal : nullptr;
        plane.metal = metal.prepare(
            plane, configuration.batch_size, scratch_owner, plan_owner);
    }
    fixture.heterogeneous_cpu_frames =
        fixture.planes[0].horizontal->half_bandwidth >= 5
        ? configuration.wide_cpu_frames : configuration.narrow_cpu_frames;
    return fixture;
}

template <class Sample>
[[nodiscard]] CaseResult benchmark_fixture(
    Fixture<Sample> &fixture, MetalYuvRunner &metal,
    const dsmvc::CpuExecutor &neon, BatchPool &pool,
    DedicatedWorker &gpu_worker, const Configuration &configuration,
    bool &all_correct) {
    for (std::size_t warmup = 0U; warmup < configuration.warmups; ++warmup) {
        for (const Route route : all_routes) {
            (void)run_route(route, fixture, metal, neon, pool, gpu_worker);
        }
    }

    CaseResult result;
    result.name = std::string{fixture.format_name} + '-' + fixture.spec.name;
    result.format = fixture.format_name;
    result.half_bandwidth = fixture.planes[0].horizontal->half_bandwidth;
    result.heterogeneous_cpu_frames = fixture.heterogeneous_cpu_frames;
    result.cpu_concurrency = result.half_bandwidth >= 5
        ? configuration.wide_cpu_concurrency
        : configuration.narrow_cpu_concurrency;
    for (std::size_t plane_index = 0U;
         plane_index < fixture.planes.size(); ++plane_index) {
        const auto &plane = fixture.planes[plane_index];
        result.working_set_bytes += plane.metal.working_set_bytes;
        if (plane_index < 2U) {
            result.plan_buffer_bytes += plane.metal.horizontal.bytes
                + plane.metal.vertical.bytes;
        }
        result.host_copy_bytes += static_cast<std::size_t>(
            plane.geometry.source_height) * plane.geometry.input_stride
            * sizeof(Sample);
        result.host_copy_bytes += static_cast<std::size_t>(
            plane.geometry.destination_height) * plane.geometry.output_stride
            * sizeof(Sample);
        if (plane_index < 2U) {
            result.horizontal_transpose_reuse[plane_index] =
                transpose_reuse_stats(*plane.horizontal);
        }
    }
    result.metal_logical_accesses = logical_metal_accesses(fixture);

    (void)run_route(Route::neon, fixture, metal, neon, pool, gpu_worker);
    for (auto &plane : fixture.planes) plane.oracle = plane.output;
    result.errors[route_index(Route::neon)] = {};
    result.identities[route_index(Route::neon)] = output_identity(fixture);
    for (const Route route : {Route::metal, Route::heterogeneous}) {
        (void)run_route(route, fixture, metal, neon, pool, gpu_worker);
        const std::size_t index = route_index(route);
        result.errors[index] = compare_output(fixture);
        result.identities[index] = output_identity(fixture);
        all_correct = all_correct
            && result.errors[index].maximum_error <= maximum_sample_error;
    }

    for (std::size_t sample = 0U; sample < configuration.samples; ++sample) {
        const std::size_t offset = sample % all_routes.size();
        for (std::size_t slot = 0U; slot < all_routes.size(); ++slot) {
            const Route route = all_routes[(slot + offset) % all_routes.size()];
            result.measurements[route_index(route)].push_back(
                run_route(route, fixture, metal, neon, pool, gpu_worker));
        }
    }

    std::cout << result.name;
    for (const Route route : all_routes) {
        const std::size_t index = route_index(route);
        const Summary wall = summarize_measurements(
            result.measurements[index],
            [](const Measurement &measurement) { return measurement.wall_ms; });
        std::cout << ' ' << route_name(route) << "_wall_ms="
                  << std::fixed << std::setprecision(3) << wall.median
                  << " max_sample_error=" << result.errors[index].maximum_error;
    }
    std::cout << '\n';
    return result;
}

[[nodiscard]] std::string json_string(std::string_view value) {
    std::string result;
    result.reserve(value.size() + 2U);
    result.push_back('"');
    for (const char character : value) {
        switch (character) {
        case '"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default: result.push_back(character); break;
        }
    }
    result.push_back('"');
    return result;
}

void write_summary(std::ostream &output, const Summary &summary) {
    output << "{\"raw\":[";
    for (std::size_t index = 0U; index < summary.raw.size(); ++index) {
        if (index != 0U) output << ',';
        output << summary.raw[index];
    }
    output << "],\"median\":" << summary.median
           << ",\"mad\":" << summary.mad
           << ",\"minimum\":" << summary.minimum
           << ",\"maximum\":" << summary.maximum << '}';
}

void write_transpose_reuse(std::ostream &output,
                           const TransposeReuseStats &stats) {
    const double entries = static_cast<double>(stats.entries);
    const double adjacent_reuse_ratio = entries > 0.0
        ? static_cast<double>(stats.adjacent_reused_entries) / entries : 0.0;
    const double full_reuse_reduction_ratio = entries > 0.0
        ? 1.0 - static_cast<double>(stats.unique_indices) / entries : 0.0;
    output << "{\"entries\":" << stats.entries
           << ",\"adjacent_reused_entries\":"
           << stats.adjacent_reused_entries
           << ",\"one_step_loads\":" << stats.one_step_loads
           << ",\"unique_indices\":" << stats.unique_indices
           << ",\"max_entries_per_destination\":"
           << stats.max_entries_per_destination
           << ",\"max_adjacent_union\":" << stats.max_adjacent_union
           << ",\"noncontiguous_destinations\":"
           << stats.noncontiguous_destinations
           << ",\"max_index_span\":" << stats.max_index_span
           << ",\"max_start_advance\":" << stats.max_start_advance
           << ",\"max_end_advance\":" << stats.max_end_advance
           << ",\"adjacent_reuse_ratio\":" << adjacent_reuse_ratio
           << ",\"full_reuse_reduction_ratio\":"
           << full_reuse_reduction_ratio
           << ",\"monotonic_bounds\":"
           << (stats.monotonic_bounds ? "true" : "false") << '}';
}

void write_results(const Configuration &configuration,
                   const MetalYuvRunner &metal,
                   const std::vector<CaseResult> &results) {
    if (configuration.json_output.empty()) return;
    if (configuration.json_output.has_parent_path()) {
        std::filesystem::create_directories(
            configuration.json_output.parent_path());
    }
    std::ofstream output(configuration.json_output);
    if (!output) throw std::runtime_error("failed to open YUV JSON output");
    output << std::setprecision(17)
           << "{\"schema_version\":1,\"benchmark\":\"dsmvc-metal-yuv-routes\""
           << ",\"device\":{\"name\":" << json_string(metal.device_name())
           << ",\"unified_memory\":true}"
           << ",\"setup_ms\":" << metal.setup_ms()
           << ",\"recipe\":{\"source\":\"1920x1080\""
           << ",\"destination\":\"1692x952\""
           << ",\"subsampling\":\"420\""
           << ",\"range\":\"limited\""
           << ",\"samples\":" << configuration.samples
           << ",\"warmups\":" << configuration.warmups
           << ",\"batch_size\":" << configuration.batch_size
           << ",\"narrow_cpu_frames\":" << configuration.narrow_cpu_frames
           << ",\"wide_cpu_frames\":" << configuration.wide_cpu_frames
           << ",\"narrow_cpu_concurrency\":"
           << configuration.narrow_cpu_concurrency
           << ",\"wide_cpu_concurrency\":"
           << configuration.wide_cpu_concurrency
           << ",\"threads_per_threadgroup\":"
           << configuration.threads_per_threadgroup
           << ",\"source_strides_bytes\":{\"p8\":[1920,960,960]"
              ",\"p10\":[3840,1920,1920]}"
           << ",\"destination_strides_bytes\":{\"p8\":[1696,864,864]"
              ",\"p10\":[3392,1696,1696]}"
           << ",\"stride_source\":\"VapourSynth R78 fixed-geometry probe\""
           << ",\"persistent_shared_buffers\":true"
           << ",\"plan_upload_in_timing\":false"
           << ",\"logical_access_model\":"
              "\"source-level buffer accesses before cache, coalescing, and "
              "compiler effects; fixed B5/B7 recurrence results retained in "
              "per-thread registers; argument buffers excluded\"}"
           << ",\"error_limits\":{\"absolute\":" << maximum_sample_error
           << ",\"relative\":1},\"cases\":[";
    for (std::size_t case_index = 0U; case_index < results.size(); ++case_index) {
        if (case_index != 0U) output << ',';
        const auto &result = results[case_index];
        output << "{\"name\":" << json_string(result.name)
               << ",\"format\":" << json_string(result.format)
               << ",\"half_bandwidth\":" << result.half_bandwidth
               << ",\"working_set_bytes\":" << result.working_set_bytes
               << ",\"working_set_bytes_per_frame\":"
               << result.working_set_bytes / configuration.batch_size
               << ",\"plan_buffer_bytes\":" << result.plan_buffer_bytes
               << ",\"total_requested_metal_buffer_bytes\":"
               << result.working_set_bytes + result.plan_buffer_bytes
               << ",\"horizontal_transpose_reuse\":{"
               << "\"luma\":";
        write_transpose_reuse(output, result.horizontal_transpose_reuse[0]);
        output << ",\"chroma\":";
        write_transpose_reuse(output, result.horizontal_transpose_reuse[1]);
        output << '}'
               << ",\"metal_logical_access_bytes_per_metal_frame\":{"
               << "\"image_reads\":"
               << result.metal_logical_accesses.image_reads
               << ",\"image_writes\":"
               << result.metal_logical_accesses.image_writes
               << ",\"plan_reads\":"
               << result.metal_logical_accesses.plan_reads
               << ",\"total\":" << result.metal_logical_accesses.total()
               << '}'
               << ",\"heterogeneous_cpu_frames\":"
               << result.heterogeneous_cpu_frames
               << ",\"heterogeneous_metal_frames\":"
               << configuration.batch_size - result.heterogeneous_cpu_frames
               << ",\"cpu_concurrency\":" << result.cpu_concurrency
               << ",\"routes\":{";
        for (std::size_t index = 0U; index < all_routes.size(); ++index) {
            if (index != 0U) output << ',';
            const Route route = all_routes[index];
            const auto &error = result.errors[index];
            output << json_string(route_name(route)) << ":{\"correctness\":{"
                   << "\"finite\":true,\"maximum_absolute_error\":"
                   << error.maximum_error
                   << ",\"maximum_relative_error\":0"
                   << ",\"differing_samples\":" << error.differing_samples
                   << ",\"output_identity\":\"" << std::hex
                   << result.identities[index] << std::dec << "\"}"
                   << ",\"host_copy_bytes_per_frame\":"
                   << (route == Route::neon ? 0U
                       : route == Route::metal ? result.host_copy_bytes
                       : result.host_copy_bytes
                           * (configuration.batch_size
                              - result.heterogeneous_cpu_frames)
                           / configuration.batch_size)
                   << ",\"command_buffers_per_batch\":"
                   << (route == Route::neon ? 0 : 1)
                   << ",\"compute_encoders_per_batch\":"
                   << (route == Route::neon ? 0 : 9)
                   << ",\"timings_ms_per_frame\":{";
            const auto &measurements = result.measurements[index];
            const auto emit = [&](const char *name, auto projection, bool &first) {
                if (!first) output << ',';
                first = false;
                output << json_string(name) << ':';
                write_summary(output, summarize_measurements(measurements, projection));
            };
            bool first = true;
            emit("wall", [](const Measurement &value) { return value.wall_ms; }, first);
            emit("upload", [](const Measurement &value) { return value.upload_ms; }, first);
            emit("encode", [](const Measurement &value) { return value.encode_ms; }, first);
            emit("submit_wait", [](const Measurement &value) {
                return value.submit_wait_ms;
            }, first);
            emit("gpu", [](const Measurement &value) { return value.gpu_ms; }, first);
            emit("download", [](const Measurement &value) {
                return value.download_ms;
            }, first);
            output << "}}";
        }
        output << "}}";
    }
    output << "]}\n";
    if (!output) throw std::runtime_error("failed to write YUV JSON output");
}

template <class Sample>
[[nodiscard]] bool profile_format(
    const char *format_name, std::uint32_t bits,
    const Configuration &configuration, MetalYuvRunner &metal,
    BatchPool &narrow_pool, BatchPool &wide_pool,
    DedicatedWorker &gpu_worker) {
    for (std::size_t index = 0U; index < case_specs.size(); ++index) {
        const std::string case_name = std::string{format_name}
            + '-' + case_specs[index].name;
        if (case_name != configuration.profile_case) continue;

        dsmvc::CpuExecutor neon(dsmvc::CpuPath::neon);
        auto fixture = make_fixture<Sample>(
            case_specs[index], index, format_name, bits,
            configuration, metal, neon);
        neon.seal();
        BatchPool &pool = fixture.planes[0].horizontal->half_bandwidth >= 5
            ? wide_pool : narrow_pool;
        const Route route = parse_route(configuration.profile_route);
        for (std::size_t warmup = 0U;
             warmup < configuration.warmups; ++warmup) {
            (void)run_route(
                route, fixture, metal, neon, pool, gpu_worker);
        }

        std::vector<double> wall_samples;
        wall_samples.reserve(configuration.profile_iterations);
        os_log_t profile_log = os_log_create(
            "com.dsmvc.benchmarks", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
        const os_signpost_id_t profile_id =
            os_signpost_id_generate(profile_log);
        os_signpost_interval_begin(
            profile_log, profile_id, "MeasuredLoop",
            "route=%{public}s case=%{public}s batch=%zu iterations=%zu",
            route_name(route), case_name.c_str(), configuration.batch_size,
            configuration.profile_iterations);
        for (std::size_t iteration = 0U;
             iteration < configuration.profile_iterations; ++iteration) {
            wall_samples.push_back(run_route(
                route, fixture, metal, neon, pool, gpu_worker).wall_ms);
        }
        os_signpost_interval_end(
            profile_log, profile_id, "MeasuredLoop");

        const std::uint64_t identity = output_identity(fixture);
        const Summary wall = summarize(std::move(wall_samples));
        std::cout << "profile route=" << route_name(route)
                  << " case=" << case_name
                  << " batch_size=" << configuration.batch_size
                  << " iterations=" << configuration.profile_iterations
                  << " median_ms_per_frame=" << std::fixed
                  << std::setprecision(6) << wall.median
                  << " relative_mad="
                  << (wall.median > 0.0 ? wall.mad / wall.median : 0.0)
                  << " output_sink=" << std::hex << identity << std::dec
                  << '\n';
        return true;
    }
    return false;
}

template <class Sample>
void benchmark_format(const char *format_name, std::uint32_t bits,
                      const Configuration &configuration,
                      MetalYuvRunner &metal, BatchPool &narrow_pool,
                      BatchPool &wide_pool,
                      DedicatedWorker &gpu_worker,
                      std::vector<CaseResult> &results, bool &all_correct) {
    for (std::size_t index = 0U; index < case_specs.size(); ++index) {
        dsmvc::CpuExecutor neon(dsmvc::CpuPath::neon);
        auto fixture = make_fixture<Sample>(
            case_specs[index], index, format_name, bits,
            configuration, metal, neon);
        neon.seal();
        BatchPool &pool = fixture.planes[0].horizontal->half_bandwidth >= 5
            ? wide_pool : narrow_pool;
        results.push_back(benchmark_fixture(
            fixture, metal, neon, pool, gpu_worker,
            configuration, all_correct));
    }
}

} // namespace

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            const Configuration configuration = parse_arguments(argc, argv);
            if (!dsmvc::cpu_neon_compiled() || !dsmvc::cpu_neon_available()) {
                throw std::runtime_error("Metal YUV benchmark requires NEON");
            }
            MetalYuvRunner metal(configuration.threads_per_threadgroup);
            BatchPool narrow_pool(configuration.narrow_cpu_concurrency);
            BatchPool wide_pool(configuration.wide_cpu_concurrency);
            DedicatedWorker gpu_worker;
            if (configuration.profile_iterations != 0U) {
                const bool profiled = profile_format<std::uint8_t>(
                    "yuv420p8", 8U, configuration, metal,
                    narrow_pool, wide_pool, gpu_worker)
                    || profile_format<std::uint16_t>(
                        "yuv420p10", 10U, configuration, metal,
                        narrow_pool, wide_pool, gpu_worker);
                if (!profiled) {
                    throw std::invalid_argument(
                        "--profile-case does not name a YUV benchmark case");
                }
                return EXIT_SUCCESS;
            }
            std::vector<CaseResult> results;
            results.reserve(case_specs.size() * 2U);
            bool all_correct = true;
            benchmark_format<std::uint8_t>(
                "yuv420p8", 8U, configuration, metal,
                narrow_pool, wide_pool, gpu_worker, results, all_correct);
            benchmark_format<std::uint16_t>(
                "yuv420p10", 10U, configuration, metal,
                narrow_pool, wide_pool, gpu_worker, results, all_correct);
            write_results(configuration, metal, results);
            std::cout << "device=" << metal.device_name()
                      << " unified_memory=true setup_ms=" << std::fixed
                      << std::setprecision(3) << metal.setup_ms()
                      << " correctness=" << (all_correct ? "pass" : "fail") << '\n';
            if (configuration.assert_gates && !all_correct) return EXIT_FAILURE;
            return EXIT_SUCCESS;
        } catch (const std::exception &error) {
            std::cerr << "Metal YUV benchmark failure: " << error.what() << '\n';
            return EXIT_FAILURE;
        }
    }
}
