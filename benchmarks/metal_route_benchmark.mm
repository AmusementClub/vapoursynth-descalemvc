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
#include <sstream>
#include <stdexcept>
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
constexpr std::int32_t input_stride = source_width;
constexpr std::int32_t intermediate_stride = 1696;
constexpr std::int32_t output_stride = 1696;
constexpr double horizontal_active_length = 1691.5555555555557;
constexpr double horizontal_shift = 0.2222222222221717;
constexpr double vertical_active_length = 951.5;
constexpr double vertical_shift = 0.25;
constexpr double absolute_error_limit = 5.0e-5;
constexpr double relative_error_limit = 2.0e-3;

enum class Route : std::uint8_t {
    neon_neon,
    metal_metal,
    neon_metal,
    metal_neon,
    heterogeneous,
};

constexpr std::array<Route, 5> all_routes{
    Route::neon_neon,
    Route::metal_metal,
    Route::neon_metal,
    Route::metal_neon,
    Route::heterogeneous,
};

[[nodiscard]] constexpr std::size_t route_index(Route route) noexcept {
    return static_cast<std::size_t>(route);
}

[[nodiscard]] const char *route_name(Route route) noexcept {
    switch (route) {
    case Route::neon_neon: return "neon-neon";
    case Route::metal_metal: return "metal-metal";
    case Route::neon_metal: return "neon-metal";
    case Route::metal_neon: return "metal-neon";
    case Route::heterogeneous: return "neon+metal";
    }
    return "unknown";
}

[[nodiscard]] Route parse_route(std::string_view name) {
    for (const Route route : all_routes) {
        if (name == route_name(route)) return route;
    }
    throw std::invalid_argument(
        "profile route must be neon-neon, metal-metal, neon-metal, "
        "metal-neon, or neon+metal");
}

struct Configuration {
    std::size_t samples = 21U;
    std::size_t warmups = 5U;
    std::size_t threads_per_threadgroup = 32U;
    std::size_t batch_size = 1U;
    std::size_t heterogeneous_cpu_frames = 0U;
    std::size_t profile_iterations = 0U;
    std::string profile_route = "neon+metal";
    std::string profile_case = "b7-spline64";
    bool assert_gates = false;
    std::filesystem::path json_output;
};

struct CaseSpec {
    const char *name = "";
    dsmvc::KernelKind kind = dsmvc::KernelKind::bicubic;
    std::int32_t taps = 0;
};

constexpr std::array<CaseSpec, 4> case_specs{{
    {"b1-bilinear", dsmvc::KernelKind::bilinear, 0},
    {"b3-bicubic", dsmvc::KernelKind::bicubic, 0},
    {"b5-lanczos3", dsmvc::KernelKind::lanczos, 3},
    {"b7-spline64", dsmvc::KernelKind::spline64, 0},
}};

struct Summary {
    std::vector<double> raw;
    double median = 0.0;
    double mad = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

struct Measurement {
    double wall_ms = 0.0;
    double upload_ms = 0.0;
    double cpu_horizontal_ms = 0.0;
    double encode_ms = 0.0;
    double submit_wait_ms = 0.0;
    double gpu_ms = 0.0;
    double download_ms = 0.0;
    double cpu_vertical_ms = 0.0;
};

struct ErrorStats {
    bool finite = true;
    double maximum_absolute = 0.0;
    double maximum_relative = 0.0;
};

struct AxisJob {
    std::uint32_t source_size = 0;
    std::uint32_t destination_size = 0;
    std::uint32_t vector_count = 0;
    std::uint32_t input_stride = 0;
    std::uint32_t output_stride = 0;
    std::uint32_t direction = 0;
    std::uint32_t half_bandwidth = 0;
    std::uint32_t reserved = 0;
    std::uint32_t batch_count = 0;
    std::uint32_t input_frame_stride = 0;
    std::uint32_t output_frame_stride = 0;
    std::uint32_t reserved_2 = 0;
};

static_assert(sizeof(AxisJob) == 48U);

struct PlanBuffers {
    id<MTLBuffer> transpose_offsets = nil;
    id<MTLBuffer> transpose_indices = nil;
    id<MTLBuffer> transpose_weights = nil;
    id<MTLBuffer> lower_ld = nil;
    id<MTLBuffer> upper_l = nil;
    id<MTLBuffer> inverse_diagonal = nil;
    std::uint32_t half_bandwidth = 0;
    std::size_t bytes = 0U;
};

struct PreparedMetalCase {
    PlanBuffers horizontal;
    PlanBuffers vertical;
    id<MTLBuffer> input = nil;
    id<MTLBuffer> intermediate = nil;
    id<MTLBuffer> output = nil;
    std::size_t working_set_bytes = 0U;
};

struct Fixture {
    CaseSpec spec;
    std::shared_ptr<const dsmvc::AxisPlan> horizontal;
    std::shared_ptr<const dsmvc::AxisPlan> vertical;
    std::vector<float> input;
    std::vector<float> oracle;
    std::vector<float> output;
    PreparedMetalCase metal;
    std::size_t batch_size = 1U;
    std::size_t heterogeneous_cpu_frames = 1U;
};

struct CaseResult {
    std::string name;
    std::int32_t half_bandwidth = 0;
    std::size_t plan_bytes = 0U;
    std::size_t working_set_bytes = 0U;
    std::array<std::vector<Measurement>, all_routes.size()> measurements;
    std::array<ErrorStats, all_routes.size()> errors;
    std::array<std::uint64_t, all_routes.size()> identities{};
};

class FrameBatchPool {
    struct JobState {
        JobState(std::size_t count, std::function<void(std::size_t)> function)
            : job(std::move(function)), task_count(count) {}

        std::function<void(std::size_t)> job;
        std::size_t task_count = 0U;
        std::atomic<std::size_t> next_task{1U};
        std::mutex mutex;
        std::exception_ptr error;
    };

public:
    explicit FrameBatchPool(std::size_t parallelism)
        : parallelism_(std::max<std::size_t>(parallelism, 1U)),
          start_barrier_(static_cast<std::ptrdiff_t>(parallelism_)),
          finish_barrier_(static_cast<std::ptrdiff_t>(parallelism_)) {
        workers_.reserve(parallelism_ - 1U);
        for (std::size_t index = 1U; index < parallelism_; ++index) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~FrameBatchPool() {
        stopping_ = true;
        start_barrier_.arrive_and_wait();
        for (auto &worker : workers_) worker.join();
    }

    FrameBatchPool(const FrameBatchPool &) = delete;
    FrameBatchPool &operator=(const FrameBatchPool &) = delete;

    template <class Function>
    void run(std::size_t task_count, Function &&function) {
        if (task_count == 0U || task_count > parallelism_) {
            throw std::invalid_argument("invalid persistent frame-batch task count");
        }
        auto state = std::make_shared<JobState>(
            task_count, std::forward<Function>(function));
        current_state_ = state;
        start_barrier_.arrive_and_wait();
        try {
            state->job(0U);
        } catch (...) {
            const std::scoped_lock lock(state->mutex);
            state->error = std::current_exception();
        }
        finish_barrier_.arrive_and_wait();
        current_state_.reset();
        if (state->error) std::rethrow_exception(state->error);
    }

private:
    void worker_loop() {
        for (;;) {
            start_barrier_.arrive_and_wait();
            if (stopping_) return;
            const auto state = current_state_;
            for (;;) {
                const std::size_t task = state->next_task.fetch_add(
                    1U, std::memory_order_relaxed);
                if (task >= state->task_count) break;
                try {
                    state->job(task);
                } catch (...) {
                    const std::scoped_lock lock(state->mutex);
                    if (!state->error) state->error = std::current_exception();
                }
            }
            finish_barrier_.arrive_and_wait();
        }
    }

    std::size_t parallelism_ = 1U;
    std::barrier<> start_barrier_;
    std::barrier<> finish_barrier_;
    std::vector<std::thread> workers_;
    std::shared_ptr<JobState> current_state_;
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
        } else if (argument == "--json-out") {
            result.json_output = std::filesystem::path{next(argument)};
        } else if (argument == "--threads-per-threadgroup") {
            result.threads_per_threadgroup = parse_positive(next(argument), argument);
        } else if (argument == "--batch-size") {
            result.batch_size = parse_positive(next(argument), argument);
        } else if (argument == "--heterogeneous-cpu-frames") {
            result.heterogeneous_cpu_frames = parse_positive(next(argument), argument);
        } else if (argument == "--profile-iterations") {
            result.profile_iterations = parse_positive(next(argument), argument);
        } else if (argument == "--profile-route") {
            result.profile_route = std::string{next(argument)};
        } else if (argument == "--profile-case") {
            result.profile_case = std::string{next(argument)};
        } else if (argument == "--assert") {
            result.assert_gates = true;
        } else {
            throw std::invalid_argument(
                "usage: dsmvc_metal_route_benchmark [--samples N] "
                "[--warmups N] [--threads-per-threadgroup N] "
                "[--batch-size N] [--heterogeneous-cpu-frames N] "
                "[--profile-iterations N] [--profile-route ROUTE] "
                "[--profile-case CASE] [--json-out PATH] [--assert]");
        }
    }
    if (result.threads_per_threadgroup > 1024U) {
        throw std::invalid_argument("--threads-per-threadgroup exceeds 1024");
    }
    if (result.batch_size > 64U) {
        throw std::invalid_argument("--batch-size exceeds 64");
    }
    if (result.heterogeneous_cpu_frames == 0U) {
        result.heterogeneous_cpu_frames = (result.batch_size + 1U) / 2U;
    }
    if (result.heterogeneous_cpu_frames > result.batch_size) {
        throw std::invalid_argument(
            "--heterogeneous-cpu-frames exceeds --batch-size");
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

[[nodiscard]] Summary summarize(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("cannot summarize no samples");
    for (const double value : values) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::runtime_error("profile sample is not finite and nonnegative");
        }
    }
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t middle = sorted.size() / 2U;
    const double median = (sorted.size() & 1U) != 0U
        ? sorted[middle] : (sorted[middle - 1U] + sorted[middle]) / 2.0;
    std::vector<double> deviations;
    deviations.reserve(sorted.size());
    for (const double value : values) deviations.push_back(std::abs(value - median));
    std::sort(deviations.begin(), deviations.end());
    const double mad = (deviations.size() & 1U) != 0U
        ? deviations[middle]
        : (deviations[middle - 1U] + deviations[middle]) / 2.0;
    return {std::move(values), median, mad, sorted.front(), sorted.back()};
}

template <class Function>
[[nodiscard]] double timed_ms(Function &&function) {
    const auto start = Clock::now();
    std::forward<Function>(function)();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

[[nodiscard]] dsmvc::AxisRequest make_request(const CaseSpec &spec,
                                               bool horizontal) {
    dsmvc::AxisRequest request;
    request.source_size = horizontal ? source_width : source_height;
    request.destination_size = horizontal ? destination_width : destination_height;
    request.active_length = horizontal
        ? horizontal_active_length : vertical_active_length;
    request.shift = horizontal ? horizontal_shift : vertical_shift;
    request.kernel.kind = spec.kind;
    request.kernel.taps = spec.taps;
    request.kernel.b = 0.0;
    request.kernel.c = 0.5;
    request.border = dsmvc::BorderMode::mirror;
    return request;
}

void fill_input(std::vector<float> &input, std::size_t case_index,
                std::size_t batch_size) {
    const std::size_t frame_elements = static_cast<std::size_t>(source_height)
        * input_stride;
    for (std::size_t frame = 0; frame < batch_size; ++frame) {
        const float phase = static_cast<float>(case_index + frame + 1U) * 0.071F;
        for (std::int32_t row = 0; row < source_height; ++row) {
            for (std::int32_t column = 0; column < source_width; ++column) {
                input[frame * frame_elements + static_cast<std::size_t>(row)
                      * input_stride + static_cast<std::size_t>(column)] =
                    std::sin(static_cast<float>(column) * 0.0031F + phase)
                    + 0.25F * std::cos(static_cast<float>(row) * 0.017F - phase)
                    + 0.03125F * static_cast<float>((row + column) & 31);
            }
        }
    }
}

[[nodiscard]] ErrorStats compare_output(const std::vector<float> &reference,
                                        const std::vector<float> &candidate,
                                        std::size_t batch_size) {
    ErrorStats result;
    const std::size_t frame_elements = static_cast<std::size_t>(destination_height)
        * output_stride;
    for (std::size_t frame = 0; frame < batch_size; ++frame) {
        for (std::int32_t row = 0; row < destination_height; ++row) {
            for (std::int32_t column = 0; column < destination_width; ++column) {
                const auto index = frame * frame_elements
                    + static_cast<std::size_t>(row) * output_stride
                    + static_cast<std::size_t>(column);
                const float left = reference[index];
                const float right = candidate[index];
                if (!std::isfinite(left) || !std::isfinite(right)) {
                    result.finite = false;
                    continue;
                }
                const double absolute = std::abs(
                    static_cast<double>(left) - static_cast<double>(right));
                const double relative = absolute / std::max(
                    1.0e-3, std::abs(static_cast<double>(left)));
                result.maximum_absolute = std::max(result.maximum_absolute, absolute);
                result.maximum_relative = std::max(result.maximum_relative, relative);
            }
        }
    }
    return result;
}

[[nodiscard]] std::uint64_t output_identity(const std::vector<float> &output,
                                            std::size_t batch_size) {
    std::uint64_t hash = 1469598103934665603ULL;
    const std::size_t frame_elements = static_cast<std::size_t>(destination_height)
        * output_stride;
    for (std::size_t frame = 0; frame < batch_size; ++frame) {
        for (std::int32_t row = 0; row < destination_height; ++row) {
            const auto *bytes = reinterpret_cast<const unsigned char *>(
                output.data() + frame * frame_elements
                + static_cast<std::ptrdiff_t>(row) * output_stride);
            const std::size_t count = static_cast<std::size_t>(destination_width)
                * sizeof(float);
            for (std::size_t index = 0; index < count; ++index) {
                hash ^= bytes[index];
                hash *= 1099511628211ULL;
            }
        }
    }
    return hash;
}

class MetalRouteRunner {
public:
    MetalRouteRunner(std::size_t threads_per_threadgroup, std::size_t batch_size)
        : threads_per_threadgroup_(threads_per_threadgroup),
          batch_size_(batch_size) {
        const auto start = Clock::now();
        device_ = MTLCreateSystemDefaultDevice();
        if (device_ == nil) throw std::runtime_error("no Metal device is available");
        if (!device_.hasUnifiedMemory) {
            throw std::runtime_error("Metal route benchmark requires unified memory");
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
        pipelines_[0] = make_pipeline(@"inverse_axis_generic");
        pipelines_[1] = make_pipeline(@"inverse_axis_h1");
        pipelines_[2] = make_pipeline(@"inverse_axis_h3");
        pipelines_[3] = make_pipeline(@"inverse_axis_h5");
        pipelines_[4] = make_pipeline(@"inverse_axis_h7");
        setup_ms_ = std::chrono::duration<double, std::milli>(
            Clock::now() - start).count();
    }

    [[nodiscard]] const char *device_name() const noexcept {
        const char *name = device_.name.UTF8String;
        return name == nullptr ? "Metal device" : name;
    }

    [[nodiscard]] std::uint64_t registry_id() const noexcept {
        return device_.registryID;
    }

    [[nodiscard]] std::size_t maximum_buffer_bytes() const noexcept {
        return static_cast<std::size_t>(device_.maxBufferLength);
    }

    [[nodiscard]] double setup_ms() const noexcept { return setup_ms_; }

    [[nodiscard]] PreparedMetalCase prepare(const dsmvc::AxisPlan &horizontal,
                                            const dsmvc::AxisPlan &vertical) {
        PreparedMetalCase result;
        result.horizontal = prepare_plan(horizontal, "horizontal");
        result.vertical = prepare_plan(vertical, "vertical");
        const std::size_t input_bytes = static_cast<std::size_t>(source_height)
            * input_stride * sizeof(float) * batch_size_;
        const std::size_t intermediate_bytes = static_cast<std::size_t>(source_height)
            * intermediate_stride * sizeof(float) * batch_size_;
        const std::size_t output_bytes = static_cast<std::size_t>(destination_height)
            * output_stride * sizeof(float) * batch_size_;
        result.input = make_empty_buffer(input_bytes, @"dsmvc route input");
        result.intermediate = make_empty_buffer(
            intermediate_bytes, @"dsmvc route intermediate");
        result.output = make_empty_buffer(output_bytes, @"dsmvc route output");
        result.working_set_bytes = input_bytes + intermediate_bytes + output_bytes;
        return result;
    }

    [[nodiscard]] Measurement run_metal_metal(
        PreparedMetalCase &prepared, const std::vector<float> &input,
        std::vector<float> &output) {
        return run_metal_metal_frames(
            prepared, input.data(), output.data(), batch_size_);
    }

    [[nodiscard]] Measurement run_metal_metal_frames(
        PreparedMetalCase &prepared, const float *input, float *output,
        std::size_t frame_count) {
        @autoreleasepool {
            Measurement result;
            const auto wall_start = Clock::now();
            const std::size_t input_elements = static_cast<std::size_t>(source_height)
                * input_stride * frame_count;
            const std::size_t output_elements =
                static_cast<std::size_t>(destination_height)
                * output_stride * frame_count;
            result.upload_ms = timed_ms([&] {
                std::memcpy(prepared.input.contents, input,
                            input_elements * sizeof(float));
            });
            id<MTLCommandBuffer> command = [queue_ commandBuffer];
            if (command == nil) throw std::runtime_error("Metal command buffer failed");
            command.label = @"dsmvc Metal-Metal route";
            result.encode_ms = timed_ms([&] {
                encode_horizontal(command, prepared, frame_count);
                encode_vertical(command, prepared, frame_count);
            });
            finish(command, result);
            result.download_ms = timed_ms([&] {
                std::memcpy(output, prepared.output.contents,
                            output_elements * sizeof(float));
            });
            result.wall_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - wall_start).count();
            return result;
        }
    }

    [[nodiscard]] Measurement run_neon_metal(
        PreparedMetalCase &prepared, const Fixture &fixture,
        const dsmvc::CpuExecutor &neon, FrameBatchPool &frame_pool,
        std::vector<float> &output) {
        @autoreleasepool {
            Measurement result;
            const auto wall_start = Clock::now();
            result.cpu_horizontal_ms = timed_ms([&] {
                frame_pool.run(batch_size_, [&](std::size_t frame) {
                    const std::size_t input_frame = static_cast<std::size_t>(source_height)
                        * input_stride;
                    const std::size_t intermediate_frame =
                        static_cast<std::size_t>(source_height) * intermediate_stride;
                    neon.inverse_rows(
                        *fixture.horizontal,
                        fixture.input.data() + frame * input_frame, input_stride,
                        static_cast<float *>(prepared.intermediate.contents)
                            + frame * intermediate_frame,
                        intermediate_stride, source_height);
                });
            });
            id<MTLCommandBuffer> command = [queue_ commandBuffer];
            if (command == nil) throw std::runtime_error("Metal command buffer failed");
            command.label = @"dsmvc NEON-Metal route";
            result.encode_ms = timed_ms([&] {
                encode_vertical(command, prepared, batch_size_);
            });
            finish(command, result);
            result.download_ms = timed_ms([&] {
                std::memcpy(output.data(), prepared.output.contents,
                            output.size() * sizeof(float));
            });
            result.wall_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - wall_start).count();
            return result;
        }
    }

    [[nodiscard]] Measurement run_metal_neon(
        PreparedMetalCase &prepared, const Fixture &fixture,
        const dsmvc::CpuExecutor &neon, FrameBatchPool &frame_pool,
        std::vector<float> &output) {
        @autoreleasepool {
            Measurement result;
            const auto wall_start = Clock::now();
            result.upload_ms = timed_ms([&] {
                std::memcpy(prepared.input.contents, fixture.input.data(),
                            fixture.input.size() * sizeof(float));
            });
            id<MTLCommandBuffer> command = [queue_ commandBuffer];
            if (command == nil) throw std::runtime_error("Metal command buffer failed");
            command.label = @"dsmvc Metal-NEON route";
            result.encode_ms = timed_ms([&] {
                encode_horizontal(command, prepared, batch_size_);
            });
            finish(command, result);
            result.cpu_vertical_ms = timed_ms([&] {
                frame_pool.run(batch_size_, [&](std::size_t frame) {
                    const std::size_t intermediate_frame =
                        static_cast<std::size_t>(source_height) * intermediate_stride;
                    const std::size_t output_frame =
                        static_cast<std::size_t>(destination_height) * output_stride;
                    neon.inverse_columns(
                        *fixture.vertical,
                        static_cast<const float *>(prepared.intermediate.contents)
                            + frame * intermediate_frame,
                        intermediate_stride,
                        output.data() + frame * output_frame, output_stride,
                        destination_width);
                });
            });
            result.wall_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - wall_start).count();
            return result;
        }
    }

private:
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLLibrary> library_ = nil;
    std::array<id<MTLComputePipelineState>, 5> pipelines_{};
    std::size_t threads_per_threadgroup_ = 32U;
    std::size_t batch_size_ = 1U;
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

    [[nodiscard]] id<MTLBuffer> make_empty_buffer(std::size_t bytes,
                                                  NSString *label) {
        if (bytes == 0U || bytes > maximum_buffer_bytes()) {
            throw std::length_error("invalid Metal working-buffer size");
        }
        id<MTLBuffer> buffer = [device_ newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
        if (buffer == nil) throw std::runtime_error("Metal buffer allocation failed");
        buffer.label = label;
        return buffer;
    }

    template <class T>
    [[nodiscard]] id<MTLBuffer> make_plan_buffer(const std::vector<T> &values,
                                                 NSString *label) {
        if (values.empty()) throw std::invalid_argument("Metal plan buffer is empty");
        const std::size_t bytes = values.size() * sizeof(T);
        id<MTLBuffer> buffer = [device_ newBufferWithBytes:values.data()
                                                   length:bytes
                                                  options:MTLResourceStorageModeShared];
        if (buffer == nil) throw std::runtime_error("Metal plan upload failed");
        buffer.label = label;
        return buffer;
    }

    [[nodiscard]] PlanBuffers prepare_plan(const dsmvc::AxisPlan &plan,
                                           std::string_view direction) {
        if (!plan.valid()) throw std::invalid_argument("cannot upload invalid axis plan");
        PlanBuffers result;
        const std::string prefix = "dsmvc " + std::string{direction} + " ";
        const auto label = [&](std::string_view suffix) {
            const std::string text = prefix + std::string{suffix};
            NSString *value = [NSString stringWithUTF8String:text.c_str()];
            if (value == nil) throw std::runtime_error("Metal label is not UTF-8");
            return value;
        };
        result.transpose_offsets = make_plan_buffer(
            plan.transpose_offsets, label("transpose offsets"));
        result.transpose_indices = make_plan_buffer(
            plan.transpose_indices, label("transpose indices"));
        result.transpose_weights = make_plan_buffer(
            plan.transpose_weights, label("transpose weights"));
        result.lower_ld = make_plan_buffer(plan.lower_ld, label("lower LD"));
        result.upper_l = make_plan_buffer(plan.upper_l, label("upper L"));
        result.inverse_diagonal = make_plan_buffer(
            plan.inverse_diagonal, label("inverse diagonal"));
        result.half_bandwidth = static_cast<std::uint32_t>(plan.half_bandwidth);
        result.bytes = plan.transpose_offsets.size() * sizeof(std::uint32_t)
            + plan.transpose_indices.size() * sizeof(std::int32_t)
            + plan.transpose_weights.size() * sizeof(float)
            + plan.lower_ld.size() * sizeof(float)
            + plan.upper_l.size() * sizeof(float)
            + plan.inverse_diagonal.size() * sizeof(float);
        return result;
    }

    [[nodiscard]] id<MTLComputePipelineState> pipeline(
        std::uint32_t half_bandwidth) const {
        switch (half_bandwidth) {
        case 1U: return pipelines_[1];
        case 3U: return pipelines_[2];
        case 5U: return pipelines_[3];
        case 7U: return pipelines_[4];
        default: return pipelines_[0];
        }
    }

    void encode_axis(id<MTLCommandBuffer> command, id<MTLBuffer> input,
                     id<MTLBuffer> output, const PlanBuffers &plan,
                     const AxisJob &job, NSString *label) {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) throw std::runtime_error("Metal encoder creation failed");
        encoder.label = label;
        id<MTLComputePipelineState> state = pipeline(plan.half_bandwidth);
        [encoder setComputePipelineState:state];
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBytes:&job length:sizeof(job) atIndex:1];
        [encoder setBuffer:plan.transpose_offsets offset:0 atIndex:2];
        [encoder setBuffer:plan.transpose_indices offset:0 atIndex:3];
        [encoder setBuffer:plan.transpose_weights offset:0 atIndex:4];
        [encoder setBuffer:plan.lower_ld offset:0 atIndex:5];
        [encoder setBuffer:plan.upper_l offset:0 atIndex:6];
        [encoder setBuffer:plan.inverse_diagonal offset:0 atIndex:7];
        [encoder setBuffer:output offset:0 atIndex:8];
        const NSUInteger threads = std::min<NSUInteger>(
            static_cast<NSUInteger>(threads_per_threadgroup_),
            state.maxTotalThreadsPerThreadgroup);
        const std::size_t dispatch_count = static_cast<std::size_t>(job.vector_count)
            * job.batch_count;
        if (dispatch_count > std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("Metal batch dispatch exceeds 32-bit range");
        }
        [encoder dispatchThreads:MTLSizeMake(dispatch_count, 1U, 1U)
             threadsPerThreadgroup:MTLSizeMake(threads, 1U, 1U)];
        [encoder endEncoding];
    }

    void encode_horizontal(id<MTLCommandBuffer> command,
                           const PreparedMetalCase &prepared,
                           std::size_t frame_count) {
        const AxisJob job{
            static_cast<std::uint32_t>(source_width),
            static_cast<std::uint32_t>(destination_width),
            static_cast<std::uint32_t>(source_height),
            static_cast<std::uint32_t>(input_stride),
            static_cast<std::uint32_t>(intermediate_stride),
            0U,
            prepared.horizontal.half_bandwidth,
            0U,
            static_cast<std::uint32_t>(frame_count),
            static_cast<std::uint32_t>(source_height * input_stride),
            static_cast<std::uint32_t>(source_height * intermediate_stride),
            0U,
        };
        encode_axis(command, prepared.input, prepared.intermediate,
                    prepared.horizontal, job, @"dsmvc horizontal inverse");
    }

    void encode_vertical(id<MTLCommandBuffer> command,
                         const PreparedMetalCase &prepared,
                         std::size_t frame_count) {
        const AxisJob job{
            static_cast<std::uint32_t>(source_height),
            static_cast<std::uint32_t>(destination_height),
            static_cast<std::uint32_t>(destination_width),
            static_cast<std::uint32_t>(intermediate_stride),
            static_cast<std::uint32_t>(output_stride),
            1U,
            prepared.vertical.half_bandwidth,
            0U,
            static_cast<std::uint32_t>(frame_count),
            static_cast<std::uint32_t>(source_height * intermediate_stride),
            static_cast<std::uint32_t>(destination_height * output_stride),
            0U,
        };
        encode_axis(command, prepared.intermediate, prepared.output,
                    prepared.vertical, job, @"dsmvc vertical inverse");
    }

    static void finish(id<MTLCommandBuffer> command, Measurement &measurement) {
        measurement.submit_wait_ms = timed_ms([&] {
            [command commit];
            [command waitUntilCompleted];
        });
        if (command.status == MTLCommandBufferStatusError) {
            throw std::runtime_error(ns_error(command.error, "Metal execution failed"));
        }
        const CFTimeInterval start = command.GPUStartTime;
        const CFTimeInterval end = command.GPUEndTime;
        if (end >= start && start > 0.0) measurement.gpu_ms = (end - start) * 1000.0;
    }

};

[[nodiscard]] Measurement run_neon_neon(
    const Fixture &fixture, const dsmvc::CpuExecutor &neon,
    FrameBatchPool &frame_pool, std::vector<float> &output) {
    Measurement result;
    result.wall_ms = timed_ms([&] {
        const std::size_t input_frame = static_cast<std::size_t>(source_height)
            * input_stride;
        const std::size_t output_frame = static_cast<std::size_t>(destination_height)
            * output_stride;
        frame_pool.run(fixture.batch_size, [&](std::size_t frame) {
            neon.inverse_2d(
                *fixture.horizontal, *fixture.vertical,
                fixture.input.data() + frame * input_frame, input_stride,
                output.data() + frame * output_frame, output_stride);
        });
    });
    return result;
}

[[nodiscard]] Measurement run_heterogeneous(
    Fixture &fixture, MetalRouteRunner &metal,
    const dsmvc::CpuExecutor &neon, FrameBatchPool &frame_pool) {
    Measurement result;
    const std::size_t cpu_frames = fixture.heterogeneous_cpu_frames;
    const std::size_t gpu_frames = fixture.batch_size - cpu_frames;
    if (gpu_frames == 0U) {
        return run_neon_neon(fixture, neon, frame_pool, fixture.output);
    }

    const std::size_t input_frame = static_cast<std::size_t>(source_height)
        * input_stride;
    const std::size_t output_frame = static_cast<std::size_t>(destination_height)
        * output_stride;
    std::exception_ptr gpu_error;
    Measurement gpu_measurement;
    const auto wall_start = Clock::now();
    {
        std::jthread gpu_worker([&] {
            try {
                gpu_measurement = metal.run_metal_metal_frames(
                    fixture.metal,
                    fixture.input.data() + cpu_frames * input_frame,
                    fixture.output.data() + cpu_frames * output_frame,
                    gpu_frames);
            } catch (...) {
                gpu_error = std::current_exception();
            }
        });

        frame_pool.run(cpu_frames, [&](std::size_t frame) {
            neon.inverse_2d(
                *fixture.horizontal, *fixture.vertical,
                fixture.input.data() + frame * input_frame, input_stride,
                fixture.output.data() + frame * output_frame, output_stride);
        });
    }
    if (gpu_error) std::rethrow_exception(gpu_error);
    result = gpu_measurement;
    result.wall_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - wall_start).count();
    return result;
}

[[nodiscard]] Measurement run_route(
    Route route, Fixture &fixture, MetalRouteRunner &metal,
    const dsmvc::CpuExecutor &neon, FrameBatchPool &frame_pool) {
    Measurement result;
    switch (route) {
    case Route::neon_neon:
        result = run_neon_neon(fixture, neon, frame_pool, fixture.output);
        break;
    case Route::metal_metal:
        result = metal.run_metal_metal(
            fixture.metal, fixture.input, fixture.output);
        break;
    case Route::neon_metal:
        result = metal.run_neon_metal(
            fixture.metal, fixture, neon, frame_pool, fixture.output);
        break;
    case Route::metal_neon:
        result = metal.run_metal_neon(
            fixture.metal, fixture, neon, frame_pool, fixture.output);
        break;
    case Route::heterogeneous:
        result = run_heterogeneous(fixture, metal, neon, frame_pool);
        break;
    }
    const double divisor = static_cast<double>(fixture.batch_size);
    result.wall_ms /= divisor;
    result.upload_ms /= divisor;
    result.cpu_horizontal_ms /= divisor;
    result.encode_ms /= divisor;
    result.submit_wait_ms /= divisor;
    result.gpu_ms /= divisor;
    result.download_ms /= divisor;
    result.cpu_vertical_ms /= divisor;
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
    for (std::size_t index = 0; index < summary.raw.size(); ++index) {
        if (index != 0U) output << ',';
        output << summary.raw[index];
    }
    output << "],\"median\":" << summary.median
           << ",\"mad\":" << summary.mad
           << ",\"minimum\":" << summary.minimum
           << ",\"maximum\":" << summary.maximum << '}';
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

[[nodiscard]] std::size_t host_copy_bytes(
    Route route, std::size_t batch_size,
    std::size_t heterogeneous_cpu_frames) noexcept {
    const std::size_t upload = static_cast<std::size_t>(source_height)
        * input_stride * sizeof(float);
    const std::size_t download = static_cast<std::size_t>(destination_height)
        * output_stride * sizeof(float);
    switch (route) {
    case Route::neon_neon: return 0U;
    case Route::metal_metal: return upload + download;
    case Route::neon_metal: return download;
    case Route::metal_neon: return upload;
    case Route::heterogeneous:
        return (batch_size - heterogeneous_cpu_frames)
            * (upload + download) / batch_size;
    }
    return 0U;
}

void write_results(const Configuration &configuration,
                   const MetalRouteRunner &metal,
                   const std::vector<CaseResult> &results) {
    if (configuration.json_output.empty()) return;
    if (configuration.json_output.has_parent_path()) {
        std::filesystem::create_directories(
            configuration.json_output.parent_path());
    }
    std::ofstream output(configuration.json_output);
    if (!output) throw std::runtime_error("failed to open Metal route JSON output");
    output << std::setprecision(17)
           << "{\"schema_version\":1,\"benchmark\":\"dsmvc-metal-routes\""
           << ",\"device\":{\"name\":" << json_string(metal.device_name())
           << ",\"registry_id\":" << metal.registry_id()
           << ",\"unified_memory\":true"
           << ",\"maximum_buffer_bytes\":" << metal.maximum_buffer_bytes()
           << "},\"setup_ms\":" << metal.setup_ms()
           << ",\"recipe\":{\"source\":\"1920x1080\""
           << ",\"destination\":\"1692x952\""
           << ",\"samples\":" << configuration.samples
           << ",\"warmups\":" << configuration.warmups
           << ",\"threads_per_threadgroup\":"
           << configuration.threads_per_threadgroup
           << ",\"batch_size\":" << configuration.batch_size
           << ",\"heterogeneous_cpu_frames\":"
           << configuration.heterogeneous_cpu_frames
           << ",\"heterogeneous_metal_frames\":"
           << configuration.batch_size - configuration.heterogeneous_cpu_frames
           << ",\"plan_upload_in_timing\":false"
           << ",\"persistent_shared_buffers\":true}"
           << ",\"error_limits\":{\"absolute\":" << absolute_error_limit
           << ",\"relative\":" << relative_error_limit << "},\"cases\":[";
    for (std::size_t case_index = 0; case_index < results.size(); ++case_index) {
        if (case_index != 0U) output << ',';
        const auto &result = results[case_index];
        output << "{\"name\":" << json_string(result.name)
               << ",\"half_bandwidth\":" << result.half_bandwidth
               << ",\"plan_bytes\":" << result.plan_bytes
               << ",\"working_set_bytes\":" << result.working_set_bytes
               << ",\"working_set_bytes_per_frame\":"
               << result.working_set_bytes / configuration.batch_size
               << ",\"routes\":{";
        for (std::size_t index = 0; index < all_routes.size(); ++index) {
            if (index != 0U) output << ',';
            const Route route = all_routes[index];
            const auto &measurements = result.measurements[index];
            const auto &error = result.errors[index];
            output << json_string(route_name(route)) << ":{\"correctness\":{"
                   << "\"finite\":" << (error.finite ? "true" : "false")
                   << ",\"maximum_absolute_error\":" << error.maximum_absolute
                   << ",\"maximum_relative_error\":" << error.maximum_relative
                   << ",\"output_identity\":\"" << std::hex
                   << result.identities[index] << std::dec << "\"}"
                   << ",\"host_copy_bytes_per_frame\":"
                   << host_copy_bytes(
                          route, configuration.batch_size,
                          configuration.heterogeneous_cpu_frames)
                   << ",\"command_buffers_per_batch\":"
                   << (route == Route::neon_neon
                       || (route == Route::heterogeneous
                           && configuration.heterogeneous_cpu_frames
                               == configuration.batch_size) ? 0 : 1)
                   << ",\"compute_encoders_per_batch\":"
                   << (route == Route::metal_metal
                           || (route == Route::heterogeneous
                               && configuration.heterogeneous_cpu_frames
                                   != configuration.batch_size) ? 2
                       : route == Route::neon_neon ? 0 : 1)
                   << ",\"timings_ms_per_frame\":{";
            const auto emit = [&](const char *name, auto projection, bool &first) {
                if (!first) output << ',';
                first = false;
                output << json_string(name) << ':';
                write_summary(output, summarize_measurements(measurements, projection));
            };
            bool first = true;
            emit("wall", [](const Measurement &m) { return m.wall_ms; }, first);
            emit("upload", [](const Measurement &m) { return m.upload_ms; }, first);
            emit("cpu_horizontal", [](const Measurement &m) {
                return m.cpu_horizontal_ms;
            }, first);
            emit("encode", [](const Measurement &m) { return m.encode_ms; }, first);
            emit("submit_wait", [](const Measurement &m) {
                return m.submit_wait_ms;
            }, first);
            emit("gpu", [](const Measurement &m) { return m.gpu_ms; }, first);
            emit("download", [](const Measurement &m) { return m.download_ms; }, first);
            emit("cpu_vertical", [](const Measurement &m) {
                return m.cpu_vertical_ms;
            }, first);
            output << "}}";
        }
        output << "}}";
    }
    output << "]}\n";
    if (!output) throw std::runtime_error("failed to write Metal route JSON output");
}

[[nodiscard]] bool correctness_pass(const ErrorStats &error) noexcept {
    return error.finite
        && error.maximum_absolute <= absolute_error_limit
        && error.maximum_relative <= relative_error_limit;
}

} // namespace

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            const Configuration configuration = parse_arguments(argc, argv);
            if (!dsmvc::cpu_neon_compiled() || !dsmvc::cpu_neon_available()) {
                throw std::runtime_error("Metal route benchmark requires NEON");
            }

            MetalRouteRunner metal(
                configuration.threads_per_threadgroup, configuration.batch_size);
            dsmvc::CpuExecutor scalar(dsmvc::CpuPath::scalar);
            dsmvc::CpuExecutor neon(dsmvc::CpuPath::neon);
            std::vector<Fixture> fixtures;
            fixtures.reserve(case_specs.size());
            for (std::size_t index = 0; index < case_specs.size(); ++index) {
                if (configuration.profile_iterations != 0U
                    && configuration.profile_case != case_specs[index].name) {
                    continue;
                }
                Fixture fixture;
                fixture.spec = case_specs[index];
                fixture.horizontal = std::make_shared<const dsmvc::AxisPlan>(
                    dsmvc::build_axis_plan(make_request(fixture.spec, true)));
                fixture.vertical = std::make_shared<const dsmvc::AxisPlan>(
                    dsmvc::build_axis_plan(make_request(fixture.spec, false)));
                fixture.input.resize(
                    static_cast<std::size_t>(source_height) * input_stride
                    * configuration.batch_size);
                fixture.oracle.resize(
                    static_cast<std::size_t>(destination_height) * output_stride
                    * configuration.batch_size);
                fixture.output.resize(fixture.oracle.size());
                fixture.batch_size = configuration.batch_size;
                fixture.heterogeneous_cpu_frames =
                    configuration.heterogeneous_cpu_frames;
                fill_input(fixture.input, index, configuration.batch_size);
                scalar.prepare(fixture.horizontal);
                scalar.prepare(fixture.vertical);
                neon.prepare(fixture.horizontal);
                neon.prepare(fixture.vertical);
                fixture.metal = metal.prepare(*fixture.horizontal, *fixture.vertical);
                fixtures.push_back(std::move(fixture));
            }
            if (fixtures.empty()) {
                throw std::invalid_argument(
                    "--profile-case does not name a benchmark case");
            }
            scalar.seal();
            neon.seal();
            FrameBatchPool frame_pool(configuration.batch_size);

            if (configuration.profile_iterations != 0U) {
                Fixture &fixture = fixtures.front();
                const Route route = parse_route(configuration.profile_route);
                for (std::size_t warmup = 0;
                     warmup < configuration.warmups; ++warmup) {
                    (void)run_route(route, fixture, metal, neon, frame_pool);
                }
                std::vector<double> wall_samples;
                wall_samples.reserve(configuration.profile_iterations);
                std::uint64_t identity = 0U;
                os_log_t profile_log = os_log_create(
                    "com.dsmvc.benchmarks", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
                const os_signpost_id_t profile_id =
                    os_signpost_id_generate(profile_log);
                os_signpost_interval_begin(
                    profile_log, profile_id, "MeasuredLoop",
                    "route=%{public}s case=%{public}s batch=%zu iterations=%zu",
                    route_name(route), fixture.spec.name,
                    configuration.batch_size, configuration.profile_iterations);
                for (std::size_t iteration = 0;
                     iteration < configuration.profile_iterations; ++iteration) {
                    const Measurement measurement = run_route(
                        route, fixture, metal, neon, frame_pool);
                    wall_samples.push_back(measurement.wall_ms);
                    identity ^= output_identity(fixture.output, fixture.batch_size);
                }
                os_signpost_interval_end(
                    profile_log, profile_id, "MeasuredLoop",
                    "output_sink=%{public}llx",
                    static_cast<unsigned long long>(identity));
                const Summary wall = summarize(std::move(wall_samples));
                std::cout << "profile route=" << route_name(route)
                          << " case=" << fixture.spec.name
                          << " batch_size=" << configuration.batch_size
                          << " iterations=" << configuration.profile_iterations
                          << " median_ms_per_frame=" << std::fixed
                          << std::setprecision(6) << wall.median
                          << " relative_mad="
                          << (wall.median > 0.0 ? wall.mad / wall.median : 0.0)
                          << " output_sink=" << std::hex << identity << std::dec
                          << '\n';
                return EXIT_SUCCESS;
            }

            std::vector<CaseResult> results;
            results.reserve(fixtures.size());
            bool all_correct = true;
            for (Fixture &fixture : fixtures) {
                const std::size_t input_frame = static_cast<std::size_t>(source_height)
                    * input_stride;
                const std::size_t output_frame =
                    static_cast<std::size_t>(destination_height) * output_stride;
                for (std::size_t frame = 0; frame < configuration.batch_size; ++frame) {
                    scalar.inverse_2d(
                        *fixture.horizontal, *fixture.vertical,
                        fixture.input.data() + frame * input_frame, input_stride,
                        fixture.oracle.data() + frame * output_frame, output_stride);
                }

                for (std::size_t warmup = 0; warmup < configuration.warmups; ++warmup) {
                    for (const Route route : all_routes) {
                        (void)run_route(route, fixture, metal, neon, frame_pool);
                    }
                }

                CaseResult result;
                result.name = fixture.spec.name;
                result.half_bandwidth = fixture.horizontal->half_bandwidth;
                result.plan_bytes = fixture.metal.horizontal.bytes
                    + fixture.metal.vertical.bytes;
                result.working_set_bytes = fixture.metal.working_set_bytes;
                for (const Route route : all_routes) {
                    (void)run_route(route, fixture, metal, neon, frame_pool);
                    const std::size_t index = route_index(route);
                    result.errors[index] = compare_output(
                        fixture.oracle, fixture.output, fixture.batch_size);
                    result.identities[index] = output_identity(
                        fixture.output, fixture.batch_size);
                    all_correct = all_correct
                        && correctness_pass(result.errors[index]);
                }

                for (std::size_t sample = 0; sample < configuration.samples; ++sample) {
                    const std::size_t offset = sample % all_routes.size();
                    for (std::size_t slot = 0; slot < all_routes.size(); ++slot) {
                        const Route route = all_routes[(slot + offset) % all_routes.size()];
                        result.measurements[route_index(route)].push_back(
                            run_route(route, fixture, metal, neon, frame_pool));
                    }
                }

                std::cout << result.name
                          << " half_bandwidth=" << result.half_bandwidth
                          << " batch_size=" << configuration.batch_size;
                for (const Route route : all_routes) {
                    const auto index = route_index(route);
                    const Summary wall = summarize_measurements(
                        result.measurements[index],
                        [](const Measurement &m) { return m.wall_ms; });
                    const Summary gpu = summarize_measurements(
                        result.measurements[index],
                        [](const Measurement &m) { return m.gpu_ms; });
                    std::cout << ' ' << route_name(route)
                              << "_wall_ms=" << std::fixed << std::setprecision(3)
                              << wall.median
                              << " gpu_ms=" << gpu.median
                              << " max_abs=" << std::scientific
                              << result.errors[index].maximum_absolute;
                }
                std::cout << '\n';
                results.push_back(std::move(result));
            }

            write_results(configuration, metal, results);
            std::cout << "device=" << metal.device_name()
                      << " unified_memory=true setup_ms=" << std::fixed
                      << std::setprecision(3) << metal.setup_ms()
                      << " correctness=" << (all_correct ? "pass" : "fail") << '\n';
            if (configuration.assert_gates && !all_correct) return EXIT_FAILURE;
            return EXIT_SUCCESS;
        } catch (const std::exception &error) {
            std::cerr << "Metal route benchmark failure: " << error.what() << '\n';
            return EXIT_FAILURE;
        }
    }
}
