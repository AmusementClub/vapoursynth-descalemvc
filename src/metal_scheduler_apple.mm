#include "metal_scheduler_apple.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <os/signpost.h>

#include <dispatch/dispatch.h>

#include "dsmvc_metal_routes_metallib.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dsmvc::metal {

struct Client::Impl {
    explicit Impl(std::uint64_t identity_value) : identity(identity_value) {}

    const std::uint64_t identity;
    std::once_flag release_once;
    std::mutex mutex;
    std::condition_variable ready;
    std::size_t active = 0;
    bool closing = false;
    bool used = false;
};

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t arena_alignment = 256U;
constexpr std::size_t ring_size = 3U;
constexpr std::size_t narrow_gpu_batch = 4U;
constexpr std::size_t wide_gpu_batch = 7U;
constexpr std::size_t scheduling_period = 16U;
constexpr std::size_t narrow_cpu_frames = 12U;
constexpr std::size_t wide_cpu_frames = 7U;
constexpr std::size_t maximum_queued_jobs = 64U;
constexpr std::size_t maximum_cached_plans = 128U;
constexpr std::size_t maximum_batch_descriptors = 48U;
constexpr std::size_t shared_input_clients = 4U;
constexpr std::size_t maximum_recent_inputs = 64U;
constexpr std::size_t maximum_recent_clients = 16U;
constexpr std::uint64_t automatic_work_floor = 1920ULL * 1080ULL * 6ULL;
constexpr std::uint64_t shared_input_work_floor = 1920ULL * 1080ULL * 2ULL;
constexpr auto batch_timeout = std::chrono::microseconds{500};
constexpr auto recent_input_window = std::chrono::milliseconds{50};
constexpr std::size_t resident_alignment = 256U;
constexpr std::size_t resident_tile = 16U;

std::atomic<bool> fail_next_resident_producer{false};

[[nodiscard]] std::uint64_t next_client_identity() noexcept {
    static std::atomic<std::uint64_t> sequence{1U};
    return sequence.fetch_add(1U, std::memory_order_relaxed);
}

struct AxisJob {
    std::uint32_t source_size = 0;
    std::uint32_t destination_size = 0;
    std::uint32_t vector_count = 0;
    std::uint32_t input_stride = 0;
    std::uint32_t output_stride = 0;
    std::uint32_t direction = 0;
    std::uint32_t half_bandwidth = 0;
    std::uint32_t reserved = 0;
    std::uint32_t batch_count = 1;
    std::uint32_t input_frame_stride = 0;
    std::uint32_t output_frame_stride = 0;
    std::uint32_t reserved_2 = 0;
};

struct AxisBatchJob {
    AxisJob axis;
    std::uint32_t input_offset = 0;
    std::uint32_t output_offset = 0;
    std::uint32_t transpose_offsets_offset = 0;
    std::uint32_t transpose_indices_offset = 0;
    std::uint32_t transpose_weights_offset = 0;
    std::uint32_t lower_ld_offset = 0;
    std::uint32_t upper_l_offset = 0;
    std::uint32_t inverse_diagonal_offset = 0;
};

struct ConvertJob {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t input_stride = 0;
    std::uint32_t output_stride = 0;
    std::uint32_t batch_count = 1;
    std::uint32_t input_frame_stride = 0;
    std::uint32_t output_frame_stride = 0;
    std::uint32_t reserved = 0;
};

static_assert(sizeof(AxisJob) == 48U);
static_assert(sizeof(AxisBatchJob) == 80U);
static_assert(sizeof(ConvertJob) == 32U);

[[nodiscard]] std::size_t checked_add(
    std::size_t left, std::size_t right, const char *label) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::length_error(std::string{label} + " size overflow");
    }
    return left + right;
}

[[nodiscard]] std::size_t checked_multiply(
    std::size_t left, std::size_t right, const char *label) {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::length_error(std::string{label} + " size overflow");
    }
    return left * right;
}

[[nodiscard]] std::size_t checked_align_up(
    std::size_t value, std::size_t alignment, const char *label) {
    if (alignment == 0U) {
        throw std::invalid_argument(std::string{label} + " alignment is zero");
    }
    return checked_multiply(
        checked_add(value, alignment - 1U, label) / alignment,
        alignment, label);
}

[[nodiscard]] std::uint32_t checked_u32(
    std::size_t value, const char *label) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error(std::string{label} + " exceeds 32-bit Metal range");
    }
    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::string ns_error(NSError *error, const char *fallback) {
    if (error == nil || error.localizedDescription == nil) return fallback;
    const char *text = error.localizedDescription.UTF8String;
    return text == nullptr ? fallback : text;
}

[[nodiscard]] os_log_t profile_log() noexcept {
    static os_log_t log = os_log_create(
        "com.dsmvc.plugin", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
    return log;
}

struct CopyStats {
    std::size_t calls = 0;
    std::size_t bytes = 0;
};

void copy_rows(
    void *destination, std::ptrdiff_t destination_stride,
    const void *source, std::ptrdiff_t source_stride,
    std::size_t row_bytes, std::uint32_t rows, CopyStats &stats) {
    if (!destination || !source || rows == 0U
        || destination_stride < static_cast<std::ptrdiff_t>(row_bytes)
        || source_stride < static_cast<std::ptrdiff_t>(row_bytes)) {
        throw std::invalid_argument("invalid Metal staging plane");
    }
    auto *destination_bytes = static_cast<std::byte *>(destination);
    const auto *source_bytes = static_cast<const std::byte *>(source);
    if (destination_stride == source_stride) {
        const std::size_t bytes = (static_cast<std::size_t>(rows) - 1U)
            * static_cast<std::size_t>(source_stride) + row_bytes;
        std::memcpy(destination_bytes, source_bytes, bytes);
        ++stats.calls;
        stats.bytes += bytes;
        return;
    }
    for (std::uint32_t row = 0; row < rows; ++row) {
        std::memcpy(
            destination_bytes + static_cast<std::ptrdiff_t>(row)
                * destination_stride,
            source_bytes + static_cast<std::ptrdiff_t>(row) * source_stride,
            row_bytes);
    }
    stats.calls += rows;
    stats.bytes += static_cast<std::size_t>(rows) * row_bytes;
}

struct Request {
    enum class State { pending, cpu, submitted, complete, failed };

    FrameJob job;
    std::uint64_t client_identity = 0U;
    bool resident_preferred = false;
    std::mutex mutex;
    std::condition_variable ready;
    State state = State::pending;
    RunResult result;
    std::exception_ptr error;
};

struct PlanBuffers {
    std::shared_ptr<const AxisPlan> lifetime;
    id<MTLBuffer> transpose_offsets = nil;
    id<MTLBuffer> transpose_indices = nil;
    id<MTLBuffer> transpose_weights = nil;
    id<MTLBuffer> lower_ld = nil;
    id<MTLBuffer> upper_l = nil;
    id<MTLBuffer> inverse_diagonal = nil;
    std::size_t bytes = 0;
    std::uint64_t last_use = 0;
};

struct InputKey {
    const void *pointer = nullptr;
    const void *lifetime_identity = nullptr;
    std::ptrdiff_t stride = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t sample_bytes = 0;
    std::uint32_t layout = 0;
    std::array<std::uint32_t, 5> conversion{};

    bool operator==(const InputKey &) const noexcept = default;
};

struct InputKeyHash {
    std::size_t operator()(const InputKey &key) const noexcept {
        std::size_t result = std::hash<const void *>{}(key.pointer);
        auto mix = [&result](std::size_t value) {
            result ^= value + 0x9e3779b97f4a7c15ULL + (result << 6U)
                + (result >> 2U);
        };
        mix(std::hash<const void *>{}(key.lifetime_identity));
        mix(std::hash<std::ptrdiff_t>{}(key.stride));
        mix(key.width);
        mix(key.height);
        mix(key.sample_bytes);
        mix(key.layout);
        for (const auto value : key.conversion) mix(value);
        return result;
    }
};

[[nodiscard]] InputKey input_key(const PlaneJob &plane, bool transposed) noexcept {
    InputKey key;
    key.pointer = plane.source;
    key.lifetime_identity = plane.source_lifetime.get();
    key.stride = plane.source_stride_bytes;
    key.width = plane.source_width;
    key.height = plane.source_height;
    key.sample_bytes = plane.sample_bytes;
    key.layout = (transposed ? 1U : 0U) | (plane.integer_samples ? 2U : 0U);
    if (plane.integer_samples) {
        key.conversion = {
            std::bit_cast<std::uint32_t>(plane.conversion.input_offset),
            std::bit_cast<std::uint32_t>(plane.conversion.input_scale),
            std::bit_cast<std::uint32_t>(plane.conversion.output_scale),
            std::bit_cast<std::uint32_t>(plane.conversion.output_offset),
            plane.conversion.output_maximum,
        };
    }
    return key;
}

struct ResidentInput {
    InputKey key;
    std::shared_ptr<const void> source_lifetime;
    std::unordered_set<std::uint64_t> owners;
    std::size_t offset = 0U;
    std::size_t bytes = 0U;
    std::size_t source_bytes = 0U;
    std::uint32_t stride = 0U;
    std::uint64_t last_use = 0U;
    std::atomic<std::size_t> pins{0U};
    bool ready = false;
};

class ResidentLease final {
public:
    ResidentLease(
        std::shared_ptr<ResidentInput> entry,
        std::condition_variable *ready) noexcept
        : entry_(std::move(entry)), ready_(ready) {
        if (entry_) entry_->pins.fetch_add(1U, std::memory_order_relaxed);
    }

    ResidentLease(const ResidentLease &) = delete;
    ResidentLease &operator=(const ResidentLease &) = delete;

    ResidentLease(ResidentLease &&other) noexcept
        : entry_(std::move(other.entry_)), ready_(other.ready_) {
        other.ready_ = nullptr;
    }

    ResidentLease &operator=(ResidentLease &&other) noexcept {
        if (this != &other) {
            release();
            entry_ = std::move(other.entry_);
            ready_ = other.ready_;
            other.ready_ = nullptr;
        }
        return *this;
    }

    ~ResidentLease() { release(); }

private:
    void release() noexcept {
        if (entry_) {
            entry_->pins.fetch_sub(1U, std::memory_order_relaxed);
            entry_.reset();
            if (ready_) ready_->notify_all();
        }
        ready_ = nullptr;
    }

    std::shared_ptr<ResidentInput> entry_;
    std::condition_variable *ready_ = nullptr;
};

struct FreeRegion {
    std::size_t offset = 0U;
    std::size_t bytes = 0U;
};

struct PlaneLayout {
    std::size_t input_offset = 0;
    std::size_t intermediate_offset = 0;
    std::size_t result_offset = 0;
    std::size_t output_offset = 0;
    std::uint32_t input_stride = 0;
    std::uint32_t intermediate_stride = 0;
    std::uint32_t result_stride = 0;
    std::uint32_t output_stride = 0;
    bool resident_input = false;
    bool transposed_input = false;
};

struct JobLayout {
    std::shared_ptr<Request> request;
    std::vector<PlaneLayout> planes;
};

struct PlaneReference {
    std::size_t job_index = 0;
    std::size_t plane_index = 0;

    bool operator==(const PlaneReference &) const noexcept = default;
};

struct PlaneBatch {
    std::vector<PlaneReference> planes;
};

struct PackedPlanOffsets {
    std::uint32_t transpose_offsets = 0;
    std::uint32_t transpose_indices = 0;
    std::uint32_t transpose_weights = 0;
    std::uint32_t lower_ld = 0;
    std::uint32_t upper_l = 0;
    std::uint32_t inverse_diagonal = 0;
};

struct HeterogeneousAxisBatch {
    std::vector<std::size_t> plane_batches;
    std::size_t pipeline = 0;
    bool horizontal = false;
    bool resident_input = false;
};

struct HeterogeneousTwoAxisBatch {
    std::vector<std::size_t> plane_batches;
    std::size_t horizontal_pipeline = 0;
    std::size_t vertical_pipeline = 0;
    bool resident_input = false;
};

struct Submission {
    std::vector<JobLayout> jobs;
    std::vector<PlaneBatch> plane_batches;
    std::vector<HeterogeneousAxisBatch> heterogeneous_axis_batches;
    std::vector<HeterogeneousTwoAxisBatch> heterogeneous_two_axis_batches;
    std::unordered_map<const AxisPlan *, PackedPlanOffsets> packed_plans;
    std::vector<ResidentLease> resident_leases;
    CopyStats stats;
    std::size_t unique_input_planes = 0;
    std::size_t axis_dispatches = 0;
    std::size_t conversion_dispatches = 0;
    std::size_t heterogeneous_axis_dispatches = 0;
    std::size_t heterogeneous_axis_descriptors = 0;
    std::size_t resident_producers = 0;
    std::size_t resident_hits = 0;
    std::size_t resident_evictions = 0;
    std::size_t eliminated_staging_bytes = 0;
    std::uint64_t submission_gap_nanoseconds = 0U;
    bool profile_signposts = false;
    os_signpost_id_t signpost_id = OS_SIGNPOST_ID_INVALID;
};

class DeviceContext final {
public:
    struct AdmissionSnapshot {
        bool resident_ready = false;
        std::size_t in_flight = 0U;
    };

    using Completion = std::function<void(
        const std::vector<std::shared_ptr<Request>> &, RunResult,
        std::exception_ptr)>;

    DeviceContext() {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (device_ == nil || !device_.hasUnifiedMemory) {
                throw std::runtime_error(
                    "Metal backend requires an Apple unified-memory device");
            }
            queue_ = [device_ newCommandQueue];
            if (queue_ == nil) {
                throw std::runtime_error("Metal command queue creation failed");
            }
            dispatch_data_t data = dispatch_data_create(
                dsmvc_metal_routes_metallib,
                dsmvc_metal_routes_metallib_size,
                dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            NSError *error = nil;
            library_ = [device_ newLibraryWithData:data error:&error];
            if (library_ == nil) {
                throw std::runtime_error(
                    ns_error(error, "embedded Metal library load failed"));
            }
            const std::array<NSString *, 5> float_names{
                @"inverse_axis_generic", @"inverse_axis_h1",
                @"inverse_axis_h3", @"inverse_axis_h5", @"inverse_axis_h7"};
            const std::array<NSString *, 5> u8_names{
                @"inverse_axis_u8_generic", @"inverse_axis_u8_h1",
                @"inverse_axis_u8_h3", @"inverse_axis_u8_h5",
                @"inverse_axis_u8_h7"};
            const std::array<NSString *, 5> u16_names{
                @"inverse_axis_u16_generic", @"inverse_axis_u16_h1",
                @"inverse_axis_u16_h3", @"inverse_axis_u16_h5",
                @"inverse_axis_u16_h7"};
            const std::array<NSString *, 5> transposed_names{
                @"inverse_axis_transposed_generic",
                @"inverse_axis_transposed_h1", @"inverse_axis_transposed_h3",
                @"inverse_axis_transposed_h5", @"inverse_axis_transposed_h7"};
            for (std::size_t index = 0; index < float_names.size(); ++index) {
                float_pipelines_[index] = make_pipeline(float_names[index]);
                u8_pipelines_[index] = make_pipeline(u8_names[index]);
                u16_pipelines_[index] = make_pipeline(u16_names[index]);
                transposed_pipelines_[index] = make_pipeline(
                    transposed_names[index]);
            }
            convert_u8_ = make_pipeline(@"convert_f32_to_u8");
            convert_u16_ = make_pipeline(@"convert_f32_to_u16");

            const std::size_t recommended = static_cast<std::size_t>(
                device_.recommendedMaxWorkingSetSize);
            const std::size_t configured = memory_budget_from_environment();
            constexpr std::size_t maximum_default = 1024ULL * 1024ULL * 1024ULL;
            memory_budget_ = configured != 0U ? configured : std::min(
                maximum_default,
                recommended == 0U ? maximum_default : recommended / 8U);
            constexpr std::size_t maximum_plan_cache =
                64ULL * 1024ULL * 1024ULL;
            plan_cache_budget_ = std::min(
                maximum_plan_cache, memory_budget_ / 8U);
            constexpr std::size_t maximum_resident_cache =
                256ULL * 1024ULL * 1024ULL;
            const std::size_t configured_resident =
                resident_budget_from_environment();
            resident_capacity_ = std::min(
                configured_resident != 0U
                    ? configured_resident : maximum_resident_cache,
                memory_budget_ / 4U);
            if (resident_capacity_ >= resident_alignment) {
                resident_input_ = make_buffer(
                    resident_capacity_, MTLResourceStorageModeShared,
                    @"dsmvc transposed resident input cache");
                resident_free_regions_.push_back({0U, resident_capacity_});
            }
        }
    }

    ~DeviceContext() { wait_idle(); }

    DeviceContext(const DeviceContext &) = delete;
    DeviceContext &operator=(const DeviceContext &) = delete;

    [[nodiscard]] AdmissionSnapshot admission_snapshot(
        const FrameJob &job) const noexcept {
        AdmissionSnapshot snapshot;
        snapshot.in_flight = in_flight_snapshot_.load(std::memory_order_acquire);
        if (job.planes.empty()
            || resident_entry_count_.load(std::memory_order_relaxed) == 0U) {
            return snapshot;
        }

        const std::scoped_lock lock(resident_mutex_);
        snapshot.resident_ready = std::all_of(
            job.planes.begin(), job.planes.end(), [&](const PlaneJob &plane) {
                if (!plane.process_horizontal || !plane.source_lifetime) {
                    return false;
                }
                const auto found = resident_inputs_.find(input_key(plane, true));
                return found != resident_inputs_.end() && found->second->ready;
            });
        return snapshot;
    }

    [[nodiscard]] bool ring_has_capacity() const noexcept {
        return in_flight_snapshot_.load(std::memory_order_acquire) < ring_size;
    }

    void submit(
        std::vector<std::shared_ptr<Request>> requests,
        Completion completion) {
        std::size_t slot_index = 0;
        {
            std::unique_lock lock(slots_mutex_);
            slots_ready_.wait(lock, [&] {
                return std::any_of(slots_.begin(), slots_.end(),
                                   [](const Slot &slot) { return !slot.in_use; });
            });
            for (; slot_index < slots_.size(); ++slot_index) {
                if (!slots_[slot_index].in_use) break;
            }
            slots_[slot_index].in_use = true;
            ++in_flight_;
            in_flight_snapshot_.store(in_flight_, std::memory_order_release);
            ++submissions_;
            maximum_in_flight_ = std::max(maximum_in_flight_, in_flight_);
        }

        try {
            submit_on_slot(slot_index, std::move(requests), std::move(completion));
        } catch (...) {
            record_metal_error();
            release_slot(slot_index);
            throw;
        }
    }

    void wait_idle() noexcept {
        std::unique_lock lock(slots_mutex_);
        slots_ready_.wait(lock, [&] { return in_flight_ == 0U; });
    }

    void release_client(std::uint64_t identity) noexcept {
        try {
            std::unique_lock lock(resident_mutex_);
            std::vector<std::shared_ptr<ResidentInput>> orphaned;
            for (const auto &[key, entry] : resident_inputs_) {
                (void)key;
                if (entry->owners.erase(identity) != 0U
                    && entry->owners.empty()) {
                    orphaned.push_back(entry);
                }
            }
            resident_ready_.wait(lock, [&] {
                return std::all_of(
                    orphaned.begin(), orphaned.end(), [&](const auto &entry) {
                        const auto found = resident_inputs_.find(entry->key);
                        return found == resident_inputs_.end()
                            || found->second != entry || !entry->owners.empty()
                            || entry->pins.load(std::memory_order_relaxed) == 0U;
                    });
            });
            for (const auto &entry : orphaned) {
                const auto found = resident_inputs_.find(entry->key);
                if (found != resident_inputs_.end() && found->second == entry
                    && entry->owners.empty()
                    && entry->pins.load(std::memory_order_relaxed) == 0U) {
                    (void)erase_resident_locked(entry, false);
                }
            }
            lock.unlock();
            resident_ready_.notify_all();
        } catch (...) {
            // Client teardown is noexcept; retain the entry on cleanup failure.
        }
    }

    [[nodiscard]] SchedulerDiagnostics diagnostics() const noexcept {
        const std::scoped_lock lock(slots_mutex_);
        return {
            ring_size,
            submissions_,
            completions_,
            maximum_in_flight_,
            plan_entry_count_.load(std::memory_order_relaxed),
            plan_resident_bytes_.load(std::memory_order_relaxed),
            plan_evictions_.load(std::memory_order_relaxed),
            resident_entry_count_.load(std::memory_order_relaxed),
            resident_logical_bytes_.load(std::memory_order_relaxed),
            resident_capacity_,
            resident_evictions_.load(std::memory_order_relaxed),
            resident_pinned_eviction_blocks_.load(std::memory_order_relaxed),
            resident_producers_.load(std::memory_order_relaxed),
            resident_hits_.load(std::memory_order_relaxed),
            metal_errors_.load(std::memory_order_relaxed),
            consecutive_metal_errors_.load(std::memory_order_relaxed),
            maximum_consecutive_metal_errors_.load(std::memory_order_relaxed),
        };
    }

private:
    struct Slot {
        id<MTLBuffer> input = nil;
        id<MTLBuffer> scratch = nil;
        id<MTLBuffer> output = nil;
        id<MTLBuffer> plan = nil;
        std::size_t input_capacity = 0;
        std::size_t scratch_capacity = 0;
        std::size_t output_capacity = 0;
        std::size_t plan_capacity = 0;
        bool in_use = false;
    };

    [[nodiscard]] static std::size_t memory_budget_from_environment() noexcept {
        const char *value = std::getenv("DSMVC_METAL_MEMORY_MB");
        if (!value) return 0U;
        char *end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        if (end == value || *end != '\0' || parsed < 64U || parsed > 8192U) {
            return 0U;
        }
        return static_cast<std::size_t>(parsed) * 1024U * 1024U;
    }

    [[nodiscard]] static std::size_t resident_budget_from_environment() noexcept {
        const char *value = std::getenv("DSMVC_METAL_RESIDENT_MB");
        if (!value) return 0U;
        char *end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        if (end == value || *end != '\0' || parsed < 1U || parsed > 256U) {
            return 0U;
        }
        return static_cast<std::size_t>(parsed) * 1024U * 1024U;
    }

    void record_metal_error() noexcept {
        metal_errors_.fetch_add(1U, std::memory_order_relaxed);
        const std::size_t consecutive = consecutive_metal_errors_.fetch_add(
            1U, std::memory_order_relaxed) + 1U;
        auto maximum = maximum_consecutive_metal_errors_.load(
            std::memory_order_relaxed);
        while (maximum < consecutive
               && !maximum_consecutive_metal_errors_.compare_exchange_weak(
                   maximum, consecutive, std::memory_order_relaxed)) {}
    }

    void record_metal_success() noexcept {
        consecutive_metal_errors_.store(0U, std::memory_order_relaxed);
    }

    [[nodiscard]] id<MTLComputePipelineState> make_pipeline(NSString *name) {
        id<MTLFunction> function = [library_ newFunctionWithName:name];
        if (function == nil) {
            throw std::runtime_error(
                "embedded Metal library is missing "
                + std::string{name.UTF8String});
        }
        NSError *error = nil;
        id<MTLComputePipelineState> pipeline =
            [device_ newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            throw std::runtime_error(
                ns_error(error, "Metal pipeline creation failed"));
        }
        return pipeline;
    }

    [[nodiscard]] id<MTLComputePipelineState> float_batch_pipeline(
        std::size_t index, bool transposed) {
        auto &pipelines = transposed
            ? transposed_batch_pipelines_ : float_batch_pipelines_;
        if (pipelines[index] != nil) {
            return pipelines[index];
        }
        const std::array<NSString *, 5> regular_names{
            @"inverse_axis_batch_generic", @"inverse_axis_batch_h1",
            @"inverse_axis_batch_h3", @"inverse_axis_batch_h5",
            @"inverse_axis_batch_h7"};
        const std::array<NSString *, 5> transposed_names{
            @"inverse_axis_transposed_batch_generic",
            @"inverse_axis_transposed_batch_h1",
            @"inverse_axis_transposed_batch_h3",
            @"inverse_axis_transposed_batch_h5",
            @"inverse_axis_transposed_batch_h7"};
        pipelines[index] = make_pipeline(
            transposed ? transposed_names[index] : regular_names[index]);
        return pipelines[index];
    }

    [[nodiscard]] id<MTLBuffer> make_buffer(
        std::size_t bytes, MTLResourceOptions options, NSString *label) {
        id<MTLBuffer> buffer = [device_ newBufferWithLength:std::max<std::size_t>(bytes, 256U)
                                                    options:options];
        if (buffer == nil) throw std::bad_alloc();
        buffer.label = label;
        return buffer;
    }

    [[nodiscard]] std::optional<std::size_t> take_resident_region_locked(
        std::size_t bytes) {
        const auto found = std::find_if(
            resident_free_regions_.begin(), resident_free_regions_.end(),
            [bytes](const FreeRegion &region) { return region.bytes >= bytes; });
        if (found == resident_free_regions_.end()) return std::nullopt;
        const std::size_t offset = found->offset;
        found->offset = checked_add(found->offset, bytes, "resident region");
        found->bytes -= bytes;
        if (found->bytes == 0U) resident_free_regions_.erase(found);
        return offset;
    }

    void add_resident_region_locked(std::size_t offset, std::size_t bytes) {
        resident_free_regions_.push_back({offset, bytes});
        std::sort(
            resident_free_regions_.begin(), resident_free_regions_.end(),
            [](const FreeRegion &left, const FreeRegion &right) {
                return left.offset < right.offset;
            });
        std::vector<FreeRegion> merged;
        merged.reserve(resident_free_regions_.size());
        for (const FreeRegion &region : resident_free_regions_) {
            if (!merged.empty()
                && checked_add(
                    merged.back().offset, merged.back().bytes,
                    "resident free region") == region.offset) {
                merged.back().bytes = checked_add(
                    merged.back().bytes, region.bytes, "resident free region");
            } else {
                merged.push_back(region);
            }
        }
        resident_free_regions_ = std::move(merged);
    }

    [[nodiscard]] bool erase_resident_locked(
        const std::shared_ptr<ResidentInput> &entry, bool record_eviction) {
        const auto found = resident_inputs_.find(entry->key);
        if (found == resident_inputs_.end() || found->second != entry) return false;
        if (entry->bytes > resident_bytes_) {
            throw std::logic_error("resident cache accounting underflow");
        }
        add_resident_region_locked(entry->offset, entry->bytes);
        resident_bytes_ -= entry->bytes;
        resident_inputs_.erase(found);
        resident_entry_count_.store(
            resident_inputs_.size(), std::memory_order_relaxed);
        resident_logical_bytes_.store(resident_bytes_, std::memory_order_relaxed);
        if (record_eviction) {
            resident_evictions_.fetch_add(1U, std::memory_order_relaxed);
        }
        return true;
    }

    [[nodiscard]] bool evict_one_resident_locked() {
        auto oldest = resident_inputs_.end();
        for (auto candidate = resident_inputs_.begin();
             candidate != resident_inputs_.end(); ++candidate) {
            if (!candidate->second->ready
                || candidate->second->pins.load(std::memory_order_relaxed) != 0U) {
                continue;
            }
            if (oldest == resident_inputs_.end()
                || candidate->second->last_use < oldest->second->last_use) {
                oldest = candidate;
            }
        }
        if (oldest == resident_inputs_.end()) {
            if (!resident_inputs_.empty()) {
                resident_pinned_eviction_blocks_.fetch_add(
                    1U, std::memory_order_relaxed);
            }
            return false;
        }
        return erase_resident_locked(oldest->second, true);
    }

    template <class Sample>
    static void transpose_normalized(
        float *destination, std::uint32_t destination_stride,
        const PlaneJob &plane) {
        const auto *source = static_cast<const std::byte *>(plane.source);
        const auto conversion = plane.conversion;
        for (std::uint32_t row_base = 0U; row_base < plane.source_height;
             row_base += resident_tile) {
            const std::uint32_t row_end = std::min<std::uint32_t>(
                row_base + resident_tile, plane.source_height);
            for (std::uint32_t column_base = 0U;
                 column_base < plane.source_width; column_base += resident_tile) {
                const std::uint32_t column_end = std::min<std::uint32_t>(
                    column_base + resident_tile, plane.source_width);
                for (std::uint32_t row = row_base; row < row_end; ++row) {
                    const auto *source_row = reinterpret_cast<const Sample *>(
                        source + static_cast<std::ptrdiff_t>(row)
                            * plane.source_stride_bytes);
                    for (std::uint32_t column = column_base;
                         column < column_end; ++column) {
                        float value = static_cast<float>(source_row[column]);
                        if (plane.integer_samples) {
                            value = (value - conversion.input_offset)
                                * conversion.input_scale;
                        }
                        destination[static_cast<std::size_t>(column)
                                        * destination_stride + row] = value;
                    }
                }
            }
        }
    }

    static void produce_resident_input(
        const PlaneJob &plane, ResidentInput &entry, id<MTLBuffer> buffer) {
        if (fail_next_resident_producer.exchange(false, std::memory_order_relaxed)) {
            throw std::runtime_error("simulated resident input producer failure");
        }
        auto *destination = reinterpret_cast<float *>(
            static_cast<std::byte *>(buffer.contents) + entry.offset);
        std::memset(destination, 0, entry.bytes);
        if (!plane.integer_samples && plane.sample_bytes == sizeof(float)) {
            transpose_normalized<float>(destination, entry.stride, plane);
        } else if (plane.integer_samples && plane.sample_bytes == 1U) {
            transpose_normalized<std::uint8_t>(destination, entry.stride, plane);
        } else if (plane.integer_samples && plane.sample_bytes == 2U) {
            transpose_normalized<std::uint16_t>(destination, entry.stride, plane);
        } else {
            throw std::invalid_argument(
                "resident transpose does not support the source sample layout");
        }
    }

    [[nodiscard]] std::shared_ptr<ResidentInput> acquire_resident_input(
        const PlaneJob &plane, std::uint64_t client_identity,
        Submission &submission) {
        if (resident_input_ == nil || !plane.source_lifetime
            || !plane.process_horizontal) {
            return {};
        }
        const std::size_t padded_height = checked_align_up(
            plane.source_height, resident_tile, "resident height");
        const std::size_t samples = checked_multiply(
            plane.source_width, padded_height, "resident input");
        const std::size_t bytes = checked_align_up(
            checked_multiply(samples, sizeof(float), "resident input"),
            resident_alignment, "resident input");
        if (bytes > resident_capacity_) return {};
        const std::size_t source_bytes = checked_multiply(
            checked_multiply(
                plane.source_width, plane.sample_bytes, "resident source"),
            plane.source_height, "resident source");
        const InputKey key = input_key(plane, true);

        auto entry = std::make_shared<ResidentInput>();
        entry->key = key;
        entry->source_lifetime = plane.source_lifetime;
        entry->bytes = bytes;
        entry->source_bytes = source_bytes;
        entry->stride = checked_u32(padded_height, "resident input stride");
        std::size_t evictions = 0U;
        {
            std::unique_lock lock(resident_mutex_);
            for (;;) {
                const auto found = resident_inputs_.find(key);
                if (found == resident_inputs_.end()) break;
                const auto existing = found->second;
                if (!existing->ready) {
                    resident_ready_.wait(lock, [&] {
                        const auto current = resident_inputs_.find(key);
                        return current == resident_inputs_.end()
                            || current->second->ready;
                    });
                    continue;
                }
                existing->last_use = ++resident_use_sequence_;
                existing->owners.insert(client_identity);
                resident_hits_.fetch_add(1U, std::memory_order_relaxed);
                ++submission.resident_hits;
                submission.eliminated_staging_bytes = checked_add(
                    submission.eliminated_staging_bytes, source_bytes,
                    "eliminated staging traffic");
                submission.resident_leases.emplace_back(existing, &resident_ready_);
                return existing;
            }

            auto region = take_resident_region_locked(bytes);
            while (!region && evict_one_resident_locked()) {
                ++evictions;
                region = take_resident_region_locked(bytes);
            }
            if (!region) return {};
            entry->offset = *region;
            entry->owners.insert(client_identity);
            const auto inserted = resident_inputs_.emplace(key, entry);
            if (!inserted.second) {
                throw std::logic_error("resident producer single-flight failed");
            }
            resident_bytes_ = checked_add(
                resident_bytes_, bytes, "resident cache");
            resident_entry_count_.store(
                resident_inputs_.size(), std::memory_order_relaxed);
            resident_logical_bytes_.store(
                resident_bytes_, std::memory_order_relaxed);
        }

        try {
            produce_resident_input(plane, *entry, resident_input_);
        } catch (...) {
            const auto producer_error = std::current_exception();
            std::exception_ptr cleanup_error;
            try {
                const std::scoped_lock lock(resident_mutex_);
                const auto found = resident_inputs_.find(key);
                if (found != resident_inputs_.end() && found->second == entry) {
                    (void)erase_resident_locked(entry, false);
                }
            } catch (...) {
                cleanup_error = std::current_exception();
            }
            resident_ready_.notify_all();
            std::rethrow_exception(cleanup_error ? cleanup_error : producer_error);
        }

        {
            const std::scoped_lock lock(resident_mutex_);
            entry->last_use = ++resident_use_sequence_;
            entry->ready = true;
            resident_producers_.fetch_add(1U, std::memory_order_relaxed);
            submission.resident_leases.emplace_back(entry, &resident_ready_);
        }
        resident_ready_.notify_all();
        ++submission.resident_producers;
        submission.resident_evictions = checked_add(
            submission.resident_evictions, evictions, "resident eviction count");
        return entry;
    }

    void ensure_slot_capacity(
        Slot &slot, std::size_t input_bytes, std::size_t scratch_bytes,
        std::size_t output_bytes, std::size_t plan_bytes) {
        const std::size_t new_input = input_bytes > slot.input_capacity
            ? checked_align_up(input_bytes, 4096U, "Metal input ring")
            : slot.input_capacity;
        const std::size_t new_scratch = scratch_bytes > slot.scratch_capacity
            ? checked_align_up(scratch_bytes, 4096U, "Metal scratch ring")
            : slot.scratch_capacity;
        const std::size_t new_output = output_bytes > slot.output_capacity
            ? checked_align_up(output_bytes, 4096U, "Metal output ring")
            : slot.output_capacity;
        const std::size_t new_plan = plan_bytes > slot.plan_capacity
            ? checked_align_up(plan_bytes, 4096U, "Metal plan ring")
            : slot.plan_capacity;
        const std::size_t old_total = checked_add(
            checked_add(slot.input_capacity, slot.scratch_capacity, "Metal ring"),
            checked_add(slot.output_capacity, slot.plan_capacity, "Metal ring"),
            "Metal ring");
        const std::size_t new_total = checked_add(
            checked_add(new_input, new_scratch, "Metal ring"),
            checked_add(new_output, new_plan, "Metal ring"), "Metal ring");
        if (old_total > ring_bytes_) {
            throw std::logic_error("Metal ring accounting underflow");
        }
        const std::size_t new_ring_bytes = checked_add(
            ring_bytes_ - old_total, new_total, "Metal ring");
        evict_plans_until([&] {
            return checked_add(
                checked_add(new_ring_bytes, plan_bytes_, "Metal UMA budget"),
                resident_capacity_, "Metal UMA budget") <= memory_budget_;
        });
        if (checked_add(
                checked_add(new_ring_bytes, plan_bytes_, "Metal UMA budget"),
                resident_capacity_, "Metal UMA budget") > memory_budget_) {
            throw std::length_error("Metal staging ring exceeds memory budget");
        }
        if (new_input != slot.input_capacity) {
            slot.input = make_buffer(
                new_input, MTLResourceStorageModeShared,
                @"dsmvc UMA input arena");
            slot.input_capacity = new_input;
        }
        if (new_scratch != slot.scratch_capacity) {
            slot.scratch = make_buffer(
                new_scratch, MTLResourceStorageModePrivate,
                @"dsmvc GPU scratch arena");
            slot.scratch_capacity = new_scratch;
        }
        if (new_output != slot.output_capacity) {
            slot.output = make_buffer(
                new_output, MTLResourceStorageModeShared,
                @"dsmvc UMA output arena");
            slot.output_capacity = new_output;
        }
        if (new_plan != slot.plan_capacity) {
            slot.plan = make_buffer(
                new_plan, MTLResourceStorageModeShared,
                @"dsmvc UMA plan arena");
            slot.plan_capacity = new_plan;
        }
        ring_bytes_ = new_ring_bytes;
    }

    template <class Predicate>
    void evict_plans_until(Predicate within_limit) {
        while (!within_limit() && !plans_.empty()) {
            const auto oldest = std::min_element(
                plans_.begin(), plans_.end(), [](const auto &left, const auto &right) {
                    return left.second.last_use < right.second.last_use;
                });
            plan_bytes_ -= oldest->second.bytes;
            plans_.erase(oldest);
            plan_entry_count_.store(plans_.size(), std::memory_order_relaxed);
            plan_resident_bytes_.store(plan_bytes_, std::memory_order_relaxed);
            plan_evictions_.fetch_add(1U, std::memory_order_relaxed);
        }
    }

    template <class T>
    [[nodiscard]] static std::size_t plan_vector_bytes(
        const std::vector<T> &values) {
        return std::max(
            sizeof(T), checked_multiply(
                values.size(), sizeof(T), "Metal plan vector"));
    }

    [[nodiscard]] static std::size_t plan_resource_bytes(
        const AxisPlan &plan) {
        std::size_t bytes = 0U;
        for (const std::size_t value : {
                 plan_vector_bytes(plan.transpose_offsets),
                 plan_vector_bytes(plan.transpose_indices),
                 plan_vector_bytes(plan.transpose_weights),
                 plan_vector_bytes(plan.lower_ld),
                 plan_vector_bytes(plan.upper_l),
                 plan_vector_bytes(plan.inverse_diagonal)}) {
            bytes = checked_add(bytes, value, "Metal plan resources");
        }
        return bytes;
    }

    [[nodiscard]] PlanBuffers &prepare_plan(
        const std::shared_ptr<const AxisPlan> &plan) {
        if (!plan || !plan->valid()) {
            throw std::invalid_argument("invalid Metal axis plan");
        }
        const auto found = plans_.find(plan.get());
        if (found != plans_.end()) {
            found->second.last_use = ++plan_use_sequence_;
            return found->second;
        }
        const std::size_t required = plan_resource_bytes(*plan);
        evict_plans_until([&] {
            const bool within_entry_limit = plans_.size() < maximum_cached_plans;
            const bool within_cache_budget = plans_.empty()
                || checked_add(plan_bytes_, required, "Metal plan cache")
                    <= plan_cache_budget_;
            const bool within_total_budget =
                checked_add(
                    checked_add(ring_bytes_, plan_bytes_, "Metal UMA budget"),
                    checked_add(required, resident_capacity_, "Metal UMA budget"),
                    "Metal UMA budget") <= memory_budget_;
            return within_entry_limit && within_cache_budget
                && within_total_budget;
        });
        if (checked_add(
                checked_add(ring_bytes_, plan_bytes_, "Metal UMA budget"),
                checked_add(required, resident_capacity_, "Metal UMA budget"),
                "Metal UMA budget") > memory_budget_) {
            throw std::length_error("Metal plan exceeds UMA memory budget");
        }
        PlanBuffers buffers;
        buffers.lifetime = plan;
        buffers.transpose_offsets = make_plan_buffer(plan->transpose_offsets);
        buffers.transpose_indices = make_plan_buffer(plan->transpose_indices);
        buffers.transpose_weights = make_plan_buffer(plan->transpose_weights);
        buffers.lower_ld = make_plan_buffer(plan->lower_ld);
        buffers.upper_l = make_plan_buffer(plan->upper_l);
        buffers.inverse_diagonal = make_plan_buffer(plan->inverse_diagonal);
        buffers.bytes = required;
        buffers.last_use = ++plan_use_sequence_;
        auto [inserted, _] = plans_.emplace(plan.get(), std::move(buffers));
        plan_bytes_ = checked_add(plan_bytes_, required, "Metal plan cache");
        plan_entry_count_.store(plans_.size(), std::memory_order_relaxed);
        plan_resident_bytes_.store(plan_bytes_, std::memory_order_relaxed);
        return inserted->second;
    }

    template <class T>
    [[nodiscard]] id<MTLBuffer> make_plan_buffer(const std::vector<T> &values) {
        const std::size_t bytes = checked_multiply(
            values.size(), sizeof(T), "Metal plan buffer");
        id<MTLBuffer> buffer = nil;
        if (values.empty()) {
            buffer = [device_ newBufferWithLength:sizeof(T)
                                          options:MTLResourceStorageModeShared];
        } else {
            buffer = [device_ newBufferWithBytes:values.data()
                                             length:bytes
                                            options:MTLResourceStorageModeShared];
        }
        if (buffer == nil) throw std::bad_alloc();
        return buffer;
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

    void bind_plan(
        id<MTLComputeCommandEncoder> encoder, const PlanBuffers &plan) {
        [encoder setBuffer:plan.transpose_offsets offset:0 atIndex:2];
        [encoder setBuffer:plan.transpose_indices offset:0 atIndex:3];
        [encoder setBuffer:plan.transpose_weights offset:0 atIndex:4];
        [encoder setBuffer:plan.lower_ld offset:0 atIndex:5];
        [encoder setBuffer:plan.upper_l offset:0 atIndex:6];
        [encoder setBuffer:plan.inverse_diagonal offset:0 atIndex:7];
    }

    void dispatch(
        id<MTLComputeCommandEncoder> encoder,
        id<MTLComputePipelineState> pipeline, std::size_t count) {
        const NSUInteger width = std::min<NSUInteger>(
            128U, pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(count, 1U, 1U)
             threadsPerThreadgroup:MTLSizeMake(width, 1U, 1U)];
    }

    void dispatch_2d(
        id<MTLComputeCommandEncoder> encoder,
        id<MTLComputePipelineState> pipeline,
        std::size_t width_count, std::size_t height_count) {
        const NSUInteger width = std::min<NSUInteger>(
            128U, pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(width_count, height_count, 1U)
             threadsPerThreadgroup:MTLSizeMake(width, 1U, 1U)];
    }

    enum class Arena { input, intermediate, result, output };

    [[nodiscard]] static const PlaneJob &plane_for(
        const Submission &submission, PlaneReference reference) {
        return submission.jobs[reference.job_index]
            .request->job.planes[reference.plane_index];
    }

    [[nodiscard]] static const PlaneLayout &layout_for(
        const Submission &submission, PlaneReference reference) {
        return submission.jobs[reference.job_index].planes[reference.plane_index];
    }

    [[nodiscard]] static PlaneLayout &layout_for(
        Submission &submission, PlaneReference reference) {
        return submission.jobs[reference.job_index].planes[reference.plane_index];
    }

    [[nodiscard]] static std::size_t arena_offset(
        const PlaneLayout &layout, Arena arena) noexcept {
        switch (arena) {
        case Arena::input: return layout.input_offset;
        case Arena::intermediate: return layout.intermediate_offset;
        case Arena::result: return layout.result_offset;
        case Arena::output: return layout.output_offset;
        }
        return 0U;
    }

    [[nodiscard]] static std::uint32_t arena_stride(
        const PlaneLayout &layout, Arena arena) noexcept {
        switch (arena) {
        case Arena::input: return layout.input_stride;
        case Arena::intermediate: return layout.intermediate_stride;
        case Arena::result: return layout.result_stride;
        case Arena::output: return layout.output_stride;
        }
        return 0U;
    }

    [[nodiscard]] static std::size_t arena_element_bytes(
        const PlaneJob &plane, const PlaneLayout &layout, Arena arena) noexcept {
        if (arena == Arena::input && layout.transposed_input) {
            return sizeof(float);
        }
        return arena == Arena::intermediate || arena == Arena::result
            ? sizeof(float) : plane.sample_bytes;
    }

    [[nodiscard]] static bool same_operation(
        const Submission &submission, PlaneReference left,
        PlaneReference right) noexcept {
        const PlaneJob &a = plane_for(submission, left);
        const PlaneJob &b = plane_for(submission, right);
        const PlaneLayout &a_layout = layout_for(submission, left);
        const PlaneLayout &b_layout = layout_for(submission, right);
        return a.source_width == b.source_width
            && a.source_height == b.source_height
            && a.destination_width == b.destination_width
            && a.destination_height == b.destination_height
            && a.sample_bytes == b.sample_bytes
            && a.integer_samples == b.integer_samples
            && a.process_horizontal == b.process_horizontal
            && a.process_vertical == b.process_vertical
            && a.horizontal.get() == b.horizontal.get()
            && a.vertical.get() == b.vertical.get()
            && a_layout.input_stride == b_layout.input_stride
            && a_layout.intermediate_stride == b_layout.intermediate_stride
            && a_layout.result_stride == b_layout.result_stride
            && a_layout.output_stride == b_layout.output_stride
            && a_layout.resident_input == b_layout.resident_input
            && a_layout.transposed_input == b_layout.transposed_input;
    }

    [[nodiscard]] static bool follows_regular_offsets(
        const Submission &submission, const PlaneBatch &batch,
        PlaneReference candidate) noexcept {
        if (batch.planes.empty()) return true;
        const auto follows = [&](Arena arena) {
            const std::size_t first = arena_offset(
                layout_for(submission, batch.planes[0]), arena);
            if (batch.planes.size() == 1U) {
                return arena_offset(layout_for(submission, candidate), arena)
                    >= first;
            }
            const std::size_t second = arena_offset(
                layout_for(submission, batch.planes[1]), arena);
            if (second < first) return false;
            const std::size_t delta = second - first;
            const std::size_t index = batch.planes.size();
            if (delta != 0U
                && index > (std::numeric_limits<std::size_t>::max() - first)
                    / delta) {
                return false;
            }
            return arena_offset(layout_for(submission, candidate), arena)
                == first + index * delta;
        };
        return follows(Arena::input) && follows(Arena::output);
    }

    static void append_regular_batches(
        Submission &submission, std::vector<PlaneReference> references) {
        while (!references.empty()) {
            std::optional<std::size_t> repeated_input;
            for (std::size_t left = 0; left < references.size(); ++left) {
                const std::size_t offset = arena_offset(
                    layout_for(submission, references[left]), Arena::input);
                const bool repeated = std::any_of(
                    references.begin() + static_cast<std::ptrdiff_t>(left + 1U),
                    references.end(), [&](PlaneReference reference) {
                        return arena_offset(
                            layout_for(submission, reference), Arena::input)
                            == offset;
                    });
                if (repeated) {
                    repeated_input = offset;
                    break;
                }
            }

            std::vector<PlaneReference> candidates;
            for (PlaneReference reference : references) {
                if (!repeated_input
                    || arena_offset(layout_for(submission, reference), Arena::input)
                        == *repeated_input) {
                    candidates.push_back(reference);
                }
            }

            PlaneBatch batch;
            batch.planes.push_back(candidates.front());
            for (std::size_t index = 1; index < candidates.size(); ++index) {
                if (follows_regular_offsets(
                        submission, batch, candidates[index])) {
                    batch.planes.push_back(candidates[index]);
                }
            }
            for (PlaneReference reference : batch.planes) {
                const auto found = std::find(
                    references.begin(), references.end(), reference);
                if (found != references.end()) references.erase(found);
            }
            submission.plane_batches.push_back(std::move(batch));
        }
    }

    [[nodiscard]] static std::size_t prepare_plane_batches(
        Submission &submission) {
        std::size_t maximum_planes = 0U;
        for (const JobLayout &job : submission.jobs) {
            maximum_planes = std::max(maximum_planes, job.planes.size());
        }
        for (std::size_t plane_index = 0; plane_index < maximum_planes;
             ++plane_index) {
            std::vector<std::vector<PlaneReference>> classes;
            for (std::size_t job_index = 0; job_index < submission.jobs.size();
                 ++job_index) {
                if (plane_index >= submission.jobs[job_index].planes.size()) continue;
                const PlaneReference reference{job_index, plane_index};
                const auto found = std::find_if(
                    classes.begin(), classes.end(), [&](const auto &group) {
                        return same_operation(submission, group.front(), reference);
                    });
                if (found == classes.end()) {
                    classes.push_back({reference});
                } else {
                    found->push_back(reference);
                }
            }
            for (auto &group : classes) {
                append_regular_batches(submission, std::move(group));
            }
        }

        std::size_t scratch_bytes = 0U;
        for (const PlaneBatch &batch : submission.plane_batches) {
            const PlaneJob &plane = plane_for(submission, batch.planes.front());
            const PlaneLayout &first_layout =
                layout_for(submission, batch.planes.front());
            std::size_t batch_scratch = 0U;
            if (plane.process_horizontal && plane.process_vertical) {
                const std::size_t frame_bytes = checked_align_up(
                    checked_multiply(
                        checked_multiply(
                            plane.source_height, first_layout.intermediate_stride,
                            "Metal intermediate arena"),
                        sizeof(float), "Metal intermediate arena"),
                    arena_alignment, "Metal intermediate arena");
                for (std::size_t index = 0; index < batch.planes.size(); ++index) {
                    layout_for(submission, batch.planes[index]).intermediate_offset =
                        checked_multiply(
                            index, frame_bytes, "Metal intermediate arena");
                }
                batch_scratch = checked_multiply(
                    batch.planes.size(), frame_bytes, "Metal intermediate arena");
            }
            if (plane.integer_samples) {
                const std::size_t base = checked_align_up(
                    batch_scratch, arena_alignment, "Metal result arena");
                const std::size_t frame_bytes = checked_align_up(
                    checked_multiply(
                        checked_multiply(
                            plane.destination_height, first_layout.result_stride,
                            "Metal result arena"),
                        sizeof(float), "Metal result arena"),
                    arena_alignment, "Metal result arena");
                for (std::size_t index = 0; index < batch.planes.size(); ++index) {
                    layout_for(submission, batch.planes[index]).result_offset =
                        checked_add(
                            base, checked_multiply(
                                index, frame_bytes, "Metal result arena"),
                            "Metal result arena");
                }
                batch_scratch = checked_add(
                    base, checked_multiply(
                        batch.planes.size(), frame_bytes, "Metal result arena"),
                    "Metal result arena");
            }
            scratch_bytes = std::max(scratch_bytes, batch_scratch);
        }
        return scratch_bytes;
    }

    [[nodiscard]] static std::size_t prepare_heterogeneous_axis_batches(
        Submission &submission) {
        for (std::size_t batch_index = 0;
             batch_index < submission.plane_batches.size(); ++batch_index) {
            const PlaneBatch &batch = submission.plane_batches[batch_index];
            const PlaneJob &plane = plane_for(
                submission, batch.planes.front());
            const PlaneLayout &layout = layout_for(
                submission, batch.planes.front());
            if (plane.integer_samples || plane.sample_bytes != sizeof(float)) {
                continue;
            }
            if (plane.process_horizontal && plane.process_vertical) {
                const std::size_t horizontal_pipeline = pipeline_index(
                    static_cast<std::uint32_t>(
                        plane.horizontal->half_bandwidth));
                const std::size_t vertical_pipeline = pipeline_index(
                    static_cast<std::uint32_t>(
                        plane.vertical->half_bandwidth));
                const auto found = std::find_if(
                    submission.heterogeneous_two_axis_batches.begin(),
                    submission.heterogeneous_two_axis_batches.end(),
                    [&](const HeterogeneousTwoAxisBatch &candidate) {
                        return candidate.horizontal_pipeline
                                == horizontal_pipeline
                            && candidate.vertical_pipeline == vertical_pipeline
                            && candidate.resident_input == layout.resident_input
                            && candidate.plane_batches.size()
                                < maximum_batch_descriptors;
                    });
                if (found == submission.heterogeneous_two_axis_batches.end()) {
                    submission.heterogeneous_two_axis_batches.push_back(
                        HeterogeneousTwoAxisBatch{
                            {batch_index}, horizontal_pipeline,
                            vertical_pipeline, layout.resident_input});
                } else {
                    found->plane_batches.push_back(batch_index);
                }
                continue;
            }
            if (plane.process_horizontal == plane.process_vertical) continue;
            const bool horizontal = plane.process_horizontal;
            const auto &plan = horizontal ? plane.horizontal : plane.vertical;
            const std::size_t pipeline = pipeline_index(
                static_cast<std::uint32_t>(plan->half_bandwidth));
            const auto found = std::find_if(
                submission.heterogeneous_axis_batches.begin(),
                submission.heterogeneous_axis_batches.end(),
                [&](const HeterogeneousAxisBatch &candidate) {
                    return candidate.horizontal == horizontal
                        && candidate.pipeline == pipeline
                        && candidate.resident_input == layout.resident_input
                        && candidate.plane_batches.size()
                            < maximum_batch_descriptors;
                });
            if (found == submission.heterogeneous_axis_batches.end()) {
                submission.heterogeneous_axis_batches.push_back(
                    HeterogeneousAxisBatch{
                        {batch_index}, pipeline, horizontal,
                        layout.resident_input});
            } else {
                found->plane_batches.push_back(batch_index);
            }
        }
        std::erase_if(
            submission.heterogeneous_axis_batches,
            [](const HeterogeneousAxisBatch &batch) {
                return batch.plane_batches.size() < 2U;
            });
        std::erase_if(
            submission.heterogeneous_two_axis_batches,
            [](const HeterogeneousTwoAxisBatch &batch) {
                return batch.plane_batches.size() < 2U;
            });

        std::size_t scratch_bytes = 0U;
        for (const HeterogeneousTwoAxisBatch &heterogeneous :
             submission.heterogeneous_two_axis_batches) {
            std::size_t group_bytes = 0U;
            for (std::size_t batch_index : heterogeneous.plane_batches) {
                const PlaneBatch &batch = submission.plane_batches[batch_index];
                const PlaneJob &plane = plane_for(
                    submission, batch.planes.front());
                const PlaneLayout &first_layout = layout_for(
                    submission, batch.planes.front());
                const std::size_t frame_bytes = checked_align_up(
                    checked_multiply(
                        checked_multiply(
                            plane.source_height, first_layout.intermediate_stride,
                            "heterogeneous scratch arena"),
                        sizeof(float), "heterogeneous scratch arena"),
                    arena_alignment, "heterogeneous scratch arena");
                for (PlaneReference reference : batch.planes) {
                    PlaneLayout &layout = layout_for(submission, reference);
                    layout.intermediate_offset = group_bytes;
                    group_bytes = checked_add(
                        group_bytes, frame_bytes,
                        "heterogeneous scratch arena");
                }
            }
            scratch_bytes = std::max(scratch_bytes, group_bytes);
        }
        return scratch_bytes;
    }

    template <class T>
    [[nodiscard]] static std::uint32_t append_plan_values(
        std::vector<std::byte> &arena, const std::vector<T> &values,
        const char *label) {
        constexpr std::size_t plan_alignment = 16U;
        const std::size_t offset = checked_align_up(
            arena.size(), plan_alignment, "Metal plan arena");
        const std::size_t count = std::max<std::size_t>(1U, values.size());
        const std::size_t end = checked_add(
            offset, checked_multiply(count, sizeof(T), "Metal plan arena"),
            "Metal plan arena");
        arena.resize(end, std::byte{});
        if (!values.empty()) {
            std::memcpy(arena.data() + offset, values.data(),
                        checked_multiply(
                            values.size(), sizeof(T), "Metal plan arena copy"));
        }
        return checked_u32(offset / sizeof(T), label);
    }

    [[nodiscard]] std::vector<std::byte> pack_heterogeneous_plans(
        Submission &submission) const {
        std::vector<std::byte> arena;
        const auto pack_plan = [&](const std::shared_ptr<const AxisPlan> &plan) {
            if (submission.packed_plans.contains(plan.get())) return;
            PackedPlanOffsets offsets;
            offsets.transpose_offsets = append_plan_values(
                arena, plan->transpose_offsets, "transpose offset arena");
            offsets.transpose_indices = append_plan_values(
                arena, plan->transpose_indices, "transpose index arena");
            offsets.transpose_weights = append_plan_values(
                arena, plan->transpose_weights, "transpose weight arena");
            offsets.lower_ld = append_plan_values(
                arena, plan->lower_ld, "lower factor arena");
            offsets.upper_l = append_plan_values(
                arena, plan->upper_l, "upper factor arena");
            offsets.inverse_diagonal = append_plan_values(
                arena, plan->inverse_diagonal, "diagonal arena");
            submission.packed_plans.emplace(plan.get(), offsets);
        };
        for (const HeterogeneousAxisBatch &heterogeneous :
             submission.heterogeneous_axis_batches) {
            for (std::size_t batch_index : heterogeneous.plane_batches) {
                const PlaneBatch &batch = submission.plane_batches[batch_index];
                const PlaneJob &plane = plane_for(
                    submission, batch.planes.front());
                const auto &plan = heterogeneous.horizontal
                    ? plane.horizontal : plane.vertical;
                pack_plan(plan);
            }
        }
        for (const HeterogeneousTwoAxisBatch &heterogeneous :
             submission.heterogeneous_two_axis_batches) {
            for (std::size_t batch_index : heterogeneous.plane_batches) {
                const PlaneBatch &batch = submission.plane_batches[batch_index];
                const PlaneJob &plane = plane_for(
                    submission, batch.planes.front());
                pack_plan(plane.horizontal);
                pack_plan(plane.vertical);
            }
        }
        if (arena.empty()) return arena;
        arena.resize(checked_align_up(
            arena.size(), arena_alignment, "Metal plan arena"), std::byte{});
        if (arena.size() > plan_cache_budget_) {
            throw std::length_error(
                "heterogeneous Metal plan arena exceeds cache budget");
        }
        return arena;
    }

    [[nodiscard]] static std::uint32_t batch_frame_stride(
        const Submission &submission, const PlaneBatch &batch, Arena arena,
        const char *label) {
        if (batch.planes.size() < 2U) return 0U;
        const PlaneJob &plane = plane_for(submission, batch.planes.front());
        const std::size_t first = arena_offset(
            layout_for(submission, batch.planes[0]), arena);
        const std::size_t second = arena_offset(
            layout_for(submission, batch.planes[1]), arena);
        if (second < first) {
            throw std::logic_error("Metal batch arena offsets are not monotonic");
        }
        const std::size_t delta = second - first;
        for (std::size_t index = 2; index < batch.planes.size(); ++index) {
            if (arena_offset(layout_for(submission, batch.planes[index]), arena)
                != first + index * delta) {
                throw std::logic_error("Metal batch arena offsets are not regular");
            }
        }
        const std::size_t element_bytes = arena_element_bytes(
            plane, layout_for(submission, batch.planes.front()), arena);
        if (delta % element_bytes != 0U) {
            throw std::logic_error("Metal batch arena stride is misaligned");
        }
        return checked_u32(delta / element_bytes, label);
    }

    void encode_axis_batch(
        id<MTLCommandBuffer> command, Submission &submission,
        const PlaneBatch &batch, const std::shared_ptr<const AxisPlan> &plan,
        bool horizontal, id<MTLBuffer> input, Arena input_arena,
        id<MTLBuffer> output, Arena output_arena, bool integer_input) {
        const PlaneJob &plane = plane_for(submission, batch.planes.front());
        const PlaneLayout &layout = layout_for(
            submission, batch.planes.front());
        const bool transposed = horizontal && layout.transposed_input;
        PlanBuffers &prepared = prepare_plan(plan);
        const std::uint32_t vector_count = horizontal
            ? plane.source_height : plane.destination_width;
        const AxisJob job{
            checked_u32(plan->source_size, "axis source size"),
            checked_u32(plan->destination_size, "axis destination size"),
            vector_count, arena_stride(layout, input_arena),
            arena_stride(layout, output_arena),
            horizontal ? 0U : 1U,
            checked_u32(plan->half_bandwidth, "axis half bandwidth"),
            transposed ? 1U : 0U,
            checked_u32(batch.planes.size(), "Metal batch size"),
            batch_frame_stride(
                submission, batch, input_arena, "input frame stride"),
            batch_frame_stride(
                submission, batch, output_arena, "output frame stride"),
            0U};

        id<MTLComputePipelineState> pipeline = nil;
        const std::size_t index = pipeline_index(job.half_bandwidth);
        if (!integer_input) {
            pipeline = transposed
                ? transposed_pipelines_[index] : float_pipelines_[index];
        } else if (plane.sample_bytes == 1U) {
            pipeline = u8_pipelines_[index];
        } else {
            pipeline = u16_pipelines_[index];
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) throw std::runtime_error("Metal encoder creation failed");
        encoder.label = horizontal ? @"dsmvc horizontal inverse"
                                   : @"dsmvc vertical inverse";
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input
                    offset:arena_offset(layout, input_arena) atIndex:0];
        [encoder setBytes:&job length:sizeof(job) atIndex:1];
        bind_plan(encoder, prepared);
        [encoder setBuffer:output
                    offset:arena_offset(layout, output_arena) atIndex:8];
        if (integer_input) {
            std::vector<IntegerConversion> conversions;
            conversions.reserve(batch.planes.size());
            for (PlaneReference reference : batch.planes) {
                conversions.push_back(plane_for(submission, reference).conversion);
            }
            [encoder setBytes:conversions.data()
                       length:conversions.size() * sizeof(conversions.front())
                      atIndex:9];
        }
        dispatch(
            encoder, pipeline,
            static_cast<std::size_t>(vector_count) * batch.planes.size());
        [encoder endEncoding];
        ++submission.axis_dispatches;
    }

    void encode_heterogeneous_axis_batch(
        id<MTLCommandBuffer> command, Submission &submission,
        const std::vector<std::size_t> &plane_batches,
        std::size_t pipeline_index_value, bool horizontal,
        id<MTLBuffer> input, Arena input_arena,
        id<MTLBuffer> output, Arena output_arena,
        id<MTLBuffer> packed_plans) {
        std::vector<AxisBatchJob> jobs;
        jobs.reserve(plane_batches.size());
        std::size_t maximum_vectors = 0U;
        for (std::size_t batch_index : plane_batches) {
            const PlaneBatch &batch = submission.plane_batches[batch_index];
            const PlaneJob &plane = plane_for(
                submission, batch.planes.front());
            const PlaneLayout &layout = layout_for(
                submission, batch.planes.front());
            const bool transposed = horizontal && layout.transposed_input;
            const auto &plan = horizontal ? plane.horizontal : plane.vertical;
            const auto packed = submission.packed_plans.find(plan.get());
            if (packed == submission.packed_plans.end()) {
                throw std::logic_error("Metal plan is absent from packed arena");
            }
            const std::uint32_t vector_count = horizontal
                ? plane.source_height : plane.destination_width;
            const AxisJob axis{
                checked_u32(plan->source_size, "axis source size"),
                checked_u32(plan->destination_size, "axis destination size"),
                vector_count, arena_stride(layout, input_arena),
                arena_stride(layout, output_arena),
                horizontal ? 0U : 1U,
                checked_u32(plan->half_bandwidth, "axis half bandwidth"),
                transposed ? 1U : 0U,
                checked_u32(batch.planes.size(), "Metal batch size"),
                batch_frame_stride(
                    submission, batch, input_arena, "input frame stride"),
                batch_frame_stride(
                    submission, batch, output_arena, "output frame stride"),
                0U};
            const std::size_t input_offset = arena_offset(layout, input_arena);
            const std::size_t output_offset = arena_offset(layout, output_arena);
            if (input_offset % sizeof(float) != 0U
                || output_offset % sizeof(float) != 0U) {
                throw std::logic_error(
                    "heterogeneous Metal arena offset is misaligned");
            }
            jobs.push_back(AxisBatchJob{
                axis,
                checked_u32(
                    input_offset / sizeof(float), "input arena offset"),
                checked_u32(
                    output_offset / sizeof(float), "output arena offset"),
                packed->second.transpose_offsets,
                packed->second.transpose_indices,
                packed->second.transpose_weights,
                packed->second.lower_ld,
                packed->second.upper_l,
                packed->second.inverse_diagonal});
            maximum_vectors = std::max(
                maximum_vectors,
                static_cast<std::size_t>(vector_count) * batch.planes.size());
        }
        if (jobs.empty() || jobs.size() > maximum_batch_descriptors) {
            throw std::logic_error("invalid heterogeneous Metal batch");
        }

        const PlaneBatch &first_batch = submission.plane_batches[
            plane_batches.front()];
        const bool transposed = horizontal && layout_for(
            submission, first_batch.planes.front()).transposed_input;
        id<MTLComputePipelineState> pipeline =
            float_batch_pipeline(pipeline_index_value, transposed);
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) throw std::runtime_error("Metal encoder creation failed");
        encoder.label = horizontal
            ? @"dsmvc heterogeneous horizontal inverse"
            : @"dsmvc heterogeneous vertical inverse";
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBytes:jobs.data()
                   length:jobs.size() * sizeof(jobs.front()) atIndex:1];
        for (NSUInteger index = 2U; index <= 7U; ++index) {
            [encoder setBuffer:packed_plans offset:0 atIndex:index];
        }
        [encoder setBuffer:output offset:0 atIndex:8];
        dispatch_2d(encoder, pipeline, maximum_vectors, jobs.size());
        [encoder endEncoding];
        ++submission.axis_dispatches;
        ++submission.heterogeneous_axis_dispatches;
        submission.heterogeneous_axis_descriptors += jobs.size();
    }

    void encode_conversion_batch(
        id<MTLCommandBuffer> command, Submission &submission,
        const PlaneBatch &batch, id<MTLBuffer> scratch, id<MTLBuffer> output) {
        const PlaneJob &plane = plane_for(submission, batch.planes.front());
        const PlaneLayout &layout = layout_for(
            submission, batch.planes.front());
        const ConvertJob job{
            plane.destination_width, plane.destination_height,
            layout.result_stride, layout.output_stride,
            checked_u32(batch.planes.size(), "Metal conversion batch size"),
            batch_frame_stride(
                submission, batch, Arena::result, "conversion input stride"),
            batch_frame_stride(
                submission, batch, Arena::output, "conversion output stride"),
            0U};
        id<MTLComputePipelineState> pipeline = plane.sample_bytes == 1U
            ? convert_u8_ : convert_u16_;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) throw std::runtime_error("Metal encoder creation failed");
        encoder.label = @"dsmvc integer conversion";
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:scratch offset:layout.result_offset atIndex:0];
        [encoder setBytes:&job length:sizeof(job) atIndex:1];
        std::vector<IntegerConversion> conversions;
        conversions.reserve(batch.planes.size());
        for (PlaneReference reference : batch.planes) {
            conversions.push_back(plane_for(submission, reference).conversion);
        }
        [encoder setBytes:conversions.data()
                   length:conversions.size() * sizeof(conversions.front())
                  atIndex:2];
        [encoder setBuffer:output offset:layout.output_offset atIndex:3];
        dispatch(
            encoder, pipeline,
            static_cast<std::size_t>(plane.destination_width)
                * plane.destination_height * batch.planes.size());
        [encoder endEncoding];
        ++submission.conversion_dispatches;
    }

    [[nodiscard]] std::shared_ptr<Submission> prepare_submission(
        Slot &slot, const std::vector<std::shared_ptr<Request>> &requests) {
        auto submission = std::make_shared<Submission>();
        std::unordered_map<InputKey, std::size_t, InputKeyHash> staged_inputs;
        std::unordered_set<InputKey, InputKeyHash> unique_inputs;
        std::size_t input_bytes = 0;
        std::size_t output_bytes = 0;

        for (const auto &request : requests) {
            if (request->job.planes.empty()) {
                throw std::invalid_argument("Metal frame job has no planes");
            }
            submission->profile_signposts = submission->profile_signposts
                || request->job.profile_signposts;
            JobLayout job_layout;
            job_layout.request = request;
            job_layout.planes.reserve(request->job.planes.size());
            for (const PlaneJob &plane : request->job.planes) {
                if (!plane.source || !plane.destination
                    || plane.source_width == 0U || plane.source_height == 0U
                    || plane.destination_width == 0U
                    || plane.destination_height == 0U
                    || (plane.sample_bytes != 1U && plane.sample_bytes != 2U
                        && plane.sample_bytes != 4U)
                    || (!plane.process_horizontal && !plane.process_vertical)
                    || (plane.process_horizontal && !plane.horizontal)
                    || (plane.process_vertical && !plane.vertical)) {
                    throw std::invalid_argument("invalid general Metal plane job");
                }
                PlaneLayout layout;
                const bool source_stride_usable = plane.source_stride_bytes > 0
                    && plane.source_stride_bytes
                        % static_cast<std::ptrdiff_t>(plane.sample_bytes) == 0
                    && plane.source_stride_bytes >= static_cast<std::ptrdiff_t>(
                        static_cast<std::size_t>(plane.source_width)
                            * plane.sample_bytes);
                if (!source_stride_usable) {
                    throw std::invalid_argument("invalid Metal source stride");
                }
                layout.input_stride = checked_u32(
                    static_cast<std::size_t>(plane.source_stride_bytes)
                        / plane.sample_bytes,
                    "input stride");
                layout.intermediate_stride = checked_u32(
                    checked_align_up(
                        checked_multiply(
                            plane.destination_width, sizeof(float),
                            "intermediate stride"),
                        64U, "intermediate stride")
                        / sizeof(float),
                    "intermediate stride");
                layout.result_stride = layout.intermediate_stride;
                const bool destination_stride_usable =
                    plane.destination_stride_bytes > 0
                    && plane.destination_stride_bytes
                        % static_cast<std::ptrdiff_t>(plane.sample_bytes) == 0
                    && plane.destination_stride_bytes >= static_cast<std::ptrdiff_t>(
                        static_cast<std::size_t>(plane.destination_width)
                            * plane.sample_bytes);
                if (!destination_stride_usable) {
                    throw std::invalid_argument("invalid Metal destination stride");
                }
                layout.output_stride = checked_u32(
                    static_cast<std::size_t>(plane.destination_stride_bytes)
                        / plane.sample_bytes,
                    "output stride");

                if (const auto resident = acquire_resident_input(
                        plane, request->client_identity, *submission)) {
                    layout.input_offset = resident->offset;
                    layout.input_stride = resident->stride;
                    layout.resident_input = true;
                    layout.transposed_input = true;
                    unique_inputs.emplace(resident->key);
                } else {
                    const InputKey key = input_key(plane, false);
                    unique_inputs.emplace(key);
                    const auto found = staged_inputs.find(key);
                    if (found == staged_inputs.end()) {
                        layout.input_offset = checked_align_up(
                            input_bytes, arena_alignment, "input arena");
                        staged_inputs.emplace(key, layout.input_offset);
                        const std::size_t plane_bytes = checked_multiply(
                            checked_multiply(
                                plane.source_height, layout.input_stride,
                                "input arena"),
                            plane.sample_bytes, "input arena");
                        input_bytes = checked_add(
                            layout.input_offset, plane_bytes, "input arena");
                    } else {
                        layout.input_offset = found->second;
                    }
                }

                layout.output_offset = checked_align_up(
                    output_bytes, arena_alignment, "output arena");
                const std::size_t plane_output_bytes = checked_multiply(
                    checked_multiply(
                        plane.destination_height, layout.output_stride,
                        "output arena"),
                    plane.sample_bytes, "output arena");
                output_bytes = checked_add(
                    layout.output_offset, plane_output_bytes, "output arena");
                job_layout.planes.push_back(layout);
            }
            submission->jobs.push_back(std::move(job_layout));
        }
        submission->unique_input_planes = unique_inputs.size();
        const std::size_t scratch_bytes = prepare_plane_batches(*submission);
        const std::size_t heterogeneous_scratch_bytes =
            prepare_heterogeneous_axis_batches(*submission);
        const std::vector<std::byte> plan_arena =
            pack_heterogeneous_plans(*submission);
        ensure_slot_capacity(
            slot, input_bytes, std::max(scratch_bytes, heterogeneous_scratch_bytes),
            output_bytes, plan_arena.size());
        if (!plan_arena.empty()) {
            std::memcpy(slot.plan.contents, plan_arena.data(), plan_arena.size());
        }

        auto *input_base = input_bytes == 0U ? nullptr
            : static_cast<std::byte *>(slot.input.contents);
        std::unordered_map<InputKey, bool, InputKeyHash> copied;
        for (std::size_t job_index = 0; job_index < submission->jobs.size(); ++job_index) {
            const auto &job = submission->jobs[job_index];
            const auto &planes = job.request->job.planes;
            for (std::size_t plane_index = 0; plane_index < planes.size(); ++plane_index) {
                const PlaneJob &plane = planes[plane_index];
                const PlaneLayout &layout = job.planes[plane_index];
                if (layout.resident_input) continue;
                const InputKey key = input_key(plane, false);
                if (copied.emplace(key, true).second) {
                    copy_rows(
                        input_base + layout.input_offset,
                        static_cast<std::ptrdiff_t>(layout.input_stride)
                            * plane.sample_bytes,
                        plane.source, plane.source_stride_bytes,
                        static_cast<std::size_t>(plane.source_width)
                            * plane.sample_bytes,
                        plane.source_height, submission->stats);
                }
            }
        }
        return submission;
    }

    void encode_submission(
        id<MTLCommandBuffer> command, Slot &slot,
        Submission &submission) {
        std::vector<bool> consumed(submission.plane_batches.size(), false);
        for (const HeterogeneousTwoAxisBatch &heterogeneous :
             submission.heterogeneous_two_axis_batches) {
            id<MTLBuffer> horizontal_input = heterogeneous.resident_input
                ? resident_input_ : slot.input;
            encode_heterogeneous_axis_batch(
                command, submission, heterogeneous.plane_batches,
                heterogeneous.horizontal_pipeline, true,
                horizontal_input, Arena::input,
                slot.scratch, Arena::intermediate, slot.plan);
            encode_heterogeneous_axis_batch(
                command, submission, heterogeneous.plane_batches,
                heterogeneous.vertical_pipeline, false,
                slot.scratch, Arena::intermediate,
                slot.output, Arena::output, slot.plan);
            for (std::size_t batch_index : heterogeneous.plane_batches) {
                consumed[batch_index] = true;
            }
        }
        for (const HeterogeneousAxisBatch &heterogeneous :
             submission.heterogeneous_axis_batches) {
            id<MTLBuffer> input = heterogeneous.horizontal
                    && heterogeneous.resident_input
                ? resident_input_ : slot.input;
            encode_heterogeneous_axis_batch(
                command, submission, heterogeneous.plane_batches,
                heterogeneous.pipeline, heterogeneous.horizontal,
                input, Arena::input,
                slot.output, Arena::output, slot.plan);
            for (std::size_t batch_index : heterogeneous.plane_batches) {
                consumed[batch_index] = true;
            }
        }
        for (std::size_t batch_index = 0;
             batch_index < submission.plane_batches.size(); ++batch_index) {
            if (consumed[batch_index]) continue;
            const PlaneBatch &batch = submission.plane_batches[batch_index];
            const PlaneJob &plane = plane_for(
                submission, batch.planes.front());
            const PlaneLayout &layout = layout_for(
                submission, batch.planes.front());
            id<MTLBuffer> current_buffer = layout.resident_input
                ? resident_input_ : slot.input;
            Arena current_arena = Arena::input;
            bool current_integer = plane.integer_samples
                && !layout.resident_input;

            if (plane.process_horizontal) {
                const bool final_float = !plane.process_vertical
                    && !plane.integer_samples;
                id<MTLBuffer> target = final_float ? slot.output : slot.scratch;
                const Arena target_arena = final_float
                    ? Arena::output
                    : (plane.process_vertical ? Arena::intermediate
                                              : Arena::result);
                encode_axis_batch(
                    command, submission, batch, plane.horizontal, true,
                    current_buffer, current_arena, target, target_arena,
                    current_integer);
                current_buffer = target;
                current_arena = target_arena;
                current_integer = false;
            }
            if (plane.process_vertical) {
                const bool final_float = !plane.integer_samples;
                id<MTLBuffer> target = final_float ? slot.output : slot.scratch;
                const Arena target_arena = final_float
                    ? Arena::output : Arena::result;
                encode_axis_batch(
                    command, submission, batch, plane.vertical, false,
                    current_buffer, current_arena, target, target_arena,
                    current_integer);
            }
            if (plane.integer_samples) {
                encode_conversion_batch(
                    command, submission, batch, slot.scratch, slot.output);
            }
        }
    }

    void download_submission(Slot &slot, Submission &submission) {
        const auto *output_base = static_cast<const std::byte *>(slot.output.contents);
        for (const JobLayout &job_layout : submission.jobs) {
            const auto &planes = job_layout.request->job.planes;
            for (std::size_t index = 0; index < planes.size(); ++index) {
                const PlaneJob &plane = planes[index];
                const PlaneLayout &layout = job_layout.planes[index];
                copy_rows(
                    plane.destination, plane.destination_stride_bytes,
                    output_base + layout.output_offset,
                    static_cast<std::ptrdiff_t>(layout.output_stride)
                        * plane.sample_bytes,
                    static_cast<std::size_t>(plane.destination_width)
                        * plane.sample_bytes,
                    plane.destination_height, submission.stats);
            }
        }
    }

    void submit_on_slot(
        std::size_t slot_index,
        std::vector<std::shared_ptr<Request>> requests,
        Completion completion) {
        @autoreleasepool {
            Slot &slot = slots_[slot_index];
            auto submission = prepare_submission(slot, requests);
            if (submission->profile_signposts) {
                const os_log_t log = profile_log();
                submission->signpost_id = os_signpost_id_generate(log);
                os_signpost_interval_begin(
                    log, submission->signpost_id, "DSMVCMetalBatch",
                    "jobs=%zu unique_inputs=%zu", requests.size(),
                    submission->unique_input_planes);
            }
            id<MTLCommandBuffer> command = [queue_ commandBuffer];
            if (command == nil) {
                throw std::runtime_error("Metal command buffer creation failed");
            }
            command.label = @"dsmvc general UMA batch";
            encode_submission(command, slot, *submission);
            {
                const std::scoped_lock lock(submission_timing_mutex_);
                const auto commit_time = Clock::now();
                if (last_submission_commit_) {
                    const auto gap = std::chrono::duration_cast<
                        std::chrono::nanoseconds>(
                            commit_time - *last_submission_commit_).count();
                    submission->submission_gap_nanoseconds =
                        static_cast<std::uint64_t>(
                            std::max<std::int64_t>(gap, 0));
                }
                last_submission_commit_ = commit_time;
            }
            const std::size_t batch_size = requests.size();
            [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                std::exception_ptr error;
                if (completed.status == MTLCommandBufferStatusError) {
                    error = std::make_exception_ptr(std::runtime_error(
                        ns_error(completed.error, "Metal execution failed")));
                } else {
                    try {
                        download_submission(slot, *submission);
                    } catch (...) {
                        error = std::current_exception();
                    }
                }
                if (error) record_metal_error();
                else record_metal_success();
                if (submission->profile_signposts) {
                    os_signpost_interval_end(
                        profile_log(), submission->signpost_id,
                        "DSMVCMetalBatch",
                        "copies=%zu bytes=%zu", submission->stats.calls,
                        submission->stats.bytes);
                }
                std::uint64_t gpu_interval_nanoseconds = 0U;
                if (completed.GPUEndTime >= completed.GPUStartTime
                    && completed.GPUStartTime > 0.0) {
                    const double interval =
                        (completed.GPUEndTime - completed.GPUStartTime) * 1.0e9;
                    if (interval > 0.0
                        && interval <= static_cast<double>(
                            std::numeric_limits<std::uint64_t>::max())) {
                        gpu_interval_nanoseconds =
                            static_cast<std::uint64_t>(interval);
                    }
                }
                RunResult result{
                    batch_size, submission->stats.calls,
                    submission->stats.bytes,
                    submission->unique_input_planes,
                    submission->axis_dispatches,
                    submission->conversion_dispatches,
                    submission->heterogeneous_axis_dispatches,
                    submission->heterogeneous_axis_descriptors,
                    submission->resident_producers,
                    submission->resident_hits,
                    submission->resident_evictions,
                    resident_logical_bytes_.load(std::memory_order_relaxed),
                    submission->eliminated_staging_bytes,
                    gpu_interval_nanoseconds,
                    submission->submission_gap_nanoseconds};
                std::vector<std::shared_ptr<Request>> completed_requests;
                completed_requests.reserve(submission->jobs.size());
                for (const JobLayout &job : submission->jobs) {
                    completed_requests.push_back(job.request);
                }
                // The command buffer and host download are complete; release
                // cache pins before waking clients that may immediately close.
                submission->resident_leases.clear();
                try {
                    completion(completed_requests, result, error);
                } catch (...) {
                    // A host notification failure must not permanently consume a slot.
                }
                release_slot(slot_index);
            }];
            [command commit];
        }
    }

    void release_slot(std::size_t index) noexcept {
        {
            const std::scoped_lock lock(slots_mutex_);
            slots_[index].in_use = false;
            if (in_flight_ != 0U) --in_flight_;
            in_flight_snapshot_.store(in_flight_, std::memory_order_release);
            ++completions_;
        }
        slots_ready_.notify_all();
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLLibrary> library_ = nil;
    std::array<id<MTLComputePipelineState>, 5> float_pipelines_{};
    std::array<id<MTLComputePipelineState>, 5> float_batch_pipelines_{};
    std::array<id<MTLComputePipelineState>, 5> transposed_pipelines_{};
    std::array<id<MTLComputePipelineState>, 5> transposed_batch_pipelines_{};
    std::array<id<MTLComputePipelineState>, 5> u8_pipelines_{};
    std::array<id<MTLComputePipelineState>, 5> u16_pipelines_{};
    id<MTLComputePipelineState> convert_u8_ = nil;
    id<MTLComputePipelineState> convert_u16_ = nil;
    id<MTLBuffer> resident_input_ = nil;
    std::unordered_map<const AxisPlan *, PlanBuffers> plans_;
    std::unordered_map<InputKey, std::shared_ptr<ResidentInput>, InputKeyHash>
        resident_inputs_;
    std::vector<FreeRegion> resident_free_regions_;
    std::array<Slot, ring_size> slots_{};
    mutable std::mutex slots_mutex_;
    mutable std::mutex resident_mutex_;
    std::condition_variable resident_ready_;
    std::mutex submission_timing_mutex_;
    std::condition_variable slots_ready_;
    std::size_t in_flight_ = 0;
    std::atomic<std::size_t> in_flight_snapshot_{0U};
    std::size_t submissions_ = 0;
    std::size_t completions_ = 0;
    std::size_t maximum_in_flight_ = 0;
    std::size_t ring_bytes_ = 0;
    std::size_t plan_bytes_ = 0;
    std::size_t memory_budget_ = 0;
    std::size_t plan_cache_budget_ = 0;
    std::size_t resident_capacity_ = 0;
    std::size_t resident_bytes_ = 0;
    std::uint64_t plan_use_sequence_ = 0;
    std::uint64_t resident_use_sequence_ = 0;
    std::optional<Clock::time_point> last_submission_commit_;
    std::atomic<std::size_t> plan_entry_count_{0U};
    std::atomic<std::size_t> plan_resident_bytes_{0U};
    std::atomic<std::size_t> plan_evictions_{0U};
    std::atomic<std::size_t> resident_entry_count_{0U};
    std::atomic<std::size_t> resident_logical_bytes_{0U};
    std::atomic<std::size_t> resident_evictions_{0U};
    std::atomic<std::size_t> resident_pinned_eviction_blocks_{0U};
    std::atomic<std::size_t> resident_producers_{0U};
    std::atomic<std::size_t> resident_hits_{0U};
    std::atomic<std::size_t> metal_errors_{0U};
    std::atomic<std::size_t> consecutive_metal_errors_{0U};
    std::atomic<std::size_t> maximum_consecutive_metal_errors_{0U};
};

class Scheduler final {
public:
    Scheduler() : worker_([this] { worker_loop(); }) {}

    ~Scheduler() {
        {
            const std::scoped_lock lock(mutex_);
            stopping_ = true;
            flush_to_cpu(narrow_);
            flush_to_cpu(wide_);
        }
        ready_.notify_all();
        if (worker_.joinable()) worker_.join();
        if (device_) device_->wait_idle();
    }

    [[nodiscard]] SchedulerDiagnostics diagnostics() noexcept {
        const std::scoped_lock lock(mutex_);
        return device_ ? device_->diagnostics()
                       : SchedulerDiagnostics{ring_size};
    }

    void release_client(std::uint64_t identity) noexcept {
        {
            const std::scoped_lock lock(admission_mutex_);
            for (auto input = recent_inputs_.begin();
                 input != recent_inputs_.end();) {
                std::erase_if(input->second.clients, [&](const auto &entry) {
                    return entry.identity == identity;
                });
                if (input->second.clients.empty()) {
                    input = recent_inputs_.erase(input);
                } else {
                    ++input;
                }
            }
        }
        DeviceContext *device = nullptr;
        {
            const std::scoped_lock lock(mutex_);
            device = device_.get();
        }
        if (device) device->release_client(identity);
    }

    RunResult run(
        const std::shared_ptr<Client::Impl> &client, FrameJob job,
        std::function<void()> cpu_work, bool automatic) {
        if (!client || !cpu_work) {
            throw std::invalid_argument("Metal scheduler requires client and CPU work");
        }
        ClientCall call(client);
        ActiveCall active(active_calls_);
        const std::size_t recent_shared_clients =
            observe_shared_input(client, job, automatic);
        const bool wide = job.maximum_half_bandwidth >= 5U;
        const bool regular_automatic = automatic
            && active.count() >= 8U
            && eligible_for_automatic_gpu(job, automatic_work_floor);
        const bool shared_input_automatic = automatic
            && recent_shared_clients >= shared_input_clients
            && eligible_for_automatic_gpu(job, shared_input_work_floor);
        DeviceContext::AdmissionSnapshot admission;
        if (automatic
            && eligible_for_automatic_gpu(job, shared_input_work_floor)) {
            DeviceContext *device = nullptr;
            {
                const std::scoped_lock lock(mutex_);
                device = device_.get();
            }
            if (device) admission = device->admission_snapshot(job);
        }
        const bool resident_automatic = automatic && admission.resident_ready
            && eligible_for_automatic_gpu(job, shared_input_work_floor);
        if ((automatic && !regular_automatic && !shared_input_automatic
             && !resident_automatic)
            || (!automatic && !eligible_for_gpu(job))
            || (!shared_input_automatic && !resident_automatic
                && choose_cpu(wide))) {
            cpu_work();
            return {};
        }

        auto request = std::make_shared<Request>();
        request->job = std::move(job);
        request->client_identity = client->identity;
        request->resident_preferred = resident_automatic;
        Queue *queue = wide ? &wide_ : &narrow_;
        bool fallback_to_cpu = false;
        std::exception_ptr scheduler_error;
        {
            const std::scoped_lock lock(mutex_);
            if (terminal_error_) {
                scheduler_error = terminal_error_;
            } else if (stopping_ || queued_jobs_ >= maximum_queued_jobs
                       || (automatic && device_
                           && !device_->ring_has_capacity()
                           && queued_jobs_ != 0U)) {
                fallback_to_cpu = true;
            } else {
                if (queue->requests.empty()) {
                    queue->deadline = Clock::now() + batch_timeout;
                }
                queue->requests.push_back(request);
                ++queued_jobs_;
            }
        }
        if (scheduler_error) {
            if (!automatic) std::rethrow_exception(scheduler_error);
            cpu_work();
            return {};
        }
        if (fallback_to_cpu) {
            cpu_work();
            return {};
        }
        ready_.notify_one();

        std::unique_lock request_lock(request->mutex);
        request->ready.wait(request_lock, [&] {
            return request->state != Request::State::pending
                && request->state != Request::State::submitted;
        });
        if (request->state == Request::State::cpu) {
            request_lock.unlock();
            cpu_work();
            return {};
        }
        if (request->state == Request::State::failed) {
            const auto error = request->error;
            request_lock.unlock();
            if (automatic) {
                cpu_work();
                return {};
            }
            std::rethrow_exception(error);
        }
        return request->result;
    }

private:
    struct Queue {
        std::deque<std::shared_ptr<Request>> requests;
        std::optional<Clock::time_point> deadline;
    };

    struct RecentClient {
        std::uint64_t identity = 0U;
        std::weak_ptr<Client::Impl> client;
        Clock::time_point last_seen;
    };

    struct RecentInput {
        std::deque<RecentClient> clients;
        Clock::time_point last_seen;
    };

    class ClientCall final {
    public:
        explicit ClientCall(const std::shared_ptr<Client::Impl> &client)
            : client_(client) {
            std::unique_lock lock(client_->mutex);
            if (client_->closing) {
                throw std::runtime_error("Metal scheduler client is closed");
            }
            client_->used = true;
            ++client_->active;
        }
        ~ClientCall();

    private:
        std::shared_ptr<Client::Impl> client_;
    };

    class ActiveCall final {
    public:
        explicit ActiveCall(std::atomic<std::size_t> &active) noexcept
            : active_(active), count_(
                  active_.fetch_add(1U, std::memory_order_relaxed) + 1U) {}
        ~ActiveCall() {
            active_.fetch_sub(1U, std::memory_order_relaxed);
        }
        [[nodiscard]] std::size_t count() const noexcept { return count_; }

    private:
        std::atomic<std::size_t> &active_;
        std::size_t count_ = 0U;
    };

    void prune_recent_inputs(Clock::time_point now) {
        for (auto input = recent_inputs_.begin(); input != recent_inputs_.end();) {
            auto &clients = input->second.clients;
            std::erase_if(clients, [&](const RecentClient &recent) {
                return recent.client.expired()
                    || now - recent.last_seen > recent_input_window;
            });
            if (clients.empty()
                || now - input->second.last_seen > recent_input_window) {
                input = recent_inputs_.erase(input);
            } else {
                ++input;
            }
        }
    }

    [[nodiscard]] std::size_t observe_shared_input(
        const std::shared_ptr<Client::Impl> &client,
        const FrameJob &job, bool automatic) {
        if (!automatic || job.planes.empty()) return 0U;
        const PlaneJob &plane = job.planes.front();
        if (!plane.source || plane.source_stride_bytes <= 0
            || plane.source_width == 0U || plane.source_height == 0U
            || plane.sample_bytes == 0U) {
            return 0U;
        }
        const InputKey key = input_key(plane, false);
        const auto now = Clock::now();
        const std::scoped_lock lock(admission_mutex_);
        prune_recent_inputs(now);
        auto input = recent_inputs_.find(key);
        if (input == recent_inputs_.end()
            && recent_inputs_.size() >= maximum_recent_inputs) {
            const auto oldest = std::min_element(
                recent_inputs_.begin(), recent_inputs_.end(),
                [](const auto &left, const auto &right) {
                    return left.second.last_seen < right.second.last_seen;
                });
            recent_inputs_.erase(oldest);
        }
        RecentInput &recent = recent_inputs_[key];
        const auto existing = std::find_if(
            recent.clients.begin(), recent.clients.end(),
            [&](const RecentClient &entry) {
                return entry.identity == client->identity;
            });
        if (existing != recent.clients.end()) {
            recent.clients.erase(existing);
        } else if (recent.clients.size() >= maximum_recent_clients) {
            recent.clients.pop_front();
        }
        recent.clients.push_back(RecentClient{
            client->identity, client, now});
        recent.last_seen = now;
        return recent.clients.size();
    }

    [[nodiscard]] static bool eligible_for_gpu(const FrameJob &job) noexcept {
        return !job.planes.empty();
    }

    [[nodiscard]] static bool eligible_for_automatic_gpu(
        const FrameJob &job, std::uint64_t work_floor) noexcept {
        return job.maximum_half_bandwidth >= 5U
            && job.estimated_work >= work_floor;
    }

    [[nodiscard]] bool choose_cpu(bool wide) noexcept {
        std::atomic<std::size_t> &sequence = wide ? wide_sequence_ : narrow_sequence_;
        const std::size_t cpu_frames = wide ? wide_cpu_frames : narrow_cpu_frames;
        return sequence.fetch_add(1U, std::memory_order_relaxed)
                % scheduling_period < cpu_frames;
    }

    void flush_to_cpu(Queue &queue) {
        while (!queue.requests.empty()) {
            auto request = std::move(queue.requests.front());
            queue.requests.pop_front();
            {
                const std::scoped_lock lock(request->mutex);
                request->state = Request::State::cpu;
            }
            request->ready.notify_all();
            if (queued_jobs_ != 0U) --queued_jobs_;
        }
        queue.deadline.reset();
    }

    [[nodiscard]] bool queue_ready(
        const Queue &queue, std::size_t target, Clock::time_point now) const {
        return queue.requests.size() >= target
            || (!queue.requests.empty() && queue.deadline && now >= *queue.deadline);
    }

    std::vector<std::shared_ptr<Request>> take_batch(
        Queue &queue, std::size_t target, Clock::time_point now,
        DeviceContext *device) {
        if (!queue_ready(queue, target, now)) return {};
        const bool resident_singleton = queue.requests.size() == 1U
            && queue.requests.front()->resident_preferred && device
            && device->ring_has_capacity();
        if (queue.requests.size() < 2U && !resident_singleton) {
            flush_to_cpu(queue);
            return {};
        }
        const std::size_t count = std::min(target, queue.requests.size());
        std::vector<std::shared_ptr<Request>> batch;
        batch.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            batch.push_back(std::move(queue.requests.front()));
            queue.requests.pop_front();
            --queued_jobs_;
        }
        queue.deadline = queue.requests.empty()
            ? std::nullopt
            : std::optional<Clock::time_point>{now + batch_timeout};
        for (const auto &request : batch) {
            const std::scoped_lock lock(request->mutex);
            request->state = Request::State::submitted;
        }
        return batch;
    }

    void complete(
        const std::vector<std::shared_ptr<Request>> &requests,
        RunResult result, std::exception_ptr error) {
        for (const auto &request : requests) {
            {
                const std::scoped_lock lock(request->mutex);
                request->result = result;
                request->error = error;
                request->state = error ? Request::State::failed
                                       : Request::State::complete;
            }
            request->ready.notify_all();
        }
        ready_.notify_one();
    }

    void worker_loop() noexcept {
        try {
            DeviceContext *device = nullptr;
            std::unique_lock lock(mutex_);
            while (true) {
                const auto now = Clock::now();
                auto batch = take_batch(wide_, wide_gpu_batch, now, device);
                if (batch.empty()) {
                    batch = take_batch(narrow_, narrow_gpu_batch, now, device);
                }
                if (!batch.empty()) {
                    lock.unlock();
                    try {
                        if (!device) {
                            auto initialized = std::make_unique<DeviceContext>();
                            {
                                const std::scoped_lock device_lock(mutex_);
                                device_ = std::move(initialized);
                                device = device_.get();
                            }
                        }
                        device->submit(
                            batch,
                            [this](const auto &requests, RunResult result,
                                   std::exception_ptr error) {
                                complete(requests, result, error);
                            });
                    } catch (...) {
                        const auto error = std::current_exception();
                        complete(batch, {}, error);
                        if (!device) std::rethrow_exception(error);
                    }
                    lock.lock();
                    continue;
                }
                if (stopping_ && queued_jobs_ == 0U) break;
                std::optional<Clock::time_point> deadline;
                for (const Queue *queue : {&narrow_, &wide_}) {
                    if (queue->deadline
                        && (!deadline || *queue->deadline < *deadline)) {
                        deadline = queue->deadline;
                    }
                }
                if (deadline) {
                    ready_.wait_until(lock, *deadline);
                } else {
                    ready_.wait(lock);
                }
            }
            lock.unlock();
            if (device) device->wait_idle();
        } catch (...) {
            const auto error = std::current_exception();
            std::vector<std::shared_ptr<Request>> failed;
            {
                const std::scoped_lock lock(mutex_);
                for (Queue *queue : {&narrow_, &wide_}) {
                    while (!queue->requests.empty()) {
                        failed.push_back(std::move(queue->requests.front()));
                        queue->requests.pop_front();
                    }
                    queue->deadline.reset();
                }
                queued_jobs_ = 0U;
                stopping_ = true;
                terminal_error_ = error;
            }
            complete(failed, {}, error);
        }
    }

    std::mutex mutex_;
    std::condition_variable ready_;
    Queue narrow_;
    Queue wide_;
    std::size_t queued_jobs_ = 0;
    bool stopping_ = false;
    std::exception_ptr terminal_error_;
    std::atomic<std::size_t> narrow_sequence_{0U};
    std::atomic<std::size_t> wide_sequence_{0U};
    std::atomic<std::size_t> active_calls_{0U};
    std::mutex admission_mutex_;
    std::unordered_map<InputKey, RecentInput, InputKeyHash> recent_inputs_;
    std::unique_ptr<DeviceContext> device_;
    std::thread worker_;
};

Scheduler &scheduler() {
    static Scheduler instance;
    return instance;
}

} // namespace

Scheduler::ClientCall::~ClientCall() {
    {
        const std::scoped_lock lock(client_->mutex);
        if (client_->active != 0U) --client_->active;
    }
    client_->ready.notify_all();
}

Client::Client() : impl_(std::make_shared<Impl>(next_client_identity())) {}

Client::~Client() { close(); }

void Client::close() noexcept {
    if (!impl_) return;
    const auto impl = impl_;
    bool used = false;
    {
        std::unique_lock lock(impl->mutex);
        impl->closing = true;
        impl->ready.wait(lock, [&] { return impl->active == 0U; });
        used = impl->used;
    }
    std::call_once(impl->release_once, [identity = impl->identity, used] {
        if (used) scheduler().release_client(identity);
    });
}

bool available() noexcept {
    return dsmvc::metal_available();
}

std::shared_ptr<Client> make_client() {
    if (!available()) {
        throw std::runtime_error("no Apple unified-memory Metal device is available");
    }
    return std::make_shared<Client>();
}

SchedulerDiagnostics diagnostics() noexcept {
    return scheduler().diagnostics();
}

void fail_next_resident_producer_for_testing() noexcept {
    fail_next_resident_producer.store(true, std::memory_order_relaxed);
}

RunResult run(
    const std::shared_ptr<Client> &client, FrameJob job,
    std::function<void()> cpu_work, bool automatic) {
    if (!client) throw std::invalid_argument("Metal client is required");
    return scheduler().run(
        client->impl_, std::move(job), std::move(cpu_work), automatic);
}

} // namespace dsmvc::metal
