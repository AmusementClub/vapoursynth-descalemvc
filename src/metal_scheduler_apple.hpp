#pragma once

#include <dsmvc/engine.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace dsmvc::metal {

struct PlaneJob {
    const void *source = nullptr;
    std::ptrdiff_t source_stride_bytes = 0;
    void *destination = nullptr;
    std::ptrdiff_t destination_stride_bytes = 0;
    std::uint32_t source_width = 0;
    std::uint32_t source_height = 0;
    std::uint32_t destination_width = 0;
    std::uint32_t destination_height = 0;
    std::uint32_t sample_bytes = 0;
    bool integer_samples = false;
    bool process_horizontal = false;
    bool process_vertical = false;
    IntegerConversion conversion{};
    std::shared_ptr<const void> source_lifetime;
    std::shared_ptr<const AxisPlan> horizontal;
    std::shared_ptr<const AxisPlan> vertical;
};

struct FrameJob {
    std::vector<PlaneJob> planes;
    std::uint32_t maximum_half_bandwidth = 0;
    std::uint64_t estimated_work = 0;
    bool profile_signposts = false;
};

struct RunResult {
    std::size_t metal_batch_size = 0;
    std::size_t staging_memcpy_calls = 0;
    std::size_t staging_copied_bytes = 0;
    std::size_t unique_input_planes = 0;
    std::size_t axis_dispatches = 0;
    std::size_t conversion_dispatches = 0;
    std::size_t heterogeneous_axis_dispatches = 0;
    std::size_t heterogeneous_axis_descriptors = 0;
    std::size_t resident_producers = 0;
    std::size_t resident_hits = 0;
    std::size_t resident_evictions = 0;
    std::size_t resident_bytes = 0;
    std::size_t eliminated_staging_bytes = 0;
    std::uint64_t gpu_interval_nanoseconds = 0;
    std::uint64_t submission_gap_nanoseconds = 0;
};

struct SchedulerDiagnostics {
    std::size_t ring_slots = 0;
    std::size_t submissions = 0;
    std::size_t completions = 0;
    std::size_t maximum_in_flight = 0;
    std::size_t plan_cache_entries = 0;
    std::size_t plan_cache_bytes = 0;
    std::size_t plan_cache_evictions = 0;
    std::size_t resident_cache_entries = 0;
    std::size_t resident_cache_bytes = 0;
    std::size_t resident_cache_capacity = 0;
    std::size_t resident_cache_evictions = 0;
    std::size_t resident_cache_pinned_eviction_blocks = 0;
    std::size_t resident_cache_producers = 0;
    std::size_t resident_cache_hits = 0;
    std::size_t metal_errors = 0;
    std::size_t consecutive_metal_errors = 0;
    std::size_t maximum_consecutive_metal_errors = 0;
};

class Client final {
public:
    struct Impl;

    Client();
    ~Client();

    Client(const Client &) = delete;
    Client &operator=(const Client &) = delete;

    void close() noexcept;

private:
    std::shared_ptr<Impl> impl_;

    friend RunResult run(
        const std::shared_ptr<Client> &, FrameJob, std::function<void()>, bool);
};

[[nodiscard]] bool available() noexcept;
[[nodiscard]] std::shared_ptr<Client> make_client();
[[nodiscard]] SchedulerDiagnostics diagnostics() noexcept;
void fail_next_resident_producer_for_testing() noexcept;
#if defined(DSMVC_METAL_SCHEDULER_TESTING)
void set_batch_timeout_for_testing(std::uint64_t microseconds) noexcept;
#endif

RunResult run(
    const std::shared_ptr<Client> &client, FrameJob job,
    std::function<void()> cpu_work, bool automatic);

} // namespace dsmvc::metal
