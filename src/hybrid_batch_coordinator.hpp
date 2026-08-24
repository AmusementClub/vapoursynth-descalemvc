#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>

namespace dsmvc::experimental {

class HybridBatchCoordinator final {
public:
    using CpuWork = std::function<void()>;
    using GpuBatchWork = std::function<void(std::span<void *const>)>;

    HybridBatchCoordinator(
        std::size_t batch_size, std::size_t cpu_frames,
        std::size_t activation_threshold,
        std::chrono::microseconds flush_timeout, GpuBatchWork gpu_work);
    ~HybridBatchCoordinator();

    HybridBatchCoordinator(const HybridBatchCoordinator &) = delete;
    HybridBatchCoordinator &operator=(const HybridBatchCoordinator &) = delete;

    void run(void *payload, CpuWork cpu_work);
    void shutdown() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dsmvc::experimental
