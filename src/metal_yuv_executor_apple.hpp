#pragma once

#include <dsmvc/engine.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>

namespace dsmvc::experimental {

struct MetalYuvFrame {
    std::array<const void *, 3> source_planes{};
    std::array<std::ptrdiff_t, 3> source_strides_bytes{};
    std::array<void *, 3> destination_planes{};
    std::array<std::ptrdiff_t, 3> destination_strides_bytes{};
    std::array<IntegerConversion, 3> conversions{};
};

struct MetalYuvStagingStats {
    std::size_t memcpy_calls = 0U;
    std::size_t copied_bytes = 0U;
};

namespace detail {

// Keep the row-wise fallback explicit: external VapourSynth strides may not
// match the executor's tightly selected Metal staging stride.
inline void copy_strided_rows(
    void *destination, std::ptrdiff_t destination_stride,
    const void *source, std::ptrdiff_t source_stride, std::size_t row_bytes,
    std::uint32_t row_count, MetalYuvStagingStats &stats) {
    if (row_count == 0U) return;

    auto *destination_bytes = static_cast<std::byte *>(destination);
    const auto *source_bytes = static_cast<const std::byte *>(source);
    if (destination_stride == source_stride) {
        const std::size_t copied_bytes =
            (static_cast<std::size_t>(row_count) - 1U)
                * static_cast<std::size_t>(source_stride)
            + row_bytes;
        std::memcpy(destination_bytes, source_bytes, copied_bytes);
        ++stats.memcpy_calls;
        stats.copied_bytes += copied_bytes;
        return;
    }

    for (std::uint32_t row = 0U; row < row_count; ++row) {
        std::memcpy(
            destination_bytes + static_cast<std::ptrdiff_t>(row)
                * destination_stride,
            source_bytes + static_cast<std::ptrdiff_t>(row) * source_stride,
            row_bytes);
    }
    stats.memcpy_calls += row_count;
    stats.copied_bytes += static_cast<std::size_t>(row_count) * row_bytes;
}

} // namespace detail

class MetalYuvExecutor final {
public:
    MetalYuvExecutor(
        std::array<std::shared_ptr<const AxisPlan>, 2> horizontal,
        std::array<std::shared_ptr<const AxisPlan>, 2> vertical,
        std::uint32_t sample_bytes, std::size_t maximum_batch_size,
        std::size_t threads_per_threadgroup = 128U,
        bool profile_signposts = false);
    ~MetalYuvExecutor();

    MetalYuvExecutor(const MetalYuvExecutor &) = delete;
    MetalYuvExecutor &operator=(const MetalYuvExecutor &) = delete;

    void execute(std::span<const MetalYuvFrame> frames);

    [[nodiscard]] const std::string &device_name() const noexcept;
    [[nodiscard]] std::size_t requested_buffer_bytes() const noexcept;
    [[nodiscard]] MetalYuvStagingStats last_staging_stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dsmvc::experimental
