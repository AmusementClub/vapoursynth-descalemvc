#include "solve_partition.hpp"

#include <algorithm>
#include <cstdlib>
#include <string_view>

namespace dsmvc::solve_partition {
namespace {

[[nodiscard]] bool enabled() noexcept {
    static const bool value = [] {
        const char *text = std::getenv("DSMVC_GPU_SOLVE_PARTITION");
        return !text || std::string_view{text} != "0";
    }();
    return value;
}

[[nodiscard]] std::uint32_t minimum_chunk() noexcept {
    static const std::uint32_t value = [] {
        constexpr std::uint32_t fallback = 32U;
        const char *text = std::getenv("DSMVC_GPU_SOLVE_MIN_CHUNK");
        if (!text) return fallback;
        const long requested = std::atol(text);
        return requested >= 8L && requested <= 4096L
            ? static_cast<std::uint32_t>(requested) : fallback;
    }();
    return value;
}

} // namespace

Partition choose(const AxisPlan &plan) noexcept {
    Partition partition;
    const auto width = plan.half_bandwidth;
    if (!enabled() || plan.requires_float64()
        || (width != 1 && width != 3 && width != 5 && width != 7)
        || plan.destination_size <= 0) {
        return partition;
    }
    const auto size = static_cast<std::uint32_t>(plan.destination_size);
    const std::uint32_t shortest = minimum_chunk();
    std::uint32_t chunks = 1U;
    while (chunks < maximum_chunks
           && (size + chunks * 2U - 1U) / (chunks * 2U) >= shortest) {
        chunks *= 2U;
    }
    if (chunks < 2U) return partition;
    partition.chunks = chunks;
    partition.chunk_length = (size + chunks - 1U) / chunks;
    return partition;
}

Partition build(const AxisPlan &plan) {
    Partition partition = choose(plan);
    if (partition.chunks == 0U) return partition;
    const auto size = static_cast<std::size_t>(plan.destination_size);
    const auto width = static_cast<std::size_t>(plan.half_bandwidth);
    const std::size_t matrix = width * width;
    partition.transfers.assign(transfer_floats(partition, plan), 0.0F);
    std::vector<double> ring(width);
    const auto shift_in = [&](double value) {
        for (std::size_t distance = width - 1U; distance > 0U; --distance) {
            ring[distance] = ring[distance - 1U];
        }
        ring[0] = value;
    };
    for (std::size_t chunk = 0U; chunk < partition.chunks; ++chunk) {
        const std::size_t begin = std::min(size, chunk * partition.chunk_length);
        const std::size_t end = std::min(size, begin + partition.chunk_length);
        const std::size_t backward_end = std::min(end, size - 1U);
        for (std::size_t column = 0U; column < width; ++column) {
            // Forward: y[i] = -inverse_diagonal[i] * sum lower[d][i] y[i - d].
            std::fill(ring.begin(), ring.end(), 0.0);
            ring[column] = 1.0;
            for (std::size_t index = begin; index < end; ++index) {
                double sum = 0.0;
                const std::size_t available = std::min(width, index);
                for (std::size_t distance = 1U; distance <= available; ++distance) {
                    sum -= static_cast<double>(
                               plan.lower_ld[(distance - 1U) * size + index])
                        * ring[distance - 1U];
                }
                shift_in(sum * static_cast<double>(plan.inverse_diagonal[index]));
            }
            for (std::size_t row = 0U; row < width; ++row) {
                partition.transfers[chunk * matrix + row * width + column] =
                    static_cast<float>(ring[row]);
            }
            // Backward: x[i] = -sum upper[d][i] x[i + d], from the chunk's end.
            std::fill(ring.begin(), ring.end(), 0.0);
            ring[column] = 1.0;
            for (std::size_t index = backward_end; index-- > begin;) {
                double sum = 0.0;
                const std::size_t available = std::min(width, size - index - 1U);
                for (std::size_t distance = 1U; distance <= available; ++distance) {
                    sum += static_cast<double>(
                               plan.upper_l[(distance - 1U) * size + index])
                        * ring[distance - 1U];
                }
                shift_in(-sum);
            }
            for (std::size_t row = 0U; row < width; ++row) {
                partition.transfers[(partition.chunks + chunk) * matrix
                                    + row * width + column] =
                    static_cast<float>(ring[row]);
            }
        }
    }
    return partition;
}

} // namespace dsmvc::solve_partition
