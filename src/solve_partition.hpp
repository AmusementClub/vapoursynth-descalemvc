#pragma once

#include "dsmvc/engine.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

// Partitioned solve of an axis plan's Float32 LDL^T system for the GPU
// backends. The sequential solve walks each vector from end to end, so a
// frame offers only one thread per row or column. A partition splits every
// vector into `chunks` pieces that are solved concurrently:
//
//   1. each piece is solved from a zero carry-in (its last half_bandwidth
//      outputs are kept);
//   2. one thread per vector carries the true state from piece to piece with
//      the transfer matrices below (the response of a piece's last outputs to
//      its carry-in);
//   3. each piece is solved again from its true carry-in.
//
// The same is done for the backward substitution. The result equals the
// sequential solve up to Float32 rounding of the carried state; descale plans
// forget their carry-in quickly, so with pieces of a few dozen samples the
// output is usually bit-identical.
namespace dsmvc::solve_partition {

inline constexpr std::uint32_t maximum_chunks = 32U;

struct Partition {
    std::uint32_t chunks = 0U; // 0: keep the sequential kernels
    std::uint32_t chunk_length = 0U;
    // Forward matrices for every chunk, then backward ones: chunks * W * W
    // floats each, entry [(chunk * W + row) * W + column], W = half_bandwidth.
    std::vector<float> transfers;
};

// Chunks and chunk length for a plan, without the matrices. Partitions are
// used for Float32 plans with half_bandwidth 1, 3, 5 or 7 whose vectors hold
// at least two chunks of DSMVC_GPU_SOLVE_MIN_CHUNK samples (default 32, at
// least 8). DSMVC_GPU_SOLVE_PARTITION=0 disables them.
[[nodiscard]] Partition choose(const AxisPlan &plan) noexcept;

[[nodiscard]] Partition build(const AxisPlan &plan);

[[nodiscard]] inline std::size_t transfer_floats(const Partition &partition,
                                                 const AxisPlan &plan) noexcept {
    const auto width = static_cast<std::size_t>(plan.half_bandwidth);
    return static_cast<std::size_t>(partition.chunks) * width * width * 2U;
}

} // namespace dsmvc::solve_partition
