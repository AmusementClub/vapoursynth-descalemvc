#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <getnative/axis_plan.hpp>

namespace dsmvc {

enum class KernelKind : std::uint8_t {
    bilinear,
    bicubic,
    lanczos,
    spline16,
    spline36,
    spline64,
    custom,
};

enum class BackendKind : std::uint8_t {
    automatic,
    cpu,
    metal,
    vulkan,
    cuda,
};

enum class CpuPath : std::uint8_t {
    automatic,
    scalar,
    avx2,
};

struct KernelSpec {
    KernelKind kind = KernelKind::bicubic;
    std::int32_t taps = 3;
    double b = 0.0;
    double c = 0.5;
};

struct AxisRequest {
    std::int32_t source_size = 0;
    std::int32_t destination_size = 0;
    double active_length = 0.0;
    double shift = 0.0;
    KernelSpec kernel{};
    getnative::BorderMode border = getnative::BorderMode::mirror;
};

using CustomKernel = std::function<double(double)>;

// Immutable inverse-only planner output. The CPU descale executor never uses
// GetNative's forward projection table, so keeping it in every VS node wastes
// planner time and resident memory.
struct AxisPlan {
    std::int32_t source_size = 0;
    std::int32_t destination_size = 0;
    std::int32_t support = 0;
    std::int32_t half_bandwidth = 0;
    double active_length = 0.0;
    double shift = 0.0;

    std::vector<std::uint32_t> transpose_offsets;
    std::vector<std::int32_t> transpose_indices;
    std::vector<float> transpose_weights;
    std::vector<float> lower_ld;
    std::vector<float> upper_l;
    std::vector<float> inverse_diagonal;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::size_t storage_bytes() const noexcept;
};

struct PlannerCacheStats {
    std::uint64_t plan_hits = 0;
    std::uint64_t plan_builds = 0;
    std::uint64_t geometry_hits = 0;
    std::uint64_t geometry_builds = 0;
    std::size_t plan_entries = 0;
    std::size_t plan_resident_bytes = 0;
    std::size_t geometry_entries = 0;
    std::size_t geometry_resident_bytes = 0;
};

struct BackendCapability {
    BackendKind kind{};
    const char *name = "";
    bool compiled = false;
    bool device_available = false;
};

[[nodiscard]] AxisPlan build_axis_plan(
    const AxisRequest &request, const CustomKernel &custom_kernel = {});
[[nodiscard]] std::shared_ptr<const AxisPlan> get_or_build_axis_plan(
    const AxisRequest &request, const CustomKernel &custom_kernel = {});
[[nodiscard]] PlannerCacheStats planner_cache_stats();
void clear_planner_caches();

void inverse_axis_f32(const AxisPlan &plan,
                      const float *input, std::ptrdiff_t input_stride,
                      float *output, std::ptrdiff_t output_stride);

inline void inverse_axis_f32(const AxisPlan &plan,
                             std::span<const float> input,
                             std::span<float> output) {
    if (input.size() < static_cast<std::size_t>(plan.source_size)
        || output.size() < static_cast<std::size_t>(plan.destination_size)) {
        throw std::invalid_argument("inverse axis spans are too small");
    }
    inverse_axis_f32(plan, input.data(), 1, output.data(), 1);
}

[[nodiscard]] BackendKind parse_backend(std::string_view name);
[[nodiscard]] BackendKind resolve_backend(BackendKind requested);
[[nodiscard]] std::vector<BackendCapability> backend_capabilities();
[[nodiscard]] const char *backend_name(BackendKind kind) noexcept;

[[nodiscard]] bool cpu_avx2_compiled() noexcept;
[[nodiscard]] bool cpu_avx2_available() noexcept;

class CpuExecutor {
public:
    explicit CpuExecutor(CpuPath requested = CpuPath::automatic);
    ~CpuExecutor();
    CpuExecutor(const CpuExecutor &) noexcept = default;
    CpuExecutor &operator=(const CpuExecutor &) noexcept = default;
    CpuExecutor(CpuExecutor &&) noexcept = default;
    CpuExecutor &operator=(CpuExecutor &&) noexcept = default;

    [[nodiscard]] CpuPath path() const noexcept;
    [[nodiscard]] const char *name() const noexcept;

    void prepare(std::shared_ptr<const AxisPlan> plan) const;
    void seal() const;

    void inverse_rows(const AxisPlan &plan,
                      const float *input, std::ptrdiff_t input_row_stride,
                      float *output, std::ptrdiff_t output_row_stride,
                      std::int32_t row_count) const;

    void inverse_columns(const AxisPlan &plan,
                         const float *input, std::ptrdiff_t input_row_stride,
                         float *output, std::ptrdiff_t output_row_stride,
                         std::int32_t column_count) const;

private:
    struct Impl;
    CpuPath path_ = CpuPath::scalar;
    std::shared_ptr<Impl> impl_;
};

} // namespace dsmvc
