#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
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

struct BackendCapability {
    BackendKind kind{};
    const char *name = "";
    bool compiled = false;
    bool device_available = false;
};

[[nodiscard]] getnative::AxisPlan build_axis_plan(
    const AxisRequest &request, const CustomKernel &custom_kernel = {});

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

    void prepare(const getnative::AxisPlan &plan) const;
    void seal() const;

    void inverse_rows(const getnative::AxisPlan &plan,
                      const float *input, std::ptrdiff_t input_row_stride,
                      float *output, std::ptrdiff_t output_row_stride,
                      std::int32_t row_count) const;

    void inverse_columns(const getnative::AxisPlan &plan,
                         const float *input, std::ptrdiff_t input_row_stride,
                         float *output, std::ptrdiff_t output_row_stride,
                         std::int32_t column_count) const;

private:
    struct Impl;
    CpuPath path_ = CpuPath::scalar;
    std::shared_ptr<Impl> impl_;
};

} // namespace dsmvc
