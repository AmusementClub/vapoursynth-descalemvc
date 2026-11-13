#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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
    // Automatic may fall back to scalar. Explicit SIMD values require that
    // exact implementation to be compiled and supported by the current CPU.
    automatic,
    scalar,
    avx2,
    neon,
    avx512,
};

enum class BorderMode : std::uint8_t {
    zero = 0,
    repeat = 1,
    reflect101 = 2,
    symmetric = 3,
    // Original descale performs one edge-duplicating reflection, not a
    // periodic extension. Kept for the legacy border_handling parameter.
    mirror = 4,
};

enum class F64Mode : std::uint8_t {
    automatic = 0,
    float32_only = 1,
    float64_only = 2,
};

struct KernelSpec {
    KernelKind kind = KernelKind::bicubic;
    std::int32_t taps = 3;
    double b = 0.0;
    double c = 0.5;
    double blur = 1.0;
};

struct AxisRequest {
    std::int32_t source_size = 0;
    std::int32_t destination_size = 0;
    double active_length = 0.0;
    double shift = 0.0;
    KernelSpec kernel{};
    BorderMode border = BorderMode::symmetric;
    F64Mode f64_mode = F64Mode::automatic;
};

using CustomKernel = std::function<double(double)>;

// Immutable inverse-only planner output. The CPU descale executor does not
// need a forward projection table, so plans retain only inverse coefficients.
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

    // The Float32 normal-equation path is retained for well-conditioned axes.
    // This is either a conservative lower bound or a Hager 1-norm estimate.
    // Unsafe axes also retain the original Float64 transpose, normal matrix,
    // and LDLT bands. The unfactored normal matrix supports independent
    // high-precision residual evaluation without reconstructing it from F32.
    double normal_rcond = 1.0;
    double normal_inf_norm = 0.0;
    std::vector<double> transpose_weights_f64;
    std::vector<double> normal_bands_f64;
    std::vector<double> ldlt_bands_f64;
    std::vector<double> inverse_diagonal_f64;

    // Checks every coefficient once per object: a successful result is
    // remembered, and copies and moves start unchecked. Plans are immutable
    // once validated; modify a copy instead.
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool requires_float64() const noexcept;
    [[nodiscard]] std::size_t storage_bytes() const noexcept;

private:
    class ValidationMemo {
    public:
        ValidationMemo() noexcept = default;
        ValidationMemo(const ValidationMemo &) noexcept {}
        ValidationMemo &operator=(const ValidationMemo &) noexcept {
            valid_.store(false, std::memory_order_relaxed);
            return *this;
        }
        [[nodiscard]] bool known() const noexcept {
            return valid_.load(std::memory_order_acquire);
        }
        void remember() const noexcept {
            valid_.store(true, std::memory_order_release);
        }

    private:
        mutable std::atomic<bool> valid_{false};
    };

    [[nodiscard]] bool check_valid() const noexcept;
    ValidationMemo validation_;
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

struct CpuPlanPackingStats {
    std::uint64_t pack_executions = 0;
    std::uint64_t single_flight_waits = 0;
    std::uint64_t single_flight_wait_nanoseconds = 0;
    std::uint64_t lazy_requests = 0;
    std::uint64_t lazy_hits = 0;
    std::uint64_t maximum_concurrent_packs = 0;
};

struct BackendCapability {
    BackendKind kind{};
    const char *name = "";
    bool compiled = false;
    bool device_available = false;
};

struct IntegerConversion {
    float input_offset = 0.0F;
    float input_scale = 1.0F;
    float output_scale = 1.0F;
    float output_offset = 0.0F;
    std::uint32_t output_maximum = 255U;
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
[[nodiscard]] bool cpu_avx512_compiled() noexcept;
[[nodiscard]] bool cpu_avx512_available() noexcept;
[[nodiscard]] bool cpu_neon_compiled() noexcept;
[[nodiscard]] bool cpu_neon_available() noexcept;
[[nodiscard]] bool metal_compiled() noexcept;
[[nodiscard]] bool metal_available() noexcept;
[[nodiscard]] bool vulkan_compiled() noexcept;
[[nodiscard]] bool vulkan_available() noexcept;
[[nodiscard]] bool cuda_compiled() noexcept;
[[nodiscard]] bool cuda_available() noexcept;

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
    [[nodiscard]] CpuPlanPackingStats packing_stats() const noexcept;

    void prepare(std::shared_ptr<const AxisPlan> plan) const;
    void defer(std::shared_ptr<const AxisPlan> plan) const;
    void seal() const;

    // Buffer contract for this class and Executor below: strides cover the
    // logical width of each row. Callers need only allocate (row_count - 1) *
    // row_stride + logical_width elements; the final row does not require
    // pitch padding. Executors do not access or modify row padding outside the
    // logical width.

    void inverse_rows(const AxisPlan &plan,
                      const float *input, std::ptrdiff_t input_row_stride,
                      float *output, std::ptrdiff_t output_row_stride,
                      std::int32_t row_count) const;

    void inverse_columns(const AxisPlan &plan,
                         const float *input, std::ptrdiff_t input_row_stride,
                         float *output, std::ptrdiff_t output_row_stride,
                         std::int32_t column_count) const;

    void inverse_2d(const AxisPlan &horizontal, const AxisPlan &vertical,
                    const float *input, std::ptrdiff_t input_row_stride,
                    float *output, std::ptrdiff_t output_row_stride) const;

    void inverse_2d_u8(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const std::uint8_t *input, std::ptrdiff_t input_row_stride,
        std::uint8_t *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion) const;

    void inverse_2d_u16(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const std::uint16_t *input, std::ptrdiff_t input_row_stride,
        std::uint16_t *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion) const;

    void inverse_2d_u8_streamed(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const std::uint8_t *input, std::ptrdiff_t input_row_stride,
        std::uint8_t *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion) const;

    void inverse_2d_u16_streamed(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const std::uint16_t *input, std::ptrdiff_t input_row_stride,
        std::uint16_t *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion) const;

private:
    struct Impl;
    CpuPath path_ = CpuPath::scalar;
    std::shared_ptr<Impl> impl_;

    template <class Sample>
    void inverse_2d_integer(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const Sample *input, std::ptrdiff_t input_row_stride,
        Sample *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion) const;

    template <class Sample>
    void inverse_2d_integer_streamed(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const Sample *input, std::ptrdiff_t input_row_stride,
        Sample *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion) const;
};

// One plane of a device-resident (GPU) frame: a linear pitched Vulkan buffer
// laid out like the CPU plane, plus the timeline pair to wait on before reading.
// Vulkan handles travel as integers so this header stays free of Vulkan.
struct GpuPlane {
    std::uint64_t buffer = 0;           // VkBuffer
    std::uint64_t buffer_bytes = 0;     // stride * height
    std::uint64_t ready_semaphore = 0;  // VkSemaphore (timeline); 0 = ready now
    std::uint64_t ready_value = 0;
    std::ptrdiff_t pitch_bytes = 0;     // distance between rows
};

// A plane of a GPU frame exported by the core for a foreign API (CUDA): the
// whole allocation is imported once and the plane lives at `offset` in it. The
// handle (a POSIX file descriptor, or a Windows handle) is consumed by the call
// that receives it, whatever the outcome.
struct ExportedPlane {
    std::uint64_t memory_id = 0;     // stable for the allocation's lifetime
    std::uint64_t memory_size = 0;
    std::uint64_t offset = 0;
    std::uint64_t plane_bytes = 0;   // stride * height
    int handle_type = 0;             // VkExternalMemoryHandleTypeFlagBits
    std::intptr_t handle = -1;
    std::ptrdiff_t pitch_bytes = 0;
    // An input plane's producer pair, for an executor that has imported the
    // semaphore (import_exported_semaphore): it then waits for the pair on
    // the device. Zero when the caller has waited on the host instead.
    std::uint64_t ready_semaphore = 0;  // VkSemaphore (timeline)
    std::uint64_t ready_value = 0;
};

// Closes the handle of an exported plane that was never passed to an executor.
void close_exported_plane(const ExportedPlane &plane) noexcept;

enum class ResidentKind : std::uint8_t { none, vulkan, cuda };

enum class SampleKind : std::uint8_t { float32, u8, u16 };

// One plane of a multi-plane 2D frame for Executor::inverse_2d_frame. Host
// frames give pointers and strides in samples, GPU frames the planes;
// `conversion` is read for integer samples only.
struct FramePlane2D {
    const AxisPlan *horizontal = nullptr;
    const AxisPlan *vertical = nullptr;
    const void *input = nullptr;
    std::ptrdiff_t input_row_stride = 0;
    void *output = nullptr;
    std::ptrdiff_t output_row_stride = 0;
    GpuPlane gpu_input;
    GpuPlane gpu_output;
    IntegerConversion conversion;
};

// The Vulkan device of a VapourSynth core, borrowed for as long as the
// executor made from it lives. The queue is shared with the core's own filters:
// every submission is bracketed by lock_queue / unlock_queue.
struct VulkanCoreDevice {
    void *instance = nullptr;           // VkInstance
    void *physical_device = nullptr;    // VkPhysicalDevice
    void *device = nullptr;             // VkDevice
    std::uint32_t queue_family = 0;
    std::uint32_t queue_index = 0;
    void (*lock_queue)(void *user) = nullptr;
    void (*unlock_queue)(void *user) = nullptr;
    void *user = nullptr;
};

// Backend-neutral executor used by the plugin. Automatic dispatch deliberately
// remains CPU-first; callers opt in to a GPU backend explicitly.
class Executor {
public:
    explicit Executor(
        BackendKind requested = BackendKind::automatic,
        CpuPath cpu_path = CpuPath::automatic);
    ~Executor();
    Executor(const Executor &) noexcept = default;
    Executor &operator=(const Executor &) noexcept = default;
    Executor(Executor &&) noexcept = default;
    Executor &operator=(Executor &&) noexcept = default;

    // Runs on the Vulkan device of a core that hands out GPU frames. Throws
    // when this build has no Vulkan or the device cannot run the kernels.
    [[nodiscard]] static Executor on_core_vulkan(const VulkanCoreDevice &device);
    // CUDA on the GPU that backs a core's Vulkan device (matched by UUID).
    // Throws when the CUDA device in use is not that GPU.
    [[nodiscard]] static Executor on_core_cuda(const std::uint8_t device_uuid[16]);

    [[nodiscard]] BackendKind backend() const noexcept;
    [[nodiscard]] const char *name() const noexcept;
    [[nodiscard]] bool input_cache_enabled() const noexcept;
    [[nodiscard]] CpuPlanPackingStats cpu_plan_packing_stats() const noexcept;
    // True for an executor made by on_core_vulkan: it takes GPU planes only.
    [[nodiscard]] bool resident() const noexcept;
    [[nodiscard]] ResidentKind resident_kind() const noexcept;

    void prepare(std::shared_ptr<const AxisPlan> plan) const;
    void defer(std::shared_ptr<const AxisPlan> plan) const;
    void seal() const;

    // GPU frames (resident executors, Float32). The call returns once the
    // output plane is written; the input plane's ready pair is waited on.
    void inverse_rows_resident(
        const AxisPlan &plan, const GpuPlane &input, const GpuPlane &output,
        std::int32_t row_count) const;
    void inverse_columns_resident(
        const AxisPlan &plan, const GpuPlane &input, const GpuPlane &output,
        std::int32_t column_count) const;
    void inverse_2d_resident(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const GpuPlane &input, const GpuPlane &output) const;
    // Fused 8-16 bit integer 2D descale on GPU frames.
    void inverse_2d_u8_resident(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const GpuPlane &input, const GpuPlane &output,
        const IntegerConversion &conversion) const;
    void inverse_2d_u16_resident(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const GpuPlane &input, const GpuPlane &output,
        const IntegerConversion &conversion) const;

    // The same for a CUDA resident executor, through exported memory. The
    // caller has waited for the input frame's producers, or names their pair
    // in the input plane; the output is written and synchronized before the
    // call returns.
    //
    // A timeline semaphore of the core, exported for this executor to wait on
    // (keyed by its VkSemaphore, which must stay alive as long as the
    // executor). The handle is consumed whatever the outcome; returns false
    // when it cannot be imported, and the caller waits on the host.
    [[nodiscard]] bool has_exported_semaphore(std::uint64_t semaphore) const;
    [[nodiscard]] bool import_exported_semaphore(
        std::uint64_t semaphore, int handle_type, std::intptr_t handle) const;
    void inverse_rows_exported(
        const AxisPlan &plan, const ExportedPlane &input,
        const ExportedPlane &output, std::int32_t row_count) const;
    void inverse_columns_exported(
        const AxisPlan &plan, const ExportedPlane &input,
        const ExportedPlane &output, std::int32_t column_count) const;
    void inverse_2d_exported(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const ExportedPlane &input, const ExportedPlane &output) const;
    void inverse_2d_u8_exported(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const ExportedPlane &input, const ExportedPlane &output,
        const IntegerConversion &conversion) const;
    void inverse_2d_u16_exported(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const ExportedPlane &input, const ExportedPlane &output,
        const IntegerConversion &conversion) const;

    // All planes (2 to 3, one sample kind, 2D) of a frame in one submission,
    // for the Vulkan backend: host frames, or GPU frames on an executor made
    // by on_core_vulkan. Returns false without doing anything when the backend
    // or the frame is not eligible (another backend, a Float64 plan, a frame
    // that does not fit one workspace, DSMVC_VULKAN_FRAME_SUBMIT=0); the
    // caller then runs the planes one by one, which also reports invalid
    // arguments.
    [[nodiscard]] bool inverse_2d_frame(
        SampleKind kind, std::span<const FramePlane2D> planes,
        const std::shared_ptr<const void> &input_lifetime = {}) const;

    void inverse_rows(const AxisPlan &plan,
                      const float *input, std::ptrdiff_t input_row_stride,
                      float *output, std::ptrdiff_t output_row_stride,
                      std::int32_t row_count,
                      std::shared_ptr<const void> input_lifetime = {}) const;

    void inverse_columns(const AxisPlan &plan,
                         const float *input, std::ptrdiff_t input_row_stride,
                         float *output, std::ptrdiff_t output_row_stride,
                         std::int32_t column_count,
                         std::shared_ptr<const void> input_lifetime = {}) const;

    void inverse_2d(const AxisPlan &horizontal, const AxisPlan &vertical,
                    const float *input, std::ptrdiff_t input_row_stride,
                    float *output, std::ptrdiff_t output_row_stride,
                    std::shared_ptr<const void> input_lifetime = {}) const;

    void inverse_2d_u8(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const std::uint8_t *input, std::ptrdiff_t input_row_stride,
        std::uint8_t *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion,
        std::shared_ptr<const void> input_lifetime = {}) const;

    void inverse_2d_u16(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const std::uint16_t *input, std::ptrdiff_t input_row_stride,
        std::uint16_t *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion,
        std::shared_ptr<const void> input_lifetime = {}) const;

    void inverse_2d_u8_streamed(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const std::uint8_t *input, std::ptrdiff_t input_row_stride,
        std::uint8_t *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion,
        std::shared_ptr<const void> input_lifetime = {}) const;

    void inverse_2d_u16_streamed(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const std::uint16_t *input, std::ptrdiff_t input_row_stride,
        std::uint16_t *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion,
        std::shared_ptr<const void> input_lifetime = {}) const;

private:
    struct Impl;
    explicit Executor(std::shared_ptr<Impl> impl);
    std::shared_ptr<Impl> impl_;
};

} // namespace dsmvc
