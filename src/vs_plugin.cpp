#include <dsmvc/engine.hpp>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#if defined(DSMVC_HAS_CUDA)
#include "nvtx.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <charconv>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace {

using dsmvc::AxisRequest;
using dsmvc::BackendKind;
using dsmvc::BorderMode;
using dsmvc::CpuPath;
using dsmvc::CustomKernel;
using dsmvc::Executor;
using dsmvc::IntegerConversion;
using dsmvc::KernelKind;
using dsmvc::KernelSpec;

constexpr const char *plugin_id = "com.dsmvc.descale";

struct ParsedArguments {
    KernelSpec kernel{};
    VSFunction *custom_kernel = nullptr;
    int width = 0;
    int height = 0;
    double src_left = 0.0;
    double src_top = 0.0;
    double src_width = 0.0;
    double src_height = 0.0;
    BorderMode border = BorderMode::mirror;
    int border_value = 0;
    int force = 0;
    int force_h = 0;
    int force_v = 0;
    int opt = 0;
    CpuPath cpu_path = CpuPath::automatic;
    BackendKind backend = BackendKind::automatic;
    std::string backend_text = "auto";
    std::string function_name = "Descale";
};

struct FilterData {
    FilterData(BackendKind backend, CpuPath path) : executor(backend, path) {}

    VSNode *node = nullptr;
    VSVideoInfo vi{};
    int source_width = 0;
    int source_height = 0;
    int destination_width = 0;
    int destination_height = 0;
    int subsampling_w = 0;
    int subsampling_h = 0;
    int num_planes = 0;
    int color_family = cfGray;
    int bits_per_sample = 32;
    int bytes_per_sample = 4;
    int core_threads = 1;
    bool process_horizontal = false;
    bool process_vertical = false;
    bool fused_integer = false;
    AxisRequest horizontal_requests[2];
    AxisRequest vertical_requests[2];
    bool has_horizontal_request[2]{};
    bool has_vertical_request[2]{};
    std::shared_ptr<const dsmvc::AxisPlan> horizontal[2];
    std::shared_ptr<const dsmvc::AxisPlan> vertical[2];
    std::once_flag planning_once;
    std::exception_ptr planning_error;
    VSFunction *custom_kernel = nullptr;
    CustomKernel custom_callback;
    Executor executor{};
    std::atomic<std::uint32_t> active_2d_frames{0};
};

class ActiveFrameGuard {
public:
    explicit ActiveFrameGuard(std::atomic<std::uint32_t> *counter) noexcept
        : counter_(counter), first_(
              !counter || counter->fetch_add(1, std::memory_order_relaxed) == 0) {}

    ~ActiveFrameGuard() {
        if (counter_) counter_->fetch_sub(1, std::memory_order_relaxed);
    }

    ActiveFrameGuard(const ActiveFrameGuard &) = delete;
    ActiveFrameGuard &operator=(const ActiveFrameGuard &) = delete;

    [[nodiscard]] bool first() const noexcept { return first_; }

private:
    std::atomic<std::uint32_t> *counter_ = nullptr;
    bool first_ = true;
};

struct MemoryPhaseConfig {
    std::size_t limit = 0U;
    bool all_kernels = false;
};

const MemoryPhaseConfig &memory_phase_config() noexcept {
    static const MemoryPhaseConfig config = [] {
        const auto hardware = std::max(
            std::thread::hardware_concurrency(), 1U);
        MemoryPhaseConfig result{
            std::clamp<std::size_t>(hardware / 2U, 1U, 32U), false};
        const char *environment = std::getenv("DSMVC_MEMORY_CONCURRENCY");
        if (!environment) return result;

        const std::string_view text(environment);
        std::size_t parsed = 0U;
        const auto conversion = std::from_chars(
            text.data(), text.data() + text.size(), parsed);
        if (conversion.ec != std::errc{}
            || conversion.ptr != text.data() + text.size()) {
            return result;
        }
        result.limit = parsed == 0U
            ? 0U : std::min<std::size_t>(parsed, hardware);
        result.all_kernels = true;
        return result;
    }();
    return config;
}

class MemoryPhaseLimiter {
public:
    explicit MemoryPhaseLimiter(std::size_t limit) noexcept : limit_(limit) {}

    [[nodiscard]] bool acquire() {
        if (limit_ == 0U) return false;
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [&] { return active_ < limit_; });
        ++active_;
        return true;
    }

    void release() noexcept {
        {
            const std::scoped_lock lock(mutex_);
            --active_;
        }
        ready_.notify_one();
    }

private:
    std::size_t limit_ = 0U;
    std::size_t active_ = 0U;
    std::mutex mutex_;
    std::condition_variable ready_;
};

MemoryPhaseLimiter &shared_memory_phase_limiter() {
    static MemoryPhaseLimiter limiter(memory_phase_config().limit);
    return limiter;
}

class MemoryPhaseGuard {
public:
    explicit MemoryPhaseGuard(bool enabled)
        : limiter_(enabled ? &shared_memory_phase_limiter() : nullptr),
          acquired_(limiter_ && limiter_->acquire()) {}

    ~MemoryPhaseGuard() {
        if (acquired_) limiter_->release();
    }

    MemoryPhaseGuard(const MemoryPhaseGuard &) = delete;
    MemoryPhaseGuard &operator=(const MemoryPhaseGuard &) = delete;

private:
    MemoryPhaseLimiter *limiter_ = nullptr;
    bool acquired_ = false;
};

std::string lower_copy(const char *value) {
    std::string result = value ? value : "";
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char character) {
                       return static_cast<char>(std::tolower(character));
                   });
    return result;
}

int get_int(const VSMap *map, const char *key, int fallback, const VSAPI *vsapi) {
    int error = 0;
    const auto value = vsapi->mapGetInt(map, key, 0, &error);
    return error ? fallback : vsh::int64ToIntS(value);
}

double get_float(const VSMap *map, const char *key, double fallback,
                 const VSAPI *vsapi) {
    int error = 0;
    const double value = vsapi->mapGetFloat(map, key, 0, &error);
    return error ? fallback : value;
}

std::string get_data(const VSMap *map, const char *key,
                     const char *fallback, const VSAPI *vsapi) {
    int error = 0;
    const char *value = vsapi->mapGetData(map, key, 0, &error);
    return error || !value ? fallback : value;
}

KernelKind parse_kernel_name(const std::string &name) {
    if (name == "bilinear") return KernelKind::bilinear;
    if (name == "bicubic") return KernelKind::bicubic;
    if (name == "lanczos") return KernelKind::lanczos;
    if (name == "spline16") return KernelKind::spline16;
    if (name == "spline36") return KernelKind::spline36;
    if (name == "spline64") return KernelKind::spline64;
    throw std::invalid_argument("invalid kernel specified");
}

const char *kernel_name(KernelKind kind) noexcept {
    switch (kind) {
    case KernelKind::bilinear: return "bilinear";
    case KernelKind::bicubic: return "bicubic";
    case KernelKind::lanczos: return "lanczos";
    case KernelKind::spline16: return "spline16";
    case KernelKind::spline36: return "spline36";
    case KernelKind::spline64: return "spline64";
    case KernelKind::custom: return "custom";
    }
    return "unknown";
}

void set_error(VSMap *out, const std::string &message, const VSAPI *vsapi) {
    const std::string prefixed = "dsmvc: " + message;
    vsapi->mapSetError(out, prefixed.c_str());
}

VSFunction *get_function(const VSMap *in, const char *key, const VSAPI *vsapi) {
    int error = 0;
    VSFunction *function = vsapi->mapGetFunction(in, key, 0, &error);
    return error ? nullptr : function;
}

class FunctionGuard {
public:
    FunctionGuard(VSFunction *&function, const VSAPI *vsapi)
        : function_(function), vsapi_(vsapi) {}
    ~FunctionGuard() {
        if (!dismissed_ && function_) vsapi_->freeFunction(function_);
    }
    void dismiss() noexcept { dismissed_ = true; }

private:
    VSFunction *&function_;
    const VSAPI *vsapi_;
    bool dismissed_ = false;
};

ParsedArguments parse_arguments(const VSMap *in, std::intptr_t fixed_mode,
                                const VSVideoInfo &source_vi, const VSAPI *vsapi) {
    ParsedArguments parsed;
    FunctionGuard custom_guard(parsed.custom_kernel, vsapi);
    parsed.width = get_int(in, "width", 0, vsapi);
    parsed.height = get_int(in, "height", 0, vsapi);
    parsed.src_left = get_float(in, "src_left", 0.0, vsapi);
    parsed.src_top = get_float(in, "src_top", 0.0, vsapi);
    parsed.src_width = get_float(in, "src_width", static_cast<double>(parsed.width), vsapi);
    parsed.src_height = get_float(in, "src_height", static_cast<double>(parsed.height), vsapi);
    parsed.border_value = get_int(in, "border_handling", 0, vsapi);
    parsed.border = parsed.border_value == 1 ? BorderMode::zero
        : parsed.border_value == 2 ? BorderMode::repeat
                                   : BorderMode::mirror;
    parsed.force = get_int(in, "force", 0, vsapi);
    parsed.force_h = get_int(in, "force_h", parsed.force, vsapi);
    parsed.force_v = get_int(in, "force_v", parsed.force, vsapi);
    parsed.opt = get_int(in, "opt", 0, vsapi);
    parsed.cpu_path = parsed.opt == 1 ? CpuPath::scalar
        : parsed.opt == 2 ? CpuPath::avx2 : CpuPath::automatic;
    parsed.backend_text = get_data(in, "backend", "auto", vsapi);
    parsed.backend = dsmvc::parse_backend(parsed.backend_text);
    (void)dsmvc::resolve_backend(parsed.backend);

    if (fixed_mode != 0) {
        parsed.kernel.kind = static_cast<KernelKind>(fixed_mode - 1);
    } else {
        int kernel_error = 0;
        const char *kernel = vsapi->mapGetData(in, "kernel", 0, &kernel_error);
        VSFunction *legacy_custom = get_function(in, "custom", vsapi);
        VSFunction *custom_kernel = get_function(in, "custom_kernel", vsapi);
        if (legacy_custom) {
            parsed.custom_kernel = legacy_custom;
            if (custom_kernel) vsapi->freeFunction(custom_kernel);
        } else {
            parsed.custom_kernel = custom_kernel;
        }
        if (!kernel_error && parsed.custom_kernel) {
            throw std::invalid_argument("specify either kernel or custom kernel, not both");
        }
        if (kernel_error && !parsed.custom_kernel) {
            throw std::invalid_argument("kernel or custom kernel is required");
        }
        parsed.kernel.kind = parsed.custom_kernel
            ? KernelKind::custom : parse_kernel_name(lower_copy(kernel));
    }

    parsed.kernel.b = get_float(in, "b", 0.0, vsapi);
    parsed.kernel.c = get_float(in, "c", 0.5, vsapi);
    int taps_error = 0;
    const auto taps = vsapi->mapGetInt(in, "taps", 0, &taps_error);
    int support_error = 0;
    const auto support = vsapi->mapGetInt(in, "support", 0, &support_error);
    parsed.kernel.taps = !taps_error ? vsh::int64ToIntS(taps)
        : !support_error ? vsh::int64ToIntS(support) : 3;
    if ((parsed.kernel.kind == KernelKind::lanczos
         || parsed.kernel.kind == KernelKind::custom)
        && parsed.kernel.taps < 1) {
        throw std::invalid_argument("taps must be greater than zero");
    }
    if (parsed.kernel.kind == KernelKind::custom) {
        if (taps_error && support_error) {
            throw std::invalid_argument("custom kernels require taps or support");
        }
    }

    switch (parsed.kernel.kind) {
    case KernelKind::bilinear: parsed.function_name = "Debilinear"; break;
    case KernelKind::bicubic: parsed.function_name = "Debicubic"; break;
    case KernelKind::lanczos: parsed.function_name = "Delanczos"; break;
    case KernelKind::spline16: parsed.function_name = "Despline16"; break;
    case KernelKind::spline36: parsed.function_name = "Despline36"; break;
    case KernelKind::spline64: parsed.function_name = "Despline64"; break;
    case KernelKind::custom: parsed.function_name = "Descale"; break;
    }

    if (parsed.width < 1) throw std::invalid_argument("width must be greater than zero");
    if (parsed.height < 8) throw std::invalid_argument("height must be at least 8");
    if (parsed.width > source_vi.width || parsed.height > source_vi.height) {
        throw std::invalid_argument("output dimensions must not exceed input dimensions");
    }
    if (!(parsed.src_width > 0.0) || !(parsed.src_height > 0.0)) {
        throw std::invalid_argument("src_width and src_height must be positive");
    }
    custom_guard.dismiss();
    return parsed;
}

void copy_common_arguments(VSMap *map, const ParsedArguments &parsed,
                           const VSAPI *vsapi) {
    vsapi->mapSetInt(map, "width", parsed.width, maReplace);
    vsapi->mapSetInt(map, "height", parsed.height, maReplace);
    if (parsed.kernel.kind == KernelKind::custom) {
        vsapi->mapSetFunction(
            map, "custom_kernel", parsed.custom_kernel, maReplace);
    } else {
        vsapi->mapSetData(map, "kernel", kernel_name(parsed.kernel.kind), -1,
                          dtUtf8, maReplace);
    }
    vsapi->mapSetInt(map, "taps", parsed.kernel.taps, maReplace);
    vsapi->mapSetFloat(map, "b", parsed.kernel.b, maReplace);
    vsapi->mapSetFloat(map, "c", parsed.kernel.c, maReplace);
    vsapi->mapSetFloat(map, "src_left", parsed.src_left, maReplace);
    vsapi->mapSetFloat(map, "src_top", parsed.src_top, maReplace);
    vsapi->mapSetFloat(map, "src_width", parsed.src_width, maReplace);
    vsapi->mapSetFloat(map, "src_height", parsed.src_height, maReplace);
    vsapi->mapSetInt(map, "border_handling", parsed.border_value, maReplace);
    vsapi->mapSetInt(map, "force", parsed.force, maReplace);
    vsapi->mapSetInt(map, "force_h", parsed.force_h, maReplace);
    vsapi->mapSetInt(map, "force_v", parsed.force_v, maReplace);
    vsapi->mapSetInt(map, "opt", parsed.opt, maReplace);
    vsapi->mapSetData(map, "backend", parsed.backend_text.c_str(), -1,
                      dtUtf8, maReplace);
}

VSNode *invoke_clip(VSPlugin *plugin, const char *function, VSMap *arguments,
                    const VSAPI *vsapi) {
    VSMap *result = vsapi->invoke(plugin, function, arguments);
    const char *error = vsapi->mapGetError(result);
    if (error) {
        const std::string message = error;
        vsapi->freeMap(result);
        throw std::runtime_error(message);
    }
    int node_error = 0;
    VSNode *node = vsapi->mapGetNode(result, "clip", 0, &node_error);
    vsapi->freeMap(result);
    if (node_error || !node) throw std::runtime_error("filter did not return a clip");
    return node;
}

void convert_and_invoke(VSNode *source, const VSVideoInfo &source_vi,
                        const ParsedArguments &parsed, VSMap *out,
                        VSCore *core, const VSAPI *vsapi) {
    VSPlugin *resize = vsapi->getPluginByID("com.vapoursynth.resize", core);
    VSPlugin *self = vsapi->getPluginByID(plugin_id, core);
    if (!resize || !self) throw std::runtime_error("required plugin lookup failed");
    const auto float_format_id = vsapi->queryVideoFormatID(
        source_vi.format.colorFamily, stFloat, 32,
        source_vi.format.subSamplingW, source_vi.format.subSamplingH, core);
    const auto source_format_id = vsapi->queryVideoFormatID(
        source_vi.format.colorFamily, source_vi.format.sampleType,
        source_vi.format.bitsPerSample, source_vi.format.subSamplingW,
        source_vi.format.subSamplingH, core);
    if (!float_format_id || !source_format_id) {
        throw std::runtime_error("failed to query conversion format");
    }

    VSMap *arguments = vsapi->createMap();
    vsapi->mapSetNode(arguments, "clip", source, maReplace);
    vsapi->mapSetInt(arguments, "format", float_format_id, maReplace);
    vsapi->mapSetData(
        arguments, "dither_type", "none", -1, dtUtf8, maReplace);
    VSNode *float_node = nullptr;
    VSNode *descaled_node = nullptr;
    VSNode *result_node = nullptr;
    try {
        float_node = invoke_clip(resize, "Point", arguments, vsapi);
        vsapi->freeMap(arguments);

        arguments = vsapi->createMap();
        vsapi->mapSetNode(arguments, "src", float_node, maReplace);
        copy_common_arguments(arguments, parsed, vsapi);
        descaled_node = invoke_clip(self, "Descale", arguments, vsapi);
        vsapi->freeMap(arguments);
        vsapi->freeNode(float_node);
        float_node = nullptr;

        arguments = vsapi->createMap();
        vsapi->mapSetNode(arguments, "clip", descaled_node, maReplace);
        vsapi->mapSetInt(arguments, "format", source_format_id, maReplace);
        vsapi->mapSetData(
            arguments, "dither_type", "none", -1, dtUtf8, maReplace);
        result_node = invoke_clip(resize, "Point", arguments, vsapi);
        vsapi->freeMap(arguments);
        arguments = nullptr;
        vsapi->freeNode(descaled_node);
        descaled_node = nullptr;
        vsapi->mapConsumeNode(out, "clip", result_node, maReplace);
        result_node = nullptr;
    } catch (...) {
        if (arguments) vsapi->freeMap(arguments);
        if (float_node) vsapi->freeNode(float_node);
        if (descaled_node) vsapi->freeNode(descaled_node);
        if (result_node) vsapi->freeNode(result_node);
        throw;
    }
}

double call_custom_kernel(VSFunction *function, double x, const VSAPI *vsapi) {
    VSMap *input = vsapi->createMap();
    VSMap *output = vsapi->createMap();
    vsapi->mapSetFloat(input, "x", x, maReplace);
    vsapi->callFunction(function, input, output);
    vsapi->freeMap(input);
    const char *error = vsapi->mapGetError(output);
    if (error) {
        const std::string message = "custom kernel error: " + std::string{error};
        vsapi->freeMap(output);
        throw std::runtime_error(message);
    }
    int value_error = 0;
    double value = vsapi->mapGetFloat(output, "val", 0, &value_error);
    if (value_error) {
        value = static_cast<double>(
            vsapi->mapGetInt(output, "val", 0, &value_error));
    }
    vsapi->freeMap(output);
    if (value_error) throw std::runtime_error("custom kernel must return val as float or int");
    return value;
}

std::shared_ptr<const dsmvc::AxisPlan> make_plan(
    const AxisRequest &request, const CustomKernel &custom) {
    return dsmvc::get_or_build_axis_plan(request, custom);
}

void prepare_filter_requests(FilterData &data, const ParsedArguments &parsed) {
    AxisRequest request;
    request.kernel = parsed.kernel;
    request.border = parsed.border;

    if (data.process_horizontal) {
        request.source_size = data.source_width;
        request.destination_size = data.destination_width;
        request.active_length = parsed.src_width;
        request.shift = parsed.src_left;
        data.horizontal_requests[0] = request;
        data.has_horizontal_request[0] = true;
        if (data.num_planes > 1 && data.subsampling_w > 0) {
            const auto chroma_source = data.source_width >> data.subsampling_w;
            request.source_size = chroma_source;
            request.destination_size = data.destination_width >> data.subsampling_w;
            request.shift = 0.25 - 0.25 * static_cast<double>(data.destination_width)
                / static_cast<double>(data.source_width)
                + parsed.src_left * static_cast<double>(chroma_source)
                    / static_cast<double>(data.source_width);
            request.active_length = parsed.src_width
                * static_cast<double>(chroma_source)
                / static_cast<double>(data.source_width);
            data.horizontal_requests[1] = request;
            data.has_horizontal_request[1] = true;
        }
    }

    if (data.process_vertical) {
        request.source_size = data.source_height;
        request.destination_size = data.destination_height;
        request.active_length = parsed.src_height;
        request.shift = parsed.src_top;
        data.vertical_requests[0] = request;
        data.has_vertical_request[0] = true;
        if (data.num_planes > 1 && data.subsampling_h > 0) {
            const auto chroma_source = data.source_height >> data.subsampling_h;
            request.source_size = chroma_source;
            request.destination_size = data.destination_height >> data.subsampling_h;
            request.shift = parsed.src_top * static_cast<double>(chroma_source)
                / static_cast<double>(data.source_height);
            request.active_length = parsed.src_height
                * static_cast<double>(chroma_source)
                / static_cast<double>(data.source_height);
            data.vertical_requests[1] = request;
            data.has_vertical_request[1] = true;
        }
    }
}

void ensure_filter_plans(FilterData &data, const VSAPI *vsapi) {
    std::call_once(data.planning_once, [&] {
        try {
            for (std::size_t index = 0; index < 2U; ++index) {
                if (data.has_horizontal_request[index]) {
                    data.horizontal[index] = make_plan(
                        data.horizontal_requests[index], data.custom_callback);
                }
                if (data.has_vertical_request[index]) {
                    data.vertical[index] = make_plan(
                        data.vertical_requests[index], data.custom_callback);
                }
            }
            for (const auto &plan : data.horizontal) {
                if (plan) data.executor.prepare(plan);
            }
            for (const auto &plan : data.vertical) {
                if (plan) data.executor.prepare(plan);
            }
            data.executor.seal();
        } catch (...) {
            data.planning_error = std::current_exception();
        }
        data.custom_callback = {};
        if (data.custom_kernel) {
            vsapi->freeFunction(data.custom_kernel);
            data.custom_kernel = nullptr;
        }
    });
    if (data.planning_error) std::rethrow_exception(data.planning_error);
}

int frame_range(const VSFrame *frame, int color_family,
                const VSAPI *vsapi) noexcept {
    const auto *properties = vsapi->getFramePropertiesRO(frame);
    for (const auto *name : {"_Range", "_ColorRange"}) {
        int error = 0;
        const auto value = vsapi->mapGetInt(properties, name, 0, &error);
        if (!error) return value == 1 ? 1 : 0;
    }
    return color_family == cfRGB ? 1 : 0;
}

IntegerConversion integer_conversion(const FilterData &data, int plane,
                                     int range) noexcept {
    const auto maximum = (1U << data.bits_per_sample) - 1U;
    const bool chroma = data.color_family == cfYUV && plane != 0;
    std::uint32_t offset = 0U;
    std::uint32_t scale = maximum;
    if (range == 0) {
        const auto depth_scale = 1U << (data.bits_per_sample - 8);
        offset = (chroma ? 128U : 16U) * depth_scale;
        scale = (chroma ? 224U : 219U) * depth_scale;
    } else if (chroma) {
        offset = 1U << (data.bits_per_sample - 1);
    }
    return {
        static_cast<float>(offset),
        1.0F / static_cast<float>(scale),
        static_cast<float>(scale),
        static_cast<float>(offset),
        maximum,
    };
}

const VSFrame *VS_CC filter_get_frame(
    int frame_number, int activation_reason, void *instance_data, void **,
    VSFrameContext *frame_context, VSCore *core, const VSAPI *vsapi) {
    auto *data = static_cast<FilterData *>(instance_data);
    if (activation_reason == arInitial) {
        vsapi->requestFrameFilter(frame_number, data->node, frame_context);
        return nullptr;
    }
    if (activation_reason != arAllFramesReady) return nullptr;

    const VSFrame *source = vsapi->getFrameFilter(
        frame_number, data->node, frame_context);
    const bool adaptive_2d =
        data->process_horizontal && data->process_vertical;
    VSFrame *intermediate = nullptr;
    VSFrame *destination = nullptr;
    try {
        const bool uses_cuda =
            data->executor.backend() == BackendKind::cuda;
        const bool uses_gpu = uses_cuda
            || data->executor.backend() == BackendKind::vulkan;
#if defined(DSMVC_HAS_CUDA)
        std::optional<dsmvc::cuda_detail::NvtxRange> frame_trace;
        if (uses_cuda) {
            frame_trace.emplace(
                dsmvc::cuda_detail::NvtxLabel::frame,
                static_cast<std::uint32_t>(frame_number));
        }
#endif
#if defined(DSMVC_HAS_CUDA)
        {
            std::optional<dsmvc::cuda_detail::NvtxRange> plan_trace;
            if (uses_cuda) {
                plan_trace.emplace(
                    dsmvc::cuda_detail::NvtxLabel::filter_plan_preparation,
                    static_cast<std::uint32_t>(frame_number));
            }
            ensure_filter_plans(*data, vsapi);
        }
#else
        ensure_filter_plans(*data, vsapi);
#endif
        std::shared_ptr<const void> source_lifetime;
        if (data->executor.input_cache_enabled()) {
            const VSFrame *retained = vsapi->addFrameRef(source);
            if (!retained) {
                throw std::runtime_error("failed to retain the GPU source frame");
            }
            const auto free_frame = vsapi->freeFrame;
            source_lifetime = std::shared_ptr<const void>(
                retained, [free_frame](const void *frame) noexcept {
                    free_frame(static_cast<const VSFrame *>(frame));
                });
        }
        const auto &memory_config = memory_phase_config();
        const bool wide_kernel = adaptive_2d
            && data->vertical[0]->half_bandwidth >= 5;
        MemoryPhaseGuard memory_phase(
            !uses_gpu && adaptive_2d && memory_config.limit > 0U
            && static_cast<std::size_t>(data->core_threads) > memory_config.limit
            && (wide_kernel || memory_config.all_kernels));
        const bool allow_streamed_integer = !data->fused_integer
            || data->vertical[0]->half_bandwidth < 7
            || data->core_threads > 8;
        const bool track_overlapping_frames = !uses_gpu && adaptive_2d
            && data->core_threads > 1 && allow_streamed_integer;
        ActiveFrameGuard active_frame(track_overlapping_frames
            ? &data->active_2d_frames : nullptr);
        // Keep the internally parallel path for a sole frame. Overlapping
        // frames use destination-ordered RHS generation to reduce traffic.
        const bool first_2d_frame = adaptive_2d
            && (!track_overlapping_frames || active_frame.first());
        const bool buffered_float_2d =
            !uses_gpu && !data->fused_integer && first_2d_frame;
        if (buffered_float_2d) {
            intermediate = vsapi->newVideoFrame(
                &data->vi.format, data->destination_width, data->source_height,
                nullptr, core);
        }
        destination = vsapi->newVideoFrame(
            &data->vi.format, data->destination_width, data->destination_height,
            source, core);

        const auto range = data->fused_integer
            ? frame_range(source, data->color_family, vsapi) : 0;
        if (data->fused_integer) {
            vsapi->mapSetInt(
                vsapi->getFramePropertiesRW(destination), "_Range",
                range, maReplace);
        }

        for (int plane = 0; plane < data->num_planes; ++plane) {
#if defined(DSMVC_HAS_CUDA)
            std::optional<dsmvc::cuda_detail::NvtxRange> plane_trace;
            if (uses_cuda) {
                plane_trace.emplace(
                    dsmvc::cuda_detail::NvtxLabel::plane,
                    dsmvc::cuda_detail::nvtx_frame_plane_payload(
                        static_cast<std::uint32_t>(frame_number),
                        static_cast<std::uint32_t>(plane)));
            }
#endif
            const int horizontal_index = plane != 0 && data->subsampling_w > 0;
            const int vertical_index = plane != 0 && data->subsampling_h > 0;
            if (data->fused_integer) {
                const auto conversion = integer_conversion(*data, plane, range);
                const auto source_stride = vsapi->getStride(source, plane)
                    / data->bytes_per_sample;
                const auto destination_stride = vsapi->getStride(destination, plane)
                    / data->bytes_per_sample;
                if (data->bytes_per_sample == 1) {
                    if (first_2d_frame) {
                        data->executor.inverse_2d_u8(
                            *data->horizontal[horizontal_index],
                            *data->vertical[vertical_index],
                            vsapi->getReadPtr(source, plane), source_stride,
                            vsapi->getWritePtr(destination, plane),
                            destination_stride, conversion, source_lifetime);
                    } else {
                        data->executor.inverse_2d_u8_streamed(
                            *data->horizontal[horizontal_index],
                            *data->vertical[vertical_index],
                            vsapi->getReadPtr(source, plane), source_stride,
                            vsapi->getWritePtr(destination, plane),
                            destination_stride, conversion, source_lifetime);
                    }
                } else {
                    if (first_2d_frame) {
                        data->executor.inverse_2d_u16(
                            *data->horizontal[horizontal_index],
                            *data->vertical[vertical_index],
                            reinterpret_cast<const std::uint16_t *>(
                                vsapi->getReadPtr(source, plane)),
                            source_stride,
                            reinterpret_cast<std::uint16_t *>(
                                vsapi->getWritePtr(destination, plane)),
                            destination_stride, conversion, source_lifetime);
                    } else {
                        data->executor.inverse_2d_u16_streamed(
                            *data->horizontal[horizontal_index],
                            *data->vertical[vertical_index],
                            reinterpret_cast<const std::uint16_t *>(
                                vsapi->getReadPtr(source, plane)),
                            source_stride,
                            reinterpret_cast<std::uint16_t *>(
                                vsapi->getWritePtr(destination, plane)),
                            destination_stride, conversion, source_lifetime);
                    }
                }
                continue;
            }
            const auto source_stride = vsapi->getStride(source, plane)
                / static_cast<int>(sizeof(float));
            const auto destination_stride = vsapi->getStride(destination, plane)
                / static_cast<int>(sizeof(float));
            const auto *source_ptr = reinterpret_cast<const float *>(
                vsapi->getReadPtr(source, plane));
            auto *destination_ptr = reinterpret_cast<float *>(
                vsapi->getWritePtr(destination, plane));
            if (data->process_horizontal && data->process_vertical) {
                if (buffered_float_2d) {
                    const auto intermediate_stride =
                        vsapi->getStride(intermediate, plane)
                        / static_cast<int>(sizeof(float));
                    auto *intermediate_ptr = reinterpret_cast<float *>(
                        vsapi->getWritePtr(intermediate, plane));
                    const auto row_count = data->source_height
                        >> (plane != 0 ? data->subsampling_h : 0);
                    data->executor.inverse_rows(
                        *data->horizontal[horizontal_index],
                        source_ptr, source_stride,
                        intermediate_ptr, intermediate_stride, row_count);
                    const auto column_count = data->destination_width
                        >> (plane != 0 ? data->subsampling_w : 0);
                    data->executor.inverse_columns(
                        *data->vertical[vertical_index],
                        intermediate_ptr, intermediate_stride,
                        destination_ptr, destination_stride, column_count);
                } else {
                    data->executor.inverse_2d(
                        *data->horizontal[horizontal_index],
                        *data->vertical[vertical_index],
                        source_ptr, source_stride,
                        destination_ptr, destination_stride, source_lifetime);
                }
            } else if (data->process_horizontal) {
                const auto row_count = data->source_height
                    >> (plane != 0 ? data->subsampling_h : 0);
                data->executor.inverse_rows(
                    *data->horizontal[horizontal_index], source_ptr, source_stride,
                    destination_ptr, destination_stride, row_count,
                    source_lifetime);
            } else {
                const auto column_count = data->source_width
                    >> (plane != 0 ? data->subsampling_w : 0);
                data->executor.inverse_columns(
                    *data->vertical[vertical_index], source_ptr, source_stride,
                    destination_ptr, destination_stride, column_count,
                    source_lifetime);
            }
        }
        if (intermediate) vsapi->freeFrame(intermediate);
        vsapi->freeFrame(source);
        return destination;
    } catch (const std::exception &error) {
        if (intermediate) vsapi->freeFrame(intermediate);
        if (destination) vsapi->freeFrame(destination);
        vsapi->freeFrame(source);
        const std::string message = "dsmvc: frame processing failed: "
            + std::string{error.what()};
        vsapi->setFilterError(message.c_str(), frame_context);
        return nullptr;
    }
}

void VS_CC filter_free(void *instance_data, VSCore *, const VSAPI *vsapi) {
    auto *data = static_cast<FilterData *>(instance_data);
    if (data->custom_kernel) vsapi->freeFunction(data->custom_kernel);
    if (data->node) vsapi->freeNode(data->node);
    delete data;
}

void VS_CC filter_create(const VSMap *in, VSMap *out, void *user_data,
                         VSCore *core, const VSAPI *vsapi) {
    VSNode *source = nullptr;
    VSFunction *custom = nullptr;
    try {
        int node_error = 0;
        source = vsapi->mapGetNode(in, "src", 0, &node_error);
        if (node_error || !source) throw std::invalid_argument("src is required");
        const VSVideoInfo *source_info = vsapi->getVideoInfo(source);
        if (!vsh::isConstantVideoFormat(source_info)) {
            throw std::invalid_argument("only constant format input is supported");
        }

        const auto fixed_mode = reinterpret_cast<std::intptr_t>(user_data);
        ParsedArguments parsed = parse_arguments(in, fixed_mode, *source_info, vsapi);
        custom = parsed.custom_kernel;
        if (parsed.width % (1 << source_info->format.subSamplingW) != 0) {
            throw std::invalid_argument("output width is incompatible with subsampling");
        }
        if (parsed.height % (1 << source_info->format.subSamplingH) != 0) {
            throw std::invalid_argument("output height is incompatible with subsampling");
        }

        const bool process_horizontal = parsed.width != source_info->width
            || parsed.src_left != 0.0
            || parsed.src_width != static_cast<double>(parsed.width)
            || parsed.force_h != 0;
        const bool process_vertical = parsed.height != source_info->height
            || parsed.src_top != 0.0
            || parsed.src_height != static_cast<double>(parsed.height)
            || parsed.force_v != 0;
        if (!process_horizontal && !process_vertical) {
            vsapi->mapConsumeNode(out, "clip", source, maReplace);
            source = nullptr;
            if (custom) vsapi->freeFunction(custom);
            return;
        }

        const bool fused_integer =
            source_info->format.sampleType == stInteger
            && source_info->format.bitsPerSample >= 8
            && source_info->format.bitsPerSample <= 16
            && (source_info->format.bytesPerSample == 1
                || source_info->format.bytesPerSample == 2)
            && process_horizontal && process_vertical;
        if (!fused_integer
            && (source_info->format.sampleType != stFloat
                || source_info->format.bitsPerSample != 32)) {
            convert_and_invoke(source, *source_info, parsed, out, core, vsapi);
            if (custom) vsapi->freeFunction(custom);
            vsapi->freeNode(source);
            return;
        }

        auto data = std::make_unique<FilterData>(
            parsed.backend, parsed.cpu_path);
        data->node = source;
        data->vi = *source_info;
        data->source_width = source_info->width;
        data->source_height = source_info->height;
        data->destination_width = parsed.width;
        data->destination_height = parsed.height;
        data->subsampling_w = source_info->format.subSamplingW;
        data->subsampling_h = source_info->format.subSamplingH;
        data->num_planes = source_info->format.numPlanes;
        data->color_family = source_info->format.colorFamily;
        data->bits_per_sample = source_info->format.bitsPerSample;
        data->bytes_per_sample = source_info->format.bytesPerSample;
        VSCoreInfo core_info{};
        vsapi->getCoreInfo(core, &core_info);
        data->core_threads = std::max(core_info.numThreads, 1);
        data->process_horizontal = process_horizontal;
        data->process_vertical = process_vertical;
        data->fused_integer = fused_integer;
        data->vi.width = parsed.width;
        data->vi.height = parsed.height;
        prepare_filter_requests(*data, parsed);
        if (custom) {
            data->custom_callback = [custom, vsapi](double x) {
                return call_custom_kernel(custom, x, vsapi);
            };
            data->custom_kernel = custom;
        }

        const VSFilterDependency dependency{data->node, rpStrictSpatial};
        VSNode *result = vsapi->createVideoFilter2(
            parsed.function_name.c_str(), &data->vi, filter_get_frame,
            filter_free, fmParallel, &dependency, 1, data.get(), core);
        if (!result) throw std::runtime_error("failed to create video filter");
        source = nullptr;
        custom = nullptr;
        data.release();
        if (vsapi->mapConsumeNode(out, "clip", result, maReplace) != 0) {
            throw std::runtime_error("failed to return video filter");
        }
    } catch (const std::exception &error) {
        if (custom) vsapi->freeFunction(custom);
        if (source) vsapi->freeNode(source);
        set_error(out, error.what(), vsapi);
    } catch (...) {
        if (custom) vsapi->freeFunction(custom);
        if (source) vsapi->freeNode(source);
        set_error(out, "unknown plugin error", vsapi);
    }
}

constexpr const char *common_geometry =
    "src:vnode;"
    "width:int;"
    "height:int;";

constexpr const char *common_tail =
    "src_left:float:opt;"
    "src_top:float:opt;"
    "src_width:float:opt;"
    "src_height:float:opt;"
    "border_handling:int:opt;"
    "force:int:opt;"
    "force_h:int:opt;"
    "force_v:int:opt;"
    "opt:int:opt;";

constexpr const char *backend_tail = "backend:data:opt;";
constexpr const char *clip_return = "clip:vnode;";

} // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit2(
    VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin(
        plugin_id, "dsmvc", "dsmvc optimized descale",
        VS_MAKE_VERSION(0, 1), VAPOURSYNTH_API_VERSION, 0, plugin);

    const std::string geometry = common_geometry;
    const std::string tail = common_tail;
    vspapi->registerFunction(
        "Debilinear", (geometry + tail + backend_tail).c_str(), clip_return,
        filter_create,
        reinterpret_cast<void *>(static_cast<std::intptr_t>(1)), plugin);
    vspapi->registerFunction(
        "Debicubic", (geometry + "b:float:opt;c:float:opt;" + tail
                      + backend_tail).c_str(),
        clip_return, filter_create,
        reinterpret_cast<void *>(static_cast<std::intptr_t>(2)), plugin);
    vspapi->registerFunction(
        "Delanczos", (geometry + "taps:int:opt;" + tail
                      + backend_tail).c_str(),
        clip_return, filter_create,
        reinterpret_cast<void *>(static_cast<std::intptr_t>(3)), plugin);
    vspapi->registerFunction(
        "Despline16", (geometry + tail + backend_tail).c_str(), clip_return,
        filter_create,
        reinterpret_cast<void *>(static_cast<std::intptr_t>(4)), plugin);
    vspapi->registerFunction(
        "Despline36", (geometry + tail + backend_tail).c_str(), clip_return,
        filter_create,
        reinterpret_cast<void *>(static_cast<std::intptr_t>(5)), plugin);
    vspapi->registerFunction(
        "Despline64", (geometry + tail + backend_tail).c_str(), clip_return,
        filter_create,
        reinterpret_cast<void *>(static_cast<std::intptr_t>(6)), plugin);
    vspapi->registerFunction(
        "Descale",
        (geometry
         + "kernel:data:opt;taps:int:opt;b:float:opt;c:float:opt;"
         + tail
         + "custom:func:opt;support:int:opt;custom_kernel:func:opt;"
         + backend_tail).c_str(),
        clip_return, filter_create, nullptr, plugin);
}
