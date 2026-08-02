#include <dsmvc/engine.hpp>

#include <VapourSynth.h>
#include <VSHelper.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using dsmvc::AxisRequest;
using dsmvc::BackendKind;
using dsmvc::CpuExecutor;
using dsmvc::CpuPath;
using dsmvc::CustomKernel;
using dsmvc::KernelKind;
using dsmvc::KernelSpec;

constexpr const char *plugin_id = "com.dsmvc.descale";

struct ParsedArguments {
    KernelSpec kernel{};
    VSFuncRef *custom_kernel = nullptr;
    int width = 0;
    int height = 0;
    double src_left = 0.0;
    double src_top = 0.0;
    double src_width = 0.0;
    double src_height = 0.0;
    getnative::BorderMode border = getnative::BorderMode::mirror;
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
    explicit FilterData(CpuPath path) : executor(path) {}

    VSNodeRef *node = nullptr;
    VSVideoInfo vi{};
    int source_width = 0;
    int source_height = 0;
    int destination_width = 0;
    int destination_height = 0;
    int subsampling_w = 0;
    int subsampling_h = 0;
    int num_planes = 0;
    bool process_horizontal = false;
    bool process_vertical = false;
    std::shared_ptr<const getnative::AxisPlan> horizontal[2];
    std::shared_ptr<const getnative::AxisPlan> vertical[2];
    CpuExecutor executor{};
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
    const auto value = vsapi->propGetInt(map, key, 0, &error);
    return error ? fallback : int64ToIntS(value);
}

double get_float(const VSMap *map, const char *key, double fallback,
                 const VSAPI *vsapi) {
    int error = 0;
    const double value = vsapi->propGetFloat(map, key, 0, &error);
    return error ? fallback : value;
}

std::string get_data(const VSMap *map, const char *key,
                     const char *fallback, const VSAPI *vsapi) {
    int error = 0;
    const char *value = vsapi->propGetData(map, key, 0, &error);
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
    vsapi->setError(out, prefixed.c_str());
}

VSFuncRef *get_function(const VSMap *in, const char *key, const VSAPI *vsapi) {
    int error = 0;
    VSFuncRef *function = vsapi->propGetFunc(in, key, 0, &error);
    return error ? nullptr : function;
}

class FunctionGuard {
public:
    FunctionGuard(VSFuncRef *&function, const VSAPI *vsapi)
        : function_(function), vsapi_(vsapi) {}
    ~FunctionGuard() {
        if (!dismissed_ && function_) vsapi_->freeFunc(function_);
    }
    void dismiss() noexcept { dismissed_ = true; }

private:
    VSFuncRef *&function_;
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
    parsed.border = parsed.border_value == 1 ? getnative::BorderMode::zero
        : parsed.border_value == 2 ? getnative::BorderMode::repeat
                                   : getnative::BorderMode::mirror;
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
        const char *kernel = vsapi->propGetData(in, "kernel", 0, &kernel_error);
        VSFuncRef *legacy_custom = get_function(in, "custom", vsapi);
        VSFuncRef *custom_kernel = get_function(in, "custom_kernel", vsapi);
        if (legacy_custom) {
            parsed.custom_kernel = legacy_custom;
            if (custom_kernel) vsapi->freeFunc(custom_kernel);
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
    const auto taps = vsapi->propGetInt(in, "taps", 0, &taps_error);
    int support_error = 0;
    const auto support = vsapi->propGetInt(in, "support", 0, &support_error);
    parsed.kernel.taps = !taps_error ? int64ToIntS(taps)
        : !support_error ? int64ToIntS(support) : 3;
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
    vsapi->propSetInt(map, "width", parsed.width, paReplace);
    vsapi->propSetInt(map, "height", parsed.height, paReplace);
    if (parsed.kernel.kind == KernelKind::custom) {
        vsapi->propSetFunc(map, "custom_kernel", parsed.custom_kernel, paReplace);
    } else {
        vsapi->propSetData(map, "kernel", kernel_name(parsed.kernel.kind), -1, paReplace);
    }
    vsapi->propSetInt(map, "taps", parsed.kernel.taps, paReplace);
    vsapi->propSetFloat(map, "b", parsed.kernel.b, paReplace);
    vsapi->propSetFloat(map, "c", parsed.kernel.c, paReplace);
    vsapi->propSetFloat(map, "src_left", parsed.src_left, paReplace);
    vsapi->propSetFloat(map, "src_top", parsed.src_top, paReplace);
    vsapi->propSetFloat(map, "src_width", parsed.src_width, paReplace);
    vsapi->propSetFloat(map, "src_height", parsed.src_height, paReplace);
    vsapi->propSetInt(map, "border_handling", parsed.border_value, paReplace);
    vsapi->propSetInt(map, "force", parsed.force, paReplace);
    vsapi->propSetInt(map, "force_h", parsed.force_h, paReplace);
    vsapi->propSetInt(map, "force_v", parsed.force_v, paReplace);
    vsapi->propSetInt(map, "opt", parsed.opt, paReplace);
    vsapi->propSetData(map, "backend", parsed.backend_text.c_str(), -1, paReplace);
}

VSNodeRef *invoke_clip(VSPlugin *plugin, const char *function, VSMap *arguments,
                       const VSAPI *vsapi) {
    VSMap *result = vsapi->invoke(plugin, function, arguments);
    const char *error = vsapi->getError(result);
    if (error) {
        const std::string message = error;
        vsapi->freeMap(result);
        throw std::runtime_error(message);
    }
    int node_error = 0;
    VSNodeRef *node = vsapi->propGetNode(result, "clip", 0, &node_error);
    vsapi->freeMap(result);
    if (node_error || !node) throw std::runtime_error("filter did not return a clip");
    return node;
}

void convert_and_invoke(VSNodeRef *source, const VSVideoInfo &source_vi,
                        const ParsedArguments &parsed, VSMap *out,
                        VSCore *core, const VSAPI *vsapi) {
    VSPlugin *resize = vsapi->getPluginById("com.vapoursynth.resize", core);
    VSPlugin *self = vsapi->getPluginById(plugin_id, core);
    if (!resize || !self) throw std::runtime_error("required plugin lookup failed");
    const VSFormat *float_format = vsapi->registerFormat(
        source_vi.format->colorFamily, stFloat, 32,
        source_vi.format->subSamplingW, source_vi.format->subSamplingH, core);
    if (!float_format) throw std::runtime_error("failed to register Float32 format");

    VSMap *arguments = vsapi->createMap();
    vsapi->propSetNode(arguments, "clip", source, paReplace);
    vsapi->propSetInt(arguments, "format", float_format->id, paReplace);
    vsapi->propSetData(arguments, "dither_type", "none", -1, paReplace);
    VSNodeRef *float_node = nullptr;
    VSNodeRef *descaled_node = nullptr;
    VSNodeRef *result_node = nullptr;
    try {
        float_node = invoke_clip(resize, "Point", arguments, vsapi);
        vsapi->freeMap(arguments);

        arguments = vsapi->createMap();
        vsapi->propSetNode(arguments, "src", float_node, paReplace);
        copy_common_arguments(arguments, parsed, vsapi);
        descaled_node = invoke_clip(self, "Descale", arguments, vsapi);
        vsapi->freeMap(arguments);
        vsapi->freeNode(float_node);
        float_node = nullptr;

        arguments = vsapi->createMap();
        vsapi->propSetNode(arguments, "clip", descaled_node, paReplace);
        vsapi->propSetInt(arguments, "format", source_vi.format->id, paReplace);
        vsapi->propSetData(arguments, "dither_type", "none", -1, paReplace);
        result_node = invoke_clip(resize, "Point", arguments, vsapi);
        vsapi->freeMap(arguments);
        arguments = nullptr;
        vsapi->freeNode(descaled_node);
        descaled_node = nullptr;
        vsapi->propSetNode(out, "clip", result_node, paReplace);
        vsapi->freeNode(result_node);
    } catch (...) {
        if (arguments) vsapi->freeMap(arguments);
        if (float_node) vsapi->freeNode(float_node);
        if (descaled_node) vsapi->freeNode(descaled_node);
        if (result_node) vsapi->freeNode(result_node);
        throw;
    }
}

double call_custom_kernel(VSFuncRef *function, double x, VSCore *core,
                          const VSAPI *vsapi) {
    VSMap *input = vsapi->createMap();
    VSMap *output = vsapi->createMap();
    vsapi->propSetFloat(input, "x", x, paReplace);
    vsapi->callFunc(function, input, output, core, vsapi);
    vsapi->freeMap(input);
    const char *error = vsapi->getError(output);
    if (error) {
        const std::string message = "custom kernel error: " + std::string{error};
        vsapi->freeMap(output);
        throw std::runtime_error(message);
    }
    int value_error = 0;
    double value = vsapi->propGetFloat(output, "val", 0, &value_error);
    if (value_error) {
        value = static_cast<double>(vsapi->propGetInt(output, "val", 0, &value_error));
    }
    vsapi->freeMap(output);
    if (value_error) throw std::runtime_error("custom kernel must return val as float or int");
    return value;
}

std::shared_ptr<const getnative::AxisPlan> make_plan(
    const AxisRequest &request, const CustomKernel &custom) {
    return std::make_shared<const getnative::AxisPlan>(
        dsmvc::build_axis_plan(request, custom));
}

void build_filter_plans(FilterData &data, const ParsedArguments &parsed,
                        const CustomKernel &custom) {
    AxisRequest request;
    request.kernel = parsed.kernel;
    request.border = parsed.border;

    if (data.process_horizontal) {
        request.source_size = data.source_width;
        request.destination_size = data.destination_width;
        request.active_length = parsed.src_width;
        request.shift = parsed.src_left;
        data.horizontal[0] = make_plan(request, custom);
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
            data.horizontal[1] = make_plan(request, custom);
        }
    }

    if (data.process_vertical) {
        request.source_size = data.source_height;
        request.destination_size = data.destination_height;
        request.active_length = parsed.src_height;
        request.shift = parsed.src_top;
        data.vertical[0] = make_plan(request, custom);
        if (data.num_planes > 1 && data.subsampling_h > 0) {
            const auto chroma_source = data.source_height >> data.subsampling_h;
            request.source_size = chroma_source;
            request.destination_size = data.destination_height >> data.subsampling_h;
            request.shift = parsed.src_top * static_cast<double>(chroma_source)
                / static_cast<double>(data.source_height);
            request.active_length = parsed.src_height
                * static_cast<double>(chroma_source)
                / static_cast<double>(data.source_height);
            data.vertical[1] = make_plan(request, custom);
        }
    }

    for (const auto &plan : data.horizontal) {
        if (plan) data.executor.prepare(*plan);
    }
    for (const auto &plan : data.vertical) {
        if (plan) data.executor.prepare(*plan);
    }
    data.executor.seal();
}

void VS_CC filter_init(VSMap *, VSMap *, void **instance_data, VSNode *node,
                       VSCore *, const VSAPI *vsapi) {
    auto *data = static_cast<FilterData *>(*instance_data);
    vsapi->setVideoInfo(&data->vi, 1, node);
}

const VSFrameRef *VS_CC filter_get_frame(
    int frame_number, int activation_reason, void **instance_data, void **,
    VSFrameContext *frame_context, VSCore *core, const VSAPI *vsapi) {
    auto *data = static_cast<FilterData *>(*instance_data);
    if (activation_reason == arInitial) {
        vsapi->requestFrameFilter(frame_number, data->node, frame_context);
        return nullptr;
    }
    if (activation_reason != arAllFramesReady) return nullptr;

    const VSFrameRef *source = vsapi->getFrameFilter(
        frame_number, data->node, frame_context);
    VSFrameRef *intermediate = nullptr;
    VSFrameRef *destination = nullptr;
    try {
        if (data->process_horizontal && data->process_vertical) {
            intermediate = vsapi->newVideoFrame(
                data->vi.format, data->destination_width, data->source_height,
                nullptr, core);
        }
        destination = vsapi->newVideoFrame(
            data->vi.format, data->destination_width, data->destination_height,
            source, core);

        for (int plane = 0; plane < data->num_planes; ++plane) {
            const auto source_stride = vsapi->getStride(source, plane)
                / static_cast<int>(sizeof(float));
            const auto destination_stride = vsapi->getStride(destination, plane)
                / static_cast<int>(sizeof(float));
            const auto *source_ptr = reinterpret_cast<const float *>(
                vsapi->getReadPtr(source, plane));
            auto *destination_ptr = reinterpret_cast<float *>(
                vsapi->getWritePtr(destination, plane));
            const int horizontal_index = plane != 0 && data->subsampling_w > 0;
            const int vertical_index = plane != 0 && data->subsampling_h > 0;

            if (data->process_horizontal && data->process_vertical) {
                const auto intermediate_stride = vsapi->getStride(intermediate, plane)
                    / static_cast<int>(sizeof(float));
                auto *intermediate_ptr = reinterpret_cast<float *>(
                    vsapi->getWritePtr(intermediate, plane));
                const auto row_count = data->source_height
                    >> (plane != 0 ? data->subsampling_h : 0);
                data->executor.inverse_rows(
                    *data->horizontal[horizontal_index], source_ptr, source_stride,
                    intermediate_ptr, intermediate_stride, row_count);
                const auto column_count = data->destination_width
                    >> (plane != 0 ? data->subsampling_w : 0);
                data->executor.inverse_columns(
                    *data->vertical[vertical_index], intermediate_ptr,
                    intermediate_stride, destination_ptr, destination_stride,
                    column_count);
            } else if (data->process_horizontal) {
                const auto row_count = data->source_height
                    >> (plane != 0 ? data->subsampling_h : 0);
                data->executor.inverse_rows(
                    *data->horizontal[horizontal_index], source_ptr, source_stride,
                    destination_ptr, destination_stride, row_count);
            } else {
                const auto column_count = data->source_width
                    >> (plane != 0 ? data->subsampling_w : 0);
                data->executor.inverse_columns(
                    *data->vertical[vertical_index], source_ptr, source_stride,
                    destination_ptr, destination_stride, column_count);
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
    if (data->node) vsapi->freeNode(data->node);
    delete data;
}

void VS_CC filter_create(const VSMap *in, VSMap *out, void *user_data,
                         VSCore *core, const VSAPI *vsapi) {
    VSNodeRef *source = nullptr;
    VSFuncRef *custom = nullptr;
    try {
        int node_error = 0;
        source = vsapi->propGetNode(in, "src", 0, &node_error);
        if (node_error || !source) throw std::invalid_argument("src is required");
        const VSVideoInfo *source_info = vsapi->getVideoInfo(source);
        if (!isConstantFormat(source_info)) {
            throw std::invalid_argument("only constant format input is supported");
        }
        if (source_info->format->colorFamily == cmCompat) {
            throw std::invalid_argument("compat formats are not supported");
        }

        const auto fixed_mode = reinterpret_cast<std::intptr_t>(user_data);
        ParsedArguments parsed = parse_arguments(in, fixed_mode, *source_info, vsapi);
        custom = parsed.custom_kernel;
        if (parsed.width % (1 << source_info->format->subSamplingW) != 0) {
            throw std::invalid_argument("output width is incompatible with subsampling");
        }
        if (parsed.height % (1 << source_info->format->subSamplingH) != 0) {
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
            vsapi->propSetNode(out, "clip", source, paReplace);
            if (custom) vsapi->freeFunc(custom);
            vsapi->freeNode(source);
            return;
        }

        if (source_info->format->sampleType != stFloat
            || source_info->format->bitsPerSample != 32) {
            convert_and_invoke(source, *source_info, parsed, out, core, vsapi);
            if (custom) vsapi->freeFunc(custom);
            vsapi->freeNode(source);
            return;
        }

        auto data = std::make_unique<FilterData>(parsed.cpu_path);
        data->node = source;
        source = nullptr;
        data->vi = *source_info;
        data->source_width = source_info->width;
        data->source_height = source_info->height;
        data->destination_width = parsed.width;
        data->destination_height = parsed.height;
        data->subsampling_w = source_info->format->subSamplingW;
        data->subsampling_h = source_info->format->subSamplingH;
        data->num_planes = source_info->format->numPlanes;
        data->process_horizontal = process_horizontal;
        data->process_vertical = process_vertical;
        data->vi.width = parsed.width;
        data->vi.height = parsed.height;
        CustomKernel callback;
        if (custom) {
            callback = [custom, core, vsapi](double x) {
                return call_custom_kernel(custom, x, core, vsapi);
            };
        }
        build_filter_plans(*data, parsed, callback);
        if (custom) {
            vsapi->freeFunc(custom);
            custom = nullptr;
        }

        auto *raw = data.release();
        vsapi->createFilter(in, out, parsed.function_name.c_str(), filter_init,
                            filter_get_frame, filter_free, fmParallel, 0, raw, core);
    } catch (const std::exception &error) {
        if (custom) vsapi->freeFunc(custom);
        if (source) vsapi->freeNode(source);
        set_error(out, error.what(), vsapi);
    } catch (...) {
        if (custom) vsapi->freeFunc(custom);
        if (source) vsapi->freeNode(source);
        set_error(out, "unknown plugin error", vsapi);
    }
}

constexpr const char *common_geometry =
    "src:clip;"
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

} // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin config_func, VSRegisterFunction register_func, VSPlugin *plugin) {
    config_func(plugin_id, "dsmvc", "dsmvc optimized descale",
                VAPOURSYNTH_API_VERSION, 1, plugin);

    const std::string geometry = common_geometry;
    const std::string tail = common_tail;
    register_func("Debilinear", (geometry + tail + backend_tail).c_str(), filter_create,
                  reinterpret_cast<void *>(static_cast<std::intptr_t>(1)), plugin);
    register_func("Debicubic", (geometry + "b:float:opt;c:float:opt;" + tail
                                + backend_tail).c_str(),
                  filter_create,
                  reinterpret_cast<void *>(static_cast<std::intptr_t>(2)), plugin);
    register_func("Delanczos", (geometry + "taps:int:opt;" + tail
                                + backend_tail).c_str(),
                  filter_create,
                  reinterpret_cast<void *>(static_cast<std::intptr_t>(3)), plugin);
    register_func("Despline16", (geometry + tail + backend_tail).c_str(), filter_create,
                  reinterpret_cast<void *>(static_cast<std::intptr_t>(4)), plugin);
    register_func("Despline36", (geometry + tail + backend_tail).c_str(), filter_create,
                  reinterpret_cast<void *>(static_cast<std::intptr_t>(5)), plugin);
    register_func("Despline64", (geometry + tail + backend_tail).c_str(), filter_create,
                  reinterpret_cast<void *>(static_cast<std::intptr_t>(6)), plugin);
    register_func(
        "Descale",
        (geometry
         + "kernel:data:opt;taps:int:opt;b:float:opt;c:float:opt;"
         + tail
         + "custom:func:opt;support:int:opt;custom_kernel:func:opt;"
         + backend_tail).c_str(),
        filter_create, nullptr, plugin);
}
