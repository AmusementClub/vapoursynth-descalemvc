#include <dsmvc/engine.hpp>

#include <stdexcept>

namespace dsmvc {

getnative::AxisPlan build_custom_axis_plan(const AxisRequest &request,
                                           const CustomKernel &custom_kernel);

namespace {

getnative::Filter to_filter(const KernelSpec &kernel) {
    switch (kernel.kind) {
    case KernelKind::bilinear: return getnative::Filter::bilinear();
    case KernelKind::bicubic: return getnative::Filter::bicubic(kernel.b, kernel.c);
    case KernelKind::lanczos: return getnative::Filter::lanczos(kernel.taps);
    case KernelKind::spline16: return getnative::Filter::spline16();
    case KernelKind::spline36: return getnative::Filter::spline36();
    case KernelKind::spline64: return getnative::Filter::spline64();
    case KernelKind::custom: break;
    }
    throw std::invalid_argument("custom kernels require a callback");
}

} // namespace

getnative::AxisPlan build_axis_plan(const AxisRequest &request,
                                    const CustomKernel &custom_kernel) {
    if (request.kernel.kind == KernelKind::custom) {
        if (!custom_kernel) {
            throw std::invalid_argument("custom kernel callback is missing");
        }
        return build_custom_axis_plan(request, custom_kernel);
    }

    getnative::AxisPlanRequest upstream;
    upstream.source_size = request.source_size;
    upstream.destination_size = request.destination_size;
    upstream.active_length = request.active_length;
    upstream.shift = request.shift;
    upstream.filter = to_filter(request.kernel);
    upstream.border = request.border;
    return getnative::build_axis_plan(upstream);
}

} // namespace dsmvc
