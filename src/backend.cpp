#include <dsmvc/engine.hpp>

#if defined(DSMVC_HAS_CUDA)
#include "cuda/cuda_executor.hpp"
#endif
#if defined(DSMVC_HAS_VULKAN)
#include "vulkan/vulkan_executor.hpp"
#endif
#if defined(DSMVC_HAS_METAL)
#include "metal_device_apple.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace dsmvc {

const char *backend_name(BackendKind kind) noexcept {
    switch (kind) {
    case BackendKind::automatic: return "auto";
    case BackendKind::cpu: return "cpu";
    case BackendKind::metal: return "metal";
    case BackendKind::vulkan: return "vulkan";
    case BackendKind::cuda: return "cuda";
    }
    return "unknown";
}

BackendKind parse_backend(std::string_view name) {
    std::string normalized{name};
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    if (normalized.empty() || normalized == "auto") return BackendKind::automatic;
    if (normalized == "cpu") return BackendKind::cpu;
    if (normalized == "metal") return BackendKind::metal;
    if (normalized == "vulkan") return BackendKind::vulkan;
    if (normalized == "cuda") return BackendKind::cuda;
    throw std::invalid_argument("backend must be auto, cpu, metal, vulkan, or cuda");
}

BackendKind resolve_backend(BackendKind requested) {
    if (requested == BackendKind::automatic || requested == BackendKind::cpu) {
        return BackendKind::cpu;
    }
#if defined(DSMVC_HAS_VULKAN)
    if (requested == BackendKind::vulkan) {
        vulkan_detail::require_backend_available();
        return BackendKind::vulkan;
    }
#endif
#if defined(DSMVC_HAS_CUDA)
    if (requested == BackendKind::cuda) {
        if (cuda_detail::backend_available()) return BackendKind::cuda;
        throw std::runtime_error(
            "backend 'cuda' is compiled but no CUDA device is available");
    }
#endif
    if (requested == BackendKind::metal && metal_compiled()) {
        throw std::runtime_error(
            "backend 'metal' is provided by the VapourSynth heterogeneous scheduler");
    }
    throw std::runtime_error(std::string{"backend '"} + backend_name(requested)
                             + "' is not compiled in this build");
}

std::vector<BackendCapability> backend_capabilities() {
    const bool metal_device = metal_available();
    const bool vulkan_device = vulkan_available();
    const bool cuda_device = cuda_available();
    return {
        {BackendKind::cpu, "cpu", true, true},
        {BackendKind::metal, "metal", metal_compiled(), metal_device},
        {BackendKind::vulkan, "vulkan", vulkan_compiled(), vulkan_device},
        {BackendKind::cuda, "cuda", cuda_compiled(), cuda_device},
    };
}

bool metal_compiled() noexcept {
#if defined(DSMVC_HAS_METAL)
    return true;
#else
    return false;
#endif
}

bool metal_available() noexcept {
#if defined(DSMVC_HAS_METAL)
    return metal_detail::backend_available();
#else
    return false;
#endif
}

bool vulkan_compiled() noexcept {
#if defined(DSMVC_HAS_VULKAN)
    return true;
#else
    return false;
#endif
}

bool vulkan_available() noexcept {
#if defined(DSMVC_HAS_VULKAN)
    return vulkan_detail::backend_available();
#else
    return false;
#endif
}

bool cuda_compiled() noexcept {
#if defined(DSMVC_HAS_CUDA)
    return true;
#else
    return false;
#endif
}

bool cuda_available() noexcept {
#if defined(DSMVC_HAS_CUDA)
    return cuda_detail::backend_available();
#else
    return false;
#endif
}

} // namespace dsmvc
