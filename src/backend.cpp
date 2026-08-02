#include <dsmvc/engine.hpp>

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
    throw std::runtime_error(std::string{"backend '"} + backend_name(requested)
                             + "' is not compiled in this build");
}

std::vector<BackendCapability> backend_capabilities() {
    return {
        {BackendKind::cpu, "cpu", true, true},
        {BackendKind::metal, "metal", false, false},
        {BackendKind::vulkan, "vulkan", false, false},
        {BackendKind::cuda, "cuda", false, false},
    };
}

} // namespace dsmvc
