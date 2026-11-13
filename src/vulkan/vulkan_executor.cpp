#include <dsmvc/host_blocks.hpp>
#include "vulkan_executor.hpp"
#include "../host_copy.hpp"
#include "../solve_partition.hpp"

#include "vulkan_convert_128_spv.hpp"
#include "vulkan_convert_256_spv.hpp"
#include "vulkan_convert_f64_spv.hpp"
#include "vulkan_inverse_32_spv.hpp"
#include "vulkan_inverse_64_spv.hpp"
#include "vulkan_inverse_b11_32_spv.hpp"
#include "vulkan_inverse_b1_32_spv.hpp"
#include "vulkan_inverse_b3_32_spv.hpp"
#include "vulkan_inverse_b5_32_spv.hpp"
#include "vulkan_inverse_b7_32_spv.hpp"
#include "vulkan_inverse_b9_32_spv.hpp"
#include "vulkan_inverse_f64_spv.hpp"
#include "vulkan_rhs_128_spv.hpp"
#include "vulkan_rhs_256_spv.hpp"
#include "vulkan_rhs_f64_spv.hpp"
#include "vulkan_solve_32_spv.hpp"
#include "vulkan_solve_64_spv.hpp"
#include "vulkan_solve_b11_32_spv.hpp"
#include "vulkan_solve_b1_32_spv.hpp"
#include "vulkan_solve_b3_32_spv.hpp"
#include "vulkan_solve_b5_32_spv.hpp"
#include "vulkan_solve_b7_32_spv.hpp"
#include "vulkan_solve_b9_32_spv.hpp"
#include "vulkan_solve_chunked_b1_c2_spv.hpp"
#include "vulkan_solve_chunked_b1_c4_spv.hpp"
#include "vulkan_solve_chunked_b1_c8_spv.hpp"
#include "vulkan_solve_chunked_b1_c16_spv.hpp"
#include "vulkan_solve_chunked_b1_c32_spv.hpp"
#include "vulkan_solve_chunked_b3_c2_spv.hpp"
#include "vulkan_solve_chunked_b3_c4_spv.hpp"
#include "vulkan_solve_chunked_b3_c8_spv.hpp"
#include "vulkan_solve_chunked_b3_c16_spv.hpp"
#include "vulkan_solve_chunked_b3_c32_spv.hpp"
#include "vulkan_solve_chunked_b5_c2_spv.hpp"
#include "vulkan_solve_chunked_b5_c4_spv.hpp"
#include "vulkan_solve_chunked_b5_c8_spv.hpp"
#include "vulkan_solve_chunked_b5_c16_spv.hpp"
#include "vulkan_solve_chunked_b5_c32_spv.hpp"
#include "vulkan_solve_chunked_b7_c2_spv.hpp"
#include "vulkan_solve_chunked_b7_c4_spv.hpp"
#include "vulkan_solve_chunked_b7_c8_spv.hpp"
#include "vulkan_solve_chunked_b7_c16_spv.hpp"
#include "vulkan_solve_chunked_b7_c32_spv.hpp"
#include "vulkan_solve_f64_spv.hpp"
#include "vulkan_transpose_128_spv.hpp"
#include "vulkan_transpose_256_spv.hpp"
#include "vulkan_transpose_f64_spv.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <map>
#include <utility>
#include <vector>

#ifdef __linux__
#include <chrono>
#include <dirent.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#endif

namespace dsmvc::vulkan_detail {

namespace {

constexpr VkDeviceSize maximum_allocation_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
// One plane's 2D execution records up to seven sets; a frame submits three.
constexpr std::size_t descriptor_set_count = 24U;
constexpr std::size_t frame_plane_limit = 3U;
constexpr std::size_t default_plan_cache_bytes = 16U * 1024U * 1024U;
constexpr std::size_t default_input_cache_bytes = 64U * 1024U * 1024U;
constexpr std::array<std::uint32_t, 6U> fixed_bandwidths{
    1U, 3U, 5U, 7U, 9U, 11U};

[[nodiscard]] std::size_t checked_add(
    std::size_t left, std::size_t right, const char *label) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::length_error(std::string{label} + " exceeds addressable memory");
    }
    return left + right;
}

[[nodiscard]] std::size_t checked_product(
    std::size_t left, std::size_t right, const char *label) {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::length_error(std::string{label} + " exceeds addressable memory");
    }
    return left * right;
}

[[nodiscard]] std::size_t align_up(
    std::size_t value, std::size_t alignment, const char *label) {
    if (alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
        throw std::logic_error("Vulkan alignment is not a power of two");
    }
    return checked_add(value, alignment - 1U, label) & ~(alignment - 1U);
}

[[nodiscard]] std::uint32_t checked_u32(
    std::size_t value, const char *label) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error(std::string{label} + " exceeds Vulkan shader ABI limits");
    }
    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::uint32_t divide_up(
    std::uint32_t value, std::uint32_t divisor) noexcept {
    return value / divisor + (value % divisor != 0U ? 1U : 0U);
}

[[nodiscard]] std::size_t environment_megabytes(
    const char *name, std::size_t fallback) noexcept {
    const char *text = std::getenv(name);
    if (!text || !*text) return fallback;
    std::size_t value = 0U;
    const char *end = text + std::strlen(text);
    const auto parsed = std::from_chars(text, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end
        || value > std::numeric_limits<std::size_t>::max() / (1024U * 1024U)) {
        return fallback;
    }
    return value * 1024U * 1024U;
}

[[nodiscard]] bool environment_flag(const char *name, bool fallback) noexcept {
    const char *text = std::getenv(name);
    if (!text || !*text) return fallback;
    if (std::strcmp(text, "0") == 0 || std::strcmp(text, "false") == 0
        || std::strcmp(text, "off") == 0) {
        return false;
    }
    if (std::strcmp(text, "1") == 0 || std::strcmp(text, "true") == 0
        || std::strcmp(text, "on") == 0) {
        return true;
    }
    return fallback;
}

enum class SplitRhsMode : std::uint8_t {
    disabled,
    adaptive,
    forced,
};

[[nodiscard]] SplitRhsMode split_rhs_mode() noexcept {
    const char *text = std::getenv("DSMVC_VULKAN_SPLIT_RHS");
    if (!text || !*text || std::strcmp(text, "adaptive") == 0) {
        return SplitRhsMode::adaptive;
    }
    if (std::strcmp(text, "force") == 0) return SplitRhsMode::forced;
    if (std::strcmp(text, "0") == 0) return SplitRhsMode::disabled;
    return SplitRhsMode::adaptive;
}

struct SlotConfiguration {
    std::size_t limited = 4U;
    std::size_t total = 8U;
    bool adaptive = true;
};

[[nodiscard]] SlotConfiguration slot_configuration() noexcept {
    const char *text = std::getenv("DSMVC_VULKAN_SLOTS");
    if (!text || !*text) return {};
    unsigned int value = 0U;
    const char *end = text + std::strlen(text);
    const auto parsed = std::from_chars(text, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end
        || value < 1U || value > 16U) {
        return {};
    }
    return {value, value, false};
}

[[nodiscard]] const char *result_name(VkResult result) noexcept {
    switch (result) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    default: return "unknown VkResult";
    }
}

void check_vk(VkResult result, const char *label) {
    if (result == VK_SUCCESS) return;
    throw std::runtime_error(
        std::string{label} + " failed with " + result_name(result)
        + " (" + std::to_string(static_cast<int>(result)) + ")");
}

[[nodiscard]] bool has_instance_extension(const char *name) {
    std::uint32_t count = 0U;
    check_vk(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr),
             "vkEnumerateInstanceExtensionProperties");
    std::vector<VkExtensionProperties> extensions(count);
    check_vk(vkEnumerateInstanceExtensionProperties(
                 nullptr, &count, extensions.data()),
             "vkEnumerateInstanceExtensionProperties");
    return std::ranges::any_of(extensions, [name](const auto &extension) {
        return std::strcmp(extension.extensionName, name) == 0;
    });
}

[[nodiscard]] bool has_instance_layer(const char *name) {
    std::uint32_t count = 0U;
    check_vk(vkEnumerateInstanceLayerProperties(&count, nullptr),
             "vkEnumerateInstanceLayerProperties");
    std::vector<VkLayerProperties> layers(count);
    check_vk(vkEnumerateInstanceLayerProperties(&count, layers.data()),
             "vkEnumerateInstanceLayerProperties");
    return std::ranges::any_of(layers, [name](const auto &layer) {
        return std::strcmp(layer.layerName, name) == 0;
    });
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT *data,
    void *) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::fprintf(stderr, "dsmvc Vulkan validation: %s\n",
                     data && data->pMessage ? data->pMessage : "unknown message");
    }
    return VK_FALSE;
}

struct InstanceHandle {
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    PFN_vkDestroyDebugUtilsMessengerEXT destroy_messenger = nullptr;
    bool debug_utils = false;
    bool owned = true; // false: the instance belongs to a VapourSynth core

    InstanceHandle() = default;
    InstanceHandle(const InstanceHandle &) = delete;
    InstanceHandle &operator=(const InstanceHandle &) = delete;
    InstanceHandle(InstanceHandle &&other) noexcept
        : instance(std::exchange(other.instance, VK_NULL_HANDLE)),
          messenger(std::exchange(other.messenger, VK_NULL_HANDLE)),
          destroy_messenger(std::exchange(other.destroy_messenger, nullptr)),
          debug_utils(other.debug_utils), owned(other.owned) {}
    InstanceHandle &operator=(InstanceHandle &&other) noexcept {
        if (this == &other) return *this;
        reset();
        instance = std::exchange(other.instance, VK_NULL_HANDLE);
        messenger = std::exchange(other.messenger, VK_NULL_HANDLE);
        destroy_messenger = std::exchange(other.destroy_messenger, nullptr);
        debug_utils = other.debug_utils;
        owned = other.owned;
        return *this;
    }
    ~InstanceHandle() { reset(); }

    void reset() noexcept {
        if (messenger && destroy_messenger) {
            destroy_messenger(instance, messenger, nullptr);
        }
        messenger = VK_NULL_HANDLE;
        if (instance && owned) vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
};

// The threads of this process, by id.
[[nodiscard]] std::vector<long> thread_ids() {
    std::vector<long> ids;
#ifdef __linux__
    if (DIR *tasks = opendir("/proc/self/task")) {
        while (const dirent *entry = readdir(tasks)) {
            if (entry->d_name[0] != '.') ids.push_back(std::atol(entry->d_name));
        }
        closedir(tasks);
    }
    std::ranges::sort(ids);
#endif
    return ids;
}

// The threads that existed before dsmvc first called into the Vulkan loader.
[[nodiscard]] const std::vector<long> &threads_before_vulkan() {
    static const std::vector<long> ids = thread_ids();
    return ids;
}

struct ThreadActivity {
    unsigned long long cpu = 0U; // ticks used so far
    bool running = false;        // any of them in state R
};

[[nodiscard]] ThreadActivity thread_activity(const std::vector<long> &ids) {
    ThreadActivity result;
#ifdef __linux__
    for (const long id : ids) {
        char path[48];
        std::snprintf(path, sizeof(path), "/proc/self/task/%ld/stat", id);
        std::FILE *file = std::fopen(path, "r");
        if (!file) continue; // the thread has exited
        char text[512];
        const std::size_t length = std::fread(text, 1U, sizeof(text) - 1U, file);
        std::fclose(file);
        text[length] = '\0';
        // The state (field 3) and utime, stime (14, 15) follow the name.
        const char *fields = std::strrchr(text, ')');
        char state = '\0';
        unsigned long long user = 0U;
        unsigned long long system = 0U;
        if (fields && std::sscanf(fields + 2,
                "%c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %llu %llu",
                &state, &user, &system) == 3) {
            result.cpu += user + system;
            result.running = result.running || state == 'R';
        }
    }
#else
    (void)ids;
#endif
    return result;
}

// NVIDIA's driver (seen with 615.71.09) compiles shaders on its own threads.
// When its shader cache holds an incomplete entry for a pipeline (left by a
// process that ended during a compile), it compiles that pipeline again about
// three seconds into every process, for several seconds, and stores the
// result only when the compile finishes. A process that exits during that
// compile crashes or hangs in the driver (libnvidia-gpucomp): exit handlers
// that run before dsmvc's destroy state the compiler still uses, and the
// entry stays incomplete for the next process. Waiting for the compile before
// the process exits avoids the crash and completes the entry.
//
// `threads` are the driver's (created while the owned runtime was set up).
// Returns at once (after three samples 1 ms apart) when none of them runs;
// otherwise waits until they have used no CPU for 100 ms, at most 20 s.
// DSMVC_VULKAN_TEARDOWN_WAIT=0 skips the wait; DSMVC_VULKAN_TEARDOWN_TRACE=1
// reports waits.
void wait_for_driver_threads(const std::vector<long> &threads, const char *when) noexcept {
#ifdef __linux__
    if (threads.empty() || !environment_flag("DSMVC_VULKAN_TEARDOWN_WAIT", true)) return;
    using namespace std::chrono;
    const auto start = steady_clock::now();
    ThreadActivity last = thread_activity(threads);
    bool busy = last.running;
    for (int sample = 0; sample < 2 && !busy; ++sample) {
        std::this_thread::sleep_for(milliseconds(1));
        const ThreadActivity now = thread_activity(threads);
        busy = now.running || now.cpu != last.cpu;
        last = now;
    }
    if (busy) {
        int quiet = 0;
        while (quiet < 5 && steady_clock::now() - start < seconds(20)) {
            std::this_thread::sleep_for(milliseconds(20));
            const ThreadActivity now = thread_activity(threads);
            quiet = now.cpu == last.cpu && !now.running ? quiet + 1 : 0;
            last = now;
        }
    }
    if (environment_flag("DSMVC_VULKAN_TEARDOWN_TRACE", false)) {
        std::fprintf(stderr, "dsmvc: Vulkan %s waited %.1f ms for %zu driver threads\n",
                     when, duration<double, std::milli>(steady_clock::now() - start).count(),
                     threads.size());
    }
#else
    (void)threads;
    (void)when;
#endif
}

[[nodiscard]] InstanceHandle create_instance(bool validation) {
    (void)threads_before_vulkan();
    InstanceHandle result;
    result.debug_utils = has_instance_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    std::vector<const char *> extensions;
    if (result.debug_utils) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    std::vector<const char *> layers;
    if (validation) {
        constexpr const char *validation_layer = "VK_LAYER_KHRONOS_validation";
        if (!has_instance_layer(validation_layer)) {
            throw std::runtime_error(
                "DSMVC_VULKAN_VALIDATION=1 requires VK_LAYER_KHRONOS_validation");
        }
        layers.push_back(validation_layer);
    }

    const VkApplicationInfo application{
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        nullptr,
        "dsmvc",
        VK_MAKE_API_VERSION(0, 0, 1, 0),
        "dsmvc Vulkan backend",
        VK_MAKE_API_VERSION(0, 0, 1, 0),
        VK_API_VERSION_1_2,
    };
    VkDebugUtilsMessengerCreateInfoEXT debug_info{
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        nullptr,
        0U,
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        debug_callback,
        nullptr,
    };
    const VkInstanceCreateInfo create_info{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        validation && result.debug_utils ? &debug_info : nullptr,
        0U,
        &application,
        static_cast<std::uint32_t>(layers.size()),
        layers.data(),
        static_cast<std::uint32_t>(extensions.size()),
        extensions.data(),
    };
    check_vk(vkCreateInstance(&create_info, nullptr, &result.instance),
             "vkCreateInstance(Vulkan 1.2)");

    if (validation && result.debug_utils) {
        const auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(result.instance, "vkCreateDebugUtilsMessengerEXT"));
        result.destroy_messenger =
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(
                    result.instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (create && result.destroy_messenger) {
            check_vk(create(result.instance, &debug_info, nullptr, &result.messenger),
                     "vkCreateDebugUtilsMessengerEXT");
        }
    }
    return result;
}

[[nodiscard]] bool device_has_extension(
    VkPhysicalDevice physical, const char *name) {
    std::uint32_t count = 0U;
    if (vkEnumerateDeviceExtensionProperties(
            physical, nullptr, &count, nullptr) != VK_SUCCESS) {
        return false;
    }
    std::vector<VkExtensionProperties> properties(count);
    if (count != 0U
        && vkEnumerateDeviceExtensionProperties(
               physical, nullptr, &count, properties.data()) != VK_SUCCESS) {
        return false;
    }
    return std::any_of(
        properties.begin(), properties.begin() + count,
        [name](const VkExtensionProperties &entry) {
            return std::string_view{entry.extensionName} == name;
        });
}

struct DeviceInfo {
    std::uint32_t index = 0U;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memory{};
    VulkanFloat64Capabilities float64{};
    std::uint32_t queue_family = std::numeric_limits<std::uint32_t>::max();
    bool dedicated_compute = false;
    bool eligible = false;
    std::string reason;
};

[[nodiscard]] DeviceInfo describe_device(VkPhysicalDevice physical, std::uint32_t index) {
    DeviceInfo info;
    info.index = index;
    info.physical = physical;
    VkPhysicalDeviceFloatControlsProperties float_controls{};
    float_controls.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &float_controls;
    vkGetPhysicalDeviceProperties2(info.physical, &properties);
    info.properties = properties.properties;
    vkGetPhysicalDeviceMemoryProperties(info.physical, &info.memory);
    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    vkGetPhysicalDeviceFeatures2(info.physical, &features);
    info.float64 = {
        features.features.shaderFloat64 == VK_TRUE,
        float_controls.shaderRoundingModeRTEFloat64 == VK_TRUE,
        float_controls.shaderSignedZeroInfNanPreserveFloat64 == VK_TRUE,
        float_controls.shaderDenormPreserveFloat64 == VK_TRUE,
        info.properties.limits.minStorageBufferOffsetAlignment,
        info.properties.limits.maxStorageBufferRange,
    };

    std::uint32_t queue_count = 0U;
    vkGetPhysicalDeviceQueueFamilyProperties(
        info.physical, &queue_count, nullptr);
    std::vector<VkQueueFamilyProperties> queues(queue_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        info.physical, &queue_count, queues.data());
    for (std::uint32_t family = 0U; family < queue_count; ++family) {
        if (queues[family].queueCount == 0U
            || !(queues[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            continue;
        }
        const bool dedicated = !(queues[family].queueFlags & VK_QUEUE_GRAPHICS_BIT);
        if (info.queue_family == std::numeric_limits<std::uint32_t>::max()
            || (dedicated && !info.dedicated_compute)) {
            info.queue_family = family;
            info.dedicated_compute = dedicated;
        }
    }

    const auto &limits = info.properties.limits;
    if (info.properties.apiVersion < VK_API_VERSION_1_2) {
        info.reason = "requires Vulkan 1.2";
    } else if (info.queue_family == std::numeric_limits<std::uint32_t>::max()) {
        info.reason = "no compute queue";
    } else if (limits.maxComputeWorkGroupInvocations < 128U
               || limits.maxComputeWorkGroupSize[0] < 128U
               || limits.maxComputeWorkGroupSize[1] < 8U
               || limits.maxPerStageDescriptorStorageBuffers < 3U
               || limits.maxDescriptorSetStorageBuffers < 3U
               || limits.maxPushConstantsSize < 128U
               || limits.maxStorageBufferRange < (128U * 1024U * 1024U)) {
        info.reason = "does not meet Vulkan core compute limits";
    } else {
        info.eligible = true;
        info.reason = "eligible";
    }
    return info;
}

[[nodiscard]] std::vector<DeviceInfo> enumerate_devices(VkInstance instance) {
    std::uint32_t count = 0U;
    check_vk(vkEnumeratePhysicalDevices(instance, &count, nullptr),
             "vkEnumeratePhysicalDevices");
    std::vector<VkPhysicalDevice> physical_devices(count);
    if (count != 0U) {
        check_vk(vkEnumeratePhysicalDevices(
                     instance, &count, physical_devices.data()),
                 "vkEnumeratePhysicalDevices");
    }

    std::vector<DeviceInfo> result;
    result.reserve(count);
    for (std::uint32_t index = 0U; index < count; ++index) {
        result.push_back(describe_device(physical_devices[index], index));
    }
    return result;
}

[[nodiscard]] int device_score(const DeviceInfo &device) noexcept {
    int score = 0;
    switch (device.properties.deviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: score = 400; break;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score = 300; break;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: score = 200; break;
    case VK_PHYSICAL_DEVICE_TYPE_CPU: score = 100; break;
    default: score = 0; break;
    }
    if (device.dedicated_compute) score += 10;
    return score;
}

[[nodiscard]] std::string version_string(std::uint32_t version) {
    return std::to_string(VK_API_VERSION_MAJOR(version)) + "."
        + std::to_string(VK_API_VERSION_MINOR(version)) + "."
        + std::to_string(VK_API_VERSION_PATCH(version));
}

[[nodiscard]] std::string device_inventory(
    const std::vector<DeviceInfo> &devices) {
    std::ostringstream message;
    message << "Detected Vulkan devices:";
    if (devices.empty()) message << " none";
    for (const auto &device : devices) {
        message << "\n[" << device.index << "] "
                << device.properties.deviceName << " ("
                << std::hex << std::setfill('0') << std::setw(4)
                << device.properties.vendorID << ':' << std::setw(4)
                << device.properties.deviceID << std::dec << ", Vulkan "
                << version_string(device.properties.apiVersion) << ", "
                << device.reason << ')';
    }
    return message.str();
}

[[nodiscard]] std::string float64_capability_report(const DeviceInfo &device) {
    const auto &capabilities = device.float64;
    std::ostringstream message;
    message << "device=" << device.properties.deviceName
            << " vendor_id=0x" << std::hex << device.properties.vendorID
            << " device_id=0x" << device.properties.deviceID << std::dec
            << " api=" << version_string(device.properties.apiVersion)
            << " driver=" << version_string(device.properties.driverVersion)
            << " driver_raw=0x" << std::hex
            << device.properties.driverVersion << std::dec
            << " shaderFloat64=" << capabilities.shader_float64
            << " shaderRoundingModeRTEFloat64="
            << capabilities.rounding_mode_rte_float64
            << " shaderSignedZeroInfNanPreserveFloat64="
            << capabilities.signed_zero_inf_nan_preserve_float64
            << " shaderDenormPreserveFloat64="
            << capabilities.denorm_preserve_float64
            << " denormFloat64Policy=probe-only"
            << " denormFloat64Boundary=\"subnormal intermediates may "
               "flush to zero; normal-range values unaffected\""
            << " minStorageBufferOffsetAlignment="
            << capabilities.min_storage_buffer_offset_alignment
            << " maxStorageBufferRange="
            << capabilities.max_storage_buffer_range
            << " strictFloat64=" << capabilities.strict_supported();
    if (!capabilities.strict_supported()) {
        message << " missing=\"" << capabilities.missing_requirements() << '"';
    }
    return message.str();
}

[[nodiscard]] bool parse_unsigned(
    std::string_view text, int base, std::uint32_t &value) noexcept {
    if (base == 16 && text.size() > 2U && text[0] == '0'
        && (text[1] == 'x' || text[1] == 'X')) {
        text.remove_prefix(2U);
    }
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), value, base);
    return parsed.ec == std::errc{}
        && parsed.ptr == text.data() + text.size();
}

[[nodiscard]] DeviceInfo select_device(
    const std::vector<DeviceInfo> &devices) {
    const char *requested_text = std::getenv("DSMVC_VULKAN_DEVICE");
    if (requested_text && *requested_text) {
        const std::string_view requested{requested_text};
        const auto separator = requested.find(':');
        const DeviceInfo *selected = nullptr;
        if (separator == std::string_view::npos) {
            std::uint32_t index = 0U;
            if (parse_unsigned(requested, 10, index) && index < devices.size()) {
                selected = &devices[index];
            }
        } else if (separator != 0U && separator + 1U < requested.size()
                   && requested.find(':', separator + 1U) == std::string_view::npos) {
            std::uint32_t vendor = 0U;
            std::uint32_t device = 0U;
            if (parse_unsigned(requested.substr(0U, separator), 16, vendor)
                && parse_unsigned(requested.substr(separator + 1U), 16, device)) {
                for (const auto &candidate : devices) {
                    if (candidate.properties.vendorID == vendor
                        && candidate.properties.deviceID == device
                        && (!selected
                            || device_score(candidate) > device_score(*selected))) {
                        selected = &candidate;
                    }
                }
            }
        }
        if (!selected || !selected->eligible) {
            throw std::runtime_error(
                std::string{"DSMVC_VULKAN_DEVICE='"} + requested_text
                + "' did not select an eligible Vulkan device. "
                + device_inventory(devices));
        }
        return *selected;
    }

    const DeviceInfo *selected = nullptr;
    for (const auto &candidate : devices) {
        if (candidate.eligible
            && (!selected || device_score(candidate) > device_score(*selected))) {
            selected = &candidate;
        }
    }
    if (!selected) {
        throw std::runtime_error(
            "no eligible Vulkan 1.2 compute device is available. "
            + device_inventory(devices));
    }
    return *selected;
}

class Runtime;

template <class Handle>
[[nodiscard]] std::uint64_t object_handle(Handle handle) noexcept {
    if constexpr (std::is_pointer_v<Handle>) {
        return reinterpret_cast<std::uint64_t>(handle);
    } else {
        return static_cast<std::uint64_t>(handle);
    }
}

class Buffer {
public:
    Buffer() = default;
    Buffer(Runtime &runtime, VkDeviceSize size, VkBufferUsageFlags usage,
           VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred,
           const char *name);
    ~Buffer() { reset(); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer(Buffer &&other) noexcept { swap(other); }
    Buffer &operator=(Buffer &&other) noexcept {
        if (this != &other) {
            reset();
            swap(other);
        }
        return *this;
    }

    void reset() noexcept;
    void flush(VkDeviceSize offset, VkDeviceSize size) const;
    void invalidate(VkDeviceSize offset, VkDeviceSize size) const;

    [[nodiscard]] VkBuffer handle() const noexcept { return buffer_; }
    [[nodiscard]] VkDeviceSize size() const noexcept { return size_; }
    [[nodiscard]] void *mapped() const noexcept { return mapped_; }
    [[nodiscard]] bool coherent() const noexcept { return coherent_; }

private:
    void swap(Buffer &other) noexcept {
        std::swap(runtime_, other.runtime_);
        std::swap(buffer_, other.buffer_);
        std::swap(memory_, other.memory_);
        std::swap(size_, other.size_);
        std::swap(allocation_size_, other.allocation_size_);
        std::swap(mapped_, other.mapped_);
        std::swap(coherent_, other.coherent_);
    }

    Runtime *runtime_ = nullptr;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0U;
    VkDeviceSize allocation_size_ = 0U;
    void *mapped_ = nullptr;
    bool coherent_ = false;
};

class DeviceArena;
class InputCache;
class PlanUploadPool;
class StagingPool;
struct ExecutionSlot;

// An external timeline pair a submission waits on (a GPU frame's producer).
struct ExternalWait {
    VkSemaphore semaphore = VK_NULL_HANDLE;
    std::uint64_t value = 0U;
};

class Runtime : public std::enable_shared_from_this<Runtime> {
public:
    Runtime();
    // Adopts the device of a VapourSynth core instead of creating one.
    explicit Runtime(const VulkanCoreDevice &borrowed);
    ~Runtime();
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    void check(VkResult result, const char *label);
    void throw_if_failed() const;
    [[nodiscard]] bool failed() const noexcept {
        return failed_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::uint32_t memory_type(
        std::uint32_t bits, VkMemoryPropertyFlags required,
        VkMemoryPropertyFlags preferred) const;
    void name_object(std::uint64_t object, VkObjectType type, const char *name) const noexcept;
    void begin_label(VkCommandBuffer command, const char *name) const noexcept;
    void end_label(VkCommandBuffer command) const noexcept;
    [[nodiscard]] VkPipeline create_pipeline(
        const std::uint32_t *code, std::size_t bytes, const char *name);
    // Partitioned-solve pipeline for a bandwidth of 1, 3, 5 or 7 and a power
    // of two chunk count; created on first use.
    [[nodiscard]] VkPipeline chunked_solve_pipeline(
        std::uint32_t bandwidth, std::uint32_t chunks);
    [[nodiscard]] std::uint64_t submit(
        VkCommandBuffer command, VkFence fence, std::uint64_t wait_value,
        bool signal_timeline, const ExternalWait *externals = nullptr,
        std::size_t external_count = 0U);
    void wait_fence(VkFence fence, const char *label);

    [[nodiscard]] std::size_t acquire_slot(bool heavy, bool host_limited);
    void release_slot(std::size_t index, bool heavy) noexcept;
    void attach_executor();
    void detach_executor() noexcept;

    InstanceHandle instance;
    DeviceInfo selected;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    VkPipeline transpose_pipeline = VK_NULL_HANDLE;
    VkPipeline inverse_pipeline = VK_NULL_HANDLE;
    VkPipeline rhs_pipeline = VK_NULL_HANDLE;
    VkPipeline solve_pipeline = VK_NULL_HANDLE;
    std::array<VkPipeline, fixed_bandwidths.size()> fixed_inverse_pipelines{};
    std::array<VkPipeline, fixed_bandwidths.size()> fixed_solve_pipelines{};
    std::mutex chunked_mutex;
    // [bandwidth 1/3/5/7][log2(chunks) - 1]
    std::array<std::array<VkPipeline, 5U>, 4U> chunked_pipelines{};
    VkPipeline convert_pipeline = VK_NULL_HANDLE;
    VkPipeline transpose_f64_pipeline = VK_NULL_HANDLE;
    VkPipeline inverse_f64_pipeline = VK_NULL_HANDLE;
    VkPipeline rhs_f64_pipeline = VK_NULL_HANDLE;
    VkPipeline solve_f64_pipeline = VK_NULL_HANDLE;
    VkPipeline convert_f64_pipeline = VK_NULL_HANDLE;
    VkSemaphore timeline = VK_NULL_HANDLE;
    // VK_EXT_external_memory_host (opt-in, DSMVC_VULKAN_DIRECT_IO=1): classic
    // frame pool blocks are imported as buffers so rows are copied straight
    // from and to the frames.
    bool host_import_enabled = false;
    VkDeviceSize host_import_alignment = 0U;
    std::uint32_t host_import_limit = 0U;
    std::atomic<std::uint32_t> host_imports{0U};
    PFN_vkGetMemoryHostPointerPropertiesEXT get_host_pointer_properties = nullptr;
    PFN_vkSetDebugUtilsObjectNameEXT set_object_name = nullptr;
    PFN_vkCmdBeginDebugUtilsLabelEXT command_begin_label = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT command_end_label = nullptr;
    std::uint32_t transpose_local_x = 32U;
    std::uint32_t transpose_local_y = 8U;
    std::uint32_t rhs_local_x = 32U;
    std::uint32_t rhs_local_y = 8U;
    std::uint32_t inverse_local_size = 32U;
    std::uint32_t convert_local_size = 256U;
    bool timeline_enabled = false;
    bool force_non_coherent = false;
    // DSMVC_VULKAN_FOLD_TRANSPOSE=0 restores the separate transpose passes of
    // the 2D path (see execute_2d) and the right-hand side's walk by vectors.
    const bool fold_transposes =
        environment_flag("DSMVC_VULKAN_FOLD_TRANSPOSE", true);
    // DSMVC_VULKAN_FRAME_SUBMIT=0 submits the planes of a frame one by one
    // instead of recording them into one command buffer.
    const bool frame_submit =
        environment_flag("DSMVC_VULKAN_FRAME_SUBMIT", true);
    SplitRhsMode split_rhs = SplitRhsMode::adaptive;
    std::unique_ptr<DeviceArena> plan_arena;
    std::unique_ptr<DeviceArena> input_arena;
    std::unique_ptr<InputCache> input_cache;
    std::unique_ptr<PlanUploadPool> plan_upload_pool;
    std::unique_ptr<StagingPool> staging_pool;
    std::vector<std::unique_ptr<ExecutionSlot>> slots;
    std::vector<bool> slot_busy;

    [[nodiscard]] bool borrowed() const noexcept { return !owns_device_; }

private:
    void destroy_device_objects() noexcept;
    void finish_construction();

    bool owns_device_ = true;
    std::vector<long> driver_threads_; // see wait_for_driver_threads
    // A submission was made since the device was last known idle.
    std::atomic<bool> submitted_since_idle_{false};
    std::uint32_t queue_index_ = 0U;
    void (*lock_queue_)(void *) = nullptr;
    void (*unlock_queue_)(void *) = nullptr;
    void *queue_lock_user_ = nullptr;

    mutable std::mutex failure_mutex_;
    std::string failure_message_;
    std::atomic<bool> failed_{false};
    std::mutex queue_mutex_;
    std::uint64_t next_timeline_value_ = 0U;
    std::mutex slot_mutex_;
    std::condition_variable slot_available_;
    std::size_t maximum_slot_count_ = 8U;
    std::size_t limited_slot_count_ = 4U;
    std::size_t limited_slot_busy_ = 0U;
    bool adaptive_slots_ = true;
    std::mutex executor_mutex_;
    std::size_t executor_count_ = 0U;
};

// A page-aligned block of host memory (a classic frame pool block) imported as
// a buffer with VK_EXT_external_memory_host. Importing is optional: any failure
// throws and the caller stages the frames instead, without poisoning the
// runtime.
class ImportedBlock {
public:
    ImportedBlock(Runtime &runtime, void *base, std::size_t size)
        : runtime_(&runtime) {
        if (!runtime.host_import_enabled) {
            throw std::runtime_error("Vulkan host import is not enabled");
        }
        const VkDeviceSize alignment = runtime.host_import_alignment;
        if (reinterpret_cast<std::uintptr_t>(base) % alignment != 0U
            || size % alignment != 0U || size > maximum_allocation_bytes) {
            throw std::runtime_error(
                "host block does not satisfy the Vulkan import alignment");
        }
        if (runtime.host_imports.load() >= runtime.host_import_limit) {
            throw std::runtime_error("too many imported host blocks");
        }
        try {
            create(base, size);
        } catch (...) {
            destroy();
            throw;
        }
        ++runtime.host_imports;
    }

    ~ImportedBlock() {
        if (buffer_ || memory_) {
            destroy();
            --runtime_->host_imports;
        }
    }
    ImportedBlock(const ImportedBlock &) = delete;
    ImportedBlock &operator=(const ImportedBlock &) = delete;

    [[nodiscard]] VkBuffer handle() const noexcept { return buffer_; }

private:
    static void require(VkResult result, const char *what) {
        if (result != VK_SUCCESS) {
            throw std::runtime_error(
                std::string{"Vulkan host import failed in "} + what);
        }
    }

    void create(void *base, std::size_t size) {
        VkDevice device = runtime_->device;
        VkMemoryHostPointerPropertiesEXT pointer_properties{
            VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT, nullptr, 0U};
        require(
            runtime_->get_host_pointer_properties(
                device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                base, &pointer_properties),
            "vkGetMemoryHostPointerPropertiesEXT");
        const VkExternalMemoryBufferCreateInfo external{
            VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, nullptr,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
        const VkBufferCreateInfo buffer_info{
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external, 0U, size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_SHARING_MODE_EXCLUSIVE, 0U, nullptr};
        require(vkCreateBuffer(device, &buffer_info, nullptr, &buffer_),
                "vkCreateBuffer");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, buffer_, &requirements);
        if (requirements.size > size) {
            throw std::runtime_error("imported buffer needs more memory than the block");
        }
        // Only host-coherent memory: no flush or invalidate is needed (or
        // possible) for memory the application owns.
        const std::uint32_t bits =
            requirements.memoryTypeBits & pointer_properties.memoryTypeBits;
        constexpr VkMemoryPropertyFlags wanted =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        std::uint32_t type = std::numeric_limits<std::uint32_t>::max();
        for (std::uint32_t index = 0U;
             index < runtime_->selected.memory.memoryTypeCount; ++index) {
            if (((bits >> index) & 1U) != 0U
                && (runtime_->selected.memory.memoryTypes[index].propertyFlags
                    & wanted) == wanted) {
                type = index;
                break;
            }
        }
        if (type == std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("no coherent memory type for the host block");
        }
        const VkImportMemoryHostPointerInfoEXT import{
            VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, nullptr,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, base};
        const VkMemoryAllocateInfo allocate{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &import, size, type};
        require(vkAllocateMemory(device, &allocate, nullptr, &memory_),
                "vkAllocateMemory");
        require(vkBindBufferMemory(device, buffer_, memory_, 0U),
                "vkBindBufferMemory");
    }

    void destroy() noexcept {
        if (buffer_) vkDestroyBuffer(runtime_->device, buffer_, nullptr);
        if (memory_) vkFreeMemory(runtime_->device, memory_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        memory_ = VK_NULL_HANDLE;
    }

    Runtime *runtime_ = nullptr;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
};

struct BlockRegistrar {
    static inline Runtime *runtime = nullptr;

    static host_blocks::Registered register_block(
        void *base, std::size_t size) noexcept {
        try {
            return {host_blocks::Outcome::owned,
                    std::make_shared<ImportedBlock>(*runtime, base, size)};
        } catch (...) {
            return {};
        }
    }

    // The import is destroyed when the registry drops its reference and the
    // last lease on it ends.
    static void unregister_block(
        void *, std::size_t, const std::shared_ptr<void> &) noexcept {}
};

// Bit 0: frames are copied from imported blocks; bit 1: into them.
[[nodiscard]] unsigned direct_sides() noexcept {
    static const unsigned sides = [] {
        const char *text = std::getenv("DSMVC_VULKAN_DIRECT_IO_SIDES");
        return text ? static_cast<unsigned>(std::atoi(text)) & 3U : 3U;
    }();
    return sides;
}

// A plane range inside an imported block.
struct HostRange {
    std::shared_ptr<ImportedBlock> block; // a host block imported as a buffer, or
    VkBuffer raw = VK_NULL_HANDLE;        // the buffer of a GPU frame's plane
    VkDeviceSize offset = 0U;
    explicit operator bool() const noexcept {
        return block != nullptr || raw != VK_NULL_HANDLE;
    }
    [[nodiscard]] VkBuffer buffer() const noexcept {
        return raw != VK_NULL_HANDLE ? raw : block->handle();
    }
};

// The planes of a GPU frame pair an execution reads and writes in place.
struct ResidentIo {
    const GpuPlane *source = nullptr;
    const GpuPlane *result = nullptr;
};

[[nodiscard]] HostRange resident_range(const GpuPlane &plane) noexcept {
    HostRange range;
    range.raw = reinterpret_cast<VkBuffer>(plane.buffer);
    return range;
}

[[nodiscard]] ExternalWait resident_wait(const ResidentIo *resident) noexcept {
    ExternalWait wait;
    if (resident && resident->source->ready_semaphore != 0U) {
        wait.semaphore =
            reinterpret_cast<VkSemaphore>(resident->source->ready_semaphore);
        wait.value = resident->source->ready_value;
    }
    return wait;
}

[[nodiscard]] HostRange host_range(
    unsigned side, const void *pointer, std::size_t pitch,
    std::size_t row_bytes, std::size_t rows) noexcept {
    if ((direct_sides() & side) == 0U) return {};
    auto lease = host_blocks::acquire(
        host_blocks::Domain::vulkan, pointer,
        host_blocks::plane_span(pitch, row_bytes, rows));
    if (!lease) return {};
    HostRange range;
    range.block = std::static_pointer_cast<ImportedBlock>(std::move(lease.handle));
    range.offset = lease.offset;
    return range;
}

// Copies `rows` rows of `row_bytes` between buffers whose rows are
// `source_pitch` and `destination_pitch` bytes apart.
void record_row_copies(
    VkCommandBuffer command, VkBuffer source, VkDeviceSize source_offset,
    std::size_t source_pitch, VkBuffer destination,
    VkDeviceSize destination_offset, std::size_t destination_pitch,
    std::size_t row_bytes, std::size_t rows) {
    if (source_pitch == row_bytes && destination_pitch == row_bytes) {
        const VkBufferCopy copy{
            source_offset, destination_offset, row_bytes * rows};
        vkCmdCopyBuffer(command, source, destination, 1U, &copy);
        return;
    }
    std::vector<VkBufferCopy> copies(rows);
    for (std::size_t row = 0U; row < rows; ++row) {
        copies[row] = {
            source_offset + row * source_pitch,
            destination_offset + row * destination_pitch, row_bytes};
    }
    vkCmdCopyBuffer(
        command, source, destination,
        static_cast<std::uint32_t>(copies.size()), copies.data());
}

Buffer::Buffer(
    Runtime &runtime, VkDeviceSize size, VkBufferUsageFlags usage,
    VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred,
    const char *name)
    : runtime_(&runtime), size_(size) {
    if (size == 0U || size > maximum_allocation_bytes) {
        throw std::length_error("Vulkan buffer exceeds the 2 GiB allocation guard");
    }
    const VkBufferCreateInfo buffer_info{
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        nullptr,
        0U,
        size,
        usage,
        VK_SHARING_MODE_EXCLUSIVE,
        0U,
        nullptr,
    };
    runtime.check(vkCreateBuffer(
                      runtime.device, &buffer_info, nullptr, &buffer_),
                  "vkCreateBuffer");
    try {
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(runtime.device, buffer_, &requirements);
        allocation_size_ = requirements.size;
        const std::uint32_t type = runtime.memory_type(
            requirements.memoryTypeBits, required, preferred);
        const VkMemoryAllocateInfo allocation_info{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            nullptr,
            requirements.size,
            type,
        };
        runtime.check(vkAllocateMemory(
                          runtime.device, &allocation_info, nullptr, &memory_),
                      "vkAllocateMemory");
        runtime.check(vkBindBufferMemory(runtime.device, buffer_, memory_, 0U),
                      "vkBindBufferMemory");
        const auto flags = runtime.selected.memory.memoryTypes[type].propertyFlags;
        coherent_ = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0U
            && !runtime.force_non_coherent;
        if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            runtime.check(vkMapMemory(
                              runtime.device, memory_, 0U, requirements.size,
                              0U, &mapped_),
                          "vkMapMemory");
        }
        runtime.name_object(
            object_handle(buffer_), VK_OBJECT_TYPE_BUFFER, name);
    } catch (...) {
        reset();
        throw;
    }
}

void Buffer::reset() noexcept {
    if (!runtime_) return;
    if (mapped_ && memory_) vkUnmapMemory(runtime_->device, memory_);
    mapped_ = nullptr;
    if (buffer_) vkDestroyBuffer(runtime_->device, buffer_, nullptr);
    if (memory_) vkFreeMemory(runtime_->device, memory_, nullptr);
    buffer_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
    size_ = 0U;
    allocation_size_ = 0U;
    runtime_ = nullptr;
}

void Buffer::flush(VkDeviceSize offset, VkDeviceSize size) const {
    if (coherent_ || size == 0U) return;
    const VkDeviceSize atom = runtime_->selected.properties.limits.nonCoherentAtomSize;
    const VkDeviceSize begin = offset & ~(atom - 1U);
    const VkDeviceSize requested_end = std::min(offset + size, allocation_size_);
    const VkDeviceSize end = std::min(
        (requested_end + atom - 1U) & ~(atom - 1U), allocation_size_);
    const VkMappedMemoryRange range{
        VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, memory_, begin, end - begin};
    runtime_->check(vkFlushMappedMemoryRanges(
                        runtime_->device, 1U, &range),
                    "vkFlushMappedMemoryRanges");
}

void Buffer::invalidate(VkDeviceSize offset, VkDeviceSize size) const {
    if (coherent_ || size == 0U) return;
    const VkDeviceSize atom = runtime_->selected.properties.limits.nonCoherentAtomSize;
    const VkDeviceSize begin = offset & ~(atom - 1U);
    const VkDeviceSize requested_end = std::min(offset + size, allocation_size_);
    const VkDeviceSize end = std::min(
        (requested_end + atom - 1U) & ~(atom - 1U), allocation_size_);
    const VkMappedMemoryRange range{
        VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, memory_, begin, end - begin};
    runtime_->check(vkInvalidateMappedMemoryRanges(
                        runtime_->device, 1U, &range),
                    "vkInvalidateMappedMemoryRanges");
}

class DeviceArena {
    struct Span {
        VkDeviceSize offset = 0U;
        VkDeviceSize size = 0U;
    };

public:
    struct Chunk {
        Chunk(Runtime &runtime, VkDeviceSize size, const char *name)
            : storage(runtime, size,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                          | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                          | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0U, name) {
            free.push_back({0U, size});
        }
        Buffer storage;
        std::mutex mutex;
        std::vector<Span> free;
    };

    class Allocation {
    public:
        Allocation() = default;
        Allocation(std::shared_ptr<Chunk> chunk, VkDeviceSize offset,
                   VkDeviceSize size) noexcept
            : chunk_(std::move(chunk)), offset_(offset), size_(size) {}
        ~Allocation() { reset(); }
        Allocation(const Allocation &) = delete;
        Allocation &operator=(const Allocation &) = delete;
        Allocation(Allocation &&other) noexcept { swap(other); }
        Allocation &operator=(Allocation &&other) noexcept {
            if (this != &other) {
                reset();
                swap(other);
            }
            return *this;
        }

        void reset() noexcept {
            if (!chunk_) return;
            const std::scoped_lock lock(chunk_->mutex);
            chunk_->free.push_back({offset_, size_});
            std::ranges::sort(chunk_->free, {}, &Span::offset);
            std::vector<Span> merged;
            for (const auto span : chunk_->free) {
                if (!merged.empty()
                    && merged.back().offset + merged.back().size == span.offset) {
                    merged.back().size += span.size;
                } else {
                    merged.push_back(span);
                }
            }
            chunk_->free = std::move(merged);
            chunk_.reset();
            offset_ = 0U;
            size_ = 0U;
        }

        [[nodiscard]] VkBuffer buffer() const noexcept {
            return chunk_ ? chunk_->storage.handle() : VK_NULL_HANDLE;
        }
        [[nodiscard]] VkDeviceSize offset() const noexcept { return offset_; }
        [[nodiscard]] VkDeviceSize size() const noexcept { return size_; }

    private:
        void swap(Allocation &other) noexcept {
            std::swap(chunk_, other.chunk_);
            std::swap(offset_, other.offset_);
            std::swap(size_, other.size_);
        }
        std::shared_ptr<Chunk> chunk_;
        VkDeviceSize offset_ = 0U;
        VkDeviceSize size_ = 0U;
    };

    DeviceArena(Runtime &runtime, const char *name)
        : runtime_(runtime), name_(name),
          alignment_(std::max<VkDeviceSize>(
              4U, runtime.selected.properties.limits.minStorageBufferOffsetAlignment)) {}

    [[nodiscard]] Allocation allocate(VkDeviceSize requested) {
        const VkDeviceSize size = align_up(
            static_cast<std::size_t>(requested),
            static_cast<std::size_t>(alignment_), "Vulkan arena allocation");
        if (size == 0U || size > maximum_allocation_bytes) {
            throw std::length_error("Vulkan arena allocation exceeds the 2 GiB guard");
        }
        if (size > runtime_.selected.properties.limits.maxStorageBufferRange) {
            throw std::length_error(
                "Vulkan arena allocation exceeds maxStorageBufferRange");
        }
        const std::scoped_lock arena_lock(mutex_);
        for (const auto &chunk : chunks_) {
            const std::scoped_lock chunk_lock(chunk->mutex);
            for (std::size_t index = 0U; index < chunk->free.size(); ++index) {
                Span &span = chunk->free[index];
                if (span.size < size) continue;
                const VkDeviceSize offset = span.offset;
                span.offset += size;
                span.size -= size;
                if (span.size == 0U) chunk->free.erase(chunk->free.begin() + index);
                return Allocation{chunk, offset, size};
            }
        }
        const VkDeviceSize chunk_size = std::max<VkDeviceSize>(16U * 1024U * 1024U, size);
        auto chunk = std::make_shared<Chunk>(runtime_, chunk_size, name_);
        {
            const std::scoped_lock chunk_lock(chunk->mutex);
            chunk->free.front().offset = size;
            chunk->free.front().size = chunk_size - size;
            if (chunk->free.front().size == 0U) chunk->free.clear();
        }
        chunks_.push_back(chunk);
        return Allocation{std::move(chunk), 0U, size};
    }

    void reset() noexcept {
        const std::scoped_lock lock(mutex_);
        chunks_.clear();
    }

private:
    Runtime &runtime_;
    const char *name_ = nullptr;
    VkDeviceSize alignment_ = 4U;
    std::mutex mutex_;
    std::vector<std::shared_ptr<Chunk>> chunks_;
};

class PlanUploadPool {
    static constexpr std::size_t maximum_retained_slots = 32U;

    struct Slot {
        explicit Slot(Runtime &requested_runtime) : runtime(requested_runtime) {}

        void reserve(VkDeviceSize requested) {
            if (staging.handle() && staging.size() >= requested) return;
            const VkDeviceSize capacity = std::min<VkDeviceSize>(
                maximum_allocation_bytes,
                std::max<VkDeviceSize>(requested, staging.size() * 2U));
            staging = Buffer(
                runtime, capacity, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                    | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                "dsmvc pooled plan upload staging");
        }

        Runtime &runtime;
        Buffer staging;
    };

public:
    class Allocation {
    public:
        Allocation() = default;
        ~Allocation() { reset(); }
        Allocation(const Allocation &) = delete;
        Allocation &operator=(const Allocation &) = delete;
        Allocation(Allocation &&other) noexcept
            : pool_(std::exchange(other.pool_, nullptr)),
              slot_(std::move(other.slot_)) {}
        Allocation &operator=(Allocation &&other) noexcept {
            if (this != &other) {
                reset();
                pool_ = std::exchange(other.pool_, nullptr);
                slot_ = std::move(other.slot_);
            }
            return *this;
        }

        void reset() noexcept {
            if (!pool_) return;
            pool_->release(std::move(slot_));
            pool_ = nullptr;
        }
        [[nodiscard]] Buffer &staging() noexcept { return slot_->staging; }
        [[nodiscard]] const Buffer &staging() const noexcept {
            return slot_->staging;
        }

    private:
        Allocation(PlanUploadPool &pool, std::unique_ptr<Slot> slot) noexcept
            : pool_(&pool), slot_(std::move(slot)) {}

        PlanUploadPool *pool_ = nullptr;
        std::unique_ptr<Slot> slot_;
        friend class PlanUploadPool;
    };

    explicit PlanUploadPool(Runtime &runtime) : runtime_(runtime) {}

    [[nodiscard]] Allocation acquire(VkDeviceSize bytes) {
        runtime_.throw_if_failed();
        std::unique_ptr<Slot> slot;
        {
            const std::scoped_lock lock(mutex_);
            if (!available_.empty()) {
                slot = std::move(available_.back());
                available_.pop_back();
            }
        }
        if (!slot) slot = std::make_unique<Slot>(runtime_);
        Allocation allocation(*this, std::move(slot));
        allocation.slot_->reserve(bytes);
        return allocation;
    }

private:
    void release(std::unique_ptr<Slot> slot) noexcept {
        try {
            const std::scoped_lock lock(mutex_);
            if (available_.size() < maximum_retained_slots) {
                available_.push_back(std::move(slot));
            }
        } catch (...) {
        }
    }

    Runtime &runtime_;
    std::mutex mutex_;
    std::vector<std::unique_ptr<Slot>> available_;
};

struct PushConstants {
    std::array<std::uint32_t, 32U> value{};
};
static_assert(sizeof(PushConstants) == 128U);

} // namespace


namespace {

[[nodiscard]] VulkanPlanWordLayout packed_plan_layout(
    const AxisPlan &axis, bool include_float64) {
    return VulkanPlanWordLayout::make(
        axis.transpose_offsets.size(), axis.transpose_indices.size(),
        axis.transpose_weights.size(), axis.lower_ld.size(),
        axis.inverse_diagonal.size(), axis.ldlt_bands_f64.size(),
        axis.requires_float64(), include_float64);
}

// The partitioned solve shader uses 32 x chunks invocations and
// 32 * 32 * half_bandwidth floats of shared memory; devices below that keep
// the sequential shaders.
[[nodiscard]] solve_partition::Partition device_partition(
    const Runtime &runtime, const AxisPlan &axis, bool build) {
    auto partition = build ? solve_partition::build(axis)
                           : solve_partition::choose(axis);
    const auto &limits = runtime.selected.properties.limits;
    const std::uint64_t invocations =
        static_cast<std::uint64_t>(partition.chunks) * 32U;
    const std::uint64_t shared_bytes = 32ULL * 32ULL
        * static_cast<std::uint64_t>(axis.half_bandwidth) * sizeof(float);
    if (partition.chunks != 0U
        && (invocations > limits.maxComputeWorkGroupInvocations
            || partition.chunks > limits.maxComputeWorkGroupSize[1]
            || shared_bytes > limits.maxComputeSharedMemorySize)) {
        partition = {};
    }
    return partition;
}

[[nodiscard]] std::size_t packed_plan_storage_bytes(
    const Runtime &runtime, const AxisPlan &axis, bool include_float64) {
    const std::size_t total =
        packed_plan_layout(axis, include_float64).storage_bytes()
        + solve_partition::transfer_floats(
              device_partition(runtime, axis, false), axis)
            * sizeof(float);
    if (total == 0U || total > maximum_allocation_bytes) {
        throw std::length_error("Vulkan packed plan exceeds the 2 GiB guard");
    }
    return total;
}

class PackedPlan {
public:
    PackedPlan(std::shared_ptr<Runtime> requested_runtime,
               std::shared_ptr<const AxisPlan> requested_axis,
               const AxisPlan *requested_identity, bool float64)
        : runtime(std::move(requested_runtime)), axis(std::move(requested_axis)),
          identity(requested_identity ? requested_identity : axis.get()) {
        if (!axis || !axis->valid()) {
            throw std::invalid_argument("cannot pack an invalid Vulkan axis plan");
        }
        source_size = static_cast<std::uint32_t>(axis->source_size);
        destination_size = static_cast<std::uint32_t>(axis->destination_size);
        half_bandwidth = static_cast<std::uint32_t>(axis->half_bandwidth);
        retained_float64 = axis->requires_float64();
        if (retained_float64 && !float64) {
            throw std::logic_error(
                "retained Float64 plan requested a Float32 Vulkan layout");
        }
        if (float64 && !runtime->selected.float64.strict_supported()) {
            throw std::runtime_error(runtime->selected.float64.requirement_error());
        }
        const bool include_float64 = float64;
        const auto layout = packed_plan_layout(*axis, include_float64);
        const auto partition = device_partition(*runtime, *axis, true);
        storage_bytes = layout.storage_bytes()
            + partition.transfers.size() * sizeof(float);
        if (storage_bytes > runtime->selected.properties.limits.maxStorageBufferRange) {
            throw std::length_error(
                "Vulkan packed plan exceeds maxStorageBufferRange");
        }

        storage = runtime->plan_arena->allocate(storage_bytes);
        Buffer *upload_staging = nullptr;
        if (runtime->timeline_enabled) {
            if (!runtime->plan_upload_pool) {
                throw std::logic_error(
                    "Vulkan timeline plan upload pool is unavailable");
            }
            timeline_upload = runtime->plan_upload_pool->acquire(storage_bytes);
            upload_staging = &timeline_upload.staging();
        } else {
            staging = Buffer(
                *runtime, storage_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                "dsmvc plan upload staging");
            upload_staging = &staging;
        }
        auto *packed = static_cast<std::uint32_t *>(upload_staging->mapped());
        std::memset(packed, 0, storage_bytes);
        const auto copy_words = [&](const auto &values, std::uint32_t word_offset) {
            using Value = typename std::remove_cvref_t<decltype(values)>::value_type;
            static_assert(sizeof(Value) == sizeof(std::uint32_t));
            const std::size_t bytes = values.size() * sizeof(Value);
            if (bytes != 0U) {
                std::memcpy(reinterpret_cast<std::byte *>(packed)
                                + static_cast<std::ptrdiff_t>(word_offset * 4U),
                            values.data(), bytes);
            }
        };
        const auto copy_doubles = [&](const auto &values, std::uint32_t word_offset) {
            if ((word_offset & 1U) != 0U) {
                throw std::logic_error("Vulkan Float64 plan field is misaligned");
            }
            for (std::size_t index = 0U; index < values.size(); ++index) {
                const double value = static_cast<double>(values[index]);
                const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
                packed[static_cast<std::size_t>(word_offset) + index * 2U] =
                    static_cast<std::uint32_t>(bits);
                packed[static_cast<std::size_t>(word_offset) + index * 2U + 1U] =
                    static_cast<std::uint32_t>(bits >> 32U);
            }
        };
        offsets_offset = layout.offsets;
        indices_offset = layout.indices;
        weights_offset = layout.weights_f32;
        lower_offset = layout.lower_f32;
        upper_offset = layout.upper_f32;
        diagonal_offset = layout.diagonal_f32;
        copy_words(axis->transpose_offsets, offsets_offset);
        copy_words(axis->transpose_indices, indices_offset);
        copy_words(axis->transpose_weights, weights_offset);
        copy_words(axis->lower_ld, lower_offset);
        copy_words(axis->upper_l, upper_offset);
        copy_words(axis->inverse_diagonal, diagonal_offset);
        if (partition.chunks != 0U) {
            solve_chunks = partition.chunks;
            solve_chunk_length = partition.chunk_length;
            transfers_offset = static_cast<std::uint32_t>(
                layout.storage_bytes() / sizeof(std::uint32_t));
            copy_words(partition.transfers, transfers_offset);
            // Create the pipeline now (plans are packed when a filter is
            // created), not at the first frame.
            (void)runtime->chunked_solve_pipeline(half_bandwidth, solve_chunks);
        }
        if (include_float64) {
            weights_f64_offset = layout.weights_f64;
            lower_f64_offset = layout.lower_f64;
            upper_f64_offset = layout.upper_f64;
            diagonal_f64_offset = layout.diagonal_f64;
            if (retained_float64) {
                copy_doubles(axis->transpose_weights_f64, weights_f64_offset);
                copy_doubles(axis->ldlt_bands_f64, lower_f64_offset);
                copy_doubles(axis->inverse_diagonal_f64, diagonal_f64_offset);
            } else {
                copy_doubles(axis->transpose_weights, weights_f64_offset);
                copy_doubles(axis->lower_ld, lower_f64_offset);
                copy_doubles(axis->upper_l, upper_f64_offset);
                copy_doubles(axis->inverse_diagonal, diagonal_f64_offset);
            }
        }

        upload_staging->flush(0U, storage_bytes);

        if (runtime->timeline_enabled) {
            upload_state = UploadState::pending;
        } else {
            const VkCommandPoolCreateInfo pool_info{
                VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                nullptr,
                VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                runtime->selected.queue_family,
            };
            runtime->check(vkCreateCommandPool(
                               runtime->device, &pool_info, nullptr, &upload_pool),
                           "vkCreateCommandPool(plan upload)");
            try {
                const VkCommandBufferAllocateInfo command_info{
                    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    nullptr,
                    upload_pool,
                    VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    1U,
                };
                VkCommandBuffer command = VK_NULL_HANDLE;
                runtime->check(vkAllocateCommandBuffers(
                                   runtime->device, &command_info, &command),
                               "vkAllocateCommandBuffers(plan upload)");
                const VkCommandBufferBeginInfo begin_info{
                    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    nullptr,
                    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                    nullptr,
                };
                runtime->check(vkBeginCommandBuffer(command, &begin_info),
                               "vkBeginCommandBuffer(plan upload)");
                record_upload(command, *upload_staging);
                runtime->check(vkEndCommandBuffer(command),
                               "vkEndCommandBuffer(plan upload)");

                const VkFenceCreateInfo fence_info{
                    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0U};
                runtime->check(vkCreateFence(
                                   runtime->device, &fence_info, nullptr,
                                   &upload_fence),
                               "vkCreateFence(plan upload)");
                (void)runtime->submit(command, upload_fence, 0U, false);
                runtime->wait_fence(upload_fence, "Vulkan plan upload");
                retire_upload();
            } catch (...) {
                retire_upload();
                throw;
            }
        }
    }

    ~PackedPlan() { retire_upload(); }

    [[nodiscard]] bool record_pending_upload(VkCommandBuffer command) const {
        if (!runtime->timeline_enabled) return false;
        runtime->throw_if_failed();
        std::unique_lock lock(upload_mutex);
        upload_ready.wait(lock, [&] {
            return upload_state != UploadState::recording;
        });
        if (upload_state == UploadState::complete) return false;
        upload_state = UploadState::recording;
        try {
            record_upload(command, timeline_upload.staging());
        } catch (...) {
            upload_state = UploadState::pending;
            lock.unlock();
            upload_ready.notify_all();
            throw;
        }
        return true;
    }

    void finish_pending_upload(bool complete) const noexcept {
        if (!runtime->timeline_enabled) return;
        {
            const std::scoped_lock lock(upload_mutex);
            if (upload_state != UploadState::recording) return;
            upload_state = complete ? UploadState::complete : UploadState::pending;
            if (complete) {
                const_cast<PackedPlan *>(this)->timeline_upload.reset();
            }
        }
        upload_ready.notify_all();
    }

    [[nodiscard]] VkBuffer buffer() const noexcept { return storage.buffer(); }
    [[nodiscard]] VkDeviceSize buffer_offset() const noexcept {
        return storage.offset();
    }
    [[nodiscard]] VkDeviceSize buffer_size() const noexcept {
        return storage.size();
    }

    std::shared_ptr<Runtime> runtime;
    std::shared_ptr<const AxisPlan> axis;
    const AxisPlan *identity = nullptr;
    std::uint32_t source_size = 0U;
    std::uint32_t destination_size = 0U;
    std::uint32_t half_bandwidth = 0U;
    std::uint32_t offsets_offset = 0U;
    std::uint32_t indices_offset = 0U;
    std::uint32_t weights_offset = 0U;
    std::uint32_t lower_offset = 0U;
    std::uint32_t upper_offset = 0U;
    std::uint32_t diagonal_offset = 0U;
    std::uint32_t weights_f64_offset = 0U;
    std::uint32_t lower_f64_offset = 0U;
    std::uint32_t upper_f64_offset = 0U;
    std::uint32_t diagonal_f64_offset = 0U;
    // Partitioned solve; zero chunks keeps the sequential shaders.
    std::uint32_t solve_chunks = 0U;
    std::uint32_t solve_chunk_length = 0U;
    std::uint32_t transfers_offset = 0U;
    bool retained_float64 = false;
    std::size_t storage_bytes = 0U;
    mutable std::atomic<std::uint32_t> execution_count{0U};

private:
    enum class UploadState : std::uint8_t {
        complete,
        pending,
        recording,
    };

    void record_upload(
        VkCommandBuffer command, const Buffer &upload_staging) const noexcept {
        const VkBufferCopy copy_region{0U, storage.offset(), storage_bytes};
        vkCmdCopyBuffer(
            command, upload_staging.handle(), storage.buffer(),
            1U, &copy_region);
        const VkBufferMemoryBarrier barrier{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            storage.buffer(),
            storage.offset(),
            storage.size(),
        };
        vkCmdPipelineBarrier(
            command, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U,
            0U, nullptr, 1U, &barrier, 0U, nullptr);
    }

    void retire_upload() noexcept {
        const std::scoped_lock lock(upload_mutex);
        retire_upload_locked();
    }

    void retire_upload_locked() noexcept {
        timeline_upload.reset();
        staging.reset();
        if (upload_fence) {
            vkDestroyFence(runtime->device, upload_fence, nullptr);
            upload_fence = VK_NULL_HANDLE;
        }
        if (upload_pool) {
            vkDestroyCommandPool(runtime->device, upload_pool, nullptr);
            upload_pool = VK_NULL_HANDLE;
        }
    }

    DeviceArena::Allocation storage;
    PlanUploadPool::Allocation timeline_upload;
    Buffer staging;
    VkCommandPool upload_pool = VK_NULL_HANDLE;
    VkFence upload_fence = VK_NULL_HANDLE;
    mutable UploadState upload_state = UploadState::complete;
    mutable std::mutex upload_mutex;
    mutable std::condition_variable upload_ready;
};

class PendingPlanUploads {
public:
    // The plans an execution uses, without duplicates: a frame names up to
    // four distinct ones, and an upload is recorded once.
    explicit PendingPlanUploads(
        std::shared_ptr<const PackedPlan> first,
        std::shared_ptr<const PackedPlan> second = {}) {
        add(std::move(first));
        add(std::move(second));
        order();
    }
    explicit PendingPlanUploads(
        std::span<const std::shared_ptr<const PackedPlan>> plans) {
        for (const auto &plan : plans) add(plan);
        order();
    }

    ~PendingPlanUploads() { finish(false); }
    PendingPlanUploads(const PendingPlanUploads &) = delete;
    PendingPlanUploads &operator=(const PendingPlanUploads &) = delete;

    void record(VkCommandBuffer command) {
        for (std::size_t index = 0U; index < count_; ++index) {
            pending_[index] = plans_[index]->record_pending_upload(command);
        }
    }

    void complete() noexcept { finish(true); }

private:
    static constexpr std::size_t capacity = 2U * frame_plane_limit;

    void add(std::shared_ptr<const PackedPlan> plan) {
        if (!plan) return;
        for (std::size_t index = 0U; index < count_; ++index) {
            if (plans_[index] == plan) return;
        }
        if (count_ == capacity) {
            throw std::logic_error("too many Vulkan plans in one execution");
        }
        plans_[count_++] = std::move(plan);
    }

    // Uploads are recorded in address order: two executions that share plans
    // then wait for each other's recording in the same order.
    void order() noexcept {
        std::sort(plans_.begin(), plans_.begin() + static_cast<std::ptrdiff_t>(count_),
                  [](const auto &left, const auto &right) {
                      return std::less<const PackedPlan *>{}(
                          left.get(), right.get());
                  });
    }

    void finish(bool complete) noexcept {
        if (finished_) return;
        for (std::size_t index = 0U; index < count_; ++index) {
            if (pending_[index]) plans_[index]->finish_pending_upload(complete);
        }
        finished_ = true;
    }

    std::array<std::shared_ptr<const PackedPlan>, capacity> plans_{};
    std::array<bool, capacity> pending_{};
    std::size_t count_ = 0U;
    bool finished_ = false;
};

class PackedPlanRequest {
public:
    explicit PackedPlanRequest(std::shared_ptr<const AxisPlan> requested_axis)
        : axis_(std::move(requested_axis)) {}

    [[nodiscard]] const std::shared_ptr<const AxisPlan> &axis() const noexcept {
        return axis_;
    }
    void publish(std::shared_ptr<const PackedPlan> packed) noexcept {
        {
            const std::scoped_lock lock(mutex_);
            packed_ = std::move(packed);
            complete_ = true;
        }
        ready_.notify_all();
    }
    void fail(std::exception_ptr error) noexcept {
        {
            const std::scoped_lock lock(mutex_);
            error_ = std::move(error);
            complete_ = true;
        }
        ready_.notify_all();
    }
    [[nodiscard]] std::shared_ptr<const PackedPlan> get() const {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [&] { return complete_; });
        if (error_) std::rethrow_exception(error_);
        return packed_;
    }

private:
    std::shared_ptr<const AxisPlan> axis_;
    mutable std::mutex mutex_;
    mutable std::condition_variable ready_;
    std::shared_ptr<const PackedPlan> packed_;
    std::exception_ptr error_;
    bool complete_ = false;
};

// The packed plans of every runtime, one cache per process. A plan is packed
// for one runtime's device, so the runtime is part of the key; a runtime that
// the cache would keep alive (every cached plan holds one) is purged when its
// last executor goes, which a borrowed device needs: it must not outlive the
// core's device.
struct PlanCacheKey {
    const Runtime *runtime = nullptr;
    const AxisPlan *identity = nullptr;
    bool float64 = false;

    [[nodiscard]] bool operator==(const PlanCacheKey &) const noexcept = default;
};

struct PlanCacheKeyHash {
    [[nodiscard]] std::size_t operator()(const PlanCacheKey &key) const noexcept {
        std::size_t hash = std::hash<const AxisPlan *>{}(key.identity);
        hash ^= std::hash<const Runtime *>{}(key.runtime)
            + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        return hash ^ (key.float64
            ? static_cast<std::size_t>(0x9e3779b97f4a7c15ULL) : 0U);
    }
};

struct PlanCacheEntry {
    std::shared_ptr<PackedPlanRequest> request;
    std::list<PlanCacheKey>::iterator lru;
    std::size_t bytes = 0U;
};

struct PlanCache {
    std::mutex mutex;
    std::unordered_map<PlanCacheKey, PlanCacheEntry, PlanCacheKeyHash> entries;
    std::list<PlanCacheKey> lru;
    std::size_t bytes = 0U;
    const std::size_t capacity = environment_megabytes(
        "DSMVC_VULKAN_PLAN_CACHE_MB", default_plan_cache_bytes);
};

[[nodiscard]] PlanCache &plan_cache() {
    static PlanCache cache;
    return cache;
}

void purge_packed_plans(const Runtime *runtime) noexcept {
    std::vector<std::shared_ptr<PackedPlanRequest>> evicted;
    {
        auto &cache = plan_cache();
        const std::scoped_lock lock(cache.mutex);
        for (auto it = cache.entries.begin(); it != cache.entries.end();) {
            if (it->first.runtime == runtime) {
                cache.bytes -= it->second.bytes;
                cache.lru.erase(it->second.lru);
                evicted.push_back(std::move(it->second.request));
                it = cache.entries.erase(it);
            } else {
                ++it;
            }
        }
    }
    // `evicted` is released outside the lock: the plans free their buffers.
}

[[nodiscard]] std::shared_ptr<const PackedPlan> acquire_packed_plan(
    const std::shared_ptr<Runtime> &runtime,
    const std::shared_ptr<const AxisPlan> &plan, bool float64) {
    using Key = PlanCacheKey;
    using Entry = PlanCacheEntry;
    auto &cache = plan_cache();

    const Key key{runtime.get(), plan.get(), float64};
    const std::size_t bytes = packed_plan_storage_bytes(*runtime, *plan, float64);
    std::shared_ptr<PackedPlanRequest> request;
    std::vector<std::shared_ptr<PackedPlanRequest>> evicted;
    bool producer = false;
    {
        const std::scoped_lock lock(cache.mutex);
        if (const auto found = cache.entries.find(key);
            found != cache.entries.end()) {
            if (found->second.request->axis() == plan) {
                cache.lru.splice(cache.lru.end(), cache.lru, found->second.lru);
                request = found->second.request;
            } else {
                cache.bytes -= found->second.bytes;
                cache.lru.erase(found->second.lru);
                evicted.push_back(std::move(found->second.request));
                cache.entries.erase(found);
            }
        }
        if (!request) {
            request = std::make_shared<PackedPlanRequest>(plan);
            producer = true;
            cache.bytes = checked_add(cache.bytes, bytes, "Vulkan plan cache");
            cache.lru.push_back(key);
            cache.entries.emplace(
                key, Entry{request, std::prev(cache.lru.end()), bytes});
            while (cache.bytes > cache.capacity && cache.entries.size() > 1U) {
                const Key victim_key = cache.lru.front();
                const auto victim = cache.entries.find(victim_key);
                cache.bytes -= victim->second.bytes;
                evicted.push_back(std::move(victim->second.request));
                cache.lru.pop_front();
                cache.entries.erase(victim);
            }
        }
    }
    if (!producer) return request->get();
    try {
        auto packed = std::make_shared<const PackedPlan>(
            runtime, plan, plan.get(), float64);
        request->publish(packed);
        return packed;
    } catch (...) {
        const auto error = std::current_exception();
        request->fail(error);
        std::shared_ptr<PackedPlanRequest> removed;
        {
            const std::scoped_lock lock(cache.mutex);
            const auto found = cache.entries.find(key);
            if (found != cache.entries.end()
                && found->second.request == request) {
                cache.bytes -= found->second.bytes;
                cache.lru.erase(found->second.lru);
                removed = std::move(found->second.request);
                cache.entries.erase(found);
            }
        }
        std::rethrow_exception(error);
    }
}

[[nodiscard]] bool should_split_rhs(
    const Runtime &runtime, const PackedPlan &plan) noexcept {
    if (runtime.split_rhs == SplitRhsMode::disabled
        || plan.half_bandwidth < 3U) {
        return false;
    }
    if (runtime.split_rhs == SplitRhsMode::forced) return true;
    return plan.execution_count.fetch_add(1U, std::memory_order_relaxed) != 0U;
}

void pack_host_rows(
    const void *source, std::size_t source_pitch, void *destination,
    std::size_t row_bytes, std::size_t rows) {
    host_copy::copy_rows(
        destination, row_bytes, source, source_pitch, row_bytes, rows);
}

// Packs rows tightly and zeroes the bytes between the packed image and the
// word-aligned copy size: the shaders read whole 32-bit words.
void pack_host_rows_aligned(
    const void *source, std::size_t source_pitch, void *destination,
    std::size_t row_bytes, std::size_t rows, std::size_t copy_bytes) {
    pack_host_rows(source, source_pitch, destination, row_bytes, rows);
    const std::size_t packed = row_bytes * rows;
    if (copy_bytes > packed) {
        std::memset(
            static_cast<std::byte *>(destination)
                + static_cast<std::ptrdiff_t>(packed),
            0, copy_bytes - packed);
    }
}

void unpack_host_rows(
    const void *source, std::size_t row_bytes, void *destination,
    std::size_t destination_pitch, std::size_t rows) {
    host_copy::copy_rows(
        destination, destination_pitch, source, row_bytes, row_bytes, rows);
}

void command_barrier(
    VkCommandBuffer command, VkBuffer buffer, VkDeviceSize offset,
    VkDeviceSize size, VkAccessFlags source_access,
    VkAccessFlags destination_access, VkPipelineStageFlags source_stage,
    VkPipelineStageFlags destination_stage) {
    const VkBufferMemoryBarrier barrier{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        nullptr,
        source_access,
        destination_access,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        buffer,
        offset,
        size,
    };
    vkCmdPipelineBarrier(
        command, source_stage, destination_stage, 0U,
        0U, nullptr, 1U, &barrier, 0U, nullptr);
}

// The barriers one recording step of a frame needs, issued as one command so
// that the planes of a frame do not wait for each other one at a time.
class BarrierBatch {
public:
    void add(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
             VkAccessFlags source_access, VkAccessFlags destination_access,
             VkPipelineStageFlags source_stage,
             VkPipelineStageFlags destination_stage) {
        if (count_ == barriers_.size()) {
            throw std::logic_error("too many Vulkan barriers in one step");
        }
        barriers_[count_++] = {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
            source_access,
            destination_access,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            buffer,
            offset,
            size,
        };
        source_stage_ |= source_stage;
        destination_stage_ |= destination_stage;
    }

    void flush(VkCommandBuffer command) noexcept {
        if (count_ != 0U) {
            vkCmdPipelineBarrier(
                command, source_stage_, destination_stage_, 0U,
                0U, nullptr, count_, barriers_.data(), 0U, nullptr);
        }
        count_ = 0U;
        source_stage_ = 0U;
        destination_stage_ = 0U;
    }

private:
    std::array<VkBufferMemoryBarrier, 4U * frame_plane_limit> barriers_{};
    std::uint32_t count_ = 0U;
    VkPipelineStageFlags source_stage_ = 0U;
    VkPipelineStageFlags destination_stage_ = 0U;
};

template <class Slot>
void bind_compute(
    Runtime &runtime, Slot &slot, VkPipeline pipeline,
    VkDescriptorSet set, const PushConstants &push) {
    vkCmdBindPipeline(slot.command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(
        slot.command, VK_PIPELINE_BIND_POINT_COMPUTE,
        runtime.pipeline_layout, 0U, 1U, &set, 0U, nullptr);
    vkCmdPushConstants(
        slot.command, runtime.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0U, sizeof(push), &push);
}

template <class Slot>
void record_transpose(
    Runtime &runtime, Slot &slot, const PackedPlan &descriptor_plan,
    VkBuffer cache_buffer, VkDeviceSize cache_offset, VkDeviceSize cache_size,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t source_offset, std::uint32_t destination_offset,
    std::uint32_t source_buffer, std::uint32_t destination_buffer,
    std::uint32_t sample_type, const IntegerConversion *conversion) {
    PushConstants push;
    push.value[0] = width;
    push.value[1] = height;
    push.value[2] = source_offset;
    push.value[3] = destination_offset;
    push.value[4] = source_buffer;
    push.value[5] = destination_buffer;
    push.value[6] = sample_type;
    if (conversion) {
        push.value[7] = std::bit_cast<std::uint32_t>(conversion->input_offset);
        push.value[8] = std::bit_cast<std::uint32_t>(conversion->input_scale);
    }
    const VkDescriptorSet set = slot.descriptor_set(
        descriptor_plan.buffer(), descriptor_plan.buffer_offset(),
        descriptor_plan.buffer_size(), cache_buffer, cache_offset, cache_size);
    bind_compute(runtime, slot, runtime.transpose_pipeline, set, push);
    vkCmdDispatch(
        slot.command, divide_up(width, runtime.transpose_local_x),
        divide_up(height, runtime.transpose_local_y), 1U);
}

void fill_plan_push(PushConstants &push, const PackedPlan &plan) noexcept {
    push.value[0] = plan.source_size;
    push.value[1] = plan.destination_size;
    push.value[2] = plan.half_bandwidth;
    push.value[9] = plan.offsets_offset;
    push.value[10] = plan.indices_offset;
    push.value[11] = plan.weights_offset;
    push.value[12] = plan.lower_offset;
    push.value[13] = plan.upper_offset;
    push.value[14] = plan.diagonal_offset;
}

[[nodiscard]] VkPipeline select_inverse_pipeline(
    const Runtime &runtime, std::uint32_t bandwidth, bool solve_only) noexcept {
    const auto found = std::ranges::find(fixed_bandwidths, bandwidth);
    if (found != fixed_bandwidths.end()) {
        const std::size_t index = static_cast<std::size_t>(
            found - fixed_bandwidths.begin());
        const VkPipeline pipeline = solve_only
            ? runtime.fixed_solve_pipelines[index]
            : runtime.fixed_inverse_pipelines[index];
        if (pipeline) return pipeline;
    }
    return solve_only ? runtime.solve_pipeline : runtime.inverse_pipeline;
}

// A solve is recorded whole, or as the right-hand side (rhs) and then the
// solve proper, with the caller's barrier between them.
enum class InversePhase : std::uint8_t { all, rhs, solve };

// True if an inverse with this output layout runs a separate right-hand side.
[[nodiscard]] bool has_rhs_phase(
    const PackedPlan &plan, std::uint32_t output_layout, bool split) noexcept {
    return split || (plan.solve_chunks != 0U && output_layout == 0U);
}

template <class Slot>
void record_inverse(
    Runtime &runtime, Slot &slot, const PackedPlan &plan,
    VkBuffer cache_buffer, VkDeviceSize cache_offset, VkDeviceSize cache_size,
    std::uint32_t vector_count, std::uint32_t source_offset,
    std::uint32_t output_offset, std::uint32_t source_buffer,
    std::uint32_t source_layout, std::uint32_t output_layout,
    bool split, InversePhase phase = InversePhase::all) {
    PushConstants push;
    fill_plan_push(push, plan);
    push.value[3] = vector_count;
    push.value[4] = source_offset;
    push.value[5] = output_offset;
    push.value[6] = source_buffer;
    push.value[7] = source_layout;
    push.value[8] = output_layout;
    // The partitioned solve works in place on column-major output, after the
    // right-hand side has been written there.
    const bool chunked = plan.solve_chunks != 0U && output_layout == 0U;
    if (split || chunked) {
        if (phase != InversePhase::solve) {
            VkDescriptorSet rhs_set = slot.descriptor_set(
                plan.buffer(), plan.buffer_offset(), plan.buffer_size(),
                cache_buffer, cache_offset, cache_size);
            // A row-major source is walked along its rows (see rhs.comp),
            // unless an element has too few taps for that to pay (measured:
            // bilinear).
            const bool samples_along_x = runtime.fold_transposes
                && source_layout != 0U && plan.half_bandwidth > 1U;
            push.value[18] = samples_along_x ? 1U : 0U;
            bind_compute(runtime, slot, runtime.rhs_pipeline, rhs_set, push);
            vkCmdDispatch(
                slot.command,
                divide_up(samples_along_x ? plan.destination_size
                                          : vector_count,
                          runtime.rhs_local_x),
                divide_up(samples_along_x ? vector_count
                                          : plan.destination_size,
                          runtime.rhs_local_y), 1U);
            push.value[18] = 0U;
            if (phase == InversePhase::rhs) return;
            command_barrier(
                slot.command, slot.workspace.handle(),
                static_cast<VkDeviceSize>(output_offset) * 4U,
                static_cast<VkDeviceSize>(plan.destination_size)
                    * vector_count * 4U,
                VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
        VkDescriptorSet solve_set = slot.descriptor_set(
            plan.buffer(), plan.buffer_offset(), plan.buffer_size(),
            cache_buffer, cache_offset, cache_size);
        if (chunked) {
            push.value[15] = plan.transfers_offset;
            push.value[16] = plan.solve_chunk_length;
            push.value[17] = plan.solve_chunks;
            bind_compute(
                runtime, slot,
                runtime.chunked_solve_pipeline(
                    plan.half_bandwidth, plan.solve_chunks),
                solve_set, push);
            vkCmdDispatch(slot.command, divide_up(vector_count, 32U), 1U, 1U);
            return;
        }
        bind_compute(
            runtime, slot,
            select_inverse_pipeline(runtime, plan.half_bandwidth, true),
            solve_set, push);
    } else {
        if (phase == InversePhase::rhs) return;
        VkDescriptorSet inverse_set = slot.descriptor_set(
            plan.buffer(), plan.buffer_offset(), plan.buffer_size(),
            cache_buffer, cache_offset, cache_size);
        bind_compute(
            runtime, slot,
            select_inverse_pipeline(runtime, plan.half_bandwidth, false),
            inverse_set, push);
    }
    vkCmdDispatch(
        slot.command, divide_up(vector_count, runtime.inverse_local_size), 1U, 1U);
}

template <class Slot>
void record_conversion(
    Runtime &runtime, Slot &slot, const PackedPlan &descriptor_plan,
    std::uint32_t elements, std::uint32_t source_offset,
    std::uint32_t destination_offset, std::uint32_t sample_type,
    const IntegerConversion &conversion) {
    PushConstants push;
    push.value[0] = elements;
    push.value[1] = source_offset;
    push.value[2] = destination_offset;
    push.value[3] = sample_type;
    push.value[4] = std::bit_cast<std::uint32_t>(conversion.output_scale);
    push.value[5] = std::bit_cast<std::uint32_t>(conversion.output_offset);
    push.value[6] = conversion.output_maximum;
    const VkDescriptorSet set = slot.descriptor_set(
        descriptor_plan.buffer(), descriptor_plan.buffer_offset(),
        descriptor_plan.buffer_size(), VK_NULL_HANDLE, 0U, 0U);
    bind_compute(runtime, slot, runtime.convert_pipeline, set, push);
    const std::uint32_t samples_per_word = sample_type == 1U ? 4U : 2U;
    const std::uint32_t words = divide_up(elements, samples_per_word);
    vkCmdDispatch(
        slot.command, divide_up(words, runtime.convert_local_size), 1U, 1U);
}

void fill_plan_push_f64(PushConstants &push, const PackedPlan &plan) noexcept {
    push.value[0] = plan.source_size;
    push.value[1] = plan.destination_size;
    push.value[2] = plan.half_bandwidth;
    push.value[9] = plan.offsets_offset;
    push.value[10] = plan.indices_offset;
    push.value[11] = plan.weights_f64_offset;
    push.value[12] = plan.lower_f64_offset;
    push.value[13] = plan.upper_f64_offset;
    push.value[14] = plan.diagonal_f64_offset;
    push.value[15] = plan.retained_float64 ? 1U : 0U;
}

template <class Slot>
void record_transpose_f64(
    Runtime &runtime, Slot &slot, const PackedPlan &descriptor_plan,
    VkBuffer cache_buffer, VkDeviceSize cache_offset, VkDeviceSize cache_size,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t source_offset, std::uint32_t destination_offset,
    std::uint32_t source_buffer, std::uint32_t destination_buffer,
    std::uint32_t sample_type, const IntegerConversion *conversion,
    std::uint32_t status_offset) {
    PushConstants push;
    push.value[0] = width;
    push.value[1] = height;
    push.value[2] = source_offset;
    push.value[3] = destination_offset;
    push.value[4] = source_buffer;
    push.value[5] = destination_buffer;
    push.value[6] = sample_type;
    push.value[16] = status_offset;
    if (conversion) {
        push.value[7] = std::bit_cast<std::uint32_t>(conversion->input_offset);
        push.value[8] = std::bit_cast<std::uint32_t>(conversion->input_scale);
    }
    const VkDescriptorSet set = slot.descriptor_set(
        descriptor_plan.buffer(), descriptor_plan.buffer_offset(),
        descriptor_plan.buffer_size(), cache_buffer, cache_offset, cache_size);
    bind_compute(runtime, slot, runtime.transpose_f64_pipeline, set, push);
    vkCmdDispatch(
        slot.command, divide_up(width, 32U), divide_up(height, 8U), 1U);
}

template <class Slot>
void record_inverse_f64(
    Runtime &runtime, Slot &slot, const PackedPlan &plan,
    VkBuffer cache_buffer, VkDeviceSize cache_offset, VkDeviceSize cache_size,
    std::uint32_t vector_count, std::uint32_t source_offset,
    std::uint32_t output_offset, std::uint32_t source_buffer,
    std::uint32_t source_layout, std::uint32_t output_layout,
    bool split, std::uint32_t status_offset) {
    PushConstants push;
    fill_plan_push_f64(push, plan);
    push.value[3] = vector_count;
    push.value[4] = source_offset;
    push.value[5] = output_offset;
    push.value[6] = source_buffer;
    push.value[7] = source_layout;
    push.value[8] = output_layout;
    push.value[16] = status_offset;
    if (split) {
        VkDescriptorSet rhs_set = slot.descriptor_set(
            plan.buffer(), plan.buffer_offset(), plan.buffer_size(),
            cache_buffer, cache_offset, cache_size);
        bind_compute(runtime, slot, runtime.rhs_f64_pipeline, rhs_set, push);
        vkCmdDispatch(
            slot.command, divide_up(vector_count, 32U),
            divide_up(plan.destination_size, 8U), 1U);
        command_barrier(
            slot.command, slot.workspace.handle(),
            static_cast<VkDeviceSize>(output_offset) * 4U,
            static_cast<VkDeviceSize>(plan.destination_size) * vector_count * 8U,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        VkDescriptorSet solve_set = slot.descriptor_set(
            plan.buffer(), plan.buffer_offset(), plan.buffer_size(),
            cache_buffer, cache_offset, cache_size);
        bind_compute(runtime, slot, runtime.solve_f64_pipeline, solve_set, push);
    } else {
        VkDescriptorSet inverse_set = slot.descriptor_set(
            plan.buffer(), plan.buffer_offset(), plan.buffer_size(),
            cache_buffer, cache_offset, cache_size);
        bind_compute(
            runtime, slot, runtime.inverse_f64_pipeline, inverse_set, push);
    }
    vkCmdDispatch(slot.command, divide_up(vector_count, 32U), 1U, 1U);
}

template <class Slot>
void record_conversion_f64(
    Runtime &runtime, Slot &slot, const PackedPlan &descriptor_plan,
    std::uint32_t elements, std::uint32_t source_offset,
    std::uint32_t destination_offset, std::uint32_t sample_type,
    const IntegerConversion *conversion, std::uint32_t status_offset) {
    PushConstants push;
    push.value[0] = elements;
    push.value[1] = source_offset;
    push.value[2] = destination_offset;
    push.value[3] = sample_type;
    push.value[16] = status_offset;
    if (conversion) {
        push.value[4] = std::bit_cast<std::uint32_t>(conversion->output_scale);
        push.value[5] = std::bit_cast<std::uint32_t>(conversion->output_offset);
        push.value[6] = conversion->output_maximum;
    }
    const VkDescriptorSet set = slot.descriptor_set(
        descriptor_plan.buffer(), descriptor_plan.buffer_offset(),
        descriptor_plan.buffer_size(), VK_NULL_HANDLE, 0U, 0U);
    bind_compute(runtime, slot, runtime.convert_f64_pipeline, set, push);
    const std::uint32_t samples_per_word = sample_type == 0U ? 1U
        : sample_type == 1U ? 4U : 2U;
    vkCmdDispatch(
        slot.command,
        divide_up(divide_up(elements, samples_per_word), 256U), 1U, 1U);
}

template <class Slot>
void clear_f64_status(Slot &slot, std::uint32_t status_offset) {
    const VkDeviceSize byte_offset =
        static_cast<VkDeviceSize>(status_offset) * 4U;
    vkCmdFillBuffer(slot.command, slot.workspace.handle(), byte_offset, 4U, 0U);
    command_barrier(
        slot.command, slot.workspace.handle(), byte_offset, 4U,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

template <class Slot>
void copy_f64_status(
    Slot &slot, std::uint32_t status_offset,
    Buffer &staging, VkDeviceSize staging_offset) {
    const VkDeviceSize byte_offset =
        static_cast<VkDeviceSize>(status_offset) * 4U;
    command_barrier(
        slot.command, slot.workspace.handle(), byte_offset, 4U,
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkBufferCopy copy{byte_offset, staging_offset, 4U};
    vkCmdCopyBuffer(
        slot.command, slot.workspace.handle(), staging.handle(), 1U, &copy);
}

void require_f64_status_clear(const void *address) {
    std::uint32_t status = 0U;
    std::memcpy(&status, address, sizeof(status));
    if (status != 0U) {
        throw std::runtime_error(
            "Vulkan Float64 execution produced NaN or infinity");
    }
}

class WorkspaceBuilder {
public:
    [[nodiscard]] std::uint32_t add_bytes(std::size_t bytes) {
        const std::size_t aligned = align_up(bytes, 4U, "Vulkan workspace slice");
        const std::uint32_t result = checked_u32(
            cursor_ / 4U, "Vulkan workspace offset");
        cursor_ = checked_add(cursor_, aligned, "Vulkan workspace");
        return result;
    }
    [[nodiscard]] std::size_t size() const noexcept { return cursor_; }

private:
    std::size_t cursor_ = 0U;
};

template <class Sample>
[[nodiscard]] bool valid_conversion(
    const IntegerConversion &conversion) noexcept {
    return conversion.input_scale > 0.0F
        && conversion.output_scale > 0.0F
        && std::isfinite(conversion.input_offset)
        && std::isfinite(conversion.input_scale)
        && std::isfinite(conversion.output_scale)
        && std::isfinite(conversion.output_offset)
        && conversion.output_maximum != 0U
        && conversion.output_maximum
            <= static_cast<std::uint32_t>(std::numeric_limits<Sample>::max());
}

} // namespace

namespace {

struct ExecutionSlot {
    explicit ExecutionSlot(Runtime &requested_runtime)
        : runtime(requested_runtime) {
        const VkCommandPoolCreateInfo pool_info{
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            nullptr,
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            runtime.selected.queue_family,
        };
        runtime.check(vkCreateCommandPool(
                          runtime.device, &pool_info, nullptr, &command_pool),
                      "vkCreateCommandPool(execution slot)");
        try {
            const VkCommandBufferAllocateInfo command_info{
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                nullptr,
                command_pool,
                VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                1U,
            };
            runtime.check(vkAllocateCommandBuffers(
                              runtime.device, &command_info, &command),
                          "vkAllocateCommandBuffers(execution slot)");
            const VkFenceCreateInfo fence_info{
                VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                nullptr,
                VK_FENCE_CREATE_SIGNALED_BIT,
            };
            runtime.check(vkCreateFence(
                              runtime.device, &fence_info, nullptr, &fence),
                          "vkCreateFence(execution slot)");

            const VkDescriptorPoolSize pool_size{
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                static_cast<std::uint32_t>(descriptor_set_count * 3U),
            };
            const VkDescriptorPoolCreateInfo descriptor_pool_info{
                VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                nullptr,
                0U,
                static_cast<std::uint32_t>(descriptor_set_count),
                1U,
                &pool_size,
            };
            runtime.check(vkCreateDescriptorPool(
                              runtime.device, &descriptor_pool_info, nullptr,
                              &descriptor_pool),
                          "vkCreateDescriptorPool(execution slot)");
            std::array<VkDescriptorSetLayout, descriptor_set_count> layouts{};
            layouts.fill(runtime.descriptor_layout);
            const VkDescriptorSetAllocateInfo descriptor_info{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                nullptr,
                descriptor_pool,
                static_cast<std::uint32_t>(layouts.size()),
                layouts.data(),
            };
            runtime.check(vkAllocateDescriptorSets(
                              runtime.device, &descriptor_info,
                              descriptor_sets.data()),
                          "vkAllocateDescriptorSets(execution slot)");
        } catch (...) {
            destroy_handles();
            throw;
        }
    }

    ~ExecutionSlot() { destroy_handles(); }
    ExecutionSlot(const ExecutionSlot &) = delete;
    ExecutionSlot &operator=(const ExecutionSlot &) = delete;

    void destroy_handles() noexcept {
        workspace.reset();
        staging.reset();
        if (descriptor_pool) {
            vkDestroyDescriptorPool(runtime.device, descriptor_pool, nullptr);
        }
        descriptor_pool = VK_NULL_HANDLE;
        if (fence) vkDestroyFence(runtime.device, fence, nullptr);
        fence = VK_NULL_HANDLE;
        if (command_pool) {
            vkDestroyCommandPool(runtime.device, command_pool, nullptr);
        }
        command_pool = VK_NULL_HANDLE;
        command = VK_NULL_HANDLE;
    }

    void reserve(VkDeviceSize workspace_size, VkDeviceSize staging_size) {
        runtime.throw_if_failed();
        if (workspace_size == 0U
            || workspace_size > runtime.selected.properties.limits.maxStorageBufferRange
            || workspace_size > maximum_allocation_bytes) {
            throw std::length_error(
                "Vulkan workspace exceeds maxStorageBufferRange or the 2 GiB guard");
        }
        if (!workspace.handle() || workspace.size() < workspace_size) {
            const VkDeviceSize maximum_workspace = std::min<VkDeviceSize>(
                maximum_allocation_bytes,
                runtime.selected.properties.limits.maxStorageBufferRange);
            const VkDeviceSize capacity = std::min<VkDeviceSize>(
                maximum_workspace,
                std::max<VkDeviceSize>(workspace_size, workspace.size() * 2U));
            workspace = Buffer(
                runtime, capacity,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0U,
                "dsmvc execution workspace");
        }
        if (!staging.handle() || staging.size() < staging_size) {
            const VkDeviceSize capacity = std::min<VkDeviceSize>(
                maximum_allocation_bytes,
                std::max<VkDeviceSize>(staging_size, staging.size() * 2U));
            staging = Buffer(
                runtime, capacity,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                    | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                "dsmvc mapped staging");
        }
    }

    void begin() {
        runtime.wait_fence(fence, "Vulkan execution slot reuse");
        runtime.check(vkResetFences(runtime.device, 1U, &fence),
                      "vkResetFences");
        runtime.check(vkResetCommandPool(runtime.device, command_pool, 0U),
                      "vkResetCommandPool");
        const VkCommandBufferBeginInfo begin_info{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            nullptr,
        };
        runtime.check(vkBeginCommandBuffer(command, &begin_info),
                      "vkBeginCommandBuffer");
        descriptor_cursor = 0U;
    }

    void recover_unsubmitted() noexcept {
        (void)vkResetCommandPool(runtime.device, command_pool, 0U);
        if (fence) vkDestroyFence(runtime.device, fence, nullptr);
        fence = VK_NULL_HANDLE;
        const VkFenceCreateInfo fence_info{
            VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            nullptr,
            VK_FENCE_CREATE_SIGNALED_BIT,
        };
        (void)vkCreateFence(runtime.device, &fence_info, nullptr, &fence);
    }

    [[nodiscard]] VkDescriptorSet descriptor_set(
        VkBuffer plan_buffer, VkDeviceSize plan_offset, VkDeviceSize plan_size,
        VkBuffer cache_buffer, VkDeviceSize cache_offset,
        VkDeviceSize cache_size) {
        if (descriptor_cursor >= descriptor_sets.size()) {
            throw std::logic_error("Vulkan execution exceeded its descriptor-set budget");
        }
        const VkDescriptorSet set = descriptor_sets[descriptor_cursor++];
        if (!cache_buffer) {
            cache_buffer = workspace.handle();
            cache_offset = 0U;
            cache_size = workspace.size();
        }
        const std::array<VkDescriptorBufferInfo, 3U> buffers{{
            {plan_buffer, plan_offset, plan_size},
            {workspace.handle(), 0U, workspace.size()},
            {cache_buffer, cache_offset, cache_size},
        }};
        std::array<VkWriteDescriptorSet, 3U> writes{};
        for (std::uint32_t index = 0U; index < writes.size(); ++index) {
            writes[index] = {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                set,
                index,
                0U,
                1U,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                nullptr,
                &buffers[index],
                nullptr,
            };
        }
        vkUpdateDescriptorSets(
            runtime.device, static_cast<std::uint32_t>(writes.size()),
            writes.data(), 0U, nullptr);
        return set;
    }

    Runtime &runtime;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, descriptor_set_count> descriptor_sets{};
    std::size_t descriptor_cursor = 0U;
    Buffer workspace;
    Buffer staging;
};

enum class CachedInputLayout : std::uint8_t {
    row_major,
    transposed,
    axis_f64,
    transposed_f64,
};

enum class CachedInputSample : std::uint8_t {
    float32,
    uint8,
    uint16,
};

struct InputCacheKey {
    std::uintptr_t data = 0U;
    std::ptrdiff_t row_stride = 0;
    std::uint32_t width = 0U;
    std::uint32_t height = 0U;
    std::uint32_t input_offset = 0U;
    std::uint32_t input_scale = 0U;
    CachedInputLayout layout = CachedInputLayout::row_major;
    CachedInputSample sample = CachedInputSample::float32;

    bool operator==(const InputCacheKey &) const noexcept = default;
};

struct InputCacheKeyHash {
    [[nodiscard]] std::size_t operator()(const InputCacheKey &key) const noexcept {
        std::size_t value = std::hash<std::uintptr_t>{}(key.data);
        const auto mix = [&value](std::size_t item) {
            value ^= item + 0x9e3779b97f4a7c15ULL
                + (value << 6U) + (value >> 2U);
        };
        mix(std::hash<std::ptrdiff_t>{}(key.row_stride));
        mix(key.width);
        mix(key.height);
        mix(key.input_offset);
        mix(key.input_scale);
        mix(static_cast<std::size_t>(key.layout));
        mix(static_cast<std::size_t>(key.sample));
        return value;
    }
};

class CachedInput {
public:
    CachedInput(DeviceArena &arena, VkDeviceSize bytes,
                std::shared_ptr<const void> requested_lifetime)
        : storage_(arena.allocate(bytes)), lifetime_(std::move(requested_lifetime)),
          bytes_(bytes) {}

    [[nodiscard]] VkBuffer buffer() const noexcept { return storage_.buffer(); }
    [[nodiscard]] VkDeviceSize offset() const noexcept { return storage_.offset(); }
    [[nodiscard]] VkDeviceSize size() const noexcept { return storage_.size(); }
    [[nodiscard]] VkDeviceSize bytes() const noexcept { return bytes_; }

    void publish(std::uint64_t timeline_value) noexcept {
        {
            const std::scoped_lock lock(mutex_);
            timeline_value_ = timeline_value;
            scheduled_ = true;
        }
        scheduled_ready_.notify_all();
    }

    void fail(std::exception_ptr error) noexcept {
        {
            const std::scoped_lock lock(mutex_);
            error_ = std::move(error);
            scheduled_ = true;
        }
        scheduled_ready_.notify_all();
    }

    [[nodiscard]] std::uint64_t wait_value() const {
        std::unique_lock lock(mutex_);
        scheduled_ready_.wait(lock, [&] { return scheduled_; });
        if (error_) std::rethrow_exception(error_);
        return timeline_value_;
    }

private:
    DeviceArena::Allocation storage_;
    std::shared_ptr<const void> lifetime_;
    VkDeviceSize bytes_ = 0U;
    mutable std::mutex mutex_;
    mutable std::condition_variable scheduled_ready_;
    bool scheduled_ = false;
    std::uint64_t timeline_value_ = 0U;
    std::exception_ptr error_;
};

class InputCache {
public:
    struct Acquisition {
        std::shared_ptr<CachedInput> input;
        bool producer = false;
    };

    explicit InputCache(DeviceArena &arena)
        : arena_(arena), capacity_(environment_megabytes(
              "DSMVC_VULKAN_INPUT_CACHE_MB", default_input_cache_bytes)) {}

    [[nodiscard]] bool enabled() const noexcept { return capacity_ != 0U; }

    [[nodiscard]] Acquisition acquire(
        const InputCacheKey &key, VkDeviceSize bytes,
        std::shared_ptr<const void> lifetime) {
        std::vector<std::shared_ptr<CachedInput>> evicted;
        const std::scoped_lock lock(mutex_);
        if (const auto found = entries_.find(key); found != entries_.end()) {
            lru_.splice(lru_.end(), lru_, found->second.lru);
            return {found->second.input, false};
        }
        auto input = std::make_shared<CachedInput>(
            arena_, bytes, std::move(lifetime));
        bytes_ = checked_add(
            bytes_, static_cast<std::size_t>(bytes), "Vulkan input cache");
        lru_.push_back(key);
        entries_.emplace(
            key, Entry{input, std::prev(lru_.end())});
        while (bytes_ > capacity_ && entries_.size() > 1U) {
            const InputCacheKey victim_key = lru_.front();
            const auto victim = entries_.find(victim_key);
            bytes_ -= static_cast<std::size_t>(victim->second.input->bytes());
            evicted.push_back(std::move(victim->second.input));
            lru_.pop_front();
            entries_.erase(victim);
        }
        return {std::move(input), true};
    }

    void erase(const InputCacheKey &key,
               const std::shared_ptr<CachedInput> &expected) noexcept {
        std::shared_ptr<CachedInput> removed;
        const std::scoped_lock lock(mutex_);
        const auto found = entries_.find(key);
        if (found == entries_.end() || found->second.input != expected) return;
        bytes_ -= static_cast<std::size_t>(found->second.input->bytes());
        lru_.erase(found->second.lru);
        removed = std::move(found->second.input);
        entries_.erase(found);
    }

    void reset() noexcept {
        decltype(entries_) removed;
        {
            const std::scoped_lock lock(mutex_);
            removed.swap(entries_);
            lru_.clear();
            bytes_ = 0U;
        }
    }

private:
    struct Entry {
        std::shared_ptr<CachedInput> input;
        std::list<InputCacheKey>::iterator lru;
    };
    DeviceArena &arena_;
    std::mutex mutex_;
    std::unordered_map<InputCacheKey, Entry, InputCacheKeyHash> entries_;
    std::list<InputCacheKey> lru_;
    std::size_t bytes_ = 0U;
    const std::size_t capacity_ = 0U;
};

class StagingPool {
public:
    struct Phase {
        explicit Phase(Runtime &runtime) : runtime(runtime) {}

        void reserve(VkDeviceSize requested) {
            if (buffer.handle() && buffer.size() >= requested) return;
            const VkDeviceSize capacity = std::min<VkDeviceSize>(
                maximum_allocation_bytes,
                std::max<VkDeviceSize>(requested, buffer.size() * 2U));
            buffer = Buffer(
                runtime, capacity,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                    | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                "dsmvc decoupled Float32 staging");
        }

        Runtime &runtime;
        Buffer buffer;
    };

    class Allocation {
    public:
        Allocation() = default;
        Allocation(StagingPool &pool, std::size_t index, bool limited) noexcept
            : pool_(&pool), index_(index), limited_(limited) {}
        ~Allocation() { reset(); }
        Allocation(const Allocation &) = delete;
        Allocation &operator=(const Allocation &) = delete;
        Allocation(Allocation &&other) noexcept
            : pool_(std::exchange(other.pool_, nullptr)),
              index_(other.index_), limited_(other.limited_) {}
        Allocation &operator=(Allocation &&other) noexcept {
            if (this != &other) {
                reset();
                pool_ = std::exchange(other.pool_, nullptr);
                index_ = other.index_;
                limited_ = other.limited_;
            }
            return *this;
        }

        void reset() noexcept {
            if (!pool_) return;
            pool_->release(index_, limited_);
            pool_ = nullptr;
        }
        void reserve(VkDeviceSize bytes) { pool_->phases_[index_]->reserve(bytes); }
        [[nodiscard]] Buffer &buffer() noexcept {
            return pool_->phases_[index_]->buffer;
        }

    private:
        StagingPool *pool_ = nullptr;
        std::size_t index_ = 0U;
        bool limited_ = false;
    };

    StagingPool(Runtime &runtime, std::size_t count, std::size_t limited)
        : runtime_(runtime), limited_count_(limited), busy_(count, false) {
        phases_.reserve(count);
        for (std::size_t index = 0U; index < count; ++index) {
            phases_.push_back(std::make_unique<Phase>(runtime));
        }
    }

    [[nodiscard]] Allocation acquire(bool limited) {
        runtime_.throw_if_failed();
        std::unique_lock lock(mutex_);
        available_.wait(lock, [&] {
            return runtime_.failed()
                || ((!limited || limited_busy_ < limited_count_)
                    && std::ranges::find(busy_, false) != busy_.end());
        });
        runtime_.throw_if_failed();
        const auto found = std::ranges::find(busy_, false);
        *found = true;
        if (limited) ++limited_busy_;
        return Allocation{
            *this, static_cast<std::size_t>(found - busy_.begin()), limited};
    }

    void notify_failure() noexcept { available_.notify_all(); }

private:
    void release(std::size_t index, bool limited) noexcept {
        {
            const std::scoped_lock lock(mutex_);
            busy_[index] = false;
            if (limited && limited_busy_ != 0U) --limited_busy_;
        }
        available_.notify_one();
    }

    friend class Allocation;
    Runtime &runtime_;
    std::vector<std::unique_ptr<Phase>> phases_;
    std::size_t limited_count_ = 4U;
    std::vector<bool> busy_;
    std::size_t limited_busy_ = 0U;
    std::mutex mutex_;
    std::condition_variable available_;
};

template <class Sample>
[[nodiscard]] InputCacheKey make_input_cache_key(
    const Sample *input, std::ptrdiff_t row_stride,
    std::uint32_t width, std::uint32_t height, CachedInputLayout layout,
    const IntegerConversion *conversion = nullptr) noexcept {
    CachedInputSample sample = CachedInputSample::float32;
    if constexpr (std::is_same_v<Sample, std::uint8_t>) {
        sample = CachedInputSample::uint8;
    } else if constexpr (std::is_same_v<Sample, std::uint16_t>) {
        sample = CachedInputSample::uint16;
    }
    InputCacheKey key{
        reinterpret_cast<std::uintptr_t>(input), row_stride, width, height,
        0U, 0U, layout, sample};
    if (conversion) {
        key.input_offset = std::bit_cast<std::uint32_t>(conversion->input_offset);
        key.input_scale = std::bit_cast<std::uint32_t>(conversion->input_scale);
    }
    return key;
}

Runtime::Runtime()
    : instance(create_instance(environment_flag("DSMVC_VULKAN_VALIDATION", false))),
      selected(select_device(enumerate_devices(instance.instance))),
      split_rhs(split_rhs_mode()) {
    force_non_coherent = environment_flag(
        "DSMVC_VULKAN_FORCE_NON_COHERENT", false);
    const float priority = 1.0F;
    const VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        nullptr,
        0U,
        selected.queue_family,
        1U,
        &priority,
    };
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_support{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
        nullptr,
        VK_FALSE,
    };
    VkPhysicalDeviceFeatures2 features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        &timeline_support,
        {},
    };
    vkGetPhysicalDeviceFeatures2(selected.physical, &features);
    timeline_enabled = timeline_support.timelineSemaphore == VK_TRUE
        && environment_flag("DSMVC_VULKAN_TIMELINE", true);
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_enable{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
        nullptr,
        timeline_enabled ? VK_TRUE : VK_FALSE,
    };
    std::vector<const char *> device_extensions;
    if (environment_flag("DSMVC_VULKAN_DIRECT_IO", false)
        && device_has_extension(
            selected.physical, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) {
        device_extensions.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
        host_import_enabled = true;
    }
    VkPhysicalDeviceFeatures enabled_features{};
    enabled_features.shaderFloat64 =
        selected.float64.strict_supported() ? VK_TRUE : VK_FALSE;
    const VkDeviceCreateInfo device_info{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        timeline_enabled ? &timeline_enable : nullptr,
        0U,
        1U,
        &queue_info,
        0U,
        nullptr,
        static_cast<std::uint32_t>(device_extensions.size()),
        device_extensions.data(),
        &enabled_features,
    };
    check(vkCreateDevice(selected.physical, &device_info, nullptr, &device),
          "vkCreateDevice");
    finish_construction();
#ifdef __linux__
    // Threads that appeared since dsmvc first called the loader: the driver's.
    std::ranges::set_difference(thread_ids(), threads_before_vulkan(),
                                std::back_inserter(driver_threads_));
    std::erase(driver_threads_, static_cast<long>(syscall(SYS_gettid)));
#endif
}

Runtime::Runtime(const VulkanCoreDevice &borrowed)
    : instance(), selected(describe_device(
          static_cast<VkPhysicalDevice>(borrowed.physical_device), 0U)),
      split_rhs(split_rhs_mode()) {
    instance.instance = static_cast<VkInstance>(borrowed.instance);
    instance.owned = false;
    if (!borrowed.device || !instance.instance || !selected.physical) {
        throw std::invalid_argument("invalid borrowed Vulkan device");
    }
    if (!selected.eligible) {
        throw std::runtime_error(
            "the VapourSynth core's Vulkan device cannot run dsmvc: "
            + selected.reason);
    }
    force_non_coherent = environment_flag(
        "DSMVC_VULKAN_FORCE_NON_COHERENT", false);
    owns_device_ = false;
    device = static_cast<VkDevice>(borrowed.device);
    selected.queue_family = borrowed.queue_family;
    queue_index_ = borrowed.queue_index;
    lock_queue_ = borrowed.lock_queue;
    unlock_queue_ = borrowed.unlock_queue;
    queue_lock_user_ = borrowed.user;
    // Timeline semaphores are part of the Vulkan 1.4 baseline the core requires.
    timeline_enabled = true;
    finish_construction();
}

void Runtime::finish_construction() {
    try {
        vkGetDeviceQueue(device, selected.queue_family, queue_index_, &queue);
        if (host_import_enabled) {
            get_host_pointer_properties =
                reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
                    vkGetDeviceProcAddr(
                        device, "vkGetMemoryHostPointerPropertiesEXT"));
            VkPhysicalDeviceExternalMemoryHostPropertiesEXT host_properties{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT,
                nullptr, 0U};
            VkPhysicalDeviceProperties2 properties2{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &host_properties, {}};
            vkGetPhysicalDeviceProperties2(selected.physical, &properties2);
            host_import_alignment = host_properties.minImportedHostPointerAlignment;
            host_import_limit = std::min<std::uint32_t>(
                256U, std::max<std::uint32_t>(
                    1U, selected.properties.limits.maxMemoryAllocationCount / 8U));
            if (!get_host_pointer_properties || host_import_alignment == 0U) {
                host_import_enabled = false;
            }
        }
        set_object_name = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
            vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
        command_begin_label =
            reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
                vkGetDeviceProcAddr(device, "vkCmdBeginDebugUtilsLabelEXT"));
        command_end_label = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
            vkGetDeviceProcAddr(device, "vkCmdEndDebugUtilsLabelEXT"));
        name_object(object_handle(queue), VK_OBJECT_TYPE_QUEUE, "dsmvc compute queue");

        const std::array<VkDescriptorSetLayoutBinding, 3U> bindings{{
            {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U,
             VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U,
             VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U,
             VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        }};
        const VkDescriptorSetLayoutCreateInfo descriptor_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            nullptr,
            0U,
            static_cast<std::uint32_t>(bindings.size()),
            bindings.data(),
        };
        check(vkCreateDescriptorSetLayout(
                  device, &descriptor_info, nullptr, &descriptor_layout),
              "vkCreateDescriptorSetLayout");
        const VkPushConstantRange push_range{
            VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof(PushConstants)};
        const VkPipelineLayoutCreateInfo layout_info{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            nullptr,
            0U,
            1U,
            &descriptor_layout,
            1U,
            &push_range,
        };
        check(vkCreatePipelineLayout(
                  device, &layout_info, nullptr, &pipeline_layout),
              "vkCreatePipelineLayout");
        const VkPipelineCacheCreateInfo cache_info{
            VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            nullptr,
            0U,
            0U,
            nullptr,
        };
        check(vkCreatePipelineCache(
                  device, &cache_info, nullptr, &pipeline_cache),
              "vkCreatePipelineCache");

        bool use_256 = selected.properties.limits.maxComputeWorkGroupInvocations >= 256U
            && selected.properties.limits.maxComputeWorkGroupSize[0] >= 256U
            && selected.properties.limits.maxComputeWorkGroupSize[1] >= 8U;
        if (const char *workgroup = std::getenv("DSMVC_VULKAN_WORKGROUP")) {
            if (std::strcmp(workgroup, "128") == 0) use_256 = false;
            if (std::strcmp(workgroup, "256") == 0 && !use_256) {
                throw std::runtime_error(
                    "DSMVC_VULKAN_WORKGROUP=256 exceeds the selected device limits");
            }
        }
        if (use_256) {
            transpose_pipeline = create_pipeline(
                embedded::vulkan_transpose_256_spv,
                sizeof(embedded::vulkan_transpose_256_spv), "dsmvc transpose 32x8");
            rhs_pipeline = create_pipeline(
                embedded::vulkan_rhs_256_spv,
                sizeof(embedded::vulkan_rhs_256_spv), "dsmvc split RHS 32x8");
            convert_pipeline = create_pipeline(
                embedded::vulkan_convert_256_spv,
                sizeof(embedded::vulkan_convert_256_spv), "dsmvc conversion 256");
        } else {
            transpose_local_x = 16U;
            rhs_local_x = 16U;
            convert_local_size = 128U;
            transpose_pipeline = create_pipeline(
                embedded::vulkan_transpose_128_spv,
                sizeof(embedded::vulkan_transpose_128_spv), "dsmvc transpose 16x8");
            rhs_pipeline = create_pipeline(
                embedded::vulkan_rhs_128_spv,
                sizeof(embedded::vulkan_rhs_128_spv), "dsmvc split RHS 16x8");
            convert_pipeline = create_pipeline(
                embedded::vulkan_convert_128_spv,
                sizeof(embedded::vulkan_convert_128_spv), "dsmvc conversion 128");
        }

        const bool inverse_64 = selected.properties.vendorID == 0x1002U
            && selected.properties.limits.maxComputeWorkGroupSize[0] >= 64U;
        if (inverse_64) {
            inverse_local_size = 64U;
            inverse_pipeline = create_pipeline(
                embedded::vulkan_inverse_64_spv,
                sizeof(embedded::vulkan_inverse_64_spv), "dsmvc fused inverse 64");
            solve_pipeline = create_pipeline(
                embedded::vulkan_solve_64_spv,
                sizeof(embedded::vulkan_solve_64_spv), "dsmvc split solve 64");
        } else {
            inverse_pipeline = create_pipeline(
                embedded::vulkan_inverse_32_spv,
                sizeof(embedded::vulkan_inverse_32_spv), "dsmvc fused inverse 32");
            solve_pipeline = create_pipeline(
                embedded::vulkan_solve_32_spv,
                sizeof(embedded::vulkan_solve_32_spv), "dsmvc split solve 32");
            fixed_inverse_pipelines[0] = create_pipeline(
                embedded::vulkan_inverse_b1_32_spv,
                sizeof(embedded::vulkan_inverse_b1_32_spv),
                "dsmvc fused inverse B1 32");
            fixed_inverse_pipelines[1] = create_pipeline(
                embedded::vulkan_inverse_b3_32_spv,
                sizeof(embedded::vulkan_inverse_b3_32_spv),
                "dsmvc fused inverse B3 32");
            fixed_inverse_pipelines[2] = create_pipeline(
                embedded::vulkan_inverse_b5_32_spv,
                sizeof(embedded::vulkan_inverse_b5_32_spv),
                "dsmvc fused inverse B5 32");
            fixed_inverse_pipelines[3] = create_pipeline(
                embedded::vulkan_inverse_b7_32_spv,
                sizeof(embedded::vulkan_inverse_b7_32_spv),
                "dsmvc fused inverse B7 32");
            fixed_inverse_pipelines[4] = create_pipeline(
                embedded::vulkan_inverse_b9_32_spv,
                sizeof(embedded::vulkan_inverse_b9_32_spv),
                "dsmvc fused inverse B9 32");
            fixed_inverse_pipelines[5] = create_pipeline(
                embedded::vulkan_inverse_b11_32_spv,
                sizeof(embedded::vulkan_inverse_b11_32_spv),
                "dsmvc fused inverse B11 32");
            fixed_solve_pipelines[0] = create_pipeline(
                embedded::vulkan_solve_b1_32_spv,
                sizeof(embedded::vulkan_solve_b1_32_spv),
                "dsmvc split solve B1 32");
            fixed_solve_pipelines[1] = create_pipeline(
                embedded::vulkan_solve_b3_32_spv,
                sizeof(embedded::vulkan_solve_b3_32_spv),
                "dsmvc split solve B3 32");
            fixed_solve_pipelines[2] = create_pipeline(
                embedded::vulkan_solve_b5_32_spv,
                sizeof(embedded::vulkan_solve_b5_32_spv),
                "dsmvc split solve B5 32");
            fixed_solve_pipelines[3] = create_pipeline(
                embedded::vulkan_solve_b7_32_spv,
                sizeof(embedded::vulkan_solve_b7_32_spv),
                "dsmvc split solve B7 32");
            fixed_solve_pipelines[4] = create_pipeline(
                embedded::vulkan_solve_b9_32_spv,
                sizeof(embedded::vulkan_solve_b9_32_spv),
                "dsmvc split solve B9 32");
            fixed_solve_pipelines[5] = create_pipeline(
                embedded::vulkan_solve_b11_32_spv,
                sizeof(embedded::vulkan_solve_b11_32_spv),
                "dsmvc split solve B11 32");
        }

        if (selected.float64.strict_supported()) {
            transpose_f64_pipeline = create_pipeline(
                embedded::vulkan_transpose_f64_spv,
                sizeof(embedded::vulkan_transpose_f64_spv),
                "dsmvc Float64 transpose 32x8");
            rhs_f64_pipeline = create_pipeline(
                embedded::vulkan_rhs_f64_spv,
                sizeof(embedded::vulkan_rhs_f64_spv),
                "dsmvc Float64 split RHS 32x8");
            inverse_f64_pipeline = create_pipeline(
                embedded::vulkan_inverse_f64_spv,
                sizeof(embedded::vulkan_inverse_f64_spv),
                "dsmvc Float64 fused inverse 32");
            solve_f64_pipeline = create_pipeline(
                embedded::vulkan_solve_f64_spv,
                sizeof(embedded::vulkan_solve_f64_spv),
                "dsmvc Float64 split solve 32");
            convert_f64_pipeline = create_pipeline(
                embedded::vulkan_convert_f64_spv,
                sizeof(embedded::vulkan_convert_f64_spv),
                "dsmvc Float64 conversion 256");
        }

        if (timeline_enabled) {
            const VkSemaphoreTypeCreateInfo type_info{
                VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                nullptr,
                VK_SEMAPHORE_TYPE_TIMELINE,
                0U,
            };
            const VkSemaphoreCreateInfo semaphore_info{
                VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                &type_info,
                0U,
            };
            check(vkCreateSemaphore(
                      device, &semaphore_info, nullptr, &timeline),
                  "vkCreateSemaphore(timeline)");
            plan_upload_pool = std::make_unique<PlanUploadPool>(*this);
        }

        plan_arena = std::make_unique<DeviceArena>(
            *this, "dsmvc packed-plan arena");
        input_arena = std::make_unique<DeviceArena>(
            *this, "dsmvc input-cache arena");
        input_cache = std::make_unique<InputCache>(*input_arena);

        const auto configuration = slot_configuration();
        limited_slot_count_ = configuration.limited;
        maximum_slot_count_ = configuration.total;
        adaptive_slots_ = configuration.adaptive;
        staging_pool = std::make_unique<StagingPool>(
            *this,
            std::min<std::size_t>(
                32U, std::max<std::size_t>(4U, maximum_slot_count_ * 2U)),
            4U);
        slots.reserve(maximum_slot_count_);
        slot_busy.assign(maximum_slot_count_, false);
        for (std::size_t index = 0U; index < maximum_slot_count_; ++index) {
            slots.push_back(std::make_unique<ExecutionSlot>(*this));
        }
    } catch (...) {
        destroy_device_objects();
        throw;
    }
    if (host_import_enabled) {
        BlockRegistrar::runtime = this;
        host_blocks::install(
            host_blocks::Domain::vulkan,
            {&BlockRegistrar::register_block, &BlockRegistrar::unregister_block});
    }
}

Runtime::~Runtime() {
    // Normally a no-op: the wait already ran when the last filter was freed.
    if (owns_device_ && device) wait_for_driver_threads(driver_threads_, "teardown");
    destroy_device_objects();
}

void Runtime::destroy_device_objects() noexcept {
    if (!device) return;
    // A borrowed device is the core's: nothing to unregister (only the owned
    // runtime installs the host import registrar), no device-wide wait, and
    // the device is not ours to destroy. Executions are synchronous, so no
    // submission of ours is pending.
    if (owns_device_) {
        host_blocks::uninstall(host_blocks::Domain::vulkan);
        // Normally skipped: the device went idle when the last filter was
        // freed (detach_executor). This runs in an exit handler, where a
        // layer's own static state may be gone already: synchronization
        // validation crashed or hung in this call.
        if (submitted_since_idle_.exchange(false)) {
            (void)vkDeviceWaitIdle(device);
        }
    }
    slots.clear();
    staging_pool.reset();
    plan_upload_pool.reset();
    if (input_cache) input_cache->reset();
    input_cache.reset();
    if (input_arena) input_arena->reset();
    if (plan_arena) plan_arena->reset();
    input_arena.reset();
    plan_arena.reset();
    if (timeline) vkDestroySemaphore(device, timeline, nullptr);
    timeline = VK_NULL_HANDLE;
    for (VkPipeline &pipeline : fixed_inverse_pipelines) {
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    }
    for (VkPipeline &pipeline : fixed_solve_pipelines) {
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    }
    for (auto &by_chunks : chunked_pipelines) {
        for (VkPipeline &pipeline : by_chunks) {
            if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
    }
    for (VkPipeline *pipeline : {
             &convert_f64_pipeline, &solve_f64_pipeline, &rhs_f64_pipeline,
             &inverse_f64_pipeline, &transpose_f64_pipeline,
             &convert_pipeline, &solve_pipeline, &rhs_pipeline,
             &inverse_pipeline, &transpose_pipeline}) {
        if (*pipeline) vkDestroyPipeline(device, *pipeline, nullptr);
        *pipeline = VK_NULL_HANDLE;
    }
    if (pipeline_cache) vkDestroyPipelineCache(device, pipeline_cache, nullptr);
    pipeline_cache = VK_NULL_HANDLE;
    if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    pipeline_layout = VK_NULL_HANDLE;
    if (descriptor_layout) {
        vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    }
    descriptor_layout = VK_NULL_HANDLE;
    if (owns_device_) vkDestroyDevice(device, nullptr);
    device = VK_NULL_HANDLE;
    queue = VK_NULL_HANDLE;
}

void Runtime::check(VkResult result, const char *label) {
    if (result == VK_SUCCESS) return;
    const std::string message = std::string{label} + " failed with "
        + result_name(result) + " ("
        + std::to_string(static_cast<int>(result)) + ")";
    if (result == VK_ERROR_DEVICE_LOST) {
        {
            const std::scoped_lock lock(failure_mutex_);
            if (failure_message_.empty()) failure_message_ = message;
        }
        failed_.store(true, std::memory_order_release);
        slot_available_.notify_all();
        if (staging_pool) staging_pool->notify_failure();
    }
    throw std::runtime_error(message);
}

void Runtime::throw_if_failed() const {
    if (!failed_.load(std::memory_order_acquire)) return;
    const std::scoped_lock lock(failure_mutex_);
    throw std::runtime_error(
        "Vulkan runtime is unavailable after device loss: " + failure_message_);
}

std::uint32_t Runtime::memory_type(
    std::uint32_t bits, VkMemoryPropertyFlags required,
    VkMemoryPropertyFlags preferred) const {
    std::optional<std::uint32_t> best;
    int best_score = -1;
    for (std::uint32_t index = 0U;
         index < selected.memory.memoryTypeCount; ++index) {
        if (!(bits & (1U << index))) continue;
        const auto flags = selected.memory.memoryTypes[index].propertyFlags;
        if ((flags & required) != required) continue;
        if (force_non_coherent
            && (required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            && (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            continue;
        }
        const int score = std::popcount(static_cast<unsigned int>(flags & preferred));
        if (score > best_score) {
            best = index;
            best_score = score;
        }
    }
    if (!best && force_non_coherent
        && (required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
        for (std::uint32_t index = 0U;
             index < selected.memory.memoryTypeCount; ++index) {
            if (!(bits & (1U << index))) continue;
            const auto flags = selected.memory.memoryTypes[index].propertyFlags;
            if ((flags & required) == required) return index;
        }
    }
    if (!best) throw std::runtime_error("no compatible Vulkan memory type");
    return *best;
}

void Runtime::name_object(
    std::uint64_t object, VkObjectType type, const char *name) const noexcept {
    if (!set_object_name || object == 0U || !name) return;
    const VkDebugUtilsObjectNameInfoEXT info{
        VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
        nullptr,
        type,
        object,
        name,
    };
    (void)set_object_name(device, &info);
}

void Runtime::begin_label(VkCommandBuffer command, const char *name) const noexcept {
    if (!command_begin_label) return;
    const VkDebugUtilsLabelEXT label{
        VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
        nullptr,
        name,
        {0.15F, 0.55F, 0.85F, 1.0F},
    };
    command_begin_label(command, &label);
}

void Runtime::end_label(VkCommandBuffer command) const noexcept {
    if (command_end_label) command_end_label(command);
}

VkPipeline Runtime::create_pipeline(
    const std::uint32_t *code, std::size_t bytes, const char *name) {
    const VkShaderModuleCreateInfo module_info{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        nullptr,
        0U,
        bytes,
        code,
    };
    VkShaderModule module = VK_NULL_HANDLE;
    check(vkCreateShaderModule(device, &module_info, nullptr, &module),
          "vkCreateShaderModule");
    VkPipeline pipeline = VK_NULL_HANDLE;
    try {
        const VkPipelineShaderStageCreateInfo stage{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0U,
            VK_SHADER_STAGE_COMPUTE_BIT,
            module,
            "main",
            nullptr,
        };
        const VkComputePipelineCreateInfo pipeline_info{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            nullptr,
            0U,
            stage,
            pipeline_layout,
            VK_NULL_HANDLE,
            -1,
        };
        check(vkCreateComputePipelines(
                  device, pipeline_cache, 1U, &pipeline_info, nullptr, &pipeline),
              "vkCreateComputePipelines");
        name_object(object_handle(pipeline), VK_OBJECT_TYPE_PIPELINE, name);
    } catch (...) {
        vkDestroyShaderModule(device, module, nullptr);
        throw;
    }
    vkDestroyShaderModule(device, module, nullptr);
    return pipeline;
}

VkPipeline Runtime::chunked_solve_pipeline(
    std::uint32_t bandwidth, std::uint32_t chunks) {
    static constexpr std::array<std::uint32_t, 4U> bandwidths{1U, 3U, 5U, 7U};
    const auto found = std::ranges::find(bandwidths, bandwidth);
    const unsigned exponent = static_cast<unsigned>(std::countr_zero(chunks));
    if (found == bandwidths.end() || !std::has_single_bit(chunks)
        || exponent < 1U || exponent > 5U) {
        throw std::logic_error("unsupported Vulkan partitioned solve shape");
    }
    const auto row = static_cast<std::size_t>(found - bandwidths.begin());
    const std::scoped_lock lock(chunked_mutex);
    VkPipeline &pipeline = chunked_pipelines[row][exponent - 1U];
    if (pipeline) return pipeline;
    // [bandwidth 1/3/5/7][log2(chunks) - 1]
    static const std::array<std::array<std::pair<const std::uint32_t *, std::size_t>, 5U>, 4U>
        code{{
            {{
                {embedded::vulkan_solve_chunked_b1_c2_spv,
                 sizeof(embedded::vulkan_solve_chunked_b1_c2_spv)},
                {embedded::vulkan_solve_chunked_b1_c4_spv,
                 sizeof(embedded::vulkan_solve_chunked_b1_c4_spv)},
                {embedded::vulkan_solve_chunked_b1_c8_spv,
                 sizeof(embedded::vulkan_solve_chunked_b1_c8_spv)},
                {embedded::vulkan_solve_chunked_b1_c16_spv,
                 sizeof(embedded::vulkan_solve_chunked_b1_c16_spv)},
                {embedded::vulkan_solve_chunked_b1_c32_spv,
                 sizeof(embedded::vulkan_solve_chunked_b1_c32_spv)},
            }},
            {{
                {embedded::vulkan_solve_chunked_b3_c2_spv,
                 sizeof(embedded::vulkan_solve_chunked_b3_c2_spv)},
                {embedded::vulkan_solve_chunked_b3_c4_spv,
                 sizeof(embedded::vulkan_solve_chunked_b3_c4_spv)},
                {embedded::vulkan_solve_chunked_b3_c8_spv,
                 sizeof(embedded::vulkan_solve_chunked_b3_c8_spv)},
                {embedded::vulkan_solve_chunked_b3_c16_spv,
                 sizeof(embedded::vulkan_solve_chunked_b3_c16_spv)},
                {embedded::vulkan_solve_chunked_b3_c32_spv,
                 sizeof(embedded::vulkan_solve_chunked_b3_c32_spv)},
            }},
            {{
                {embedded::vulkan_solve_chunked_b5_c2_spv,
                 sizeof(embedded::vulkan_solve_chunked_b5_c2_spv)},
                {embedded::vulkan_solve_chunked_b5_c4_spv,
                 sizeof(embedded::vulkan_solve_chunked_b5_c4_spv)},
                {embedded::vulkan_solve_chunked_b5_c8_spv,
                 sizeof(embedded::vulkan_solve_chunked_b5_c8_spv)},
                {embedded::vulkan_solve_chunked_b5_c16_spv,
                 sizeof(embedded::vulkan_solve_chunked_b5_c16_spv)},
                {embedded::vulkan_solve_chunked_b5_c32_spv,
                 sizeof(embedded::vulkan_solve_chunked_b5_c32_spv)},
            }},
            {{
                {embedded::vulkan_solve_chunked_b7_c2_spv,
                 sizeof(embedded::vulkan_solve_chunked_b7_c2_spv)},
                {embedded::vulkan_solve_chunked_b7_c4_spv,
                 sizeof(embedded::vulkan_solve_chunked_b7_c4_spv)},
                {embedded::vulkan_solve_chunked_b7_c8_spv,
                 sizeof(embedded::vulkan_solve_chunked_b7_c8_spv)},
                {embedded::vulkan_solve_chunked_b7_c16_spv,
                 sizeof(embedded::vulkan_solve_chunked_b7_c16_spv)},
                {embedded::vulkan_solve_chunked_b7_c32_spv,
                 sizeof(embedded::vulkan_solve_chunked_b7_c32_spv)},
            }},
        }};
    const auto &[words, bytes] = code[row][exponent - 1U];
    pipeline = create_pipeline(words, bytes, "dsmvc partitioned solve");
    return pipeline;
}

std::uint64_t Runtime::submit(
    VkCommandBuffer command, VkFence fence, std::uint64_t wait_value,
    bool signal_timeline, const ExternalWait *externals,
    std::size_t external_count) {
    throw_if_failed();
    const std::scoped_lock lock(queue_mutex_);
    std::uint64_t signal_value = 0U;
    if (timeline_enabled && signal_timeline) signal_value = ++next_timeline_value_;
    constexpr std::size_t maximum_waits = 1U + 2U * frame_plane_limit;
    std::array<VkSemaphore, maximum_waits> wait_semaphores{};
    std::array<std::uint64_t, maximum_waits> wait_values{};
    std::array<VkPipelineStageFlags, maximum_waits> wait_stages{};
    std::uint32_t wait_count = 0U;
    if (wait_value != 0U) {
        wait_semaphores[wait_count] = timeline;
        wait_values[wait_count] = wait_value;
        wait_stages[wait_count] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        ++wait_count;
    }
    // One submission lists a semaphore once: the largest value wins.
    const std::uint32_t first_external = wait_count;
    for (std::size_t index = 0U; index < external_count; ++index) {
        const ExternalWait &external = externals[index];
        if (external.semaphore == VK_NULL_HANDLE) continue;
        std::uint32_t slot = first_external;
        while (slot < wait_count && wait_semaphores[slot] != external.semaphore) {
            ++slot;
        }
        if (slot == wait_count) {
            if (wait_count == maximum_waits) {
                throw std::logic_error("Vulkan submission has too many waits");
            }
            wait_semaphores[slot] = external.semaphore;
            wait_stages[slot] = VK_PIPELINE_STAGE_TRANSFER_BIT
                | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            ++wait_count;
        }
        wait_values[slot] = std::max(wait_values[slot], external.value);
    }
    const VkTimelineSemaphoreSubmitInfo timeline_info{
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        nullptr,
        wait_count,
        wait_values.data(),
        signal_value != 0U ? 1U : 0U,
        signal_value != 0U ? &signal_value : nullptr,
    };
    const VkSubmitInfo submit_info{
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        (wait_count != 0U || signal_value != 0U) ? &timeline_info : nullptr,
        wait_count,
        wait_semaphores.data(),
        wait_stages.data(),
        1U,
        &command,
        signal_value != 0U ? 1U : 0U,
        signal_value != 0U ? &timeline : nullptr,
    };
    // The core's queue is shared with every other GPU filter of the graph.
    struct QueueLock {
        QueueLock(void (*lock)(void *), void (*unlock)(void *), void *user)
            : unlock_(unlock), user_(user) {
            if (lock) lock(user);
            else unlock_ = nullptr;
        }
        ~QueueLock() { if (unlock_) unlock_(user_); }
        void (*unlock_)(void *);
        void *user_;
    } queue_lock(lock_queue_, unlock_queue_, queue_lock_user_);
    submitted_since_idle_.store(true, std::memory_order_relaxed);
    check(vkQueueSubmit(queue, 1U, &submit_info, fence), "vkQueueSubmit");
    return signal_value;
}

void Runtime::wait_fence(VkFence fence, const char *label) {
    throw_if_failed();
    check(vkWaitForFences(
              device, 1U, &fence, VK_TRUE,
              std::numeric_limits<std::uint64_t>::max()),
          label);
}

std::size_t Runtime::acquire_slot(bool heavy, bool host_limited) {
    throw_if_failed();
    std::unique_lock lock(slot_mutex_);
    slot_available_.wait(lock, [&] {
        if (failed_.load(std::memory_order_acquire)) return true;
        const std::size_t limit = host_limited
            ? std::min<std::size_t>(2U, limited_slot_count_)
            : limited_slot_count_;
        return (!adaptive_slots_ || heavy || limited_slot_busy_ < limit)
            && std::ranges::find(slot_busy, false) != slot_busy.end();
    });
    throw_if_failed();
    const auto found = std::ranges::find(slot_busy, false);
    *found = true;
    if (adaptive_slots_ && !heavy) ++limited_slot_busy_;
    return static_cast<std::size_t>(found - slot_busy.begin());
}

void Runtime::release_slot(std::size_t index, bool heavy) noexcept {
    {
        const std::scoped_lock lock(slot_mutex_);
        slot_busy[index] = false;
        if (adaptive_slots_ && !heavy && limited_slot_busy_ != 0U) {
            --limited_slot_busy_;
        }
    }
    slot_available_.notify_one();
}

void Runtime::attach_executor() {
    const std::scoped_lock lock(executor_mutex_);
    ++executor_count_;
}

void Runtime::detach_executor() noexcept {
    bool reset_cache = false;
    {
        const std::scoped_lock lock(executor_mutex_);
        reset_cache = executor_count_ != 0U && --executor_count_ == 0U;
    }
    if (reset_cache && input_cache) input_cache->reset();
    // The plan cache would keep a borrowed device's runtime alive past the
    // core's device. (The owned runtime lives as long as the process.)
    if (reset_cache && !owns_device_) purge_packed_plans(this);
    // Hosts free their filters before they exit; this is the last point
    // before exit handlers run (see wait_for_driver_threads and
    // destroy_device_objects).
    if (reset_cache && owns_device_) {
        {
            const std::scoped_lock lock(queue_mutex_);
            if (submitted_since_idle_.exchange(false)) {
                (void)vkDeviceWaitIdle(device);
            }
        }
        wait_for_driver_threads(driver_threads_, "release");
    }
}

[[nodiscard]] std::shared_ptr<Runtime> shared_runtime() {
    static const auto runtime = std::make_shared<Runtime>();
    return runtime;
}

class SlotRelease {
public:
    SlotRelease(Runtime &runtime, std::size_t index, bool heavy) noexcept
        : runtime_(&runtime), index_(index), heavy_(heavy) {}
    ~SlotRelease() {
        if (runtime_) runtime_->release_slot(index_, heavy_);
    }
    SlotRelease(const SlotRelease &) = delete;
    SlotRelease &operator=(const SlotRelease &) = delete;

    void release_now() noexcept {
        if (!runtime_) return;
        runtime_->release_slot(index_, heavy_);
        runtime_ = nullptr;
    }

private:
    Runtime *runtime_ = nullptr;
    std::size_t index_ = 0U;
    bool heavy_ = false;
};

} // namespace

struct VulkanExecutor::Impl {
    Impl() : runtime(shared_runtime()) { runtime->attach_executor(); }
    explicit Impl(std::shared_ptr<Runtime> borrowed)
        : runtime(std::move(borrowed)) { runtime->attach_executor(); }
    ~Impl() { runtime->detach_executor(); }

    struct PreparedPlan {
        std::shared_ptr<const AxisPlan> axis;
        const AxisPlan *identity = nullptr;
        std::weak_ptr<const PackedPlan> packed_f32;
        std::weak_ptr<const PackedPlan> packed_f64;
    };

    [[nodiscard]] auto find(const AxisPlan &plan) const {
        return std::find_if(
            plans.begin(), plans.end(), [&plan](const auto &candidate) {
                return candidate.identity == &plan;
            });
    }

    [[nodiscard]] std::shared_ptr<const PackedPlan> get(
        const AxisPlan &plan, bool float64 = false) const {
        const auto get_prepared = [&](const auto &prepared) {
            const auto &weak = float64
                ? prepared.packed_f64 : prepared.packed_f32;
            if (auto packed = weak.lock()) return packed;
            return acquire_packed_plan(runtime, prepared.axis, float64);
        };
        if (sealed.load(std::memory_order_acquire)) {
            const auto found = find(plan);
            if (found != plans.end()) {
                return get_prepared(*found);
            }
        } else {
            const std::scoped_lock lock(mutex);
            const auto found = find(plan);
            if (found != plans.end()) {
                return get_prepared(*found);
            }
        }
        auto owned = std::make_shared<const AxisPlan>(plan);
        return std::make_shared<const PackedPlan>(
            runtime, std::move(owned), &plan, float64);
    }

    template <class Sample>
    void execute_2d_f64(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const Sample *input, std::ptrdiff_t input_row_stride,
        Sample *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion *conversion,
        std::shared_ptr<const void> input_lifetime,
        const ResidentIo *resident = nullptr) const {
        if (!runtime->selected.float64.strict_supported()) {
            throw std::runtime_error(
                runtime->selected.float64.requirement_error());
        }
        const auto packed_horizontal = get(horizontal, true);
        const auto packed_vertical = get(vertical, true);
        const auto source_width = static_cast<std::uint32_t>(
            horizontal.source_size);
        const auto source_height = static_cast<std::uint32_t>(
            vertical.source_size);
        const auto destination_width = static_cast<std::uint32_t>(
            horizontal.destination_size);
        const auto destination_height = static_cast<std::uint32_t>(
            vertical.destination_size);
        const std::size_t source_elements = checked_product(
            source_width, source_height, "Vulkan Float64 source image");
        const std::size_t intermediate_elements = checked_product(
            destination_width, source_height,
            "Vulkan Float64 intermediate image");
        const std::size_t destination_elements = checked_product(
            destination_width, destination_height,
            "Vulkan Float64 destination image");
        const std::size_t source_bytes = checked_product(
            source_elements, sizeof(Sample), "Vulkan Float64 source image");
        const std::size_t source_copy_bytes = align_up(
            source_bytes, 4U, "Vulkan Float64 packed source copy");
        const std::size_t transposed_bytes = checked_product(
            source_elements, sizeof(double), "Vulkan Float64 transposed image");
        const std::size_t intermediate_bytes = checked_product(
            intermediate_elements, sizeof(double),
            "Vulkan Float64 intermediate image");
        const std::size_t destination_bytes = checked_product(
            destination_elements, sizeof(double),
            "Vulkan Float64 destination image");
        const std::size_t result_bytes = checked_product(
            destination_elements, sizeof(Sample), "Vulkan Float64 result image");
        const std::size_t result_copy_bytes = align_up(
            result_bytes, 4U, "Vulkan Float64 packed result copy");
        const std::size_t source_row_bytes = checked_product(
            source_width, sizeof(Sample), "Vulkan Float64 source row");
        const std::size_t source_pitch = checked_product(
            static_cast<std::size_t>(input_row_stride), sizeof(Sample),
            "Vulkan Float64 source pitch");
        const std::size_t result_row_bytes = checked_product(
            destination_width, sizeof(Sample), "Vulkan Float64 result row");
        const std::size_t result_pitch = checked_product(
            static_cast<std::size_t>(output_row_stride), sizeof(Sample),
            "Vulkan Float64 result pitch");

        const bool cache_requested =
            input_lifetime && runtime->input_cache->enabled();
        std::shared_ptr<CachedInput> cached_input;
        std::optional<InputCacheKey> cache_key;
        bool cache_producer = false;
        if (cache_requested) {
            cache_key = make_input_cache_key(
                input, input_row_stride, source_width, source_height,
                CachedInputLayout::transposed_f64, conversion);
            auto acquired = runtime->input_cache->acquire(
                *cache_key, transposed_bytes, std::move(input_lifetime));
            cached_input = std::move(acquired.input);
            cache_producer = acquired.producer;
        }
        const bool reused_input = cached_input && !cache_producer;
        const std::uint64_t input_wait_value = reused_input
            ? cached_input->wait_value() : 0U;

        WorkspaceBuilder builder;
        const std::uint32_t source_offset = builder.add_bytes(
            std::max(source_copy_bytes, intermediate_bytes));
        const std::uint32_t transposed_offset =
            builder.add_bytes(transposed_bytes);
        const std::uint32_t intermediate_offset =
            builder.add_bytes(intermediate_bytes);
        const std::uint32_t destination_offset =
            builder.add_bytes(destination_bytes);
        const std::uint32_t result_offset =
            builder.add_bytes(result_copy_bytes);
        const std::uint32_t status_offset = builder.add_bytes(4U);
        const std::size_t download_offset = source_copy_bytes;
        const std::size_t status_download_offset = align_up(
            checked_add(download_offset, result_copy_bytes,
                        "Vulkan Float64 status download"),
            4U, "Vulkan Float64 status download");
        const std::size_t staging_bytes = checked_add(
            status_download_offset, 4U, "Vulkan Float64 staging buffer");
        if (builder.size() > maximum_allocation_bytes
            || staging_bytes > maximum_allocation_bytes) {
            throw std::length_error(
                "Vulkan Float64 execution exceeds the 2 GiB guard");
        }

        // Rows that lie in imported frame blocks are copied directly.
        const HostRange source_range = resident
            ? resident_range(*resident->source)
            : (source_row_bytes * source_height) % 4U == 0U
                ? host_range(1U, input, source_pitch, source_row_bytes,
                             source_height)
                : HostRange{};
        const HostRange result_range = resident
            ? resident_range(*resident->result)
            : host_range(
                2U, output, result_pitch, result_row_bytes,
                destination_height);
        const ExternalWait resident_source_wait = resident_wait(resident);
        const bool staged_io = !((reused_input || source_range) && result_range);
        const bool host_limited =
            std::is_same_v<Sample, float> && !reused_input && staged_io;
        const bool heavy = !host_limited
            && std::max(horizontal.half_bandwidth, vertical.half_bandwidth) >= 7;
        std::optional<StagingPool::Allocation> decoupled_staging;
        if constexpr (std::is_same_v<Sample, float>) {
            if (staged_io) decoupled_staging.emplace(
                runtime->staging_pool->acquire(host_limited));
            if (decoupled_staging) decoupled_staging->reserve(staging_bytes);
        }
        const std::size_t slot_index =
            runtime->acquire_slot(heavy, host_limited);
        SlotRelease slot_release(*runtime, slot_index, heavy);
        ExecutionSlot &slot = *runtime->slots[slot_index];
        slot.reserve(builder.size(), decoupled_staging ? 4U : staging_bytes);
        Buffer &staging = decoupled_staging
            ? decoupled_staging->buffer() : slot.staging;

        if (!reused_input && !source_range) {
            pack_host_rows_aligned(
                input, source_pitch, staging.mapped(),
                source_row_bytes, source_height, source_copy_bytes);
            staging.flush(0U, source_copy_bytes);
        }

        bool submitted = false;
        bool cache_published = false;
        PendingPlanUploads plan_uploads(packed_horizontal, packed_vertical);
        try {
            slot.begin();
            plan_uploads.record(slot.command);
            runtime->begin_label(slot.command, "dsmvc Float64 2D inverse");
            clear_f64_status(slot, status_offset);
            if (!reused_input) {
                runtime->begin_label(
                    slot.command, "Float64 input transpose");
                const VkBufferCopy source_copy{
                    0U,
                    static_cast<VkDeviceSize>(source_offset) * 4U,
                    source_copy_bytes,
                };
                if (source_range) {
                    if (source_copy_bytes != source_row_bytes * source_height) {
                        // Whole-word loads: zero the word that ends the source.
                        const VkDeviceSize tail = source_copy.dstOffset
                            + ((source_row_bytes * source_height)
                               & ~std::size_t{3U});
                        vkCmdFillBuffer(
                            slot.command, slot.workspace.handle(), tail, 4U, 0U);
                        command_barrier(
                            slot.command, slot.workspace.handle(), tail, 4U,
                            VK_ACCESS_TRANSFER_WRITE_BIT,
                            VK_ACCESS_TRANSFER_WRITE_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT);
                    }
                    record_row_copies(
                        slot.command, source_range.buffer(),
                        source_range.offset, source_pitch,
                        slot.workspace.handle(), source_copy.dstOffset,
                        source_row_bytes, source_row_bytes, source_height);
                } else {
                    vkCmdCopyBuffer(
                        slot.command, staging.handle(),
                        slot.workspace.handle(), 1U, &source_copy);
                }
                command_barrier(
                    slot.command, slot.workspace.handle(), source_copy.dstOffset,
                    source_copy.size, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                const std::uint32_t sample_type =
                    std::is_same_v<Sample, float> ? 0U
                    : std::is_same_v<Sample, std::uint8_t> ? 1U : 2U;
                record_transpose_f64(
                    *runtime, slot, *packed_horizontal,
                    cached_input ? cached_input->buffer() : VK_NULL_HANDLE,
                    cached_input ? cached_input->offset() : 0U,
                    cached_input ? cached_input->size() : 0U,
                    source_width, source_height, source_offset,
                    cached_input ? 0U : transposed_offset,
                    0U, cached_input ? 1U : 0U, sample_type, conversion,
                    status_offset);
                if (cached_input) {
                    command_barrier(
                        slot.command, cached_input->buffer(),
                        cached_input->offset(), cached_input->size(),
                        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                } else {
                    command_barrier(
                        slot.command, slot.workspace.handle(),
                        static_cast<VkDeviceSize>(transposed_offset) * 4U,
                        transposed_bytes, VK_ACCESS_SHADER_WRITE_BIT,
                        VK_ACCESS_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                }
                runtime->end_label(slot.command);
            }

            runtime->begin_label(
                slot.command, "Float64 horizontal inverse");
            record_inverse_f64(
                *runtime, slot, *packed_horizontal,
                cached_input ? cached_input->buffer() : VK_NULL_HANDLE,
                cached_input ? cached_input->offset() : 0U,
                cached_input ? cached_input->size() : 0U,
                source_height, cached_input ? 0U : transposed_offset,
                source_offset, cached_input ? 1U : 0U, 0U, 0U,
                should_split_rhs(*runtime, *packed_horizontal), status_offset);
            command_barrier(
                slot.command, slot.workspace.handle(),
                static_cast<VkDeviceSize>(source_offset) * 4U,
                intermediate_bytes, VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            runtime->end_label(slot.command);

            runtime->begin_label(
                slot.command, "Float64 intermediate transpose");
            record_transpose_f64(
                *runtime, slot, *packed_horizontal,
                cached_input ? cached_input->buffer() : VK_NULL_HANDLE,
                cached_input ? cached_input->offset() : 0U,
                cached_input ? cached_input->size() : 0U,
                source_height, destination_width, source_offset,
                intermediate_offset, 0U, 0U, 3U, nullptr, status_offset);
            command_barrier(
                slot.command, slot.workspace.handle(),
                static_cast<VkDeviceSize>(intermediate_offset) * 4U,
                intermediate_bytes, VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            runtime->end_label(slot.command);

            runtime->begin_label(
                slot.command, "Float64 vertical inverse");
            record_inverse_f64(
                *runtime, slot, *packed_vertical,
                cached_input ? cached_input->buffer() : VK_NULL_HANDLE,
                cached_input ? cached_input->offset() : 0U,
                cached_input ? cached_input->size() : 0U,
                destination_width, intermediate_offset, destination_offset,
                0U, 0U, 0U, should_split_rhs(*runtime, *packed_vertical),
                status_offset);
            command_barrier(
                slot.command, slot.workspace.handle(),
                static_cast<VkDeviceSize>(destination_offset) * 4U,
                destination_bytes, VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            runtime->end_label(slot.command);

            runtime->begin_label(
                slot.command, "Float64 final conversion");
            const std::uint32_t result_type =
                std::is_same_v<Sample, float> ? 0U
                : std::is_same_v<Sample, std::uint8_t> ? 1U : 2U;
            record_conversion_f64(
                *runtime, slot, *packed_vertical,
                checked_u32(destination_elements,
                            "Vulkan Float64 destination element count"),
                destination_offset, result_offset, result_type, conversion,
                status_offset);
            command_barrier(
                slot.command, slot.workspace.handle(),
                static_cast<VkDeviceSize>(result_offset) * 4U,
                result_copy_bytes, VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT);
            const VkBufferCopy result_copy{
                static_cast<VkDeviceSize>(result_offset) * 4U,
                download_offset,
                result_copy_bytes,
            };
            if (result_range) {
                record_row_copies(
                    slot.command, slot.workspace.handle(),
                    result_copy.srcOffset, result_row_bytes,
                    result_range.buffer(), result_range.offset, result_pitch,
                    result_row_bytes, destination_height);
                command_barrier(
                    slot.command, result_range.buffer(), result_range.offset,
                    host_blocks::plane_span(
                        result_pitch, result_row_bytes, destination_height),
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_HOST_BIT);
            } else {
                vkCmdCopyBuffer(
                    slot.command, slot.workspace.handle(), staging.handle(),
                    1U, &result_copy);
            }
            copy_f64_status(
                slot, status_offset, staging, status_download_offset);
            command_barrier(
                slot.command, staging.handle(), download_offset,
                staging_bytes - download_offset, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT);
            runtime->end_label(slot.command);
            runtime->end_label(slot.command);
            runtime->check(vkEndCommandBuffer(slot.command),
                           "vkEndCommandBuffer(Float64 2D inverse)");

            const std::uint64_t signal_value = runtime->submit(
                slot.command, slot.fence, input_wait_value,
                cache_producer && runtime->timeline_enabled,
                resident ? &resident_source_wait : nullptr, resident ? 1U : 0U);
            submitted = true;
            if (cache_producer && runtime->timeline_enabled) {
                cached_input->publish(signal_value);
                cache_published = true;
            }
            runtime->wait_fence(slot.fence, "Vulkan Float64 2D execution");
            if (cache_producer && !cache_published) {
                cached_input->publish(0U);
                cache_published = true;
            }
            plan_uploads.complete();
        } catch (...) {
            const auto error = std::current_exception();
            if (!submitted) slot.recover_unsubmitted();
            if (cache_producer && !cache_published) {
                cached_input->fail(error);
                runtime->input_cache->erase(*cache_key, cached_input);
            }
            std::rethrow_exception(error);
        }

        if (decoupled_staging) slot_release.release_now();
        staging.invalidate(
            download_offset, staging_bytes - download_offset);
        require_f64_status_clear(
            static_cast<const std::byte *>(staging.mapped())
                + static_cast<std::ptrdiff_t>(status_download_offset));
        if (!result_range) {
            unpack_host_rows(
                static_cast<const std::byte *>(staging.mapped())
                    + static_cast<std::ptrdiff_t>(download_offset),
                result_row_bytes, output, result_pitch, destination_height);
        }
    }

    // One plane of a Float32 2D execution: its arguments, its regions of the
    // slot's workspace and staging buffer, and its input state. A frame runs
    // its planes in one slot, each on regions of its own.
    template <class Sample>
    struct Plane2D {
        const AxisPlan *horizontal = nullptr;
        const AxisPlan *vertical = nullptr;
        const Sample *input = nullptr;
        std::ptrdiff_t input_row_stride = 0;
        Sample *output = nullptr;
        std::ptrdiff_t output_row_stride = 0;
        const IntegerConversion *conversion = nullptr;
        const ResidentIo *resident = nullptr;

        std::shared_ptr<const PackedPlan> packed_horizontal;
        std::shared_ptr<const PackedPlan> packed_vertical;
        std::uint32_t source_width = 0U;
        std::uint32_t source_height = 0U;
        std::uint32_t destination_width = 0U;
        std::uint32_t destination_height = 0U;
        std::size_t destination_elements = 0U;
        std::size_t source_bytes = 0U;
        std::size_t source_copy_bytes = 0U;
        std::size_t transposed_bytes = 0U;
        std::size_t intermediate_bytes = 0U;
        std::size_t destination_bytes = 0U;
        std::size_t result_copy_bytes = 0U;
        std::size_t source_row_bytes = 0U;
        std::size_t source_pitch = 0U;
        std::size_t result_row_bytes = 0U;
        std::size_t result_pitch = 0U;
        bool fold = false;
        bool row_major_input = false;
        std::uint32_t source_offset = 0U;
        std::uint32_t transposed_offset = 0U;
        std::uint32_t intermediate_offset = 0U;
        std::uint32_t destination_offset = 0U;
        std::uint32_t result_offset = 0U;
        // The plane's staging region: the packed source, then the result.
        std::size_t staging_base = 0U;
        std::size_t download_offset = 0U;

        std::shared_ptr<CachedInput> cached_input;
        std::optional<InputCacheKey> cache_key;
        bool cache_producer = false;
        bool cache_published = false;
        bool reused_input = false;
        std::uint64_t input_wait_value = 0U;

        bool source_in_place = false;
        VkBuffer input_buffer = VK_NULL_HANDLE;
        VkDeviceSize input_buffer_offset = 0U;
        VkDeviceSize input_buffer_size = 0U;
        std::uint32_t horizontal_source_offset = 0U;
        std::uint32_t horizontal_output_offset = 0U;
        HostRange source_range;
        HostRange result_range;
        ExternalWait resident_source_wait;
        bool staged_io = false;
        bool host_limited = false;
        bool split_horizontal = false;
        bool split_vertical = false;
    };

    // Sizes and offsets of a plane; appends its regions after those already
    // in the builder and the staging cursor.
    template <class Sample>
    void lay_out_2d(
        Plane2D<Sample> &plane, WorkspaceBuilder &builder,
        std::size_t &staging_cursor) const {
        const AxisPlan &horizontal = *plane.horizontal;
        const AxisPlan &vertical = *plane.vertical;
        plane.packed_horizontal = get(horizontal);
        plane.packed_vertical = get(vertical);
        plane.source_width = static_cast<std::uint32_t>(horizontal.source_size);
        plane.source_height = static_cast<std::uint32_t>(vertical.source_size);
        plane.destination_width =
            static_cast<std::uint32_t>(horizontal.destination_size);
        plane.destination_height =
            static_cast<std::uint32_t>(vertical.destination_size);
        const std::size_t source_elements = checked_product(
            plane.source_width, plane.source_height, "Vulkan source image");
        const std::size_t intermediate_elements = checked_product(
            plane.destination_width, plane.source_height,
            "Vulkan intermediate image");
        plane.destination_elements = checked_product(
            plane.destination_width, plane.destination_height,
            "Vulkan destination image");
        plane.source_bytes = checked_product(
            source_elements, sizeof(Sample), "Vulkan source image");
        plane.source_copy_bytes = align_up(
            plane.source_bytes, 4U, "Vulkan packed source copy");
        plane.transposed_bytes = checked_product(
            source_elements, sizeof(float), "Vulkan transposed image");
        plane.intermediate_bytes = checked_product(
            intermediate_elements, sizeof(float), "Vulkan intermediate image");
        plane.destination_bytes = checked_product(
            plane.destination_elements, sizeof(float),
            "Vulkan destination image");
        const std::size_t result_bytes = checked_product(
            plane.destination_elements, sizeof(Sample), "Vulkan result image");
        plane.result_copy_bytes = align_up(
            result_bytes, 4U, "Vulkan packed result copy");
        plane.source_row_bytes = checked_product(
            plane.source_width, sizeof(Sample), "Vulkan source row");
        plane.source_pitch = checked_product(
            static_cast<std::size_t>(plane.input_row_stride), sizeof(Sample),
            "Vulkan source pitch");
        plane.result_row_bytes = checked_product(
            plane.destination_width, sizeof(Sample), "Vulkan result row");
        plane.result_pitch = checked_product(
            static_cast<std::size_t>(plane.output_row_stride), sizeof(Sample),
            "Vulkan result pitch");

        // The right-hand side reads either layout, so no pass only
        // transposes: a Float32 source is read row-major as uploaded (an
        // integer source keeps its transpose, which also converts it), and the
        // vertical pass reads the horizontal result where it lies.
        plane.fold = runtime->fold_transposes;
        plane.row_major_input = plane.fold && std::is_same_v<Sample, float>;

        plane.source_offset = builder.add_bytes(
            std::max(plane.source_copy_bytes, plane.intermediate_bytes));
        plane.transposed_offset = builder.add_bytes(plane.transposed_bytes);
        plane.intermediate_offset =
            plane.fold ? 0U : builder.add_bytes(plane.intermediate_bytes);
        plane.destination_offset = builder.add_bytes(plane.destination_bytes);
        plane.result_offset = plane.destination_offset;
        if constexpr (!std::is_same_v<Sample, float>) {
            plane.result_offset = builder.add_bytes(plane.result_copy_bytes);
        }
        plane.staging_base = align_up(
            staging_cursor, 64U, "Vulkan staging buffer");
        plane.download_offset = checked_add(
            plane.staging_base, plane.source_copy_bytes,
            "Vulkan staging buffer");
        staging_cursor = checked_add(
            plane.download_offset, plane.result_copy_bytes,
            "Vulkan staging buffer");
    }

    // Joins the shared input of the plane: it is the cache's producer, or
    // waits for the producer to have scheduled it.
    template <class Sample>
    void acquire_input_2d(
        Plane2D<Sample> &plane,
        std::shared_ptr<const void> input_lifetime) const {
        if (!input_lifetime || !runtime->input_cache->enabled()) return;
        plane.cache_key = make_input_cache_key(
            plane.input, plane.input_row_stride, plane.source_width,
            plane.source_height,
            plane.row_major_input ? CachedInputLayout::row_major
                                  : CachedInputLayout::transposed,
            plane.conversion);
        auto acquired = runtime->input_cache->acquire(
            *plane.cache_key, plane.transposed_bytes, std::move(input_lifetime));
        plane.cached_input = std::move(acquired.input);
        plane.cache_producer = acquired.producer;
        plane.reused_input = plane.cached_input && !plane.cache_producer;
        if (plane.reused_input) {
            plane.input_wait_value = plane.cached_input->wait_value();
        }
    }

    // Where the plane's input is read from and its result written to.
    template <class Sample>
    void bind_io_2d(Plane2D<Sample> &plane) const {
        const ResidentIo *resident = plane.resident;
        // A GPU frame's plane with tight rows is read where it lies, through
        // the binding a shared input would use.
        plane.source_in_place = plane.row_major_input && resident
            && plane.source_pitch == plane.source_row_bytes
            && resident->source->buffer_bytes >= plane.source_bytes;
        plane.input_buffer = plane.cached_input ? plane.cached_input->buffer()
            : plane.source_in_place
                ? reinterpret_cast<VkBuffer>(resident->source->buffer)
                : VK_NULL_HANDLE;
        plane.input_buffer_offset =
            plane.cached_input ? plane.cached_input->offset() : 0U;
        plane.input_buffer_size = plane.cached_input
            ? plane.cached_input->size()
            : plane.source_in_place ? plane.source_bytes : 0U;
        plane.horizontal_source_offset = plane.input_buffer
            ? 0U
            : plane.row_major_input ? plane.source_offset
                                    : plane.transposed_offset;
        plane.horizontal_output_offset = plane.row_major_input
            ? plane.transposed_offset : plane.source_offset;

        // Rows that lie in imported frame blocks are copied directly.
        plane.source_range = resident
            ? resident_range(*resident->source)
            : (plane.source_row_bytes * plane.source_height) % 4U == 0U
                ? host_range(1U, plane.input, plane.source_pitch,
                             plane.source_row_bytes, plane.source_height)
                : HostRange{};
        plane.result_range = resident
            ? resident_range(*resident->result)
            : host_range(
                2U, plane.output, plane.result_pitch, plane.result_row_bytes,
                plane.destination_height);
        plane.resident_source_wait = resident_wait(resident);
        plane.staged_io = !((plane.reused_input || plane.source_range)
                            && plane.result_range);
        plane.host_limited = std::is_same_v<Sample, float>
            && !plane.reused_input && plane.staged_io;
    }

    // The steps of a plane's execution. A frame records each step for all of
    // its planes, then one barrier command for the step, so the planes'
    // dispatches overlap and the barriers are shared.
    enum class Step2D : std::uint8_t {
        upload,
        input_transpose,
        horizontal_rhs,
        horizontal_solve,
        intermediate_transpose,
        vertical_rhs,
        vertical_solve,
        conversion,
        download,
    };
    static constexpr std::array<std::pair<Step2D, const char *>, 9U> steps_2d{{
        {Step2D::upload, "input upload"},
        {Step2D::input_transpose, "input transpose"},
        {Step2D::horizontal_rhs, "horizontal right-hand side"},
        {Step2D::horizontal_solve, "horizontal inverse"},
        {Step2D::intermediate_transpose, "intermediate transpose"},
        {Step2D::vertical_rhs, "vertical right-hand side"},
        {Step2D::vertical_solve, "vertical inverse"},
        {Step2D::conversion, "integer conversion"},
        {Step2D::download, "download"},
    }};

    template <class Sample>
    void record_2d_step(
        const Plane2D<Sample> &plane, Step2D step, ExecutionSlot &slot,
        Buffer &staging, BarrierBatch &barriers) const {
        const PackedPlan &packed_horizontal = *plane.packed_horizontal;
        const PackedPlan &packed_vertical = *plane.packed_vertical;
        const auto &cached_input = plane.cached_input;
        const auto &source_range = plane.source_range;
        const auto &result_range = plane.result_range;
        const VkBuffer workspace = slot.workspace.handle();
        const bool compute_input = !plane.reused_input && !plane.source_in_place;
        constexpr VkPipelineStageFlags compute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        constexpr VkPipelineStageFlags transfer = VK_PIPELINE_STAGE_TRANSFER_BIT;
        constexpr VkAccessFlags shader_write = VK_ACCESS_SHADER_WRITE_BIT;
        constexpr VkAccessFlags shader_read = VK_ACCESS_SHADER_READ_BIT;

        switch (step) {
        case Step2D::upload: {
            if (!compute_input) return;
            const VkBuffer upload_buffer = plane.row_major_input && cached_input
                ? cached_input->buffer() : workspace;
            const VkBufferCopy source_copy{
                plane.staging_base,
                plane.row_major_input && cached_input
                    ? cached_input->offset()
                    : static_cast<VkDeviceSize>(plane.source_offset) * 4U,
                plane.source_copy_bytes,
            };
            if (source_range) {
                if (plane.source_copy_bytes != plane.source_bytes) {
                    // The shaders load whole words: zero the one that ends
                    // the packed source, as the staged path does.
                    const VkDeviceSize tail = source_copy.dstOffset
                        + (plane.source_bytes & ~std::size_t{3U});
                    vkCmdFillBuffer(slot.command, workspace, tail, 4U, 0U);
                    command_barrier(
                        slot.command, workspace, tail, 4U,
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT, transfer, transfer);
                }
                record_row_copies(
                    slot.command, source_range.buffer(), source_range.offset,
                    plane.source_pitch, upload_buffer, source_copy.dstOffset,
                    plane.source_row_bytes, plane.source_row_bytes,
                    plane.source_height);
            } else {
                vkCmdCopyBuffer(
                    slot.command, staging.handle(), upload_buffer, 1U,
                    &source_copy);
            }
            barriers.add(
                upload_buffer, source_copy.dstOffset, source_copy.size,
                VK_ACCESS_TRANSFER_WRITE_BIT, shader_read, transfer, compute);
            return;
        }
        case Step2D::input_transpose: {
            if (!compute_input || plane.row_major_input) return;
            const std::uint32_t sample_type =
                std::is_same_v<Sample, float> ? 0U
                : std::is_same_v<Sample, std::uint8_t> ? 1U : 2U;
            record_transpose(
                *runtime, slot, packed_horizontal,
                cached_input ? cached_input->buffer() : VK_NULL_HANDLE,
                cached_input ? cached_input->offset() : 0U,
                cached_input ? cached_input->size() : 0U,
                plane.source_width, plane.source_height, plane.source_offset,
                cached_input ? 0U : plane.transposed_offset,
                0U, cached_input ? 1U : 0U, sample_type, plane.conversion);
            barriers.add(
                cached_input ? cached_input->buffer() : workspace,
                cached_input
                    ? cached_input->offset()
                    : static_cast<VkDeviceSize>(plane.transposed_offset) * 4U,
                cached_input ? cached_input->size() : plane.transposed_bytes,
                shader_write, shader_read, compute, compute);
            return;
        }
        case Step2D::horizontal_rhs:
        case Step2D::horizontal_solve: {
            const bool solve = step == Step2D::horizontal_solve;
            const bool rhs_phase = has_rhs_phase(
                packed_horizontal, 0U, plane.split_horizontal);
            if (!solve && !rhs_phase) return;
            record_inverse(
                *runtime, slot, packed_horizontal,
                plane.input_buffer, plane.input_buffer_offset,
                plane.input_buffer_size,
                plane.source_height, plane.horizontal_source_offset,
                plane.horizontal_output_offset, plane.input_buffer ? 1U : 0U,
                plane.row_major_input ? 1U : 0U, 0U, plane.split_horizontal,
                solve ? InversePhase::solve : InversePhase::rhs);
            barriers.add(
                workspace,
                static_cast<VkDeviceSize>(plane.horizontal_output_offset) * 4U,
                solve ? VkDeviceSize{plane.intermediate_bytes}
                      : static_cast<VkDeviceSize>(packed_horizontal.destination_size)
                            * plane.source_height * 4U,
                shader_write,
                solve ? shader_read : shader_read | shader_write,
                compute, compute);
            return;
        }
        case Step2D::intermediate_transpose:
            if (plane.fold) return;
            record_transpose(
                *runtime, slot, packed_horizontal,
                cached_input ? cached_input->buffer() : VK_NULL_HANDLE,
                cached_input ? cached_input->offset() : 0U,
                cached_input ? cached_input->size() : 0U,
                plane.source_height, plane.destination_width,
                plane.source_offset, plane.intermediate_offset,
                0U, 0U, 0U, nullptr);
            barriers.add(
                workspace,
                static_cast<VkDeviceSize>(plane.intermediate_offset) * 4U,
                plane.intermediate_bytes, shader_write, shader_read, compute,
                compute);
            return;
        case Step2D::vertical_rhs:
        case Step2D::vertical_solve: {
            // The horizontal result is transposed: element x of row y lies at
            // x * source_height + y, the vertical pass's vector-major layout.
            const bool solve = step == Step2D::vertical_solve;
            const bool rhs_phase = has_rhs_phase(
                packed_vertical, 0U, plane.split_vertical);
            if (!solve && !rhs_phase) return;
            record_inverse(
                *runtime, slot, packed_vertical,
                cached_input ? cached_input->buffer() : VK_NULL_HANDLE,
                cached_input ? cached_input->offset() : 0U,
                cached_input ? cached_input->size() : 0U,
                plane.destination_width,
                plane.fold ? plane.horizontal_output_offset
                           : plane.intermediate_offset,
                plane.destination_offset, 0U, plane.fold ? 1U : 0U, 0U,
                plane.split_vertical,
                solve ? InversePhase::solve : InversePhase::rhs);
            if (!solve) {
                barriers.add(
                    workspace,
                    static_cast<VkDeviceSize>(plane.destination_offset) * 4U,
                    static_cast<VkDeviceSize>(packed_vertical.destination_size)
                        * plane.destination_width * 4U,
                    shader_write, shader_read | shader_write, compute, compute);
            } else if constexpr (!std::is_same_v<Sample, float>) {
                barriers.add(
                    workspace,
                    static_cast<VkDeviceSize>(plane.destination_offset) * 4U,
                    plane.destination_bytes, shader_write, shader_read,
                    compute, compute);
            }
            return;
        }
        case Step2D::conversion:
            if constexpr (!std::is_same_v<Sample, float>) {
                record_conversion(
                    *runtime, slot, packed_vertical,
                    checked_u32(plane.destination_elements,
                                "Vulkan destination element count"),
                    plane.destination_offset, plane.result_offset,
                    std::is_same_v<Sample, std::uint8_t> ? 1U : 2U,
                    *plane.conversion);
            }
            barriers.add(
                workspace,
                static_cast<VkDeviceSize>(plane.result_offset) * 4U,
                plane.result_copy_bytes, shader_write,
                VK_ACCESS_TRANSFER_READ_BIT, compute, transfer);
            return;
        case Step2D::download: {
            const VkBufferCopy result_copy{
                static_cast<VkDeviceSize>(plane.result_offset) * 4U,
                plane.download_offset,
                plane.result_copy_bytes,
            };
            if (result_range) {
                record_row_copies(
                    slot.command, workspace, result_copy.srcOffset,
                    plane.result_row_bytes, result_range.buffer(),
                    result_range.offset, plane.result_pitch,
                    plane.result_row_bytes, plane.destination_height);
                barriers.add(
                    result_range.buffer(), result_range.offset,
                    host_blocks::plane_span(
                        plane.result_pitch, plane.result_row_bytes,
                        plane.destination_height),
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
                    transfer, VK_PIPELINE_STAGE_HOST_BIT);
            } else {
                vkCmdCopyBuffer(
                    slot.command, workspace, staging.handle(), 1U,
                    &result_copy);
                barriers.add(
                    staging.handle(), plane.download_offset,
                    plane.result_copy_bytes, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_HOST_READ_BIT, transfer,
                    VK_PIPELINE_STAGE_HOST_BIT);
            }
            return;
        }
        }
    }

    // Runs the planes of one frame (or the one plane of an execution) in one
    // slot: one command buffer, one submission, one wait. A frame that does
    // not fit a slot is not run and false is returned; nothing has happened
    // then, and the caller runs the planes one by one.
    template <class Sample>
    bool execute_2d_planes(
        std::span<Plane2D<Sample>> planes,
        const std::shared_ptr<const void> &input_lifetime, bool frame) const {
        WorkspaceBuilder builder;
        std::size_t staging_bytes = 0U;
        try {
            for (auto &plane : planes) lay_out_2d(plane, builder, staging_bytes);
        } catch (const std::length_error &) {
            if (frame) return false;
            throw;
        }
        if (builder.size() > maximum_allocation_bytes
            || staging_bytes > maximum_allocation_bytes) {
            if (frame) return false;
            throw std::length_error("Vulkan execution exceeds the 2 GiB guard");
        }
        if (frame && builder.size()
                > runtime->selected.properties.limits.maxStorageBufferRange) {
            return false;
        }

        std::array<std::shared_ptr<const PackedPlan>, 2U * frame_plane_limit>
            used_plans;
        for (std::size_t index = 0U; index < planes.size(); ++index) {
            used_plans[2U * index] = planes[index].packed_horizontal;
            used_plans[2U * index + 1U] = planes[index].packed_vertical;
        }
        PendingPlanUploads plan_uploads(
            std::span<const std::shared_ptr<const PackedPlan>>(
                used_plans.data(), 2U * planes.size()));

        std::optional<StagingPool::Allocation> decoupled_staging;
        std::optional<SlotRelease> slot_release;
        ExecutionSlot *slot_in_use = nullptr;
        Buffer *staging = nullptr;
        bool slot_begun = false;
        bool submitted = false;
        try {
            // Shared inputs are joined before a slot is taken, in the order
            // of their addresses: a frame that waits for an input holds only
            // inputs at lower addresses, so the waits cannot form a cycle
            // (plane order would not do: two clips may hold the planes of one
            // frame in different orders), and no one waits for a producer
            // while holding a slot.
            std::array<Plane2D<Sample> *, frame_plane_limit> join_order{};
            for (std::size_t index = 0U; index < planes.size(); ++index) {
                std::size_t position = index;
                for (; position != 0U
                       && std::less<const Sample *>{}(
                           planes[index].input,
                           join_order[position - 1U]->input);
                     --position) {
                    join_order[position] = join_order[position - 1U];
                }
                join_order[position] = &planes[index];
            }
            for (std::size_t index = 0U; index < planes.size(); ++index) {
                acquire_input_2d(*join_order[index], input_lifetime);
            }
            for (auto &plane : planes) bind_io_2d(plane);

            bool staged_io = false;
            bool host_limited = false;
            std::int32_t bandwidth = 0;
            std::uint64_t input_wait_value = 0U;
            bool cache_producer = false;
            std::array<ExternalWait, frame_plane_limit> externals{};
            std::size_t external_count = 0U;
            for (const auto &plane : planes) {
                staged_io = staged_io || plane.staged_io;
                host_limited = host_limited || plane.host_limited;
                bandwidth = std::max({
                    bandwidth, plane.horizontal->half_bandwidth,
                    plane.vertical->half_bandwidth});
                input_wait_value =
                    std::max(input_wait_value, plane.input_wait_value);
                cache_producer = cache_producer || plane.cache_producer;
                if (plane.resident) {
                    externals[external_count++] = plane.resident_source_wait;
                }
            }
            const bool heavy = !host_limited && bandwidth >= 7;
            if constexpr (std::is_same_v<Sample, float>) {
                if (staged_io) {
                    decoupled_staging.emplace(
                        runtime->staging_pool->acquire(host_limited));
                    decoupled_staging->reserve(staging_bytes);
                }
            }
            const std::size_t slot_index =
                runtime->acquire_slot(heavy, host_limited);
            slot_release.emplace(*runtime, slot_index, heavy);
            ExecutionSlot &slot = *runtime->slots[slot_index];
            slot_in_use = &slot;
            slot.reserve(builder.size(), decoupled_staging ? 4U : staging_bytes);
            staging = decoupled_staging ? &decoupled_staging->buffer()
                                        : &slot.staging;

            for (const auto &plane : planes) {
                if (plane.reused_input || plane.source_range) continue;
                pack_host_rows_aligned(
                    plane.input, plane.source_pitch,
                    static_cast<std::byte *>(staging->mapped())
                        + static_cast<std::ptrdiff_t>(plane.staging_base),
                    plane.source_row_bytes, plane.source_height,
                    plane.source_copy_bytes);
                staging->flush(plane.staging_base, plane.source_copy_bytes);
            }

            slot_begun = true;
            slot.begin();
            plan_uploads.record(slot.command);
            for (auto &plane : planes) {
                plane.split_horizontal =
                    should_split_rhs(*runtime, *plane.packed_horizontal);
                plane.split_vertical =
                    should_split_rhs(*runtime, *plane.packed_vertical);
            }
            runtime->begin_label(slot.command, "dsmvc 2D inverse");
            BarrierBatch barriers;
            for (const auto &[step, label] : steps_2d) {
                runtime->begin_label(slot.command, label);
                for (const auto &plane : planes) {
                    record_2d_step(plane, step, slot, *staging, barriers);
                }
                barriers.flush(slot.command);
                runtime->end_label(slot.command);
            }
            runtime->end_label(slot.command);
            runtime->check(vkEndCommandBuffer(slot.command),
                           "vkEndCommandBuffer(2D inverse)");

            const bool signal = cache_producer && runtime->timeline_enabled;
            const std::uint64_t signal_value = runtime->submit(
                slot.command, slot.fence, input_wait_value, signal,
                externals.data(), external_count);
            submitted = true;
            if (signal) {
                for (auto &plane : planes) {
                    if (!plane.cache_producer) continue;
                    plane.cached_input->publish(signal_value);
                    plane.cache_published = true;
                }
            }
            runtime->wait_fence(slot.fence, "Vulkan 2D execution");
            for (auto &plane : planes) {
                if (plane.cache_producer && !plane.cache_published) {
                    plane.cached_input->publish(0U);
                    plane.cache_published = true;
                }
            }
            plan_uploads.complete();
        } catch (...) {
            const auto error = std::current_exception();
            if (slot_begun && !submitted) slot_in_use->recover_unsubmitted();
            for (auto &plane : planes) {
                if (plane.cache_producer && !plane.cache_published) {
                    plane.cached_input->fail(error);
                    runtime->input_cache->erase(
                        *plane.cache_key, plane.cached_input);
                }
            }
            std::rethrow_exception(error);
        }

        if (decoupled_staging) slot_release->release_now();
        for (const auto &plane : planes) {
            if (plane.result_range) continue;
            staging->invalidate(plane.download_offset, plane.result_copy_bytes);
            unpack_host_rows(
                static_cast<const std::byte *>(staging->mapped())
                    + static_cast<std::ptrdiff_t>(plane.download_offset),
                plane.result_row_bytes, plane.output, plane.result_pitch,
                plane.destination_height);
        }
        return true;
    }

    template <class Sample>
    void execute_2d(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const Sample *input, std::ptrdiff_t input_row_stride,
        Sample *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion *conversion,
        std::shared_ptr<const void> input_lifetime,
        const ResidentIo *resident = nullptr) const {
        if (horizontal.requires_float64() || vertical.requires_float64()) {
            execute_2d_f64(
                horizontal, vertical, input, input_row_stride,
                output, output_row_stride, conversion,
                std::move(input_lifetime), resident);
            return;
        }
        Plane2D<Sample> plane;
        plane.horizontal = &horizontal;
        plane.vertical = &vertical;
        plane.input = input;
        plane.input_row_stride = input_row_stride;
        plane.output = output;
        plane.output_row_stride = output_row_stride;
        plane.conversion = conversion;
        plane.resident = resident;
        (void)execute_2d_planes<Sample>(
            std::span<Plane2D<Sample>>(&plane, 1U), input_lifetime, false);
    }

    // The planes of a multi-plane frame in one submission. Returns false,
    // having done nothing, when the frame is not eligible (the caller then
    // runs the planes one by one, which also reports invalid arguments).
    template <class Sample>
    [[nodiscard]] bool execute_2d_frame(
        std::span<const FramePlane2D> jobs, bool gpu_frame,
        const std::shared_ptr<const void> &input_lifetime) const {
        if (!runtime->frame_submit || jobs.size() < 2U
            || jobs.size() > frame_plane_limit) {
            return false;
        }
        std::array<Plane2D<Sample>, frame_plane_limit> planes;
        std::array<ResidentIo, frame_plane_limit> resident{};
        for (std::size_t index = 0U; index < jobs.size(); ++index) {
            const FramePlane2D &job = jobs[index];
            if (!job.horizontal || !job.vertical || !job.horizontal->valid()
                || !job.vertical->valid()
                || job.horizontal->requires_float64()
                || job.vertical->requires_float64()
                || job.vertical->source_size < 1
                || job.vertical->destination_size < 1) {
                return false;
            }
            if constexpr (!std::is_same_v<Sample, float>) {
                if (!valid_conversion<Sample>(job.conversion)) return false;
            }
            Plane2D<Sample> &plane = planes[index];
            plane.horizontal = job.horizontal;
            plane.vertical = job.vertical;
            if constexpr (!std::is_same_v<Sample, float>) {
                plane.conversion = &job.conversion;
            }
            if (gpu_frame) {
                const auto usable = [](const GpuPlane &gpu) {
                    return gpu.buffer != 0U && gpu.pitch_bytes > 0
                        && gpu.pitch_bytes
                            % static_cast<std::ptrdiff_t>(sizeof(Sample)) == 0;
                };
                if (!usable(job.gpu_input) || !usable(job.gpu_output)) {
                    return false;
                }
                resident[index] = {&job.gpu_input, &job.gpu_output};
                plane.resident = &resident[index];
                plane.input_row_stride = job.gpu_input.pitch_bytes
                    / static_cast<std::ptrdiff_t>(sizeof(Sample));
                plane.output_row_stride = job.gpu_output.pitch_bytes
                    / static_cast<std::ptrdiff_t>(sizeof(Sample));
            } else {
                if (!job.input || !job.output
                    || job.input_row_stride < job.horizontal->source_size
                    || job.output_row_stride < job.horizontal->destination_size) {
                    return false;
                }
                // Planes of one input are joined one by one.
                for (std::size_t other = 0U; other < index; ++other) {
                    if (jobs[other].input == job.input) return false;
                }
                plane.input = static_cast<const Sample *>(job.input);
                plane.input_row_stride = job.input_row_stride;
                plane.output = static_cast<Sample *>(job.output);
                plane.output_row_stride = job.output_row_stride;
            }
        }
        return execute_2d_planes<Sample>(
            std::span<Plane2D<Sample>>(planes.data(), jobs.size()),
            input_lifetime, true);
    }

    void execute_one_dimensional(
        const AxisPlan &plan, const float *input,
        std::ptrdiff_t input_row_stride, float *output,
        std::ptrdiff_t output_row_stride, std::uint32_t vector_count,
        bool rows, std::shared_ptr<const void> input_lifetime,
        const ResidentIo *resident = nullptr) const;

    void execute_one_dimensional_f64(
        const AxisPlan &plan, const float *input,
        std::ptrdiff_t input_row_stride, float *output,
        std::ptrdiff_t output_row_stride, std::uint32_t vector_count,
        bool rows, std::shared_ptr<const void> input_lifetime,
        const ResidentIo *resident = nullptr) const;

    std::shared_ptr<Runtime> runtime;
    mutable std::mutex mutex;
    std::vector<PreparedPlan> plans;
    std::atomic<bool> sealed{false};
};

void VulkanExecutor::Impl::execute_one_dimensional_f64(
    const AxisPlan &plan, const float *input,
    std::ptrdiff_t input_row_stride, float *output,
    std::ptrdiff_t output_row_stride, std::uint32_t vector_count,
    bool rows, std::shared_ptr<const void> input_lifetime,
    const ResidentIo *resident) const {
    if (!runtime->selected.float64.strict_supported()) {
        throw std::runtime_error(runtime->selected.float64.requirement_error());
    }
    const auto packed = get(plan, true);
    const std::uint32_t input_width = rows
        ? static_cast<std::uint32_t>(plan.source_size) : vector_count;
    const std::uint32_t input_rows = rows
        ? vector_count : static_cast<std::uint32_t>(plan.source_size);
    const std::uint32_t output_width = rows
        ? static_cast<std::uint32_t>(plan.destination_size) : vector_count;
    const std::uint32_t output_rows = rows
        ? vector_count : static_cast<std::uint32_t>(plan.destination_size);
    const std::size_t input_elements = checked_product(
        plan.source_size, vector_count, "Vulkan Float64 1D source");
    const std::size_t output_elements = checked_product(
        plan.destination_size, vector_count, "Vulkan Float64 1D output");
    const std::size_t input_bytes = checked_product(
        input_elements, sizeof(float), "Vulkan Float64 1D source");
    const std::size_t source_f64_bytes = checked_product(
        input_elements, sizeof(double), "Vulkan Float64 1D promoted source");
    const std::size_t output_f64_bytes = checked_product(
        output_elements, sizeof(double), "Vulkan Float64 1D output");
    const std::size_t result_bytes = checked_product(
        output_elements, sizeof(float), "Vulkan Float64 1D result");
    const std::size_t input_row_bytes = checked_product(
        input_width, sizeof(float), "Vulkan Float64 1D source row");
    const std::size_t input_pitch = checked_product(
        static_cast<std::size_t>(input_row_stride), sizeof(float),
        "Vulkan Float64 1D source pitch");
    const std::size_t output_row_bytes = checked_product(
        output_width, sizeof(float), "Vulkan Float64 1D output row");
    const std::size_t output_pitch = checked_product(
        static_cast<std::size_t>(output_row_stride), sizeof(float),
        "Vulkan Float64 1D output pitch");

    const bool cache_requested = input_lifetime && runtime->input_cache->enabled();
    std::shared_ptr<CachedInput> cached_input;
    std::optional<InputCacheKey> cache_key;
    bool cache_producer = false;
    if (cache_requested) {
        cache_key = make_input_cache_key(
            input, input_row_stride, input_width, input_rows,
            CachedInputLayout::axis_f64);
        auto acquired = runtime->input_cache->acquire(
            *cache_key, source_f64_bytes, std::move(input_lifetime));
        cached_input = std::move(acquired.input);
        cache_producer = acquired.producer;
    }
    const bool reused_input = cached_input && !cache_producer;
    const std::uint64_t input_wait_value = reused_input
        ? cached_input->wait_value() : 0U;

    WorkspaceBuilder builder;
    const std::uint32_t source_offset = builder.add_bytes(input_bytes);
    const std::uint32_t source_f64_offset = builder.add_bytes(source_f64_bytes);
    const std::uint32_t output_f64_offset = builder.add_bytes(output_f64_bytes);
    const std::uint32_t result_offset = builder.add_bytes(result_bytes);
    const std::uint32_t status_offset = builder.add_bytes(4U);
    const std::size_t download_offset = input_bytes;
    const std::size_t status_download_offset = align_up(
        checked_add(download_offset, result_bytes,
                    "Vulkan Float64 1D status download"),
        4U, "Vulkan Float64 1D status download");
    const std::size_t staging_bytes = checked_add(
        status_download_offset, 4U, "Vulkan Float64 1D staging");
    // Rows that lie in imported frame blocks are copied directly.
    const HostRange source_range = resident
        ? resident_range(*resident->source)
        : host_range(1U, input, input_pitch, input_row_bytes, input_rows);
    const HostRange result_range = resident
        ? resident_range(*resident->result)
        : host_range(2U, output, output_pitch, output_row_bytes, output_rows);
    const ExternalWait resident_source_wait = resident_wait(resident);
    const std::size_t slot_index = runtime->acquire_slot(false, false);
    SlotRelease release(*runtime, slot_index, false);
    ExecutionSlot &slot = *runtime->slots[slot_index];
    slot.reserve(builder.size(), staging_bytes);
    if (!reused_input && !source_range) {
        pack_host_rows(
            input, input_pitch, slot.staging.mapped(),
            input_row_bytes, input_rows);
        slot.staging.flush(0U, input_bytes);
    }

    bool submitted = false;
    bool cache_published = false;
    PendingPlanUploads plan_uploads(packed);
    try {
        slot.begin();
        plan_uploads.record(slot.command);
        runtime->begin_label(slot.command, rows
            ? "dsmvc Float64 row inverse" : "dsmvc Float64 column inverse");
        clear_f64_status(slot, status_offset);
        if (!reused_input) {
            const VkBufferCopy source_copy{
                0U,
                static_cast<VkDeviceSize>(source_offset) * 4U,
                input_bytes,
            };
            if (source_range) {
                record_row_copies(
                    slot.command, source_range.buffer(), source_range.offset,
                    input_pitch, slot.workspace.handle(), source_copy.dstOffset,
                    input_row_bytes, input_row_bytes, input_rows);
            } else {
                vkCmdCopyBuffer(
                    slot.command, slot.staging.handle(),
                    slot.workspace.handle(), 1U, &source_copy);
            }
            command_barrier(
                slot.command, slot.workspace.handle(), source_copy.dstOffset,
                source_copy.size, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            record_transpose_f64(
                *runtime, slot, *packed,
                cached_input ? cached_input->buffer() : VK_NULL_HANDLE,
                cached_input ? cached_input->offset() : 0U,
                cached_input ? cached_input->size() : 0U,
                input_width, input_rows, source_offset,
                cached_input ? 0U : source_f64_offset,
                0U, cached_input ? 1U : 0U, 0U, nullptr, status_offset);
            if (cached_input) {
                command_barrier(
                    slot.command, cached_input->buffer(),
                    cached_input->offset(), cached_input->size(),
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            } else {
                command_barrier(
                    slot.command, slot.workspace.handle(),
                    static_cast<VkDeviceSize>(source_f64_offset) * 4U,
                    source_f64_bytes, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            }
        }

        record_inverse_f64(
            *runtime, slot, *packed,
            cached_input ? cached_input->buffer() : VK_NULL_HANDLE,
            cached_input ? cached_input->offset() : 0U,
            cached_input ? cached_input->size() : 0U,
            vector_count, cached_input ? 0U : source_f64_offset,
            output_f64_offset, cached_input ? 1U : 0U,
            rows ? 0U : 1U, rows ? 1U : 0U,
            should_split_rhs(*runtime, *packed), status_offset);
        command_barrier(
            slot.command, slot.workspace.handle(),
            static_cast<VkDeviceSize>(output_f64_offset) * 4U,
            output_f64_bytes, VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        record_conversion_f64(
            *runtime, slot, *packed,
            checked_u32(output_elements, "Vulkan Float64 1D output count"),
            output_f64_offset, result_offset, 0U, nullptr, status_offset);
        command_barrier(
            slot.command, slot.workspace.handle(),
            static_cast<VkDeviceSize>(result_offset) * 4U, result_bytes,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
        const VkBufferCopy result_copy{
            static_cast<VkDeviceSize>(result_offset) * 4U,
            download_offset,
            result_bytes,
        };
        if (result_range) {
            record_row_copies(
                slot.command, slot.workspace.handle(), result_copy.srcOffset,
                output_row_bytes, result_range.buffer(), result_range.offset,
                output_pitch, output_row_bytes, output_rows);
            command_barrier(
                slot.command, result_range.buffer(), result_range.offset,
                host_blocks::plane_span(
                    output_pitch, output_row_bytes, output_rows),
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
        } else {
            vkCmdCopyBuffer(
                slot.command, slot.workspace.handle(), slot.staging.handle(),
                1U, &result_copy);
        }
        copy_f64_status(
            slot, status_offset, slot.staging, status_download_offset);
        command_barrier(
            slot.command, slot.staging.handle(), download_offset,
            staging_bytes - download_offset, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT);
        runtime->end_label(slot.command);
        runtime->check(vkEndCommandBuffer(slot.command),
                       "vkEndCommandBuffer(Float64 1D inverse)");
        const std::uint64_t signal_value = runtime->submit(
            slot.command, slot.fence, input_wait_value,
            cache_producer && runtime->timeline_enabled,
            resident ? &resident_source_wait : nullptr, resident ? 1U : 0U);
        submitted = true;
        if (cache_producer && runtime->timeline_enabled) {
            cached_input->publish(signal_value);
            cache_published = true;
        }
        runtime->wait_fence(slot.fence, "Vulkan Float64 1D execution");
        if (cache_producer && !cache_published) {
            cached_input->publish(0U);
            cache_published = true;
        }
        plan_uploads.complete();
    } catch (...) {
        const auto error = std::current_exception();
        if (!submitted) slot.recover_unsubmitted();
        if (cache_producer && !cache_published) {
            cached_input->fail(error);
            runtime->input_cache->erase(*cache_key, cached_input);
        }
        std::rethrow_exception(error);
    }
    slot.staging.invalidate(download_offset, staging_bytes - download_offset);
    require_f64_status_clear(
        static_cast<const std::byte *>(slot.staging.mapped())
            + static_cast<std::ptrdiff_t>(status_download_offset));
    if (!result_range) {
        unpack_host_rows(
            static_cast<const std::byte *>(slot.staging.mapped())
                + static_cast<std::ptrdiff_t>(download_offset),
            output_row_bytes, output, output_pitch, output_rows);
    }
}

void VulkanExecutor::Impl::execute_one_dimensional(
    const AxisPlan &plan, const float *input,
    std::ptrdiff_t input_row_stride, float *output,
    std::ptrdiff_t output_row_stride, std::uint32_t vector_count,
    bool rows, std::shared_ptr<const void> input_lifetime,
    const ResidentIo *resident) const {
    if (plan.requires_float64()) {
        execute_one_dimensional_f64(
            plan, input, input_row_stride, output, output_row_stride,
            vector_count, rows, std::move(input_lifetime), resident);
        return;
    }
    const auto packed = get(plan);
    const std::uint32_t input_width = rows
        ? static_cast<std::uint32_t>(plan.source_size) : vector_count;
    const std::uint32_t input_rows = rows
        ? vector_count : static_cast<std::uint32_t>(plan.source_size);
    const std::uint32_t output_width = rows
        ? static_cast<std::uint32_t>(plan.destination_size) : vector_count;
    const std::uint32_t output_rows = rows
        ? vector_count : static_cast<std::uint32_t>(plan.destination_size);
    const std::size_t input_elements = checked_product(
        plan.source_size, vector_count, "Vulkan 1D source");
    const std::size_t output_elements = checked_product(
        plan.destination_size, vector_count, "Vulkan 1D output");
    const std::size_t input_bytes = checked_product(
        input_elements, sizeof(float), "Vulkan 1D source");
    const std::size_t output_bytes = checked_product(
        output_elements, sizeof(float), "Vulkan 1D output");
    const std::size_t input_row_bytes = checked_product(
        input_width, sizeof(float), "Vulkan 1D source row");
    const std::size_t input_pitch = checked_product(
        static_cast<std::size_t>(input_row_stride), sizeof(float),
        "Vulkan 1D source pitch");
    const std::size_t output_row_bytes = checked_product(
        output_width, sizeof(float), "Vulkan 1D output row");
    const std::size_t output_pitch = checked_product(
        static_cast<std::size_t>(output_row_stride), sizeof(float),
        "Vulkan 1D output pitch");

    const bool cache_requested = input_lifetime && runtime->input_cache->enabled();
    std::shared_ptr<CachedInput> cached_input;
    std::optional<InputCacheKey> cache_key;
    bool cache_producer = false;
    if (cache_requested) {
        cache_key = make_input_cache_key(
            input, input_row_stride, input_width, input_rows,
            CachedInputLayout::row_major);
        auto acquired = runtime->input_cache->acquire(
            *cache_key, input_bytes, std::move(input_lifetime));
        cached_input = std::move(acquired.input);
        cache_producer = acquired.producer;
    }
    const bool reused_input = cached_input && !cache_producer;
    const std::uint64_t input_wait_value = reused_input
        ? cached_input->wait_value() : 0U;

    WorkspaceBuilder builder;
    const std::uint32_t source_offset = builder.add_bytes(input_bytes);
    const std::uint32_t output_offset = builder.add_bytes(output_bytes);
    // Rows are solved with the partitioned solve, which works on vector-major
    // data, and transposed back; that is faster than solving them in place
    // one thread per row.
    const bool transposed_solve =
        rows && packed->solve_chunks != 0U && runtime->fold_transposes;
    const std::uint32_t solved_offset =
        transposed_solve ? builder.add_bytes(output_bytes) : output_offset;
    const std::size_t download_offset = input_bytes;
    const std::size_t staging_bytes = checked_add(
        download_offset, output_bytes, "Vulkan 1D staging");
    // Rows that lie in imported frame blocks are copied directly.
    const HostRange source_range = resident
        ? resident_range(*resident->source)
        : host_range(1U, input, input_pitch, input_row_bytes, input_rows);
    const HostRange result_range = resident
        ? resident_range(*resident->result)
        : host_range(2U, output, output_pitch, output_row_bytes, output_rows);
    const ExternalWait resident_source_wait = resident_wait(resident);
    const std::size_t slot_index = runtime->acquire_slot(false, false);
    SlotRelease release(*runtime, slot_index, false);
    ExecutionSlot &slot = *runtime->slots[slot_index];
    slot.reserve(builder.size(), staging_bytes);
    if (!reused_input && !source_range) {
        pack_host_rows(
            input, input_pitch, slot.staging.mapped(),
            input_row_bytes, input_rows);
        slot.staging.flush(0U, input_bytes);
    }

    bool submitted = false;
    bool cache_published = false;
    PendingPlanUploads plan_uploads(packed);
    try {
        slot.begin();
        plan_uploads.record(slot.command);
        runtime->begin_label(slot.command, rows
            ? "dsmvc row inverse" : "dsmvc column inverse");
        if (!reused_input) {
            const VkBufferCopy copy{
                0U,
                cached_input ? cached_input->offset()
                             : static_cast<VkDeviceSize>(source_offset) * 4U,
                input_bytes,
            };
            if (source_range) {
                record_row_copies(
                    slot.command, source_range.buffer(), source_range.offset,
                    input_pitch,
                    cached_input ? cached_input->buffer()
                                 : slot.workspace.handle(),
                    copy.dstOffset, input_row_bytes, input_row_bytes,
                    input_rows);
            } else {
                vkCmdCopyBuffer(
                    slot.command, slot.staging.handle(),
                    cached_input ? cached_input->buffer()
                                 : slot.workspace.handle(),
                    1U, &copy);
            }
            command_barrier(
                slot.command,
                cached_input ? cached_input->buffer() : slot.workspace.handle(),
                copy.dstOffset, copy.size, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
        record_inverse(
            *runtime, slot, *packed,
            cached_input ? cached_input->buffer() : VK_NULL_HANDLE,
            cached_input ? cached_input->offset() : 0U,
            cached_input ? cached_input->size() : 0U,
            vector_count, cached_input ? 0U : source_offset, solved_offset,
            cached_input ? 1U : 0U, rows ? 1U : 0U,
            rows && !transposed_solve ? 1U : 0U,
            should_split_rhs(*runtime, *packed));
        if (transposed_solve) {
            command_barrier(
                slot.command, slot.workspace.handle(),
                static_cast<VkDeviceSize>(solved_offset) * 4U, output_bytes,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            // Sample x of row y lies at x * rows + y: an image `rows` wide.
            record_transpose(
                *runtime, slot, *packed, VK_NULL_HANDLE, 0U, 0U,
                vector_count, static_cast<std::uint32_t>(plan.destination_size),
                solved_offset, output_offset, 0U, 0U, 0U, nullptr);
        }
        command_barrier(
            slot.command, slot.workspace.handle(),
            static_cast<VkDeviceSize>(output_offset) * 4U, output_bytes,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
        const VkBufferCopy result_copy{
            static_cast<VkDeviceSize>(output_offset) * 4U,
            download_offset,
            output_bytes,
        };
        if (result_range) {
            record_row_copies(
                slot.command, slot.workspace.handle(), result_copy.srcOffset,
                output_row_bytes, result_range.buffer(), result_range.offset,
                output_pitch, output_row_bytes, output_rows);
            command_barrier(
                slot.command, result_range.buffer(), result_range.offset,
                host_blocks::plane_span(
                    output_pitch, output_row_bytes, output_rows),
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
        } else {
            vkCmdCopyBuffer(
                slot.command, slot.workspace.handle(), slot.staging.handle(),
                1U, &result_copy);
            command_barrier(
                slot.command, slot.staging.handle(), download_offset,
                output_bytes, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT);
        }
        runtime->end_label(slot.command);
        runtime->check(vkEndCommandBuffer(slot.command),
                       "vkEndCommandBuffer(1D inverse)");
        const std::uint64_t signal_value = runtime->submit(
            slot.command, slot.fence, input_wait_value,
            cache_producer && runtime->timeline_enabled,
            resident ? &resident_source_wait : nullptr, resident ? 1U : 0U);
        submitted = true;
        if (cache_producer && runtime->timeline_enabled) {
            cached_input->publish(signal_value);
            cache_published = true;
        }
        runtime->wait_fence(slot.fence, "Vulkan 1D execution");
        if (cache_producer && !cache_published) {
            cached_input->publish(0U);
            cache_published = true;
        }
        plan_uploads.complete();
    } catch (...) {
        const auto error = std::current_exception();
        if (!submitted) slot.recover_unsubmitted();
        if (cache_producer && !cache_published) {
            cached_input->fail(error);
            runtime->input_cache->erase(*cache_key, cached_input);
        }
        std::rethrow_exception(error);
    }
    if (!result_range) {
        slot.staging.invalidate(download_offset, output_bytes);
        unpack_host_rows(
            static_cast<const std::byte *>(slot.staging.mapped())
                + static_cast<std::ptrdiff_t>(download_offset),
            output_row_bytes, output, output_pitch, output_rows);
    }
}

bool backend_available() noexcept {
    try {
        require_backend_available();
        return true;
    } catch (...) {
        return false;
    }
}

void require_backend_available() {
    static std::once_flag once;
    std::call_once(once, [] {
        auto instance = create_instance(false);
        (void)select_device(enumerate_devices(instance.instance));
    });
}

VulkanFloat64Capabilities selected_float64_capabilities() {
    auto instance = create_instance(false);
    return select_device(enumerate_devices(instance.instance)).float64;
}

std::string selected_float64_capability_report() {
    auto instance = create_instance(false);
    return float64_capability_report(
        select_device(enumerate_devices(instance.instance)));
}

VulkanExecutor::VulkanExecutor() : impl_(std::make_shared<Impl>()) {}
VulkanExecutor::~VulkanExecutor() = default;

const char *VulkanExecutor::name() const noexcept { return "vulkan"; }

bool VulkanExecutor::input_cache_enabled() const noexcept {
    return impl_->runtime->input_cache->enabled();
}

void VulkanExecutor::prepare(std::shared_ptr<const AxisPlan> plan) const {
    if (!plan || !plan->valid()) {
        throw std::invalid_argument("cannot prepare an invalid Vulkan axis plan");
    }
    const std::scoped_lock lock(impl_->mutex);
    if (impl_->sealed.load(std::memory_order_relaxed)) {
        throw std::logic_error("cannot add an axis to a sealed Vulkan plan cache");
    }
    if (impl_->find(*plan) == impl_->plans.end()) {
        const bool float64 = plan->requires_float64();
        auto packed = acquire_packed_plan(impl_->runtime, plan, float64);
        VulkanExecutor::Impl::PreparedPlan prepared;
        prepared.axis = std::move(plan);
        prepared.identity = packed->identity;
        if (float64) {
            prepared.packed_f64 = packed;
        } else {
            prepared.packed_f32 = packed;
        }
        impl_->plans.push_back(std::move(prepared));
    }
}

void VulkanExecutor::seal() const {
    const std::scoped_lock lock(impl_->mutex);
    impl_->sealed.store(true, std::memory_order_release);
}


namespace {

// One runtime per borrowed device, shared by every executor on that core and
// gone with the last of them: it must not outlive the core's device.
[[nodiscard]] std::shared_ptr<Runtime> borrowed_runtime(
    const VulkanCoreDevice &device) {
    static std::mutex mutex;
    static std::map<void *, std::weak_ptr<Runtime>> runtimes;
    const std::scoped_lock lock(mutex);
    if (auto found = runtimes.find(device.device); found != runtimes.end()) {
        if (auto existing = found->second.lock()) return existing;
    }
    auto created = std::make_shared<Runtime>(device);
    runtimes[device.device] = created;
    return created;
}

// A stride in elements of `element_bytes`; GPU frame pitches are multiples of
// the sample size.
[[nodiscard]] std::ptrdiff_t resident_stride(
    const GpuPlane &plane, std::size_t element_bytes = sizeof(float)) {
    if (plane.buffer == 0U || plane.pitch_bytes <= 0
        || plane.pitch_bytes % static_cast<std::ptrdiff_t>(element_bytes) != 0) {
        throw std::invalid_argument("invalid GPU plane");
    }
    return plane.pitch_bytes / static_cast<std::ptrdiff_t>(element_bytes);
}

} // namespace

VulkanExecutor::VulkanExecutor(const VulkanCoreDevice &device)
    : impl_(std::make_shared<Impl>(borrowed_runtime(device))) {}

void VulkanExecutor::inverse_rows_resident(
    const AxisPlan &plan, const GpuPlane &input, const GpuPlane &output,
    std::int32_t row_count) const {
    if (!plan.valid() || row_count < 0) {
        throw std::invalid_argument("invalid Vulkan row executor arguments");
    }
    if (row_count == 0) return;
    const ResidentIo io{&input, &output};
    // The host pointers are never dereferenced when both sides are resident.
    impl_->execute_one_dimensional(
        plan, reinterpret_cast<const float *>(&io), resident_stride(input),
        const_cast<float *>(reinterpret_cast<const float *>(&io)), resident_stride(output),
        static_cast<std::uint32_t>(row_count), true, {}, &io);
}

void VulkanExecutor::inverse_columns_resident(
    const AxisPlan &plan, const GpuPlane &input, const GpuPlane &output,
    std::int32_t column_count) const {
    if (!plan.valid() || column_count < 0) {
        throw std::invalid_argument("invalid Vulkan column executor arguments");
    }
    if (column_count == 0) return;
    const ResidentIo io{&input, &output};
    impl_->execute_one_dimensional(
        plan, reinterpret_cast<const float *>(&io), resident_stride(input),
        const_cast<float *>(reinterpret_cast<const float *>(&io)), resident_stride(output),
        static_cast<std::uint32_t>(column_count), false, {}, &io);
}

void VulkanExecutor::inverse_2d_u8_resident(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const GpuPlane &input, const GpuPlane &output,
    const IntegerConversion &conversion) const {
    if (!horizontal.valid() || !vertical.valid()
        || !valid_conversion<std::uint8_t>(conversion)) {
        throw std::invalid_argument("invalid Vulkan u8 2D arguments");
    }
    const ResidentIo io{&input, &output};
    const auto *host = reinterpret_cast<const std::uint8_t *>(&io);
    impl_->execute_2d<std::uint8_t>(
        horizontal, vertical, host, resident_stride(input, 1U),
        const_cast<std::uint8_t *>(host), resident_stride(output, 1U),
        &conversion, {}, &io);
}

void VulkanExecutor::inverse_2d_u16_resident(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const GpuPlane &input, const GpuPlane &output,
    const IntegerConversion &conversion) const {
    if (!horizontal.valid() || !vertical.valid()
        || !valid_conversion<std::uint16_t>(conversion)) {
        throw std::invalid_argument("invalid Vulkan u16 2D arguments");
    }
    const ResidentIo io{&input, &output};
    const auto *host = reinterpret_cast<const std::uint16_t *>(&io);
    impl_->execute_2d<std::uint16_t>(
        horizontal, vertical, host, resident_stride(input, 2U),
        const_cast<std::uint16_t *>(host), resident_stride(output, 2U),
        &conversion, {}, &io);
}

void VulkanExecutor::inverse_2d_resident(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const GpuPlane &input, const GpuPlane &output) const {
    if (!horizontal.valid() || !vertical.valid()) {
        throw std::invalid_argument("invalid Vulkan 2D executor arguments");
    }
    const ResidentIo io{&input, &output};
    impl_->execute_2d<float>(
        horizontal, vertical, reinterpret_cast<const float *>(&io),
        resident_stride(input), const_cast<float *>(reinterpret_cast<const float *>(&io)),
        resident_stride(output), nullptr, {}, &io);
}

bool VulkanExecutor::inverse_2d_frame(
    SampleKind kind, std::span<const FramePlane2D> planes, bool gpu_frame,
    const std::shared_ptr<const void> &input_lifetime) const {
    switch (kind) {
    case SampleKind::float32:
        return impl_->execute_2d_frame<float>(planes, gpu_frame, input_lifetime);
    case SampleKind::u8:
        return impl_->execute_2d_frame<std::uint8_t>(
            planes, gpu_frame, input_lifetime);
    case SampleKind::u16:
        return impl_->execute_2d_frame<std::uint16_t>(
            planes, gpu_frame, input_lifetime);
    }
    return false;
}

void VulkanExecutor::inverse_rows(
    const AxisPlan &plan, const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride, std::int32_t row_count,
    std::shared_ptr<const void> input_lifetime) const {
    if (!plan.valid() || !input || !output
        || input_row_stride < plan.source_size
        || output_row_stride < plan.destination_size || row_count < 0) {
        throw std::invalid_argument("invalid Vulkan row executor arguments");
    }
    if (row_count == 0) return;
    impl_->execute_one_dimensional(
        plan, input, input_row_stride, output, output_row_stride,
        static_cast<std::uint32_t>(row_count), true,
        std::move(input_lifetime));
}

void VulkanExecutor::inverse_columns(
    const AxisPlan &plan, const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride, std::int32_t column_count,
    std::shared_ptr<const void> input_lifetime) const {
    if (!plan.valid() || !input || !output || column_count < 0
        || input_row_stride < column_count
        || output_row_stride < column_count) {
        throw std::invalid_argument("invalid Vulkan column executor arguments");
    }
    if (column_count == 0) return;
    impl_->execute_one_dimensional(
        plan, input, input_row_stride, output, output_row_stride,
        static_cast<std::uint32_t>(column_count), false,
        std::move(input_lifetime));
}

void VulkanExecutor::inverse_2d(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::shared_ptr<const void> input_lifetime) const {
    if (!horizontal.valid() || !vertical.valid() || !input || !output
        || input_row_stride < horizontal.source_size
        || output_row_stride < horizontal.destination_size
        || vertical.source_size < 1 || vertical.destination_size < 1) {
        throw std::invalid_argument("invalid Vulkan 2D executor arguments");
    }
    impl_->execute_2d(
        horizontal, vertical, input, input_row_stride, output,
        output_row_stride, static_cast<const IntegerConversion *>(nullptr),
        std::move(input_lifetime));
}

void VulkanExecutor::inverse_2d_u8(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::uint8_t *input, std::ptrdiff_t input_row_stride,
    std::uint8_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion,
    std::shared_ptr<const void> input_lifetime) const {
    if (!horizontal.valid() || !vertical.valid() || !input || !output
        || input_row_stride < horizontal.source_size
        || output_row_stride < horizontal.destination_size
        || vertical.source_size < 1 || vertical.destination_size < 1
        || !valid_conversion<std::uint8_t>(conversion)) {
        throw std::invalid_argument("invalid Vulkan u8 2D arguments");
    }
    impl_->execute_2d(
        horizontal, vertical, input, input_row_stride, output,
        output_row_stride, &conversion, std::move(input_lifetime));
}

void VulkanExecutor::inverse_2d_u16(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::uint16_t *input, std::ptrdiff_t input_row_stride,
    std::uint16_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion,
    std::shared_ptr<const void> input_lifetime) const {
    if (!horizontal.valid() || !vertical.valid() || !input || !output
        || input_row_stride < horizontal.source_size
        || output_row_stride < horizontal.destination_size
        || vertical.source_size < 1 || vertical.destination_size < 1
        || !valid_conversion<std::uint16_t>(conversion)) {
        throw std::invalid_argument("invalid Vulkan u16 2D arguments");
    }
    impl_->execute_2d(
        horizontal, vertical, input, input_row_stride, output,
        output_row_stride, &conversion, std::move(input_lifetime));
}

} // namespace dsmvc::vulkan_detail
