#include "vulkan_executor.hpp"

#include "vulkan_convert_128_spv.hpp"
#include "vulkan_convert_256_spv.hpp"
#include "vulkan_inverse_32_spv.hpp"
#include "vulkan_inverse_64_spv.hpp"
#include "vulkan_rhs_128_spv.hpp"
#include "vulkan_rhs_256_spv.hpp"
#include "vulkan_solve_32_spv.hpp"
#include "vulkan_solve_64_spv.hpp"
#include "vulkan_transpose_128_spv.hpp"
#include "vulkan_transpose_256_spv.hpp"

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
#include <iomanip>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dsmvc::vulkan_detail {

namespace {

constexpr VkDeviceSize maximum_allocation_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::size_t descriptor_set_count = 8U;
constexpr std::size_t default_plan_cache_bytes = 16U * 1024U * 1024U;
constexpr std::size_t default_input_cache_bytes = 64U * 1024U * 1024U;

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

    InstanceHandle() = default;
    InstanceHandle(const InstanceHandle &) = delete;
    InstanceHandle &operator=(const InstanceHandle &) = delete;
    InstanceHandle(InstanceHandle &&other) noexcept
        : instance(std::exchange(other.instance, VK_NULL_HANDLE)),
          messenger(std::exchange(other.messenger, VK_NULL_HANDLE)),
          destroy_messenger(std::exchange(other.destroy_messenger, nullptr)),
          debug_utils(other.debug_utils) {}
    InstanceHandle &operator=(InstanceHandle &&other) noexcept {
        if (this == &other) return *this;
        reset();
        instance = std::exchange(other.instance, VK_NULL_HANDLE);
        messenger = std::exchange(other.messenger, VK_NULL_HANDLE);
        destroy_messenger = std::exchange(other.destroy_messenger, nullptr);
        debug_utils = other.debug_utils;
        return *this;
    }
    ~InstanceHandle() { reset(); }

    void reset() noexcept {
        if (messenger && destroy_messenger) {
            destroy_messenger(instance, messenger, nullptr);
        }
        messenger = VK_NULL_HANDLE;
        if (instance) vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
};

[[nodiscard]] InstanceHandle create_instance(bool validation) {
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
        DeviceInfo info;
        info.index = index;
        info.physical = physical_devices[index];
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
        result.push_back(std::move(info));
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
            << " shaderFloat64=" << capabilities.shader_float64
            << " shaderRoundingModeRTEFloat64="
            << capabilities.rounding_mode_rte_float64
            << " shaderSignedZeroInfNanPreserveFloat64="
            << capabilities.signed_zero_inf_nan_preserve_float64
            << " shaderDenormPreserveFloat64="
            << capabilities.denorm_preserve_float64
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
class StagingPool;
struct ExecutionSlot;

class Runtime : public std::enable_shared_from_this<Runtime> {
public:
    Runtime();
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
    [[nodiscard]] std::uint64_t submit(
        VkCommandBuffer command, VkFence fence, std::uint64_t wait_value,
        bool signal_timeline);
    void wait_fence(VkFence fence, const char *label);
    void wait_timeline(std::uint64_t value, const char *label);

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
    VkPipeline convert_pipeline = VK_NULL_HANDLE;
    VkSemaphore timeline = VK_NULL_HANDLE;
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
    SplitRhsMode split_rhs = SplitRhsMode::adaptive;
    std::unique_ptr<DeviceArena> plan_arena;
    std::unique_ptr<DeviceArena> input_arena;
    std::unique_ptr<InputCache> input_cache;
    std::unique_ptr<StagingPool> staging_pool;
    std::vector<std::unique_ptr<ExecutionSlot>> slots;
    std::vector<bool> slot_busy;

private:
    void destroy_device_objects() noexcept;

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

struct PushConstants {
    std::array<std::uint32_t, 32U> value{};
};
static_assert(sizeof(PushConstants) == 128U);

} // namespace


namespace {

[[nodiscard]] std::size_t packed_plan_storage_bytes(const AxisPlan &axis) {
    std::size_t total = 0U;
    const auto add = [&](const auto &values) {
        using Value = typename std::remove_cvref_t<decltype(values)>::value_type;
        static_assert(sizeof(Value) == sizeof(std::uint32_t));
        total = checked_add(
            total,
            checked_product(values.size(), sizeof(Value), "Vulkan plan field"),
            "Vulkan packed plan");
    };
    add(axis.transpose_offsets);
    add(axis.transpose_indices);
    add(axis.transpose_weights);
    add(axis.lower_ld);
    add(axis.upper_l);
    add(axis.inverse_diagonal);
    if (total == 0U || total > maximum_allocation_bytes) {
        throw std::length_error("Vulkan packed plan exceeds the 2 GiB guard");
    }
    return total;
}

class PackedPlan {
public:
    PackedPlan(std::shared_ptr<Runtime> requested_runtime,
               std::shared_ptr<const AxisPlan> requested_axis,
               const AxisPlan *requested_identity)
        : runtime(std::move(requested_runtime)), axis(std::move(requested_axis)),
          identity(requested_identity ? requested_identity : axis.get()) {
        if (!axis || !axis->valid()) {
            throw std::invalid_argument("cannot pack an invalid Vulkan axis plan");
        }
        source_size = static_cast<std::uint32_t>(axis->source_size);
        destination_size = static_cast<std::uint32_t>(axis->destination_size);
        half_bandwidth = static_cast<std::uint32_t>(axis->half_bandwidth);
        storage_bytes = packed_plan_storage_bytes(*axis);
        if (storage_bytes > runtime->selected.properties.limits.maxStorageBufferRange) {
            throw std::length_error(
                "Vulkan packed plan exceeds maxStorageBufferRange");
        }

        std::vector<std::uint32_t> packed(storage_bytes / sizeof(std::uint32_t));
        std::size_t cursor = 0U;
        const auto copy = [&](const auto &values, std::uint32_t &word_offset) {
            using Value = typename std::remove_cvref_t<decltype(values)>::value_type;
            word_offset = checked_u32(cursor / sizeof(std::uint32_t),
                                      "Vulkan plan field offset");
            const std::size_t bytes = values.size() * sizeof(Value);
            if (bytes != 0U) {
                std::memcpy(reinterpret_cast<std::byte *>(packed.data())
                                + static_cast<std::ptrdiff_t>(cursor),
                            values.data(), bytes);
            }
            cursor += bytes;
        };
        copy(axis->transpose_offsets, offsets_offset);
        copy(axis->transpose_indices, indices_offset);
        copy(axis->transpose_weights, weights_offset);
        copy(axis->lower_ld, lower_offset);
        copy(axis->upper_l, upper_offset);
        copy(axis->inverse_diagonal, diagonal_offset);

        storage = runtime->plan_arena->allocate(storage_bytes);
        staging = Buffer(
            *runtime, storage_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            "dsmvc plan upload staging");
        std::memcpy(staging.mapped(), packed.data(), storage_bytes);
        staging.flush(0U, storage_bytes);

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
            const VkBufferCopy copy_region{0U, storage.offset(), storage_bytes};
            vkCmdCopyBuffer(
                command, staging.handle(), storage.buffer(), 1U, &copy_region);
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
            runtime->check(vkEndCommandBuffer(command),
                           "vkEndCommandBuffer(plan upload)");

            if (runtime->timeline_enabled) {
                ready_value = runtime->submit(
                    command, VK_NULL_HANDLE, 0U, true);
            } else {
                const VkFenceCreateInfo fence_info{
                    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0U};
                runtime->check(vkCreateFence(
                                   runtime->device, &fence_info, nullptr,
                                   &upload_fence),
                               "vkCreateFence(plan upload)");
                (void)runtime->submit(command, upload_fence, 0U, false);
                runtime->wait_fence(upload_fence, "Vulkan plan upload");
                retire_upload();
            }
        } catch (...) {
            retire_upload();
            throw;
        }
    }

    ~PackedPlan() {
        try {
            if (ready_value != 0U) {
                runtime->wait_timeline(ready_value, "Vulkan plan retirement");
            }
        } catch (...) {
        }
        retire_upload();
    }

    [[nodiscard]] std::uint64_t wait_value() const noexcept {
        return ready_value;
    }

    void mark_upload_complete() const noexcept {
        if (ready_value == 0U) return;
        const std::scoped_lock lock(upload_mutex);
        if (!upload_pool) return;
        std::uint64_t completed = 0U;
        if (vkGetSemaphoreCounterValue(
                runtime->device, runtime->timeline, &completed) == VK_SUCCESS
            && completed >= ready_value) {
            const_cast<PackedPlan *>(this)->retire_upload_locked();
        }
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
    std::size_t storage_bytes = 0U;
    mutable std::atomic<std::uint32_t> execution_count{0U};

private:
    void retire_upload() noexcept {
        const std::scoped_lock lock(upload_mutex);
        retire_upload_locked();
    }

    void retire_upload_locked() noexcept {
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
    Buffer staging;
    VkCommandPool upload_pool = VK_NULL_HANDLE;
    VkFence upload_fence = VK_NULL_HANDLE;
    std::uint64_t ready_value = 0U;
    mutable std::mutex upload_mutex;
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

[[nodiscard]] std::shared_ptr<const PackedPlan> acquire_packed_plan(
    const std::shared_ptr<Runtime> &runtime,
    const std::shared_ptr<const AxisPlan> &plan) {
    struct Entry {
        std::shared_ptr<PackedPlanRequest> request;
        std::list<const AxisPlan *>::iterator lru;
        std::size_t bytes = 0U;
    };
    struct Cache {
        std::mutex mutex;
        std::unordered_map<const AxisPlan *, Entry> entries;
        std::list<const AxisPlan *> lru;
        std::size_t bytes = 0U;
        const std::size_t capacity = environment_megabytes(
            "DSMVC_VULKAN_PLAN_CACHE_MB", default_plan_cache_bytes);
    };
    static Cache cache;

    const std::size_t bytes = packed_plan_storage_bytes(*plan);
    std::shared_ptr<PackedPlanRequest> request;
    std::vector<std::shared_ptr<PackedPlanRequest>> evicted;
    bool producer = false;
    {
        const std::scoped_lock lock(cache.mutex);
        if (const auto found = cache.entries.find(plan.get());
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
            cache.lru.push_back(plan.get());
            cache.entries.emplace(
                plan.get(), Entry{request, std::prev(cache.lru.end()), bytes});
            while (cache.bytes > cache.capacity && cache.entries.size() > 1U) {
                const AxisPlan *victim_key = cache.lru.front();
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
            runtime, plan, plan.get());
        request->publish(packed);
        return packed;
    } catch (...) {
        const auto error = std::current_exception();
        request->fail(error);
        std::shared_ptr<PackedPlanRequest> removed;
        {
            const std::scoped_lock lock(cache.mutex);
            const auto found = cache.entries.find(plan.get());
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
    if (source_pitch == row_bytes) {
        std::memcpy(destination, source, row_bytes * rows);
        return;
    }
    const auto *source_bytes = static_cast<const std::byte *>(source);
    auto *destination_bytes = static_cast<std::byte *>(destination);
    for (std::size_t row = 0U; row < rows; ++row) {
        std::memcpy(
            destination_bytes + static_cast<std::ptrdiff_t>(row * row_bytes),
            source_bytes + static_cast<std::ptrdiff_t>(row * source_pitch),
            row_bytes);
    }
}

void unpack_host_rows(
    const void *source, std::size_t row_bytes, void *destination,
    std::size_t destination_pitch, std::size_t rows) {
    if (destination_pitch == row_bytes) {
        std::memcpy(destination, source, row_bytes * rows);
        return;
    }
    const auto *source_bytes = static_cast<const std::byte *>(source);
    auto *destination_bytes = static_cast<std::byte *>(destination);
    for (std::size_t row = 0U; row < rows; ++row) {
        std::memcpy(
            destination_bytes + static_cast<std::ptrdiff_t>(row * destination_pitch),
            source_bytes + static_cast<std::ptrdiff_t>(row * row_bytes),
            row_bytes);
    }
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

template <class Slot>
void record_inverse(
    Runtime &runtime, Slot &slot, const PackedPlan &plan,
    VkBuffer cache_buffer, VkDeviceSize cache_offset, VkDeviceSize cache_size,
    std::uint32_t vector_count, std::uint32_t source_offset,
    std::uint32_t output_offset, std::uint32_t source_buffer,
    std::uint32_t source_layout, std::uint32_t output_layout,
    bool split) {
    PushConstants push;
    fill_plan_push(push, plan);
    push.value[3] = vector_count;
    push.value[4] = source_offset;
    push.value[5] = output_offset;
    push.value[6] = source_buffer;
    push.value[7] = source_layout;
    push.value[8] = output_layout;
    if (split) {
        VkDescriptorSet rhs_set = slot.descriptor_set(
            plan.buffer(), plan.buffer_offset(), plan.buffer_size(),
            cache_buffer, cache_offset, cache_size);
        bind_compute(runtime, slot, runtime.rhs_pipeline, rhs_set, push);
        vkCmdDispatch(
            slot.command, divide_up(vector_count, runtime.rhs_local_x),
            divide_up(plan.destination_size, runtime.rhs_local_y), 1U);
        command_barrier(
            slot.command, slot.workspace.handle(),
            static_cast<VkDeviceSize>(output_offset) * 4U,
            static_cast<VkDeviceSize>(plan.destination_size) * vector_count * 4U,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        VkDescriptorSet solve_set = slot.descriptor_set(
            plan.buffer(), plan.buffer_offset(), plan.buffer_size(),
            cache_buffer, cache_offset, cache_size);
        bind_compute(runtime, slot, runtime.solve_pipeline, solve_set, push);
    } else {
        VkDescriptorSet inverse_set = slot.descriptor_set(
            plan.buffer(), plan.buffer_offset(), plan.buffer_size(),
            cache_buffer, cache_offset, cache_size);
        bind_compute(runtime, slot, runtime.inverse_pipeline, inverse_set, push);
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
    const VkDeviceCreateInfo device_info{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        timeline_enabled ? &timeline_enable : nullptr,
        0U,
        1U,
        &queue_info,
        0U,
        nullptr,
        0U,
        nullptr,
        nullptr,
    };
    check(vkCreateDevice(selected.physical, &device_info, nullptr, &device),
          "vkCreateDevice");
    try {
        vkGetDeviceQueue(device, selected.queue_family, 0U, &queue);
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
}

Runtime::~Runtime() { destroy_device_objects(); }

void Runtime::destroy_device_objects() noexcept {
    if (!device) return;
    (void)vkDeviceWaitIdle(device);
    slots.clear();
    staging_pool.reset();
    if (input_cache) input_cache->reset();
    input_cache.reset();
    if (input_arena) input_arena->reset();
    if (plan_arena) plan_arena->reset();
    input_arena.reset();
    plan_arena.reset();
    if (timeline) vkDestroySemaphore(device, timeline, nullptr);
    timeline = VK_NULL_HANDLE;
    for (VkPipeline *pipeline : {&convert_pipeline, &solve_pipeline,
                                 &rhs_pipeline, &inverse_pipeline,
                                 &transpose_pipeline}) {
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
    vkDestroyDevice(device, nullptr);
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

std::uint64_t Runtime::submit(
    VkCommandBuffer command, VkFence fence, std::uint64_t wait_value,
    bool signal_timeline) {
    throw_if_failed();
    const std::scoped_lock lock(queue_mutex_);
    std::uint64_t signal_value = 0U;
    if (timeline_enabled && signal_timeline) signal_value = ++next_timeline_value_;
    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkTimelineSemaphoreSubmitInfo timeline_info{
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        nullptr,
        wait_value != 0U ? 1U : 0U,
        wait_value != 0U ? &wait_value : nullptr,
        signal_value != 0U ? 1U : 0U,
        signal_value != 0U ? &signal_value : nullptr,
    };
    const VkSubmitInfo submit_info{
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        (wait_value != 0U || signal_value != 0U) ? &timeline_info : nullptr,
        wait_value != 0U ? 1U : 0U,
        wait_value != 0U ? &timeline : nullptr,
        wait_value != 0U ? &wait_stage : nullptr,
        1U,
        &command,
        signal_value != 0U ? 1U : 0U,
        signal_value != 0U ? &timeline : nullptr,
    };
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

void Runtime::wait_timeline(std::uint64_t value, const char *label) {
    if (!timeline_enabled || value == 0U) return;
    const VkSemaphoreWaitInfo wait_info{
        VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        nullptr,
        0U,
        1U,
        &timeline,
        &value,
    };
    check(vkWaitSemaphores(
              device, &wait_info, std::numeric_limits<std::uint64_t>::max()),
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
    ~Impl() { runtime->detach_executor(); }

    struct PreparedPlan {
        std::shared_ptr<const AxisPlan> axis;
        const AxisPlan *identity = nullptr;
        std::weak_ptr<const PackedPlan> packed;
    };

    [[nodiscard]] auto find(const AxisPlan &plan) const {
        return std::find_if(
            plans.begin(), plans.end(), [&plan](const auto &candidate) {
                return candidate.identity == &plan;
            });
    }

    [[nodiscard]] std::shared_ptr<const PackedPlan> get(
        const AxisPlan &plan) const {
        if (sealed.load(std::memory_order_acquire)) {
            const auto found = find(plan);
            if (found != plans.end()) {
                if (auto packed = found->packed.lock()) return packed;
                return acquire_packed_plan(runtime, found->axis);
            }
        } else {
            const std::scoped_lock lock(mutex);
            const auto found = find(plan);
            if (found != plans.end()) {
                if (auto packed = found->packed.lock()) return packed;
                return acquire_packed_plan(runtime, found->axis);
            }
        }
        auto owned = std::make_shared<const AxisPlan>(plan);
        return std::make_shared<const PackedPlan>(
            runtime, std::move(owned), &plan);
    }

    template <class Sample>
    void execute_2d(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const Sample *input, std::ptrdiff_t input_row_stride,
        Sample *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion *conversion,
        std::shared_ptr<const void> input_lifetime) const {
        const auto packed_horizontal = get(horizontal);
        const auto packed_vertical = get(vertical);
        const auto source_width = static_cast<std::uint32_t>(horizontal.source_size);
        const auto source_height = static_cast<std::uint32_t>(vertical.source_size);
        const auto destination_width =
            static_cast<std::uint32_t>(horizontal.destination_size);
        const auto destination_height =
            static_cast<std::uint32_t>(vertical.destination_size);
        const std::size_t source_elements = checked_product(
            source_width, source_height, "Vulkan source image");
        const std::size_t intermediate_elements = checked_product(
            destination_width, source_height, "Vulkan intermediate image");
        const std::size_t destination_elements = checked_product(
            destination_width, destination_height, "Vulkan destination image");
        const std::size_t source_bytes = checked_product(
            source_elements, sizeof(Sample), "Vulkan source image");
        const std::size_t source_copy_bytes = align_up(
            source_bytes, 4U, "Vulkan packed source copy");
        const std::size_t transposed_bytes = checked_product(
            source_elements, sizeof(float), "Vulkan transposed image");
        const std::size_t intermediate_bytes = checked_product(
            intermediate_elements, sizeof(float), "Vulkan intermediate image");
        const std::size_t destination_bytes = checked_product(
            destination_elements, sizeof(float), "Vulkan destination image");
        const std::size_t result_bytes = checked_product(
            destination_elements, sizeof(Sample), "Vulkan result image");
        const std::size_t result_copy_bytes = align_up(
            result_bytes, 4U, "Vulkan packed result copy");
        const std::size_t source_row_bytes = checked_product(
            source_width, sizeof(Sample), "Vulkan source row");
        const std::size_t source_pitch = checked_product(
            static_cast<std::size_t>(input_row_stride), sizeof(Sample),
            "Vulkan source pitch");
        const std::size_t result_row_bytes = checked_product(
            destination_width, sizeof(Sample), "Vulkan result row");
        const std::size_t result_pitch = checked_product(
            static_cast<std::size_t>(output_row_stride), sizeof(Sample),
            "Vulkan result pitch");

        const bool cache_requested = input_lifetime && runtime->input_cache->enabled();
        std::shared_ptr<CachedInput> cached_input;
        std::optional<InputCacheKey> cache_key;
        bool cache_producer = false;
        if (cache_requested) {
            cache_key = make_input_cache_key(
                input, input_row_stride, source_width, source_height,
                CachedInputLayout::transposed, conversion);
            auto acquired = runtime->input_cache->acquire(
                *cache_key, transposed_bytes, std::move(input_lifetime));
            cached_input = std::move(acquired.input);
            cache_producer = acquired.producer;
        }
        const bool reused_input = cached_input && !cache_producer;
        std::uint64_t input_wait_value = 0U;
        if (reused_input) input_wait_value = cached_input->wait_value();

        WorkspaceBuilder builder;
        const std::uint32_t source_offset = builder.add_bytes(
            std::max(source_copy_bytes, intermediate_bytes));
        const std::uint32_t transposed_offset = builder.add_bytes(transposed_bytes);
        const std::uint32_t intermediate_offset = builder.add_bytes(intermediate_bytes);
        const std::uint32_t destination_offset = builder.add_bytes(destination_bytes);
        std::uint32_t result_offset = destination_offset;
        if constexpr (!std::is_same_v<Sample, float>) {
            result_offset = builder.add_bytes(result_copy_bytes);
        }
        const std::size_t download_offset = source_copy_bytes;
        const std::size_t staging_bytes = checked_add(
            download_offset, result_copy_bytes, "Vulkan staging buffer");
        if (builder.size() > maximum_allocation_bytes
            || staging_bytes > maximum_allocation_bytes) {
            throw std::length_error("Vulkan execution exceeds the 2 GiB guard");
        }

        const bool host_limited = std::is_same_v<Sample, float> && !reused_input;
        const bool heavy = !host_limited
            && std::max(horizontal.half_bandwidth, vertical.half_bandwidth) >= 7;
        std::optional<StagingPool::Allocation> decoupled_staging;
        if constexpr (std::is_same_v<Sample, float>) {
            decoupled_staging.emplace(
                runtime->staging_pool->acquire(host_limited));
            decoupled_staging->reserve(staging_bytes);
        }
        const std::size_t slot_index = runtime->acquire_slot(heavy, host_limited);
        SlotRelease slot_release(*runtime, slot_index, heavy);
        ExecutionSlot &slot = *runtime->slots[slot_index];
        slot.reserve(builder.size(), decoupled_staging ? 4U : staging_bytes);
        Buffer &staging = decoupled_staging
            ? decoupled_staging->buffer() : slot.staging;

        if (!reused_input) {
            std::memset(staging.mapped(), 0, source_copy_bytes);
            pack_host_rows(
                input, source_pitch, staging.mapped(),
                source_row_bytes, source_height);
            staging.flush(0U, source_copy_bytes);
        }

        bool submitted = false;
        bool cache_published = false;
        try {
            slot.begin();
            runtime->begin_label(slot.command, "dsmvc 2D inverse");
            if (!reused_input) {
                runtime->begin_label(slot.command, "input transpose");
                const VkBufferCopy source_copy{
                    0U,
                    static_cast<VkDeviceSize>(source_offset) * 4U,
                    source_copy_bytes,
                };
                vkCmdCopyBuffer(
                    slot.command, staging.handle(), slot.workspace.handle(),
                    1U, &source_copy);
                command_barrier(
                    slot.command, slot.workspace.handle(), source_copy.dstOffset,
                    source_copy.size, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                const std::uint32_t sample_type =
                    std::is_same_v<Sample, float> ? 0U
                    : std::is_same_v<Sample, std::uint8_t> ? 1U : 2U;
                record_transpose(
                    *runtime, slot, *packed_horizontal,
                    cached_input ? cached_input->buffer() : VK_NULL_HANDLE,
                    cached_input ? cached_input->offset() : 0U,
                    cached_input ? cached_input->size() : 0U,
                    source_width, source_height, source_offset,
                    cached_input ? 0U : transposed_offset,
                    0U, cached_input ? 1U : 0U, sample_type, conversion);
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

            runtime->begin_label(slot.command, "horizontal inverse");
            record_inverse(
                *runtime, slot, *packed_horizontal,
                cached_input ? cached_input->buffer() : VK_NULL_HANDLE,
                cached_input ? cached_input->offset() : 0U,
                cached_input ? cached_input->size() : 0U,
                source_height, cached_input ? 0U : transposed_offset,
                source_offset, cached_input ? 1U : 0U, 0U, 0U,
                should_split_rhs(*runtime, *packed_horizontal));
            command_barrier(
                slot.command, slot.workspace.handle(),
                static_cast<VkDeviceSize>(source_offset) * 4U,
                intermediate_bytes, VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            runtime->end_label(slot.command);

            runtime->begin_label(slot.command, "intermediate transpose");
            record_transpose(
                *runtime, slot, *packed_horizontal,
                cached_input ? cached_input->buffer() : VK_NULL_HANDLE,
                cached_input ? cached_input->offset() : 0U,
                cached_input ? cached_input->size() : 0U,
                source_height, destination_width, source_offset,
                intermediate_offset, 0U, 0U, 0U, nullptr);
            command_barrier(
                slot.command, slot.workspace.handle(),
                static_cast<VkDeviceSize>(intermediate_offset) * 4U,
                intermediate_bytes, VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            runtime->end_label(slot.command);

            runtime->begin_label(slot.command, "vertical inverse");
            record_inverse(
                *runtime, slot, *packed_vertical,
                cached_input ? cached_input->buffer() : VK_NULL_HANDLE,
                cached_input ? cached_input->offset() : 0U,
                cached_input ? cached_input->size() : 0U,
                destination_width, intermediate_offset, destination_offset,
                0U, 0U, 0U, should_split_rhs(*runtime, *packed_vertical));

            if constexpr (!std::is_same_v<Sample, float>) {
                command_barrier(
                    slot.command, slot.workspace.handle(),
                    static_cast<VkDeviceSize>(destination_offset) * 4U,
                    destination_bytes, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                runtime->end_label(slot.command);
                runtime->begin_label(slot.command, "integer conversion");
                record_conversion(
                    *runtime, slot, *packed_vertical,
                    checked_u32(destination_elements,
                                "Vulkan destination element count"),
                    destination_offset, result_offset,
                    std::is_same_v<Sample, std::uint8_t> ? 1U : 2U,
                    *conversion);
            }
            command_barrier(
                slot.command, slot.workspace.handle(),
                static_cast<VkDeviceSize>(result_offset) * 4U,
                result_copy_bytes, VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT);
            runtime->end_label(slot.command);
            const VkBufferCopy result_copy{
                static_cast<VkDeviceSize>(result_offset) * 4U,
                download_offset,
                result_copy_bytes,
            };
            vkCmdCopyBuffer(
                slot.command, slot.workspace.handle(), staging.handle(),
                1U, &result_copy);
            command_barrier(
                slot.command, staging.handle(), download_offset,
                result_copy_bytes, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT);
            runtime->end_label(slot.command);
            runtime->check(vkEndCommandBuffer(slot.command),
                           "vkEndCommandBuffer(2D inverse)");

            std::uint64_t wait_value = std::max(
                packed_horizontal->wait_value(), packed_vertical->wait_value());
            wait_value = std::max(wait_value, input_wait_value);
            const std::uint64_t signal_value = runtime->submit(
                slot.command, slot.fence, wait_value,
                cache_producer && runtime->timeline_enabled);
            submitted = true;
            if (cache_producer && runtime->timeline_enabled) {
                cached_input->publish(signal_value);
                cache_published = true;
            }
            runtime->wait_fence(slot.fence, "Vulkan 2D execution");
            if (cache_producer && !cache_published) {
                cached_input->publish(0U);
                cache_published = true;
            }
            packed_horizontal->mark_upload_complete();
            packed_vertical->mark_upload_complete();
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
        staging.invalidate(download_offset, result_copy_bytes);
        unpack_host_rows(
            static_cast<const std::byte *>(staging.mapped())
                + static_cast<std::ptrdiff_t>(download_offset),
            result_row_bytes, output, result_pitch, destination_height);
    }

    void execute_one_dimensional(
        const AxisPlan &plan, const float *input,
        std::ptrdiff_t input_row_stride, float *output,
        std::ptrdiff_t output_row_stride, std::uint32_t vector_count,
        bool rows, std::shared_ptr<const void> input_lifetime) const;

    std::shared_ptr<Runtime> runtime;
    mutable std::mutex mutex;
    std::vector<PreparedPlan> plans;
    std::atomic<bool> sealed{false};
};

void VulkanExecutor::Impl::execute_one_dimensional(
    const AxisPlan &plan, const float *input,
    std::ptrdiff_t input_row_stride, float *output,
    std::ptrdiff_t output_row_stride, std::uint32_t vector_count,
    bool rows, std::shared_ptr<const void> input_lifetime) const {
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
    const std::size_t download_offset = input_bytes;
    const std::size_t staging_bytes = checked_add(
        download_offset, output_bytes, "Vulkan 1D staging");
    const std::size_t slot_index = runtime->acquire_slot(false, false);
    SlotRelease release(*runtime, slot_index, false);
    ExecutionSlot &slot = *runtime->slots[slot_index];
    slot.reserve(builder.size(), staging_bytes);
    if (!reused_input) {
        pack_host_rows(
            input, input_pitch, slot.staging.mapped(),
            input_row_bytes, input_rows);
        slot.staging.flush(0U, input_bytes);
    }

    bool submitted = false;
    bool cache_published = false;
    try {
        slot.begin();
        runtime->begin_label(slot.command, rows
            ? "dsmvc row inverse" : "dsmvc column inverse");
        if (!reused_input) {
            const VkBufferCopy copy{
                0U,
                cached_input ? cached_input->offset()
                             : static_cast<VkDeviceSize>(source_offset) * 4U,
                input_bytes,
            };
            vkCmdCopyBuffer(
                slot.command, slot.staging.handle(),
                cached_input ? cached_input->buffer() : slot.workspace.handle(),
                1U, &copy);
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
            vector_count, cached_input ? 0U : source_offset, output_offset,
            cached_input ? 1U : 0U, rows ? 1U : 0U, rows ? 1U : 0U,
            should_split_rhs(*runtime, *packed));
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
        vkCmdCopyBuffer(
            slot.command, slot.workspace.handle(), slot.staging.handle(),
            1U, &result_copy);
        command_barrier(
            slot.command, slot.staging.handle(), download_offset,
            output_bytes, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT);
        runtime->end_label(slot.command);
        runtime->check(vkEndCommandBuffer(slot.command),
                       "vkEndCommandBuffer(1D inverse)");
        const std::uint64_t wait_value = std::max(
            packed->wait_value(), input_wait_value);
        const std::uint64_t signal_value = runtime->submit(
            slot.command, slot.fence, wait_value,
            cache_producer && runtime->timeline_enabled);
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
        packed->mark_upload_complete();
    } catch (...) {
        const auto error = std::current_exception();
        if (!submitted) slot.recover_unsubmitted();
        if (cache_producer && !cache_published) {
            cached_input->fail(error);
            runtime->input_cache->erase(*cache_key, cached_input);
        }
        std::rethrow_exception(error);
    }
    slot.staging.invalidate(download_offset, output_bytes);
    unpack_host_rows(
        static_cast<const std::byte *>(slot.staging.mapped())
            + static_cast<std::ptrdiff_t>(download_offset),
        output_row_bytes, output, output_pitch, output_rows);
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
    auto instance = create_instance(false);
    (void)select_device(enumerate_devices(instance.instance));
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
        auto packed = acquire_packed_plan(impl_->runtime, plan);
        impl_->plans.push_back({std::move(plan), packed->identity, packed});
    }
}

void VulkanExecutor::seal() const {
    const std::scoped_lock lock(impl_->mutex);
    impl_->sealed.store(true, std::memory_order_release);
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
