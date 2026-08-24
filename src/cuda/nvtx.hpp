#pragma once

#include <nvtx3/nvToolsExt.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace dsmvc::cuda_detail {

enum class NvtxLabel : std::uint8_t {
    frame,
    plane,
    execute_2d,
    execute_rows,
    execute_columns,
    filter_plan_preparation,
    prepared_plan_lookup,
    plan_cache_lookup,
    plan_cache_hit,
    plan_cache_miss,
    plan_pack,
    plan_upload,
    slot_wait,
    staging_wait,
    buffer_reserve,
    input_cache_lookup,
    input_cache_hit,
    input_cache_miss,
    host_register,
    host_unregister,
    host_pack,
    source_upload,
    input_transpose,
    horizontal_plan_wait,
    horizontal_stage,
    intermediate_transpose,
    vertical_plan_wait,
    vertical_stage,
    output_conversion,
    destination_download,
    stream_synchronize,
    host_unpack,
    count,
};

namespace nvtx_detail {

enum class Category : std::uint32_t {
    filter = 1U,
    plan,
    scheduling,
    cache,
    host,
    transfer,
    horizontal,
    vertical,
    synchronization,
    conversion,
};

struct Metadata {
    const char *name;
    Category category;
    std::uint32_t color;
};

inline constexpr std::array metadata{
    Metadata{"frame", Category::filter, 0xFF7F8C8DU},
    Metadata{"plane", Category::filter, 0xFF95A5A6U},
    Metadata{"execute 2d", Category::filter, 0xFF34495EU},
    Metadata{"execute rows", Category::filter, 0xFF34495EU},
    Metadata{"execute columns", Category::filter, 0xFF34495EU},
    Metadata{"filter plan preparation", Category::plan, 0xFFF7DC6FU},
    Metadata{"prepared plan lookup", Category::plan, 0xFFF4D03FU},
    Metadata{"plan cache lookup", Category::plan, 0xFFF1C40FU},
    Metadata{"plan cache hit", Category::plan, 0xFFF39C12U},
    Metadata{"plan cache miss", Category::plan, 0xFFE67E22U},
    Metadata{"plan pack", Category::plan, 0xFFD68910U},
    Metadata{"plan upload", Category::plan, 0xFFCA6F1EU},
    Metadata{"slot wait", Category::scheduling, 0xFFE67E22U},
    Metadata{"staging wait", Category::scheduling, 0xFFD35400U},
    Metadata{"buffer reserve", Category::scheduling, 0xFFB9770EU},
    Metadata{"input cache lookup", Category::cache, 0xFFD35400U},
    Metadata{"input cache hit", Category::cache, 0xFF27AE60U},
    Metadata{"input cache miss", Category::cache, 0xFFC0392BU},
    Metadata{"host register", Category::host, 0xFF1ABC9CU},
    Metadata{"host unregister", Category::host, 0xFF148F77U},
    Metadata{"host pack", Category::host, 0xFF16A085U},
    Metadata{"source upload", Category::transfer, 0xFF2980B9U},
    Metadata{"input transpose", Category::transfer, 0xFF3498DBU},
    Metadata{"horizontal plan wait", Category::horizontal, 0xFF229954U},
    Metadata{"horizontal stage", Category::horizontal, 0xFF27AE60U},
    Metadata{"intermediate transpose", Category::transfer, 0xFF5DADE2U},
    Metadata{"vertical plan wait", Category::vertical, 0xFF7D3C98U},
    Metadata{"vertical stage", Category::vertical, 0xFF8E44ADU},
    Metadata{"output conversion", Category::conversion, 0xFF2C3E50U},
    Metadata{"destination download", Category::transfer, 0xFF2471A3U},
    Metadata{"stream synchronize", Category::synchronization, 0xFFC0392BU},
    Metadata{"host unpack", Category::host, 0xFF117864U},
};

static_assert(metadata.size() == static_cast<std::size_t>(NvtxLabel::count));

class State {
public:
    State() noexcept : domain(nvtxDomainCreateA("dsmvc.cuda")) {
        nvtxDomainNameCategoryA(
            domain, static_cast<std::uint32_t>(Category::filter), "filter");
        nvtxDomainNameCategoryA(
            domain, static_cast<std::uint32_t>(Category::plan), "plan");
        nvtxDomainNameCategoryA(
            domain, static_cast<std::uint32_t>(Category::scheduling),
            "scheduling");
        nvtxDomainNameCategoryA(
            domain, static_cast<std::uint32_t>(Category::cache), "cache");
        nvtxDomainNameCategoryA(
            domain, static_cast<std::uint32_t>(Category::host), "host");
        nvtxDomainNameCategoryA(
            domain, static_cast<std::uint32_t>(Category::transfer), "transfer");
        nvtxDomainNameCategoryA(
            domain, static_cast<std::uint32_t>(Category::horizontal),
            "horizontal");
        nvtxDomainNameCategoryA(
            domain, static_cast<std::uint32_t>(Category::vertical), "vertical");
        nvtxDomainNameCategoryA(
            domain, static_cast<std::uint32_t>(Category::synchronization),
            "synchronization");
        nvtxDomainNameCategoryA(
            domain, static_cast<std::uint32_t>(Category::conversion),
            "conversion");
        for (std::size_t index = 0U; index < metadata.size(); ++index) {
            messages[index] = nvtxDomainRegisterStringA(
                domain, metadata[index].name);
        }
    }

    nvtxDomainHandle_t domain = nullptr;
    std::array<nvtxStringHandle_t, metadata.size()> messages{};
};

[[nodiscard]] inline State &state() noexcept {
    static State instance;
    return instance;
}

} // namespace nvtx_detail

inline void initialize_nvtx() noexcept {
    (void)nvtx_detail::state();
}

class NvtxRange {
public:
    explicit NvtxRange(NvtxLabel label) noexcept {
        start(label, false, 0U);
    }

    NvtxRange(NvtxLabel label, std::uint64_t payload) noexcept {
        start(label, true, payload);
    }

    ~NvtxRange() {
        nvtxDomainRangeEnd(domain_, id_);
    }

    NvtxRange(const NvtxRange &) = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;
    NvtxRange(NvtxRange &&) = delete;
    NvtxRange &operator=(NvtxRange &&) = delete;

private:
    void start(NvtxLabel label, bool has_payload,
               std::uint64_t payload) noexcept {
        auto &state = nvtx_detail::state();
        const auto index = static_cast<std::size_t>(label);
        const auto &metadata = nvtx_detail::metadata[index];
        nvtxEventAttributes_t attributes{};
        attributes.version = NVTX_VERSION;
        attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attributes.category = static_cast<std::uint32_t>(metadata.category);
        attributes.colorType = NVTX_COLOR_ARGB;
        attributes.color = metadata.color;
        if (has_payload) {
            attributes.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
            attributes.payload.ullValue = payload;
        }
        attributes.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
        attributes.message.registered = state.messages[index];
        domain_ = state.domain;
        id_ = nvtxDomainRangeStartEx(domain_, &attributes);
    }

    nvtxDomainHandle_t domain_ = nullptr;
    nvtxRangeId_t id_ = 0U;
};

[[nodiscard]] constexpr std::uint64_t nvtx_frame_plane_payload(
    std::uint32_t frame, std::uint32_t plane) noexcept {
    return (static_cast<std::uint64_t>(frame) << 32U) | plane;
}

} // namespace dsmvc::cuda_detail
