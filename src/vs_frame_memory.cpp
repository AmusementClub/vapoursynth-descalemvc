#include "vs_frame_memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string_view>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace dsmvc::vs_frame_memory {
namespace {

// Layout of the version 1 prefix of the extension's function table. The
// extension is found with getVSFrameMemoryAPI() in the library that provides
// the VSAPI; a later table version only appends members.
using ReleaseHook = void (VS_CC *)(void *user_data, void *base, size_t size);

struct Api {
    int version;
    size_t page_size;
    int (VS_CC *get_plane_block)(
        const VSFrame *frame, int plane, void **base, size_t *size) VS_NOEXCEPT;
    int (VS_CC *add_release_hook)(
        VSCore *core, ReleaseHook hook, void *user_data) VS_NOEXCEPT;
    void (VS_CC *remove_release_hook)(VSCore *core, int id) VS_NOEXCEPT;
};

constexpr int supported_version = 1;

using Entry = const Api *(VS_CC *)(int version) VS_NOEXCEPT;

Entry find_entry(const VSAPI *vsapi) noexcept {
    if (!vsapi || !vsapi->getFrameWidth) return nullptr;
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(vsapi->getFrameWidth), &module)) {
        return nullptr;
    }
    return reinterpret_cast<Entry>(reinterpret_cast<void *>(
        GetProcAddress(module, "getVSFrameMemoryAPI")));
#else
    Dl_info info{};
    if (!dladdr(reinterpret_cast<const void *>(vsapi->getFrameWidth), &info)
        || !info.dli_fname) {
        return nullptr;
    }
    // The core is already loaded; only look inside it.
    void *handle = dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD);
    if (!handle) return nullptr;
    void *symbol = dlsym(handle, "getVSFrameMemoryAPI");
    dlclose(handle);
    return reinterpret_cast<Entry>(symbol);
#endif
}

const Api *discover(const VSAPI *vsapi) noexcept {
    static std::mutex mutex;
    static std::map<const VSAPI *, const Api *> cache;
    const std::scoped_lock lock(mutex);
    if (const auto found = cache.find(vsapi); found != cache.end()) {
        return found->second;
    }
    const Api *api = nullptr;
    if (const Entry entry = find_entry(vsapi)) {
        api = entry(supported_version);
        if (api && (api->version < supported_version
                    || !api->get_plane_block || !api->add_release_hook
                    || !api->remove_release_hook)) {
            api = nullptr;
        }
    }
    cache.emplace(vsapi, api);
    return api;
}

struct CoreState {
    int hook = 0;
    int filters = 0;
    int per_domain[host_blocks::domain_count] = {};
};

std::mutex &states_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<VSCore *, CoreState> &states() {
    static std::map<VSCore *, CoreState> map;
    return map;
}

// Runs on the thread that returns a block to the system, with no core lock
// held. It must not call VapourSynth functions.
void VS_CC release_hook(void *, void *base, size_t size) {
    host_blocks::release_block(base, size);
}

const char *domain_prefix(host_blocks::Domain domain) noexcept {
    return domain == host_blocks::Domain::cuda ? "DSMVC_CUDA" : "DSMVC_VULKAN";
}

bool env_flag(host_blocks::Domain domain, const char *suffix) noexcept {
    std::string name = domain_prefix(domain);
    name += suffix;
    const char *text = std::getenv(name.c_str());
    return text && std::string_view{text} == "1";
}

// Half of the core's frame cache, unless the environment names a cap.
std::size_t byte_cap(
    VSCore *core, const VSAPI *vsapi, host_blocks::Domain domain) noexcept {
    std::string name = domain_prefix(domain);
    name += "_DIRECT_IO_MAX_MB";
    if (const char *text = std::getenv(name.c_str())) {
        char *end = nullptr;
        const unsigned long long megabytes = std::strtoull(text, &end, 10);
        if (end != text) return static_cast<std::size_t>(megabytes) << 20U;
    }
    const long long cache = vsapi->setMaxCacheSize(0, core);
    return cache > 0 ? static_cast<std::size_t>(cache) / 2U : 0U;
}

void note_planes(
    VSCore *core, const VSAPI *vsapi, host_blocks::Domain domain,
    const VSFrame *frame) noexcept {
    const Api *api = discover(vsapi);
    if (!api) return;
    const int planes = vsapi->getVideoFrameFormat(frame)->numPlanes;
    for (int plane = 0; plane < planes; ++plane) {
        void *base = nullptr;
        size_t size = 0;
        if (api->get_plane_block(frame, plane, &base, &size)) {
            host_blocks::note_use(domain, core, base, size);
        }
    }
}

} // namespace

bool attach(VSCore *core, const VSAPI *vsapi, host_blocks::Domain domain) {
    if (!host_blocks::enabled(domain)) return false;
    const Api *api = discover(vsapi);
    if (!api) return false;
    const std::scoped_lock lock(states_mutex());
    CoreState &state = states()[core];
    if (state.filters == 0) {
        state.hook = api->add_release_hook(core, release_hook, nullptr);
        if (state.hook == 0) {
            states().erase(core);
            return false;
        }
    }
    ++state.filters;
    ++state.per_domain[static_cast<std::size_t>(domain)];
    host_blocks::set_limit(core, byte_cap(core, vsapi, domain));
    return true;
}

void detach(VSCore *core, const VSAPI *vsapi, host_blocks::Domain domain) {
    const Api *api = discover(vsapi);
    if (!api) return;
    int hook = 0;
    {
        const std::scoped_lock lock(states_mutex());
        const auto found = states().find(core);
        if (found == states().end()) return;
        CoreState &state = found->second;
        --state.per_domain[static_cast<std::size_t>(domain)];
        if (--state.filters > 0) return;
        hook = state.hook;
        states().erase(found);
    }
    // The hook only takes the registry lock, never ours, so this cannot
    // deadlock; it returns once the hook is not running and never will again.
    api->remove_release_hook(core, hook);
    host_blocks::drop_owner(core);
    for (const auto d : {host_blocks::Domain::cuda, host_blocks::Domain::vulkan}) {
        if (env_flag(d, "_DIRECT_IO_STATS") && host_blocks::enabled(d)) {
            std::fprintf(stderr, "%s\n", host_blocks::describe(d).c_str());
        }
    }
}

void note_frame(
    VSCore *core, const VSAPI *vsapi, host_blocks::Domain domain,
    const VSFrame *frame) {
    note_planes(core, vsapi, domain, frame);
}

void note_frame(
    VSCore *core, const VSAPI *vsapi, host_blocks::Domain domain,
    VSFrame *frame) {
    const int planes = vsapi->getVideoFrameFormat(frame)->numPlanes;
    for (int plane = 0; plane < planes; ++plane) {
        (void)vsapi->getWritePtr(frame, plane);
    }
    note_planes(core, vsapi, domain, frame);
}

} // namespace dsmvc::vs_frame_memory
