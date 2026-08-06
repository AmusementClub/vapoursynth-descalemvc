#include "cuda_driver.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#else
#error "The CUDA driver loader requires Windows or Linux"
#endif

#include <sstream>
#include <stdexcept>
#include <string>

namespace dsmvc::cuda_detail {
namespace {

template <class Function>
[[nodiscard]] Function required_symbol(void *library, const char *name) {
#if defined(_WIN32)
    const HMODULE module = static_cast<HMODULE>(library);
    const FARPROC address = GetProcAddress(module, name);
    if (!address) {
        throw std::runtime_error(
            "CUDA Driver API is missing required symbol " + std::string{name});
    }
    return reinterpret_cast<Function>(address);
#else
    (void)dlerror();
    void *address = dlsym(library, name);
    const char *error = dlerror();
    if (error || !address) {
        throw std::runtime_error(
            "CUDA Driver API is missing required symbol " + std::string{name}
            + ": " + (error ? error : "unknown error"));
    }
    return reinterpret_cast<Function>(address);
#endif
}

} // namespace

DriverApi::~DriverApi() {
#if defined(_WIN32)
    if (library) FreeLibrary(static_cast<HMODULE>(library));
#else
    if (library) (void)dlclose(library);
#endif
}

std::shared_ptr<DriverApi> load_cuda_driver() {
#if defined(_WIN32)
    HMODULE library = LoadLibraryW(L"nvcuda.dll");
    if (!library) {
        throw std::runtime_error(
            "CUDA driver DLL nvcuda.dll is unavailable (Windows error "
            + std::to_string(GetLastError()) + ")");
    }
#else
    void *library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        const char *error = dlerror();
        throw std::runtime_error(
            "CUDA driver library libcuda.so.1 is unavailable: "
            + std::string{error ? error : "unknown error"});
    }
#endif

    auto api = std::shared_ptr<DriverApi>(new DriverApi{});
    api->library = static_cast<void *>(library);
#define DSMVC_CUDA_SYMBOL(member, type, name) \
    api->member = required_symbol<DriverApi::type>(library, name)
    DSMVC_CUDA_SYMBOL(init, Init, "cuInit");
    DSMVC_CUDA_SYMBOL(driver_get_version, DriverGetVersion, "cuDriverGetVersion");
    DSMVC_CUDA_SYMBOL(device_get_count, DeviceGetCount, "cuDeviceGetCount");
    DSMVC_CUDA_SYMBOL(device_get, DeviceGet, "cuDeviceGet");
    DSMVC_CUDA_SYMBOL(device_get_name, DeviceGetName, "cuDeviceGetName");
    DSMVC_CUDA_SYMBOL(
        device_get_attribute, DeviceGetAttribute, "cuDeviceGetAttribute");
    DSMVC_CUDA_SYMBOL(ctx_create, CtxCreate, "cuCtxCreate_v2");
    DSMVC_CUDA_SYMBOL(ctx_destroy, CtxDestroy, "cuCtxDestroy_v2");
    DSMVC_CUDA_SYMBOL(ctx_get_current, CtxGetCurrent, "cuCtxGetCurrent");
    DSMVC_CUDA_SYMBOL(ctx_set_current, CtxSetCurrent, "cuCtxSetCurrent");
    DSMVC_CUDA_SYMBOL(module_load_data, ModuleLoadData, "cuModuleLoadData");
    DSMVC_CUDA_SYMBOL(module_unload, ModuleUnload, "cuModuleUnload");
    DSMVC_CUDA_SYMBOL(
        module_get_function, ModuleGetFunction, "cuModuleGetFunction");
    DSMVC_CUDA_SYMBOL(mem_alloc, MemAlloc, "cuMemAlloc_v2");
    DSMVC_CUDA_SYMBOL(mem_free, MemFree, "cuMemFree_v2");
    DSMVC_CUDA_SYMBOL(memcpy_htod, MemcpyHtoD, "cuMemcpyHtoD_v2");
    DSMVC_CUDA_SYMBOL(
        memcpy_htod_async, MemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2");
    DSMVC_CUDA_SYMBOL(
        memcpy_dtoh_async, MemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2");
    DSMVC_CUDA_SYMBOL(mem_host_alloc, MemHostAlloc, "cuMemHostAlloc");
    DSMVC_CUDA_SYMBOL(mem_free_host, MemFreeHost, "cuMemFreeHost");
    DSMVC_CUDA_SYMBOL(stream_create, StreamCreate, "cuStreamCreate");
    DSMVC_CUDA_SYMBOL(stream_destroy, StreamDestroy, "cuStreamDestroy_v2");
    DSMVC_CUDA_SYMBOL(
        stream_synchronize, StreamSynchronize, "cuStreamSynchronize");
    DSMVC_CUDA_SYMBOL(
        stream_wait_event, StreamWaitEvent, "cuStreamWaitEvent");
    DSMVC_CUDA_SYMBOL(event_create, EventCreate, "cuEventCreate");
    DSMVC_CUDA_SYMBOL(event_destroy, EventDestroy, "cuEventDestroy_v2");
    DSMVC_CUDA_SYMBOL(event_query, EventQuery, "cuEventQuery");
    DSMVC_CUDA_SYMBOL(event_record, EventRecord, "cuEventRecord");
    DSMVC_CUDA_SYMBOL(
        event_synchronize, EventSynchronize, "cuEventSynchronize");
    DSMVC_CUDA_SYMBOL(launch_kernel, LaunchKernel, "cuLaunchKernel");
    DSMVC_CUDA_SYMBOL(get_error_name, GetErrorName, "cuGetErrorName");
    DSMVC_CUDA_SYMBOL(get_error_string, GetErrorString, "cuGetErrorString");
#undef DSMVC_CUDA_SYMBOL
    return api;
}

std::string cuda_error(const DriverApi &api, CUresult result) {
    const char *name = nullptr;
    const char *description = nullptr;
    (void)api.get_error_name(result, &name);
    (void)api.get_error_string(result, &description);
    std::ostringstream stream;
    stream << "CUDA error " << static_cast<int>(result);
    if (name) stream << " (" << name << ')';
    if (description) stream << ": " << description;
    return stream.str();
}

void cuda_check(const DriverApi &api, CUresult result, const char *operation) {
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(
            std::string{operation} + " failed: " + cuda_error(api, result));
    }
}

} // namespace dsmvc::cuda_detail
