#include "dsmvc_cuda_fatbin.hpp"

#include <windows.h>

#include <stdexcept>
#include <string>

namespace dsmvc::cuda_detail {
namespace {

constexpr WORD fatbin_resource_id = 101;
constexpr WORD rcdata_resource_type = 10;
const int module_anchor = 0;

[[noreturn]] void throw_resource_error(const char *operation) {
    throw std::runtime_error(
        std::string{operation} + " failed for the embedded CUDA fatbin"
        + " (Windows error " + std::to_string(GetLastError()) + ")");
}

} // namespace

const void *embedded_cuda_fatbin() {
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&module_anchor), &module)) {
        throw_resource_error("GetModuleHandleExW");
    }
    const HRSRC resource = FindResourceW(
        module, MAKEINTRESOURCEW(fatbin_resource_id),
        MAKEINTRESOURCEW(rcdata_resource_type));
    if (!resource) throw_resource_error("FindResourceW");
    const HGLOBAL loaded = LoadResource(module, resource);
    if (!loaded) throw_resource_error("LoadResource");
    const void *data = LockResource(loaded);
    if (!data || SizeofResource(module, resource) == 0U) {
        throw_resource_error("LockResource");
    }
    return data;
}

} // namespace dsmvc::cuda_detail
