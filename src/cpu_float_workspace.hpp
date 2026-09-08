#pragma once

#include "checked_size.hpp"

#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>

namespace dsmvc::detail {

// Keep SIMD alignment without treating adjacent struct members as one float
// array. The extra elements leave room for the aligned view after any resize.
template <std::size_t Alignment>
[[nodiscard]] inline float *aligned_float_workspace(
    std::vector<float> &storage, std::size_t count, std::string_view label) {
    static_assert(Alignment >= alignof(float)
                  && (Alignment & (Alignment - 1U)) == 0U);
    constexpr auto padding = Alignment / sizeof(float) - 1U;
    storage.resize(checked_size_add(count, padding, label));
    void *pointer = storage.data();
    auto bytes = storage.size() * sizeof(float);
    return static_cast<float *>(std::align(
        Alignment, count * sizeof(float), pointer, bytes));
}

} // namespace dsmvc::detail
