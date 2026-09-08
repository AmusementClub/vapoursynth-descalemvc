#pragma once

#include "cpu_packed.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace dsmvc::detail {

#if defined(_MSC_VER)
#define DSMVC_WINDOW_INLINE __forceinline
#else
#define DSMVC_WINDOW_INLINE inline __attribute__((always_inline))
#endif

// Constant indices let the compiler retain the short recurrence window in
// registers. The descending order matches the generic wide-band solve.
template <int Distance, class Ops, bool Boundary>
[[nodiscard]] DSMVC_WINDOW_INLINE typename Ops::Vector subtract_window(
    typename Ops::Vector value, const float *factors,
    std::size_t factor_stride, std::int32_t index, std::int32_t available,
    const typename Ops::Vector *window) noexcept {
    if constexpr (!Boundary) {
        value = Ops::subtract_product(value,
            factors[static_cast<std::size_t>(Distance - 1) * factor_stride
                    + static_cast<std::size_t>(index)],
            window[Distance - 1]);
    } else if (available >= Distance) {
        value = Ops::subtract_product(value,
            factors[static_cast<std::size_t>(Distance - 1) * factor_stride
                    + static_cast<std::size_t>(index)],
            window[Distance - 1]);
    }
    if constexpr (Distance > 1) {
        return subtract_window<Distance - 1, Ops, Boundary>(
            value, factors, factor_stride, index, available, window);
    }
    return value;
}

template <int Index, class Ops>
DSMVC_WINDOW_INLINE void shift_window(typename Ops::Vector *window) noexcept {
    if constexpr (Index > 0) {
        window[Index] = window[Index - 1];
        shift_window<Index - 1, Ops>(window);
    }
}

template <int Bandwidth, class Ops, class Rhs>
void solve_horizontal_window(const PackedCpuPlan &packed,
                             std::int32_t destination_size,
                             float *work, Rhs &&rhs) noexcept {
    static_assert(Bandwidth > 0);
    using Vector = typename Ops::Vector;
    const auto factor_stride = static_cast<std::size_t>(
        packed.padded_destination_size);
    Vector window[Bandwidth];
    for (auto &value : window) value = Ops::zero();

    const auto forward = [&]<bool Boundary>(std::int32_t i) noexcept {
        auto value = subtract_window<Bandwidth, Ops, Boundary>(
            rhs(i), packed.lower_ld.data(), factor_stride, i, i, window);
        value = Ops::multiply(value,
            packed.inverse_diagonal[static_cast<std::size_t>(i)]);
        Ops::store(work + static_cast<std::size_t>(i) * Ops::lanes, value);
        shift_window<Bandwidth - 1, Ops>(window);
        window[0] = value;
    };
    const auto forward_boundary = std::min(Bandwidth, destination_size);
    for (std::int32_t i = 0; i < forward_boundary; ++i) {
        forward.template operator()<true>(i);
    }
    for (std::int32_t i = forward_boundary; i < destination_size; ++i) {
        forward.template operator()<false>(i);
    }
    for (std::int32_t i = destination_size;
         i < packed.padded_destination_size; ++i) {
        Ops::store(work + static_cast<std::size_t>(i) * Ops::lanes,
                   Ops::zero());
    }

    for (auto &value : window) value = Ops::zero();
    window[0] = Ops::load(work
        + static_cast<std::size_t>(destination_size - 1) * Ops::lanes);
    const auto backward = [&]<bool Boundary>(std::int32_t i) noexcept {
        auto *destination = work + static_cast<std::size_t>(i) * Ops::lanes;
        const auto value = subtract_window<Bandwidth, Ops, Boundary>(
            Ops::load(destination), packed.upper_l.data(), factor_stride, i,
            destination_size - i - 1, window);
        Ops::store(destination, value);
        shift_window<Bandwidth - 1, Ops>(window);
        window[0] = value;
    };
    const auto backward_boundary = std::max(destination_size - Bandwidth, 0);
    for (std::int32_t i = destination_size - 2;
         i >= backward_boundary; --i) {
        backward.template operator()<true>(i);
    }
    for (std::int32_t i = backward_boundary - 1; i >= 0; --i) {
        backward.template operator()<false>(i);
    }
}

#undef DSMVC_WINDOW_INLINE

} // namespace dsmvc::detail
