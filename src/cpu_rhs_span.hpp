#pragma once

#include <cstdint>

namespace dsmvc::detail {

#if defined(_MSC_VER)
#define DSMVC_RHS_INLINE __forceinline
#else
#define DSMVC_RHS_INLINE inline __attribute__((always_inline))
#endif

template <int Tap, int Count, class Accumulate>
DSMVC_RHS_INLINE void accumulate_rhs_span(Accumulate &accumulate) noexcept {
    accumulate.template operator()<Tap>();
    if constexpr (Tap + 1 < Count) {
        accumulate_rhs_span<Tap + 1, Count>(accumulate);
    }
}

// Transposed resampling weights can have more source terms than the kernel's
// nominal support. Specialize common wide-band spans without changing their
// ascending FMA order or adding operations for zero-padded coefficients.
template <int Bandwidth, class Accumulate>
[[nodiscard]] DSMVC_RHS_INLINE bool try_accumulate_rhs_span(
    std::int32_t count, Accumulate &&accumulate) noexcept {
    if constexpr (Bandwidth == 5) {
        if (count == 7) {
            accumulate_rhs_span<0, 7>(accumulate);
            return true;
        }
        if (count == 6) {
            accumulate_rhs_span<0, 6>(accumulate);
            return true;
        }
    } else if constexpr (Bandwidth == 7) {
        if (count == 9) {
            accumulate_rhs_span<0, 9>(accumulate);
            return true;
        }
        if (count == 10) {
            accumulate_rhs_span<0, 10>(accumulate);
            return true;
        }
    } else if constexpr (Bandwidth == 11) {
        if (count == 14) {
            accumulate_rhs_span<0, 14>(accumulate);
            return true;
        }
        if (count == 13) {
            accumulate_rhs_span<0, 13>(accumulate);
            return true;
        }
    }
    return false;
}

#undef DSMVC_RHS_INLINE

} // namespace dsmvc::detail
