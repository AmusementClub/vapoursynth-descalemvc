#include "host_copy.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#define DSMVC_HOST_COPY_STREAMING 1
#endif

namespace dsmvc::host_copy {
namespace {

#if defined(DSMVC_HOST_COPY_STREAMING)
void copy_streaming(
    std::byte *destination, const std::byte *source, std::size_t bytes) noexcept {
    constexpr std::size_t line = 64U;
    const std::size_t misaligned =
        static_cast<std::size_t>(reinterpret_cast<std::uintptr_t>(destination) % line);
    if (misaligned != 0U) {
        const std::size_t head = line - misaligned < bytes ? line - misaligned : bytes;
        std::memcpy(destination, source, head);
        destination += head;
        source += head;
        bytes -= head;
    }
    for (; bytes >= line; bytes -= line) {
        const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(source));
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(source + 16));
        const __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i *>(source + 32));
        const __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i *>(source + 48));
        _mm_stream_si128(reinterpret_cast<__m128i *>(destination), a);
        _mm_stream_si128(reinterpret_cast<__m128i *>(destination + 16), b);
        _mm_stream_si128(reinterpret_cast<__m128i *>(destination + 32), c);
        _mm_stream_si128(reinterpret_cast<__m128i *>(destination + 48), d);
        destination += line;
        source += line;
    }
    if (bytes != 0U) std::memcpy(destination, source, bytes);
}
#endif

} // namespace

std::size_t streaming_minimum() noexcept {
    static const std::size_t value = [] {
        constexpr std::size_t fallback = std::size_t{1U} << 20U;
        const char *text = std::getenv("DSMVC_HOST_COPY_STREAM_MIN");
        if (!text) return fallback;
        char *end = nullptr;
        const unsigned long long requested = std::strtoull(text, &end, 10);
        if (end == text) return fallback;
        return requested == 0U ? std::numeric_limits<std::size_t>::max()
                               : static_cast<std::size_t>(requested);
    }();
    return value;
}

void copy_rows(
    void *destination, std::size_t destination_pitch, const void *source,
    std::size_t source_pitch, std::size_t row_bytes, std::size_t rows) noexcept {
    auto *destination_bytes = static_cast<std::byte *>(destination);
    const auto *source_bytes = static_cast<const std::byte *>(source);
    const bool contiguous =
        source_pitch == row_bytes && destination_pitch == row_bytes;
#if defined(DSMVC_HOST_COPY_STREAMING)
    if (rows != 0U && row_bytes >= streaming_minimum() / rows) {
        if (contiguous) {
            copy_streaming(destination_bytes, source_bytes, row_bytes * rows);
        } else {
            for (std::size_t row = 0U; row < rows; ++row) {
                copy_streaming(
                    destination_bytes + row * destination_pitch,
                    source_bytes + row * source_pitch, row_bytes);
            }
        }
        _mm_sfence();
        return;
    }
#endif
    if (contiguous) {
        std::memcpy(destination_bytes, source_bytes, row_bytes * rows);
        return;
    }
    for (std::size_t row = 0U; row < rows; ++row) {
        std::memcpy(
            destination_bytes + row * destination_pitch,
            source_bytes + row * source_pitch, row_bytes);
    }
}

} // namespace dsmvc::host_copy
