#pragma once

#include <cstddef>

// Copies of whole frame planes between host frame memory and transfer staging.
namespace dsmvc::host_copy {

// Copies `rows` rows of `row_bytes` between planes whose rows are
// `source_pitch` and `destination_pitch` bytes apart. Planes of
// streaming_minimum() bytes or more are written with non-temporal stores on
// x86-64: such a plane does not fit the copying core's cache, and an ordinary
// store first reads every destination line from memory, which is a third of
// the traffic of a copy on a host whose frame rate is bound by memory
// bandwidth. (glibc only does this itself above 3/4 of the shared cache: 24 MB
// on a Ryzen 9 5950X, more than a 1080p Float32 plane.)
void copy_rows(
    void *destination, std::size_t destination_pitch, const void *source,
    std::size_t source_pitch, std::size_t row_bytes, std::size_t rows) noexcept;

// DSMVC_HOST_COPY_STREAM_MIN (bytes; 0 never streams). Default 1 MiB.
[[nodiscard]] std::size_t streaming_minimum() noexcept;

} // namespace dsmvc::host_copy
