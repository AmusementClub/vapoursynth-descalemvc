// Unit tests for the frame plane copy: every alignment of both ends, with and
// without row padding, below and above the streaming threshold.
#include "host_copy.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check_copy(
    std::size_t destination_shift, std::size_t source_shift,
    std::size_t row_bytes, std::size_t rows, std::size_t destination_padding,
    std::size_t source_padding) {
    constexpr std::uint8_t guard = 0xA5U;
    const std::size_t destination_pitch = row_bytes + destination_padding;
    const std::size_t source_pitch = row_bytes + source_padding;
    std::vector<std::uint8_t> source(source_shift + source_pitch * rows + 64U);
    for (std::size_t index = 0U; index < source.size(); ++index) {
        source[index] = static_cast<std::uint8_t>(index * 131U + 7U);
    }
    std::vector<std::uint8_t> destination(
        destination_shift + destination_pitch * rows + 64U, guard);
    dsmvc::host_copy::copy_rows(
        destination.data() + destination_shift, destination_pitch,
        source.data() + source_shift, source_pitch, row_bytes, rows);
    bool good = true;
    for (std::size_t index = 0U; index < destination.size(); ++index) {
        std::uint8_t expected = guard;
        if (index >= destination_shift) {
            const std::size_t offset = index - destination_shift;
            const std::size_t row = offset / destination_pitch;
            const std::size_t column = offset % destination_pitch;
            if (row < rows && column < row_bytes) {
                expected = source[source_shift + row * source_pitch + column];
            }
        }
        good = good && destination[index] == expected;
    }
    if (!good) {
        std::fprintf(
            stderr, "copy_rows wrong: shifts %zu/%zu, %zu x %zu, padding %zu/%zu\n",
            destination_shift, source_shift, row_bytes, rows,
            destination_padding, source_padding);
        ++failures;
    }
}

} // namespace

int main() {
    const std::size_t large = dsmvc::host_copy::streaming_minimum();
    for (const std::size_t destination_shift : {0U, 1U, 31U, 63U}) {
        for (const std::size_t source_shift : {0U, 3U, 64U}) {
            for (const std::size_t padding : {0U, 5U, 64U}) {
                // Small planes, short rows and rows shorter than a cache line.
                check_copy(destination_shift, source_shift, 1U, 3U, padding, 0U);
                check_copy(destination_shift, source_shift, 63U, 7U, padding, padding);
                check_copy(destination_shift, source_shift, 1000U, 9U, 0U, padding);
                // Planes that stream, with tails that are not whole lines.
                check_copy(destination_shift, source_shift, large / 8U + 13U,
                           9U, padding, 0U);
                check_copy(destination_shift, source_shift, large + 127U, 1U,
                           0U, padding);
                check_copy(destination_shift, source_shift, 37U,
                           large / 37U + 2U, padding, padding);
            }
        }
    }
    check_copy(0U, 0U, 64U, 0U, 0U, 0U);
    if (failures != 0) return 1;
    std::puts("host copy tests passed");
    return 0;
}
