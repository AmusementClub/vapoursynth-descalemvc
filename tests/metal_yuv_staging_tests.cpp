#include "metal_yuv_executor_apple.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

void fill_rows(std::vector<std::byte> &storage, std::size_t stride,
               std::size_t row_bytes, std::size_t rows) {
    for (std::size_t row = 0U; row < rows; ++row) {
        for (std::size_t column = 0U; column < row_bytes; ++column) {
            storage[row * stride + column] = static_cast<std::byte>(
                (row * 37U + column * 11U + 3U) & 0xffU);
        }
    }
}

void check_logical_rows(const std::vector<std::byte> &source,
                        const std::vector<std::byte> &destination,
                        std::size_t source_stride, std::size_t destination_stride,
                        std::size_t row_bytes, std::size_t rows) {
    for (std::size_t row = 0U; row < rows; ++row) {
        for (std::size_t column = 0U; column < row_bytes; ++column) {
            require(
                destination[row * destination_stride + column]
                    == source[row * source_stride + column],
                "logical row data differs");
        }
    }
}

void test_equal_stride_bulk_copy() {
    constexpr std::size_t stride = 16U;
    constexpr std::size_t row_bytes = 10U;
    constexpr std::size_t rows = 3U;
    std::vector<std::byte> source(stride * rows, std::byte{0x5a});
    std::vector<std::byte> destination(stride * rows, std::byte{0x2d});
    fill_rows(source, stride, row_bytes, rows);

    dsmvc::experimental::MetalYuvStagingStats stats;
    dsmvc::experimental::detail::copy_strided_rows(
        destination.data(), static_cast<std::ptrdiff_t>(stride), source.data(),
        static_cast<std::ptrdiff_t>(stride), row_bytes,
        static_cast<std::uint32_t>(rows), stats);

    require(stats.memcpy_calls == 1U, "equal stride did not bulk copy");
    require(stats.copied_bytes == (rows - 1U) * stride + row_bytes,
            "equal stride byte accounting differs");
    check_logical_rows(source, destination, stride, stride, row_bytes, rows);
    require(destination[stride - 1U] == source[stride - 1U],
            "equal stride bulk copy did not preserve the copied row span");
    require(destination[(rows - 1U) * stride + row_bytes]
                == std::byte{0x2d},
            "equal stride bulk copy crossed the final logical row");
}

void test_different_stride_row_copy() {
    constexpr std::size_t source_stride = 16U;
    constexpr std::size_t destination_stride = 20U;
    constexpr std::size_t row_bytes = 10U;
    constexpr std::size_t rows = 3U;
    std::vector<std::byte> source(source_stride * rows, std::byte{0x5a});
    std::vector<std::byte> destination(destination_stride * rows,
                                       std::byte{0x2d});
    fill_rows(source, source_stride, row_bytes, rows);

    dsmvc::experimental::MetalYuvStagingStats stats;
    dsmvc::experimental::detail::copy_strided_rows(
        destination.data(), static_cast<std::ptrdiff_t>(destination_stride),
        source.data(), static_cast<std::ptrdiff_t>(source_stride), row_bytes,
        static_cast<std::uint32_t>(rows), stats);

    require(stats.memcpy_calls == rows,
            "different stride did not use one copy per row");
    require(stats.copied_bytes == rows * row_bytes,
            "different stride byte accounting differs");
    check_logical_rows(
        source, destination, source_stride, destination_stride, row_bytes, rows);
    for (std::size_t row = 0U; row < rows; ++row) {
        require(destination[row * destination_stride + row_bytes]
                    == std::byte{0x2d},
                "different stride copied source padding into destination");
    }
}

void test_empty_copy() {
    std::vector<std::byte> source(8U, std::byte{0x5a});
    std::vector<std::byte> destination(8U, std::byte{0x2d});
    dsmvc::experimental::MetalYuvStagingStats stats;
    dsmvc::experimental::detail::copy_strided_rows(
        destination.data(), 8, source.data(), 8, 8U, 0U, stats);
    require(stats.memcpy_calls == 0U && stats.copied_bytes == 0U,
            "empty copy changed staging statistics");
}

} // namespace

int main() {
    try {
        test_equal_stride_bulk_copy();
        test_different_stride_row_copy();
        test_empty_copy();
        std::cout << "dsmvc Metal staging tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "dsmvc Metal staging tests failed: " << error.what()
                  << '\n';
        return 1;
    }
}
