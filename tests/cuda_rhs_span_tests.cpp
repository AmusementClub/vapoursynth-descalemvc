#include "cuda/cuda_launch.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

template <class T>
class DeviceArray {
public:
    explicit DeviceArray(const std::vector<T> &values) : count_(values.size()) {
        check(cudaMalloc(reinterpret_cast<void **>(&data_), count_ * sizeof(T)));
        check(cudaMemcpy(data_, values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~DeviceArray() { cudaFree(data_); }
    DeviceArray(const DeviceArray &) = delete;
    DeviceArray &operator=(const DeviceArray &) = delete;
    T *data() const { return data_; }
    void reset(const std::vector<T> &values) {
        if (values.size() != count_) throw std::runtime_error("device size mismatch");
        check(cudaMemcpy(data_, values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> read() const {
        std::vector<T> result(count_);
        check(cudaMemcpy(result.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
        return result;
    }
private:
    T *data_ = nullptr;
    std::size_t count_ = 0;
};

void test_band(std::uint32_t bandwidth) {
    namespace launch = dsmvc::cuda_detail::cuda_launch;
    constexpr std::uint32_t sources = 64;
    constexpr std::uint32_t vectors = 33;
    // Empty/short rows, each selected count, and both adjacent fallback counts.
    constexpr std::uint32_t destinations = 18;
    constexpr std::size_t guard = 16;
    constexpr float sentinel = -777.0F;
    std::vector<float> input(sources * vectors);
    for (std::uint32_t i = 0; i < input.size(); ++i) {
        input[i] = std::bit_cast<float>(0x3f000000U | ((i * 2654435761U) & 0x007fffffU));
    }
    std::vector<std::uint32_t> offsets{0};
    std::vector<std::int32_t> indices;
    std::vector<float> weights;
    for (std::uint32_t i = 0; i < destinations; ++i) {
        for (std::uint32_t tap = 0; tap < i; ++tap) {
            // Noncontiguous and wrapping indices distinguish sparse entry count
            // from dense source span and make accumulation order observable.
            indices.push_back(static_cast<std::int32_t>((tap * 3U + i * 7U) % sources));
            const auto bits = 0x3e800000U | (((tap + 1U) * 2246822519U + i) & 0x007fffffU)
                | ((tap & 1U) << 31U);
            weights.push_back(std::bit_cast<float>(bits));
        }
        offsets.push_back(static_cast<std::uint32_t>(indices.size()));
    }
    std::vector<float> bands(std::max(bandwidth, 1U) * destinations, 0.0F);
    std::vector<float> diagonal(destinations, 1.0F);
    std::vector<float> initial(guard + vectors * destinations + guard, sentinel);
    DeviceArray<float> device_input(input), device_weights(weights), device_bands(bands);
    DeviceArray<float> device_diagonal(diagonal), device_output(initial);
    DeviceArray<std::uint32_t> device_offsets(offsets);
    DeviceArray<std::int32_t> device_indices(indices);
    const dsmvc::cuda_kernel::AxisPlanDescriptor plan{sources, destinations, bandwidth};

    for (unsigned route = 0; route < 6; ++route) {
        device_output.reset(initial);
        float *output = device_output.data() + guard;
        const bool column_major = route != 0U && route != 3U;
        cudaError_t result = cudaSuccess;
        if (route < 2U) {
            result = launch::rhs_horizontal(device_input.data(), vectors, plan,
                device_offsets.data(), device_indices.data(), device_weights.data(),
                output, column_major, nullptr);
        } else if (route == 2U) {
            result = launch::rhs_vertical(device_input.data(), vectors, plan,
                device_offsets.data(), device_indices.data(), device_weights.data(),
                output, nullptr);
        } else if (route < 5U) {
            result = launch::inverse_horizontal(device_input.data(), vectors, plan,
                device_offsets.data(), device_indices.data(), device_weights.data(),
                device_bands.data(), device_bands.data(), device_diagonal.data(),
                output, column_major, 32U, 32U * 33U * sizeof(float), nullptr);
        } else {
            result = launch::inverse_vertical(device_input.data(), vectors, plan,
                device_offsets.data(), device_indices.data(), device_weights.data(),
                device_bands.data(), device_bands.data(), device_diagonal.data(),
                output, 32U, nullptr);
        }
        check(result);
        check(cudaDeviceSynchronize());
        const auto actual = device_output.read();
        auto expected = initial;
        for (std::uint32_t vector = 0; vector < vectors; ++vector) {
            for (std::uint32_t index = 0; index < destinations; ++index) {
                float sum = 0.0F;
                for (auto entry = offsets[index]; entry < offsets[index + 1U]; ++entry) {
                    sum = std::fma(weights[entry], input[indices[entry] * vectors + vector], sum);
                }
                const auto destination = column_major
                    ? index * vectors + vector : vector * destinations + index;
                expected[guard + destination] = sum;
            }
        }
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(expected[i])) {
                throw std::runtime_error("CUDA RHS mismatch/guard write: band="
                    + std::to_string(bandwidth) + " route=" + std::to_string(route)
                    + " index=" + std::to_string(i));
            }
        }
    }
}

} // namespace

int main() {
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver
        || (status == cudaSuccess && devices == 0)) {
        std::cout << "CUDA RHS tests skipped: no usable CUDA device\n";
        return 77;
    }
    try {
        check(status);
        for (const auto band : {0U, 1U, 3U, 5U, 7U, 9U, 11U}) test_band(band);
        std::cout << "CUDA RHS tests passed: 42 routes/bands, sparse counts 0..17, 33-vector tails and output guards\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
