#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace dsmvc::cuda_detail {

using DevicePointer = std::byte *;

struct RuntimeApi {
    [[nodiscard]] cudaError_t mem_alloc(
        DevicePointer *pointer, std::size_t bytes) const noexcept {
        return cudaMalloc(reinterpret_cast<void **>(pointer), bytes);
    }

    [[nodiscard]] cudaError_t mem_free(DevicePointer pointer) const noexcept {
        return cudaFree(pointer);
    }

    [[nodiscard]] cudaError_t memcpy_htod(
        DevicePointer destination, const void *source,
        std::size_t bytes) const noexcept {
        return cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice);
    }

    [[nodiscard]] cudaError_t memcpy_htod_async(
        DevicePointer destination, const void *source, std::size_t bytes,
        cudaStream_t stream) const noexcept {
        return cudaMemcpyAsync(
            destination, source, bytes, cudaMemcpyHostToDevice, stream);
    }

    [[nodiscard]] cudaError_t memcpy_dtoh_async(
        void *destination, DevicePointer source, std::size_t bytes,
        cudaStream_t stream) const noexcept {
        return cudaMemcpyAsync(
            destination, source, bytes, cudaMemcpyDeviceToHost, stream);
    }

    [[nodiscard]] cudaError_t memcpy_2d_htod_async(
        DevicePointer destination, std::size_t destination_pitch,
        const void *source, std::size_t source_pitch,
        std::size_t row_bytes, std::size_t rows,
        cudaStream_t stream) const noexcept {
        return cudaMemcpy2DAsync(
            destination, destination_pitch, source, source_pitch,
            row_bytes, rows, cudaMemcpyHostToDevice, stream);
    }

    [[nodiscard]] cudaError_t memcpy_2d_dtoh_async(
        void *destination, std::size_t destination_pitch,
        DevicePointer source, std::size_t source_pitch,
        std::size_t row_bytes, std::size_t rows,
        cudaStream_t stream) const noexcept {
        return cudaMemcpy2DAsync(
            destination, destination_pitch, source, source_pitch,
            row_bytes, rows, cudaMemcpyDeviceToHost, stream);
    }

    [[nodiscard]] cudaError_t mem_host_alloc(
        void **pointer, std::size_t bytes,
        unsigned int flags) const noexcept {
        return cudaHostAlloc(pointer, bytes, flags);
    }

    [[nodiscard]] cudaError_t mem_free_host(void *pointer) const noexcept {
        return cudaFreeHost(pointer);
    }

    [[nodiscard]] cudaError_t mem_host_register(
        void *pointer, std::size_t bytes,
        unsigned int flags) const noexcept {
        return cudaHostRegister(pointer, bytes, flags);
    }

    [[nodiscard]] cudaError_t mem_host_unregister(
        void *pointer) const noexcept {
        return cudaHostUnregister(pointer);
    }

    [[nodiscard]] cudaError_t stream_create(
        cudaStream_t *stream, unsigned int flags) const noexcept {
        return cudaStreamCreateWithFlags(stream, flags);
    }

    [[nodiscard]] cudaError_t stream_destroy(
        cudaStream_t stream) const noexcept {
        return cudaStreamDestroy(stream);
    }

    [[nodiscard]] cudaError_t stream_synchronize(
        cudaStream_t stream) const noexcept {
        return cudaStreamSynchronize(stream);
    }

    [[nodiscard]] cudaError_t stream_wait_event(
        cudaStream_t stream, cudaEvent_t event,
        unsigned int flags) const noexcept {
        return cudaStreamWaitEvent(stream, event, flags);
    }

    [[nodiscard]] cudaError_t event_create(
        cudaEvent_t *event, unsigned int flags) const noexcept {
        return cudaEventCreateWithFlags(event, flags);
    }

    [[nodiscard]] cudaError_t event_destroy(
        cudaEvent_t event) const noexcept {
        return cudaEventDestroy(event);
    }

    [[nodiscard]] cudaError_t event_query(cudaEvent_t event) const noexcept {
        return cudaEventQuery(event);
    }

    [[nodiscard]] cudaError_t event_record(
        cudaEvent_t event, cudaStream_t stream) const noexcept {
        return cudaEventRecord(event, stream);
    }

    [[nodiscard]] cudaError_t event_synchronize(
        cudaEvent_t event) const noexcept {
        return cudaEventSynchronize(event);
    }
};

[[nodiscard]] inline std::string cuda_error(cudaError_t result) {
    const char *name = cudaGetErrorName(result);
    const char *description = cudaGetErrorString(result);
    std::string message = "CUDA error " + std::to_string(result);
    if (name) message += " (" + std::string{name} + ')';
    if (description) message += ": " + std::string{description};
    return message;
}

inline void cuda_check(
    const RuntimeApi &, cudaError_t result, const char *operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string{operation} + " failed: " + cuda_error(result));
    }
}

inline void cuda_check(cudaError_t result, const char *operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string{operation} + " failed: " + cuda_error(result));
    }
}

} // namespace dsmvc::cuda_detail
