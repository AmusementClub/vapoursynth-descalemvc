#include "metal_device_apple.hpp"

#import <Metal/Metal.h>

namespace dsmvc::metal_detail {

bool backend_available() noexcept {
    static const bool result = []() noexcept {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            return device != nil && device.hasUnifiedMemory;
        }
    }();
    return result;
}

} // namespace dsmvc::metal_detail
