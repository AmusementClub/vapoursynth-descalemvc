#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

// Registry of host memory blocks that a GPU backend has registered (CUDA) or
// imported (Vulkan) so that frame planes inside them can be transferred
// without a staging copy. The frame host (plugin) reports which pool blocks
// frames are being transferred from and to; a block is registered on its
// second use, within a per-owner byte cap. Executors then ask whether a plane
// range lies inside a registered block and, if so, copy straight from or to it.
// Everything is optional: a block that is not registered is staged as before.
//
// The registry knows nothing about VapourSynth. Owners are opaque tokens (the
// plugin passes the core pointer).
namespace dsmvc::host_blocks {

enum class Domain : std::uint8_t { cuda = 0, vulkan = 1 };
inline constexpr std::size_t domain_count = 2;

enum class Outcome : std::uint8_t {
    failed,
    owned,   // we registered the block and unregister it
    foreign, // somebody else registered it: usable, never unregistered by us
};

struct Registered {
    Outcome outcome = Outcome::failed;
    // Backend object for the block (a Vulkan import); null for CUDA.
    std::shared_ptr<void> handle;
};

// Called from arbitrary threads. register_block runs without the registry lock
// held, unregister_block with it held; neither may call into the registry or
// into VapourSynth.
struct Registrar {
    Registered (*register_block)(void *base, std::size_t size) noexcept = nullptr;
    void (*unregister_block)(
        void *base, std::size_t size,
        const std::shared_ptr<void> &handle) noexcept = nullptr;
};

struct Lease {
    bool covered = false;
    std::shared_ptr<void> handle; // backend object of the block, if any
    std::size_t offset = 0;       // pointer - block base
    explicit operator bool() const noexcept { return covered; }
};

struct Stats {
    std::uint64_t registrations = 0;  // blocks registered (owned or foreign)
    std::uint64_t failures = 0;       // registration attempts that failed
    std::uint64_t cap_refusals = 0;   // second uses refused by the byte cap
    std::uint64_t hits = 0;           // acquire() calls inside a registered block
    std::uint64_t misses = 0;         // acquire() calls that fell back to staging
    std::uint64_t released = 0;       // blocks unregistered
    std::uint64_t hook_releases = 0;  // of those, by the pool's release hook
    std::uint64_t registered_bytes = 0; // currently registered by us
    std::uint64_t peak_bytes = 0;
};

// Backends call install() once their runtime can register memory, and
// uninstall() before that runtime goes away (unregisters everything).
void install(Domain domain, Registrar registrar);
void uninstall(Domain domain) noexcept;
[[nodiscard]] bool enabled(Domain domain) noexcept;

// Total bytes of blocks all domains may keep registered for one owner.
void set_limit(const void *owner, std::size_t bytes);

// Report that frame data inside [base, base + size) is about to be transferred.
void note_use(Domain domain, const void *owner, void *base, std::size_t size) noexcept;

// [pointer, pointer + bytes) lies inside a block registered for `domain`.
[[nodiscard]] Lease acquire(
    Domain domain, const void *pointer, std::size_t bytes) noexcept;

// The block [base, base + size) is about to be returned to the system. May
// block while a registration of an overlapping block is in progress.
void release_block(void *base, std::size_t size) noexcept;

// Unregister everything the owner's blocks hold and forget the owner.
void drop_owner(const void *owner) noexcept;

[[nodiscard]] Stats stats(Domain domain) noexcept;
[[nodiscard]] std::string describe(Domain domain);

// Bytes spanned by `rows` rows of `row_bytes` at `pitch`.
[[nodiscard]] constexpr std::size_t plane_span(
    std::size_t pitch, std::size_t row_bytes, std::size_t rows) noexcept {
    return rows == 0U ? 0U : (rows - 1U) * pitch + row_bytes;
}

} // namespace dsmvc::host_blocks
