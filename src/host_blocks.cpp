#include "dsmvc/host_blocks.hpp"

#include <algorithm>
#include <condition_variable>
#include <map>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace dsmvc::host_blocks {
namespace {

enum class State : std::uint8_t { idle, registering, registered, failed };

struct Slot {
    State state = State::idle;
    Outcome outcome = Outcome::failed;
    std::shared_ptr<void> handle;
    std::uint32_t uses = 0;
};

struct Block {
    std::size_t size = 0;
    const void *owner = nullptr;
    Slot slots[domain_count];
};

struct Account {
    std::size_t limit = 0;
    std::size_t used = 0;
};

struct Registry {
    std::mutex mutex;
    std::condition_variable changed;
    std::map<std::uintptr_t, Block> blocks;
    std::unordered_map<const void *, Account> owners;
    Registrar registrars[domain_count];
    bool installed[domain_count] = {};
    Stats stats[domain_count];
};

// Never destroyed: a pool can release blocks after static destruction.
Registry &registry() {
    static Registry *instance = new Registry;
    return *instance;
}

constexpr std::size_t index(Domain domain) noexcept {
    return static_cast<std::size_t>(domain);
}

bool any_registering(const Block &block) noexcept {
    for (const Slot &slot : block.slots) {
        if (slot.state == State::registering) return true;
    }
    return false;
}

// Called with the lock held. Unregisters one slot we own.
void drop_slot(Registry &r, std::size_t d, std::uintptr_t key, Block &block) {
    Slot &slot = block.slots[d];
    if (slot.state == State::registered) {
        if (slot.outcome == Outcome::owned) {
            if (r.installed[d] && r.registrars[d].unregister_block) {
                r.registrars[d].unregister_block(
                    reinterpret_cast<void *>(key), block.size, slot.handle);
            }
            auto &account = r.owners[block.owner];
            account.used -= std::min(account.used, block.size);
            r.stats[d].registered_bytes -=
                std::min<std::uint64_t>(r.stats[d].registered_bytes, block.size);
            ++r.stats[d].released;
        }
    }
    slot = Slot{};
}

// Waits for an in-progress registration of the block, then drops it. The
// lock may be released while waiting, so the block is looked up again.
void erase_block(
    Registry &r, std::unique_lock<std::mutex> &lock, std::uintptr_t key) {
    for (;;) {
        const auto it = r.blocks.find(key);
        if (it == r.blocks.end()) return;
        if (!any_registering(it->second)) break;
        r.changed.wait(lock);
    }
    const auto it = r.blocks.find(key);
    for (std::size_t d = 0; d < domain_count; ++d) {
        drop_slot(r, d, key, it->second);
    }
    r.blocks.erase(it);
}

} // namespace

void install(Domain domain, Registrar registrar) {
    auto &r = registry();
    const std::scoped_lock lock(r.mutex);
    const auto d = index(domain);
    r.registrars[d] = registrar;
    r.installed[d] = registrar.register_block && registrar.unregister_block;
}

void uninstall(Domain domain) noexcept {
    auto &r = registry();
    std::unique_lock lock(r.mutex);
    const auto d = index(domain);
    if (!r.installed[d]) return;
    for (;;) {
        const bool busy = std::any_of(
            r.blocks.begin(), r.blocks.end(), [d](const auto &entry) {
                return entry.second.slots[d].state == State::registering;
            });
        if (!busy) break;
        r.changed.wait(lock);
    }
    for (auto &[key, block] : r.blocks) drop_slot(r, d, key, block);
    r.installed[d] = false;
    r.registrars[d] = Registrar{};
}

bool enabled(Domain domain) noexcept {
    auto &r = registry();
    const std::scoped_lock lock(r.mutex);
    return r.installed[index(domain)];
}

void set_limit(const void *owner, std::size_t bytes) {
    auto &r = registry();
    const std::scoped_lock lock(r.mutex);
    r.owners[owner].limit = bytes;
}

void note_use(
    Domain domain, const void *owner, void *base, std::size_t size) noexcept {
    if (!base || size == 0U) return;
    try {
        auto &r = registry();
        std::unique_lock lock(r.mutex);
        const auto d = index(domain);
        if (!r.installed[d]) return;
        const auto key = reinterpret_cast<std::uintptr_t>(base);
        auto it = r.blocks.find(key);
        if (it != r.blocks.end()
            && (it->second.owner != owner || it->second.size != size)) {
            // The pool gave this address to somebody else without us hearing
            // of it; do not trust what we recorded.
            erase_block(r, lock, key);
            it = r.blocks.end();
        }
        if (it == r.blocks.end()) {
            it = r.blocks.find(key);
            if (it == r.blocks.end()) {
                it = r.blocks.emplace(key, Block{size, owner, {}}).first;
            }
        }
        Slot &slot = it->second.slots[d];
        if (slot.state != State::idle) return;
        if (++slot.uses < 2U) return;

        auto &account = r.owners[owner];
        if (account.limit == 0U || account.used + size > account.limit) {
            --slot.uses; // retry when room appears
            ++r.stats[d].cap_refusals;
            return;
        }
        account.used += size;
        slot.state = State::registering;
        const Registrar registrar = r.registrars[d];
        lock.unlock();

        Registered result;
        try {
            result = registrar.register_block(base, size);
        } catch (...) {
            result = Registered{};
        }

        lock.lock();
        // The block cannot have been erased: erase_block waits for us.
        Block &block = r.blocks.find(key)->second;
        Slot &done = block.slots[d];
        auto &acct = r.owners[owner];
        if (result.outcome == Outcome::failed) {
            done.state = State::failed;
            acct.used -= std::min(acct.used, size);
            ++r.stats[d].failures;
        } else {
            done.state = State::registered;
            done.outcome = result.outcome;
            done.handle = std::move(result.handle);
            ++r.stats[d].registrations;
            if (result.outcome == Outcome::owned) {
                r.stats[d].registered_bytes += size;
                r.stats[d].peak_bytes = std::max(
                    r.stats[d].peak_bytes, r.stats[d].registered_bytes);
            } else {
                acct.used -= std::min(acct.used, size);
            }
        }
        lock.unlock();
        r.changed.notify_all();
    } catch (...) {
    }
}

Lease acquire(
    Domain domain, const void *pointer, std::size_t bytes) noexcept {
    Lease lease;
    if (!pointer || bytes == 0U) return lease;
    try {
        auto &r = registry();
        const std::scoped_lock lock(r.mutex);
        const auto d = index(domain);
        if (!r.installed[d]) return lease;
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        auto it = r.blocks.upper_bound(address);
        if (it != r.blocks.begin()) {
            --it;
            const auto base = it->first;
            const Slot &slot = it->second.slots[d];
            if (slot.state == State::registered
                && address - base <= it->second.size
                && bytes <= it->second.size - (address - base)) {
                lease.covered = true;
                lease.handle = slot.handle;
                lease.offset = static_cast<std::size_t>(address - base);
            }
        }
        if (lease.covered) ++r.stats[d].hits; else ++r.stats[d].misses;
    } catch (...) {
        lease = Lease{};
    }
    return lease;
}

void release_block(void *base, std::size_t size) noexcept {
    if (!base || size == 0U) return;
    try {
        auto &r = registry();
        std::unique_lock lock(r.mutex);
        const auto first = reinterpret_cast<std::uintptr_t>(base);
        const auto last = first + size;
        std::vector<std::uintptr_t> keys;
        auto it = r.blocks.lower_bound(first);
        // A recorded block that starts before `base` but reaches into it.
        if (it != r.blocks.begin()) {
            const auto previous = std::prev(it);
            if (previous->first + previous->second.size > first) {
                keys.push_back(previous->first);
            }
        }
        for (; it != r.blocks.end() && it->first < last; ++it) {
            keys.push_back(it->first);
        }
        for (const auto key : keys) {
            const auto found = r.blocks.find(key);
            if (found != r.blocks.end()) {
                for (std::size_t d = 0; d < domain_count; ++d) {
                    if (found->second.slots[d].state == State::registered
                        && found->second.slots[d].outcome == Outcome::owned) {
                        ++r.stats[d].hook_releases;
                    }
                }
            }
            erase_block(r, lock, key);
        }
    } catch (...) {
    }
}

void drop_owner(const void *owner) noexcept {
    try {
        auto &r = registry();
        std::unique_lock lock(r.mutex);
        std::vector<std::uintptr_t> keys;
        for (const auto &[key, block] : r.blocks) {
            if (block.owner == owner) keys.push_back(key);
        }
        for (const auto key : keys) erase_block(r, lock, key);
        r.owners.erase(owner);
    } catch (...) {
    }
}

Stats stats(Domain domain) noexcept {
    auto &r = registry();
    const std::scoped_lock lock(r.mutex);
    return r.stats[index(domain)];
}

std::string describe(Domain domain) {
    const Stats s = stats(domain);
    std::ostringstream text;
    text << (domain == Domain::cuda ? "cuda" : "vulkan")
         << " direct I/O: registrations=" << s.registrations
         << " failures=" << s.failures
         << " cap_refusals=" << s.cap_refusals
         << " hits=" << s.hits << " misses=" << s.misses
         << " released=" << s.released
         << " by_hook=" << s.hook_releases
         << " peak_mib=" << (s.peak_bytes >> 20U);
    return text.str();
}

} // namespace dsmvc::host_blocks
