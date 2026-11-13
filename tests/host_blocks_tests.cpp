// Unit tests for the host block registry with a fake registrar (no GPU).
#include <dsmvc/host_blocks.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <thread>
#include <vector>

using namespace dsmvc::host_blocks;

namespace {

int failures = 0;
#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__,      \
                         __LINE__, #condition);                              \
            ++failures;                                                      \
        }                                                                    \
    } while (false)

std::atomic<int> registered_now{0};
std::atomic<int> register_calls{0};
std::atomic<int> unregister_calls{0};
std::atomic<int> next_outcome{0}; // 0 owned, 1 failed, 2 foreign
std::atomic<int> register_delay_ms{0};

Registered fake_register(void *, std::size_t) noexcept {
    ++register_calls;
    if (register_delay_ms.load()) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(register_delay_ms.load()));
    }
    switch (next_outcome.load()) {
    case 1: return {Outcome::failed, {}};
    case 2: return {Outcome::foreign, std::make_shared<int>(7)};
    default:
        ++registered_now;
        return {Outcome::owned, std::make_shared<int>(1)};
    }
}

void fake_unregister(void *, std::size_t, const std::shared_ptr<void> &) noexcept {
    ++unregister_calls;
    --registered_now;
}

constexpr std::size_t block = 1U << 20;

constexpr int owner_a = 1, owner_b = 2;

// Starts a section from a clean registry: nothing registered, default limits.
void reset_counters() {
    drop_owner(&owner_a);
    drop_owner(&owner_b);
    registered_now = 0; register_calls = 0; unregister_calls = 0;
    next_outcome = 0; register_delay_ms = 0;
    set_limit(&owner_a, 3 * block);
    set_limit(&owner_b, 8 * block);
}

} // namespace

int main() {
    alignas(4096) static unsigned char memory[8 * block];
    auto base = [&](int i) { return static_cast<void *>(memory + i * block); };

    // Not installed: everything is inert.
    CHECK(!enabled(Domain::cuda));
    note_use(Domain::cuda, &owner_a, base(0), block);
    CHECK(!acquire(Domain::cuda, base(0), 16));

    install(Domain::cuda, {fake_register, fake_unregister});
    CHECK(enabled(Domain::cuda));
    CHECK(!enabled(Domain::vulkan));
    set_limit(&owner_a, 3 * block);

    // Registered on the second use, not the first.
    note_use(Domain::cuda, &owner_a, base(0), block);
    CHECK(register_calls == 0);
    CHECK(!acquire(Domain::cuda, base(0), 16));
    note_use(Domain::cuda, &owner_a, base(0), block);
    CHECK(register_calls == 1);

    // Bounds: inside, at the end, past the end, before the block.
    CHECK(acquire(Domain::cuda, base(0), block));
    CHECK(acquire(Domain::cuda, static_cast<char *>(base(0)) + block - 64, 64));
    CHECK(!acquire(Domain::cuda, static_cast<char *>(base(0)) + block - 64, 65));
    CHECK(!acquire(Domain::cuda, static_cast<char *>(base(0)) - 1, 16));
    CHECK(!acquire(Domain::cuda, base(0), 0));
    CHECK(!acquire(Domain::vulkan, base(0), 16));
    CHECK(acquire(Domain::cuda, static_cast<char *>(base(0)) + 4096, 16).offset == 4096);

    // The cap is per owner and counted in bytes: 3 blocks fit, the 4th waits.
    for (int i = 1; i <= 3; ++i) {
        note_use(Domain::cuda, &owner_a, base(i), block);
        note_use(Domain::cuda, &owner_a, base(i), block);
    }
    CHECK(register_calls == 3); // blocks 0..2; block 3 was refused
    CHECK(!acquire(Domain::cuda, base(3), 16));
    CHECK(stats(Domain::cuda).cap_refusals >= 1);

    // Releasing a block by the pool's hook frees room for the refused one.
    release_block(base(1), block);
    CHECK(unregister_calls == 1);
    CHECK(!acquire(Domain::cuda, base(1), 16));
    note_use(Domain::cuda, &owner_a, base(3), block);
    CHECK(acquire(Domain::cuda, base(3), 16));

    // A hook for a larger range removes every block inside it.
    release_block(base(0), 4 * block);
    CHECK(registered_now == 0);
    CHECK(!acquire(Domain::cuda, base(2), 16));

    // An address handed out again with a different size or owner is not trusted.
    reset_counters();
    note_use(Domain::cuda, &owner_a, base(4), block);
    note_use(Domain::cuda, &owner_a, base(4), block);
    CHECK(acquire(Domain::cuda, base(4), 16));
    note_use(Domain::cuda, &owner_b, base(4), 2 * block); // different owner and size
    CHECK(!acquire(Domain::cuda, base(4), 16));
    CHECK(unregister_calls == 1);

    // Failed registration is remembered: no retry storm, staging continues.
    reset_counters();
    next_outcome = 1;
    for (int i = 0; i < 5; ++i) note_use(Domain::cuda, &owner_b, base(5), block);
    CHECK(register_calls == 1);
    CHECK(!acquire(Domain::cuda, base(5), 16));
    CHECK(stats(Domain::cuda).failures >= 1);
    release_block(base(5), block);

    // A foreign registration is used but never unregistered or charged to the cap.
    reset_counters();
    next_outcome = 2;
    note_use(Domain::cuda, &owner_b, base(6), block);
    note_use(Domain::cuda, &owner_b, base(6), block);
    CHECK(acquire(Domain::cuda, base(6), 16));
    release_block(base(6), block);
    CHECK(unregister_calls == 0);

    // drop_owner unregisters everything the owner holds and nothing else.
    reset_counters();
    for (int i = 0; i < 2; ++i) {
        note_use(Domain::cuda, &owner_a, base(i), block);
        note_use(Domain::cuda, &owner_a, base(i), block);
    }
    note_use(Domain::cuda, &owner_b, base(7), block);
    note_use(Domain::cuda, &owner_b, base(7), block);
    CHECK(registered_now == 3);
    drop_owner(&owner_a);
    CHECK(registered_now == 1);
    CHECK(!acquire(Domain::cuda, base(0), 16));
    CHECK(acquire(Domain::cuda, base(7), 16));

    // A hook waits for a registration in progress instead of racing it.
    reset_counters();
    register_delay_ms = 150;
    std::thread registrar([&] {
        note_use(Domain::cuda, &owner_b, base(2), block);
        note_use(Domain::cuda, &owner_b, base(2), block);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    release_block(base(2), block); // must wait, then unregister
    registrar.join();
    CHECK(register_calls == 1 && unregister_calls == 1);
    CHECK(!acquire(Domain::cuda, base(2), 16));
    register_delay_ms = 0;

    // Concurrent use, hooks and lookups from many threads.
    reset_counters();
    set_limit(&owner_b, 1000U * block);
    std::atomic<bool> stop{false};
    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) {
        workers.emplace_back([&, t] {
            unsigned x = 1234567U + t;
            while (!stop.load()) {
                x = x * 1664525U + 1013904223U;
                const int i = (x >> 16) % 8;
                const unsigned op = (x >> 8) % 8;
                if (op < 4) note_use(Domain::cuda, &owner_b, base(i), block);
                else if (op < 7) (void)acquire(Domain::cuda, base(i), 64);
                else release_block(base(i), block);
            }
        });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    stop = true;
    for (auto &worker : workers) worker.join();
    drop_owner(&owner_b);
    CHECK(registered_now == 0);
    CHECK(register_calls == unregister_calls);

    uninstall(Domain::cuda);
    CHECK(!enabled(Domain::cuda));
    CHECK(registered_now == 0);

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::puts("host_blocks tests passed");
    return EXIT_SUCCESS;
}
