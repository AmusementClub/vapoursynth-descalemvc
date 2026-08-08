#include "../src/hybrid_batch_coordinator.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <exception>
#include <iostream>
#include <latch>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using dsmvc::experimental::HybridBatchCoordinator;
using namespace std::chrono_literals;

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void test_full_batch_split() {
    std::atomic<int> cpu_count{0};
    std::atomic<int> gpu_count{0};
    std::atomic<int> gpu_calls{0};
    HybridBatchCoordinator coordinator(
        4U, 2U, 0U, 100ms, [&](std::span<void *const> payloads) {
            ++gpu_calls;
            gpu_count += static_cast<int>(payloads.size());
        });
    std::barrier start{5};
    std::vector<int> payloads(4);
    std::vector<std::thread> threads;
    for (std::size_t index = 0U; index < payloads.size(); ++index) {
        threads.emplace_back([&, index] {
            start.arrive_and_wait();
            coordinator.run(&payloads[index], [&] { ++cpu_count; });
        });
    }
    start.arrive_and_wait();
    for (auto &thread : threads) thread.join();
    require(cpu_count == 2, "full batch CPU split is wrong");
    require(gpu_count == 2, "full batch GPU split is wrong");
    require(gpu_calls == 1, "full batch executed GPU more than once");
}

void test_partial_timeout_falls_back() {
    std::atomic<int> cpu_count{0};
    std::atomic<int> gpu_calls{0};
    HybridBatchCoordinator coordinator(
        4U, 2U, 0U, 20ms, [&](std::span<void *const>) { ++gpu_calls; });
    std::vector<int> payloads(3);
    coordinator.run(&payloads[0], [&] { ++cpu_count; });
    coordinator.run(&payloads[1], [&] { ++cpu_count; });
    const auto start = std::chrono::steady_clock::now();
    coordinator.run(&payloads[2], [&] { ++cpu_count; });
    const auto elapsed = std::chrono::steady_clock::now() - start;
    require(cpu_count == 3, "partial GPU batch did not fall back to CPU");
    require(gpu_calls == 0, "partial batch unexpectedly used GPU");
    require(elapsed >= 15ms, "partial batch ignored its flush deadline");
}

void test_single_caller_does_not_deadlock() {
    std::atomic<int> cpu_count{0};
    HybridBatchCoordinator coordinator(
        16U, 9U, 0U, 100ms, [](std::span<void *const>) {});
    int payload = 0;
    const auto start = std::chrono::steady_clock::now();
    coordinator.run(&payload, [&] { ++cpu_count; });
    const auto elapsed = std::chrono::steady_clock::now() - start;
    require(cpu_count == 1, "single caller did not take its CPU slot");
    require(elapsed < 50ms, "single CPU assignment waited for a full batch");
}

void test_cpu_slots_release_request_headroom() {
    std::atomic<int> cpu_count{0};
    std::atomic<int> gpu_count{0};
    HybridBatchCoordinator coordinator(
        4U, 2U, 0U, 100ms, [&](std::span<void *const> payloads) {
            gpu_count += static_cast<int>(payloads.size());
        });
    std::vector<int> payloads(4);
    coordinator.run(&payloads[0], [&] { ++cpu_count; });
    coordinator.run(&payloads[1], [&] { ++cpu_count; });

    std::barrier start{3};
    std::vector<std::thread> threads;
    for (std::size_t index = 2U; index < payloads.size(); ++index) {
        threads.emplace_back([&, index] {
            start.arrive_and_wait();
            coordinator.run(&payloads[index], [&] { ++cpu_count; });
        });
    }
    start.arrive_and_wait();
    for (auto &thread : threads) thread.join();
    require(cpu_count == 2, "released CPU slots changed the full split");
    require(gpu_count == 2, "GPU sub-batch required the whole group in flight");
}

void test_active_cpu_work_extends_gpu_fill_window() {
    std::atomic<int> cpu_count{0};
    std::atomic<int> gpu_count{0};
    HybridBatchCoordinator coordinator(
        4U, 2U, 0U, 5ms, [&](std::span<void *const> payloads) {
            gpu_count += static_cast<int>(payloads.size());
        });
    std::vector<int> payloads(4);
    std::latch cpu_started{2};
    std::latch release_cpu{1};
    std::vector<std::thread> cpu_threads;
    for (std::size_t index = 0U; index < 2U; ++index) {
        cpu_threads.emplace_back([&, index] {
            coordinator.run(&payloads[index], [&] {
                cpu_started.count_down();
                release_cpu.wait();
                ++cpu_count;
            });
        });
    }
    cpu_started.wait();

    std::thread first_gpu([&] {
        coordinator.run(&payloads[2], [&] { ++cpu_count; });
    });
    std::this_thread::sleep_for(10ms);
    std::thread second_gpu([&] {
        coordinator.run(&payloads[3], [&] { ++cpu_count; });
    });
    first_gpu.join();
    second_gpu.join();
    release_cpu.count_down();
    for (auto &thread : cpu_threads) thread.join();
    require(cpu_count == 2, "active CPU work did not preserve the split");
    require(gpu_count == 2, "active CPU work did not extend GPU admission");
}

void test_activation_threshold_uses_observed_concurrency() {
    std::atomic<int> cpu_count{0};
    std::atomic<int> gpu_count{0};
    HybridBatchCoordinator coordinator(
        4U, 2U, 4U, 5ms, [&](std::span<void *const> payloads) {
            gpu_count += static_cast<int>(payloads.size());
        });
    std::vector<int> payloads(8);
    for (std::size_t index = 0U; index < 3U; ++index) {
        coordinator.run(&payloads[index], [&] { ++cpu_count; });
    }
    require(cpu_count == 3, "low concurrency did not stay on CPU");
    require(gpu_count == 0, "low concurrency unexpectedly activated GPU");

    std::latch cpu_started{3};
    std::latch release_cpu{1};
    std::vector<std::thread> cpu_threads;
    for (std::size_t index = 3U; index < 6U; ++index) {
        cpu_threads.emplace_back([&, index] {
            coordinator.run(&payloads[index], [&] {
                cpu_started.count_down();
                release_cpu.wait();
                ++cpu_count;
            });
        });
    }
    cpu_started.wait();
    std::thread first_gpu([&] {
        coordinator.run(&payloads[6], [&] { ++cpu_count; });
    });
    std::thread second_gpu([&] {
        coordinator.run(&payloads[7], [&] { ++cpu_count; });
    });
    first_gpu.join();
    second_gpu.join();
    release_cpu.count_down();
    for (auto &thread : cpu_threads) thread.join();
    require(cpu_count == 6, "activation threshold changed CPU assignments");
    require(gpu_count == 2, "observed concurrency did not activate GPU");
}

void test_partial_auto_batch_disables_admission() {
    std::atomic<int> cpu_count{0};
    std::atomic<int> gpu_count{0};
    HybridBatchCoordinator coordinator(
        4U, 2U, 4U, 50ms, [&](std::span<void *const> payloads) {
            gpu_count += static_cast<int>(payloads.size());
        });
    std::vector<int> payloads(7);
    std::latch cpu_started{3};
    std::latch release_cpu{1};
    std::vector<std::thread> cpu_threads;
    for (std::size_t index = 0U; index < 3U; ++index) {
        cpu_threads.emplace_back([&, index] {
            coordinator.run(&payloads[index], [&] {
                cpu_started.count_down();
                release_cpu.wait();
                ++cpu_count;
            });
        });
    }
    cpu_started.wait();

    std::latch partial_entered{1};
    std::thread partial([&] {
        partial_entered.count_down();
        coordinator.run(&payloads[3], [&] { ++cpu_count; });
    });
    partial_entered.wait();
    std::this_thread::sleep_for(5ms);
    release_cpu.count_down();
    for (auto &thread : cpu_threads) thread.join();
    partial.join();
    require(cpu_count == 4, "partial auto batch did not fall back to CPU");
    require(gpu_count == 0, "partial auto batch unexpectedly used GPU");

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t index = 4U; index < payloads.size(); ++index) {
        coordinator.run(&payloads[index], [&] { ++cpu_count; });
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    require(cpu_count == 7, "disabled auto admission changed CPU assignments");
    require(elapsed < 25ms,
            "partial auto fallback left low-concurrency admission enabled");
}

void test_gpu_error_fans_out() {
    std::atomic<int> cpu_successes{0};
    std::atomic<int> gpu_errors{0};
    HybridBatchCoordinator coordinator(
        4U, 2U, 0U, 100ms, [](std::span<void *const>) {
            throw std::runtime_error("expected GPU failure");
        });
    std::barrier start{5};
    std::vector<int> payloads(4);
    std::vector<std::thread> threads;
    for (std::size_t index = 0U; index < payloads.size(); ++index) {
        threads.emplace_back([&, index] {
            start.arrive_and_wait();
            try {
                coordinator.run(
                    &payloads[index], [&] { ++cpu_successes; });
            } catch (const std::runtime_error &) {
                ++gpu_errors;
            }
        });
    }
    start.arrive_and_wait();
    for (auto &thread : threads) thread.join();
    require(cpu_successes == 2, "GPU failure affected CPU assignments");
    require(gpu_errors == 2, "GPU failure did not reach its whole sub-batch");
}

void test_gpu_batches_are_serialized() {
    std::atomic<int> active_gpu{0};
    std::atomic<int> maximum_gpu{0};
    std::atomic<int> gpu_calls{0};
    HybridBatchCoordinator coordinator(
        4U, 2U, 0U, 100ms, [&](std::span<void *const>) {
            const int active = ++active_gpu;
            maximum_gpu.store(std::max(maximum_gpu.load(), active));
            ++gpu_calls;
            std::this_thread::sleep_for(5ms);
            --active_gpu;
        });
    std::barrier start{9};
    std::vector<int> payloads(8);
    std::vector<std::thread> threads;
    for (std::size_t index = 0U; index < payloads.size(); ++index) {
        threads.emplace_back([&, index] {
            start.arrive_and_wait();
            coordinator.run(&payloads[index], [] {});
        });
    }
    start.arrive_and_wait();
    for (auto &thread : threads) thread.join();
    require(gpu_calls == 2, "two full groups did not form two GPU batches");
    require(maximum_gpu == 1, "GPU batches overlapped shared buffers");
}

void test_completed_batch_releases_callbacks() {
    HybridBatchCoordinator coordinator(
        4U, 2U, 0U, 100ms, [](std::span<void *const>) {});
    std::barrier start{5};
    std::vector<int> payloads(4);
    std::vector<std::weak_ptr<int>> captures;
    std::vector<std::thread> threads;
    for (std::size_t index = 0U; index < payloads.size(); ++index) {
        auto capture = std::make_shared<int>(static_cast<int>(index));
        captures.push_back(capture);
        threads.emplace_back([&, index, capture] {
            start.arrive_and_wait();
            coordinator.run(&payloads[index], [capture] {});
        });
    }
    start.arrive_and_wait();
    for (auto &thread : threads) thread.join();
    threads.clear();
    require(std::ranges::all_of(captures, &std::weak_ptr<int>::expired),
            "completed GPU batch retained request callbacks");
}

void test_shutdown_releases_pending_callers() {
    HybridBatchCoordinator coordinator(
        4U, 2U, 0U, 1s, [](std::span<void *const>) {});
    std::atomic<bool> failed{false};
    std::vector<int> payloads(3);
    coordinator.run(&payloads[0], [] {});
    coordinator.run(&payloads[1], [] {});
    std::thread caller([&] {
        try {
            coordinator.run(&payloads[2], [] {});
        } catch (const std::runtime_error &) {
            failed = true;
        }
    });
    std::this_thread::sleep_for(10ms);
    coordinator.shutdown();
    caller.join();
    require(failed, "shutdown did not release a pending caller with an error");
}

void test_shutdown_waits_for_active_gpu_batch() {
    std::latch gpu_started{1};
    std::latch release_gpu{1};
    HybridBatchCoordinator coordinator(
        4U, 2U, 0U, 100ms, [&](std::span<void *const>) {
            gpu_started.count_down();
            release_gpu.wait();
        });
    std::vector<int> payloads(4);
    coordinator.run(&payloads[0], [] {});
    coordinator.run(&payloads[1], [] {});

    std::barrier start{3};
    std::vector<std::thread> gpu_threads;
    for (std::size_t index = 2U; index < payloads.size(); ++index) {
        gpu_threads.emplace_back([&, index] {
            start.arrive_and_wait();
            coordinator.run(&payloads[index], [] {});
        });
    }
    start.arrive_and_wait();
    gpu_started.wait();

    std::atomic<bool> shutdown_returned{false};
    std::thread stopper([&] {
        coordinator.shutdown();
        shutdown_returned = true;
    });
    std::this_thread::sleep_for(5ms);
    const bool returned_while_active = shutdown_returned.load();
    release_gpu.count_down();
    for (auto &thread : gpu_threads) thread.join();
    stopper.join();
    require(!returned_while_active,
            "shutdown returned while the GPU callback was active");
    require(shutdown_returned,
            "shutdown did not return after the GPU callback completed");
}

} // namespace

int main() {
    try {
        test_full_batch_split();
        test_partial_timeout_falls_back();
        test_single_caller_does_not_deadlock();
        test_cpu_slots_release_request_headroom();
        test_active_cpu_work_extends_gpu_fill_window();
        test_activation_threshold_uses_observed_concurrency();
        test_partial_auto_batch_disables_admission();
        test_gpu_error_fans_out();
        test_gpu_batches_are_serialized();
        test_completed_batch_releases_callbacks();
        test_shutdown_releases_pending_callers();
        test_shutdown_waits_for_active_gpu_batch();
        std::cout << "hybrid batch coordinator tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "hybrid batch coordinator test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
