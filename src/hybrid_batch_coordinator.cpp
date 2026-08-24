#include "hybrid_batch_coordinator.hpp"

#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dsmvc::experimental {

struct HybridBatchCoordinator::Impl {
    using Clock = std::chrono::steady_clock;

    enum class State {
        pending,
        cpu,
        gpu_leader,
        gpu_follower,
        complete,
        failed,
    };

    struct Request {
        void *payload = nullptr;
        CpuWork cpu_work;
        State state = State::pending;
        std::vector<std::shared_ptr<Request>> gpu_batch;
        std::exception_ptr error;
    };

    Impl(std::size_t requested_batch_size, std::size_t requested_cpu_frames,
         std::size_t requested_activation_threshold,
         std::chrono::microseconds requested_timeout,
         GpuBatchWork requested_gpu_work)
        : batch_size(requested_batch_size), cpu_frames(requested_cpu_frames),
          activation_threshold(requested_activation_threshold),
          flush_timeout(requested_timeout),
          gpu_work(std::move(requested_gpu_work)),
          admission_enabled(activation_threshold == 0U) {
        if (batch_size < 2U) {
            throw std::invalid_argument("hybrid batch size must be at least two");
        }
        if (cpu_frames == 0U || cpu_frames >= batch_size) {
            throw std::invalid_argument(
                "hybrid CPU frame count must be inside the batch");
        }
        if (flush_timeout.count() < 0) {
            throw std::invalid_argument("hybrid flush timeout must be nonnegative");
        }
        if (!gpu_work) {
            throw std::invalid_argument("hybrid GPU batch callback is required");
        }
    }

    void begin_call() {
        const std::scoped_lock lock(mutex);
        if (stopping) throw std::runtime_error("hybrid coordinator is shut down");
        ++active_calls;
    }

    void finish_call() noexcept {
        {
            const std::scoped_lock lock(mutex);
            --active_calls;
        }
        ready.notify_all();
    }

    void finish_cpu_work() noexcept {
        {
            const std::scoped_lock lock(mutex);
            --active_cpu_work;
            if (active_cpu_work == 0U && !partial_gpu_batch.empty()) {
                deadline = Clock::now() + flush_timeout;
            }
        }
        ready.notify_all();
    }

    void start_next_gpu_batch_locked() {
        if (stopping || gpu_active || queued_gpu_batches.empty()) return;

        auto batch = std::move(queued_gpu_batches.front());
        queued_gpu_batches.pop_front();
        auto leader = batch.front();
        leader->state = State::gpu_leader;
        for (std::size_t index = 1U; index < batch.size(); ++index) {
            batch[index]->state = State::gpu_follower;
        }
        leader->gpu_batch = std::move(batch);
        gpu_active = true;
        ready.notify_all();
    }

    void flush_partial_to_cpu_locked() {
        active_cpu_work += partial_gpu_batch.size();
        for (const auto &request : partial_gpu_batch) {
            request->state = State::cpu;
        }
        partial_gpu_batch.clear();
        cpu_slots_remaining = cpu_frames;
        if (activation_threshold != 0U) admission_enabled = false;
        deadline.reset();
        ready.notify_all();
    }

    void advance_locked(Clock::time_point now) {
        start_next_gpu_batch_locked();
        if (!partial_gpu_batch.empty() && deadline && now >= *deadline
            && active_cpu_work == 0U) {
            flush_partial_to_cpu_locked();
        }
    }

    void admit_locked(const std::shared_ptr<Request> &request,
                      Clock::time_point now) {
        if (!admission_enabled) {
            if (active_calls < activation_threshold) {
                ++active_cpu_work;
                request->state = State::cpu;
                return;
            }
            admission_enabled = true;
            cpu_slots_remaining = 0U;
        }

        if (cpu_slots_remaining != 0U) {
            --cpu_slots_remaining;
            ++active_cpu_work;
            request->state = State::cpu;
            return;
        }

        const std::size_t gpu_frames = batch_size - cpu_frames;
        if (partial_gpu_batch.empty()) deadline = now + flush_timeout;
        partial_gpu_batch.push_back(request);
        if (partial_gpu_batch.size() != gpu_frames) return;

        queued_gpu_batches.push_back(std::move(partial_gpu_batch));
        partial_gpu_batch.clear();
        cpu_slots_remaining = cpu_frames;
        deadline.reset();
        start_next_gpu_batch_locked();
    }

    void execute_leader(const std::shared_ptr<Request> &leader) {
        auto batch = std::move(leader->gpu_batch);
        std::vector<void *> payloads;
        payloads.reserve(batch.size());
        for (const auto &request : batch) {
            payloads.push_back(request->payload);
        }

        std::exception_ptr error;
        try {
            gpu_work(payloads);
        } catch (...) {
            error = std::current_exception();
        }

        {
            const std::scoped_lock lock(mutex);
            for (const auto &request : batch) {
                request->error = error;
                request->state = error ? State::failed : State::complete;
            }
            gpu_active = false;
            start_next_gpu_batch_locked();
        }
        ready.notify_all();
        if (error) std::rethrow_exception(error);
    }

    void execute(void *payload, CpuWork cpu_work) {
        auto request = std::make_shared<Request>();
        request->payload = payload;
        request->cpu_work = std::move(cpu_work);

        std::unique_lock lock(mutex);
        if (stopping) throw std::runtime_error("hybrid coordinator is shut down");
        admit_locked(request, Clock::now());

        while (request->state == State::pending) {
            const auto now = Clock::now();
            advance_locked(now);
            if (request->state != State::pending) break;
            if (deadline && now < *deadline) {
                ready.wait_until(lock, *deadline);
            } else {
                ready.wait(lock);
            }
            if (stopping && request->state == State::pending) {
                request->state = State::failed;
                request->error = std::make_exception_ptr(
                    std::runtime_error("hybrid coordinator is shut down"));
            }
        }

        const State state = request->state;
        if (state == State::cpu) {
            lock.unlock();
            try {
                request->cpu_work();
            } catch (...) {
                finish_cpu_work();
                throw;
            }
            finish_cpu_work();
            return;
        }
        if (state == State::gpu_leader) {
            lock.unlock();
            execute_leader(request);
            return;
        }
        if (state == State::gpu_follower) {
            ready.wait(lock, [&] {
                return request->state == State::complete
                    || request->state == State::failed;
            });
        }
        if (request->state == State::failed) {
            std::rethrow_exception(request->error);
        }
    }

    void stop() noexcept {
        std::unique_lock lock(mutex);
        if (!stopping) {
            stopping = true;
            const auto error = std::make_exception_ptr(
                std::runtime_error("hybrid coordinator is shut down"));
            for (const auto &request : partial_gpu_batch) {
                request->error = error;
                request->state = State::failed;
            }
            partial_gpu_batch.clear();
            for (const auto &batch : queued_gpu_batches) {
                for (const auto &request : batch) {
                    request->error = error;
                    request->state = State::failed;
                }
            }
            queued_gpu_batches.clear();
            deadline.reset();
            ready.notify_all();
        }
        ready.wait(lock, [&] { return active_calls == 0U; });
    }

    const std::size_t batch_size;
    const std::size_t cpu_frames;
    const std::size_t activation_threshold;
    const std::chrono::microseconds flush_timeout;
    GpuBatchWork gpu_work;
    std::mutex mutex;
    std::condition_variable ready;
    std::vector<std::shared_ptr<Request>> partial_gpu_batch;
    std::deque<std::vector<std::shared_ptr<Request>>> queued_gpu_batches;
    std::optional<Clock::time_point> deadline;
    std::size_t active_calls = 0U;
    std::size_t active_cpu_work = 0U;
    std::size_t cpu_slots_remaining = cpu_frames;
    bool gpu_active = false;
    bool admission_enabled = false;
    bool stopping = false;
};

HybridBatchCoordinator::HybridBatchCoordinator(
    std::size_t batch_size, std::size_t cpu_frames,
    std::size_t activation_threshold,
    std::chrono::microseconds flush_timeout, GpuBatchWork gpu_work)
    : impl_(std::make_unique<Impl>(
          batch_size, cpu_frames, activation_threshold,
          flush_timeout, std::move(gpu_work))) {}

HybridBatchCoordinator::~HybridBatchCoordinator() { shutdown(); }

void HybridBatchCoordinator::run(void *payload, CpuWork cpu_work) {
    if (!cpu_work) {
        throw std::invalid_argument("hybrid CPU callback is required");
    }
    impl_->begin_call();
    try {
        impl_->execute(payload, std::move(cpu_work));
    } catch (...) {
        impl_->finish_call();
        throw;
    }
    impl_->finish_call();
}

void HybridBatchCoordinator::shutdown() noexcept {
    if (impl_) impl_->stop();
}

} // namespace dsmvc::experimental
