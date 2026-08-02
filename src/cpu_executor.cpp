#include <dsmvc/engine.hpp>

#include "cpu_packed.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace dsmvc {

namespace {

class WorkerPool {
    struct JobState {
        explicit JobState(std::size_t count,
                          std::function<void(std::size_t)> function)
            : job(std::move(function)), task_count(count) {}

        std::function<void(std::size_t)> job;
        std::size_t task_count = 0U;
        std::atomic<std::size_t> next_task{1U};
        std::mutex mutex;
        std::exception_ptr error;
    };

public:
    explicit WorkerPool(std::size_t parallelism)
        : parallelism_(std::max<std::size_t>(parallelism, 1U)),
          start_barrier_(static_cast<std::ptrdiff_t>(parallelism_)),
          finish_barrier_(static_cast<std::ptrdiff_t>(parallelism_)) {
        workers_.reserve(parallelism_ - 1U);
        for (std::size_t index = 1; index < parallelism_; ++index) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~WorkerPool() {
        stopping_ = true;
        start_barrier_.arrive_and_wait();
        for (auto &worker : workers_) worker.join();
    }

    WorkerPool(const WorkerPool &) = delete;
    WorkerPool &operator=(const WorkerPool &) = delete;

    [[nodiscard]] std::size_t parallelism() const noexcept {
        return parallelism_;
    }

    template <class Function>
    bool try_run(std::size_t task_count, Function &&function) {
        task_count = std::min(task_count, parallelism_);
        if (task_count < 2U
            || in_use_.test_and_set(std::memory_order_acquire)) {
            return false;
        }

        std::shared_ptr<JobState> state;
        try {
            state = std::make_shared<JobState>(
                task_count, std::forward<Function>(function));
        } catch (...) {
            in_use_.clear(std::memory_order_release);
            throw;
        }
        current_state_ = state;
        start_barrier_.arrive_and_wait();
        try {
            state->job(0U);
        } catch (...) {
            const std::scoped_lock lock(state->mutex);
            state->error = std::current_exception();
        }
        finish_barrier_.arrive_and_wait();
        current_state_.reset();
        in_use_.clear(std::memory_order_release);
        std::exception_ptr error;
        {
            const std::scoped_lock lock(state->mutex);
            error = state->error;
        }
        if (error) std::rethrow_exception(error);
        return true;
    }

private:
    void worker_loop() {
        for (;;) {
            start_barrier_.arrive_and_wait();
            if (stopping_) return;
            const auto state = current_state_;

            for (;;) {
                const auto task = state->next_task.fetch_add(
                    1U, std::memory_order_relaxed);
                if (task >= state->task_count) break;
                try {
                    state->job(task);
                } catch (...) {
                    const std::scoped_lock lock(state->mutex);
                    if (!state->error) state->error = std::current_exception();
                }
            }
            finish_barrier_.arrive_and_wait();
        }
    }

    std::size_t parallelism_ = 1U;
    std::barrier<> start_barrier_;
    std::barrier<> finish_barrier_;
    std::vector<std::thread> workers_;
    std::shared_ptr<JobState> current_state_;
    std::atomic_flag in_use_ = ATOMIC_FLAG_INIT;
    bool stopping_ = false;
};

[[nodiscard]] std::size_t cpu_parallelism(CpuPath path) noexcept {
    if (path != CpuPath::avx2) return 1U;
    const auto hardware = std::max(std::thread::hardware_concurrency(), 1U);
    return std::min<std::size_t>(hardware, 4U);
}

[[nodiscard]] std::shared_ptr<WorkerPool> shared_worker_pool(CpuPath path) {
    if (path != CpuPath::avx2) return {};
    static auto pool = std::make_shared<WorkerPool>(cpu_parallelism(path));
    return pool;
}

} // namespace

#if defined(DSMVC_HAS_AVX2_OBJECT)
void inverse_rows_avx2(const getnative::AxisPlan &plan,
                       const detail::PackedCpuPlan &packed,
                       const float *input, std::ptrdiff_t input_row_stride,
                       float *output, std::ptrdiff_t output_row_stride,
                       std::int32_t row_count);
void inverse_columns_avx2(const getnative::AxisPlan &plan,
                          const detail::PackedCpuPlan &packed,
                          const float *input, std::ptrdiff_t input_row_stride,
                          float *output, std::ptrdiff_t output_row_stride,
                          std::int32_t column_count);
#endif

struct CpuExecutor::Impl {
    explicit Impl(CpuPath path)
        : workers(shared_worker_pool(path)) {}

    mutable std::mutex mutex;
    mutable std::vector<std::shared_ptr<const detail::PackedCpuPlan>> plans;
    mutable bool sealed = false;
    std::shared_ptr<WorkerPool> workers;

    [[nodiscard]] auto find(const getnative::AxisPlan &plan) const {
        return std::find_if(
            plans.begin(), plans.end(), [&plan](const auto &candidate) {
                return candidate->axis == &plan;
            });
    }

    [[nodiscard]] const detail::PackedCpuPlan &get(
        const getnative::AxisPlan &plan) const {
        if (sealed) {
            const auto found = find(plan);
            if (found == plans.end()) {
                throw std::logic_error("sealed CPU plan cache is missing an axis");
            }
            return **found;
        }
        const std::scoped_lock lock(mutex);
        auto found = find(plan);
        if (found == plans.end()) {
            plans.push_back(std::make_shared<const detail::PackedCpuPlan>(
                detail::pack_cpu_plan(plan)));
            found = std::prev(plans.end());
        }
        return **found;
    }
};

namespace detail {

PackedCpuPlan pack_cpu_plan(const getnative::AxisPlan &plan) {
    PackedCpuPlan packed;
    packed.axis = &plan;
    const auto n = plan.destination_size;
    packed.padded_source_size = (plan.source_size + 7) & ~7;
    packed.padded_destination_size = (n + 7) & ~7;
    const auto padded_n = static_cast<std::size_t>(packed.padded_destination_size);
    packed.weights_left.assign(padded_n, 0);
    packed.weights_right.assign(padded_n, 0);
    for (std::int32_t row = 0; row < n; ++row) {
        const auto begin = plan.transpose_offsets[static_cast<std::size_t>(row)];
        const auto end = plan.transpose_offsets[static_cast<std::size_t>(row) + 1U];
        if (begin == end) continue;
        const auto first = plan.transpose_indices[begin];
        const auto last = plan.transpose_indices[end - 1U] + 1;
        packed.weights_left[static_cast<std::size_t>(row)] = first;
        packed.weights_right[static_cast<std::size_t>(row)] = last;
        packed.weights_columns = std::max(packed.weights_columns, last - first);
    }
    packed.weights.assign(padded_n
                              * static_cast<std::size_t>(packed.weights_columns),
                          0.0F);
    for (std::int32_t row = 0; row < n; ++row) {
        const auto begin = plan.transpose_offsets[static_cast<std::size_t>(row)];
        const auto end = plan.transpose_offsets[static_cast<std::size_t>(row) + 1U];
        const auto left = packed.weights_left[static_cast<std::size_t>(row)];
        for (auto offset = begin; offset < end; ++offset) {
            const auto column = plan.transpose_indices[offset];
            packed.weights[static_cast<std::size_t>(row)
                               * static_cast<std::size_t>(packed.weights_columns)
                           + static_cast<std::size_t>(column - left)] =
                plan.transpose_weights[offset];
        }
    }

    const auto bands = static_cast<std::size_t>(plan.half_bandwidth);
    packed.lower_ld.assign(bands * padded_n, 0.0F);
    packed.upper_l.assign(bands * padded_n, 0.0F);
    packed.inverse_diagonal.assign(padded_n, 0.0F);
    std::copy(plan.inverse_diagonal.begin(), plan.inverse_diagonal.end(),
              packed.inverse_diagonal.begin());
    for (std::size_t band = 0; band < bands; ++band) {
        const auto input_offset = band * static_cast<std::size_t>(n);
        const auto output_offset = band * padded_n;
        std::copy_n(plan.lower_ld.begin() + static_cast<std::ptrdiff_t>(input_offset),
                    n, packed.lower_ld.begin()
                           + static_cast<std::ptrdiff_t>(output_offset));
        std::copy_n(plan.upper_l.begin() + static_cast<std::ptrdiff_t>(input_offset),
                    n, packed.upper_l.begin()
                           + static_cast<std::ptrdiff_t>(output_offset));
    }
    return packed;
}

} // namespace detail

bool cpu_avx2_compiled() noexcept {
#if defined(DSMVC_HAS_AVX2_OBJECT)
    return true;
#else
    return false;
#endif
}

bool cpu_avx2_available() noexcept {
#if defined(DSMVC_HAS_AVX2_OBJECT) && defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int registers[4]{};
    __cpuid(registers, 1);
    constexpr int osxsave = 1 << 27;
    constexpr int avx = 1 << 28;
    constexpr int fma = 1 << 12;
    if ((registers[2] & (osxsave | avx | fma)) != (osxsave | avx | fma)) return false;
    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6U) != 0x6U) return false;
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
#elif defined(DSMVC_HAS_AVX2_OBJECT) && (defined(__x86_64__) || defined(__i386__))
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

CpuExecutor::CpuExecutor(CpuPath requested) {
    if (requested == CpuPath::avx2 && !cpu_avx2_available()) {
        throw std::runtime_error("opt=2 requires an AVX2 and FMA capable CPU");
    }
    path_ = requested == CpuPath::automatic
        ? (cpu_avx2_available() ? CpuPath::avx2 : CpuPath::scalar)
        : requested;
    impl_ = std::make_shared<Impl>(path_);
}

CpuExecutor::~CpuExecutor() = default;

CpuPath CpuExecutor::path() const noexcept { return path_; }

const char *CpuExecutor::name() const noexcept {
    return path_ == CpuPath::avx2 ? "avx2-fma" : "scalar";
}

void CpuExecutor::prepare(const getnative::AxisPlan &plan) const {
    if (path_ != CpuPath::avx2) return;
    const std::scoped_lock lock(impl_->mutex);
    const auto found = impl_->find(plan);
    if (found == impl_->plans.end()) {
        if (impl_->sealed) {
            throw std::logic_error("cannot add an axis to a sealed CPU plan cache");
        }
        impl_->plans.push_back(std::make_shared<const detail::PackedCpuPlan>(
            detail::pack_cpu_plan(plan)));
    }
}

void CpuExecutor::seal() const {
    if (path_ != CpuPath::avx2) return;
    const std::scoped_lock lock(impl_->mutex);
    impl_->sealed = true;
}

void CpuExecutor::inverse_rows(const getnative::AxisPlan &plan,
                               const float *input, std::ptrdiff_t input_row_stride,
                               float *output, std::ptrdiff_t output_row_stride,
                               std::int32_t row_count) const {
    if (!plan.valid() || !input || !output || input_row_stride <= 0
        || output_row_stride <= 0 || row_count < 0) {
        throw std::invalid_argument("invalid row executor arguments");
    }
#if defined(DSMVC_HAS_AVX2_OBJECT)
    if (path_ == CpuPath::avx2) {
        const auto &packed = impl_->get(plan);
        const auto complete_groups = static_cast<std::size_t>(row_count / 8);
        const auto task_count = std::min(
            impl_->workers->parallelism(), complete_groups);
        const auto enough_work = static_cast<std::size_t>(row_count)
            * static_cast<std::size_t>(plan.destination_size) >= 262144U;
        if (enough_work && impl_->workers->try_run(
                task_count, [&](std::size_t task) {
                    const auto first_group = complete_groups * task / task_count;
                    const auto last_group = complete_groups * (task + 1U) / task_count;
                    const auto first_row = static_cast<std::int32_t>(first_group * 8U);
                    const auto task_rows = static_cast<std::int32_t>(
                        (last_group - first_group) * 8U);
                    inverse_rows_avx2(
                        plan, packed,
                        input + static_cast<std::ptrdiff_t>(first_row) * input_row_stride,
                        input_row_stride,
                        output + static_cast<std::ptrdiff_t>(first_row) * output_row_stride,
                        output_row_stride, task_rows);
                })) {
            if ((row_count & 7) != 0) {
                const auto first_row = row_count - 8;
                inverse_rows_avx2(
                    plan, packed,
                    input + static_cast<std::ptrdiff_t>(first_row) * input_row_stride,
                    input_row_stride,
                    output + static_cast<std::ptrdiff_t>(first_row) * output_row_stride,
                    output_row_stride, 8);
            }
            return;
        }
        inverse_rows_avx2(plan, packed, input, input_row_stride, output,
                          output_row_stride, row_count);
        return;
    }
#endif
    for (std::int32_t row = 0; row < row_count; ++row) {
        getnative::inverse_axis_f32(
            plan, input + static_cast<std::ptrdiff_t>(row) * input_row_stride, 1,
            output + static_cast<std::ptrdiff_t>(row) * output_row_stride, 1);
    }
}

void CpuExecutor::inverse_columns(const getnative::AxisPlan &plan,
                                  const float *input, std::ptrdiff_t input_row_stride,
                                  float *output, std::ptrdiff_t output_row_stride,
                                  std::int32_t column_count) const {
    if (!plan.valid() || !input || !output || input_row_stride <= 0
        || output_row_stride <= 0 || column_count < 0) {
        throw std::invalid_argument("invalid column executor arguments");
    }
#if defined(DSMVC_HAS_AVX2_OBJECT)
    if (path_ == CpuPath::avx2) {
        const auto &packed = impl_->get(plan);
        const auto padded_columns = (column_count + 7) & ~7;
        const auto vector_columns = input_row_stride >= padded_columns
                && output_row_stride >= padded_columns
            ? padded_columns : (column_count & ~7);
        const auto column_groups = static_cast<std::size_t>(vector_columns / 8);
        const auto task_count = std::min(
            impl_->workers->parallelism(), column_groups);
        const auto enough_work = static_cast<std::size_t>(column_count)
            * static_cast<std::size_t>(plan.destination_size) >= 262144U;
        if (enough_work && impl_->workers->try_run(
                task_count, [&](std::size_t task) {
                    const auto first_group = column_groups * task / task_count;
                    const auto last_group = column_groups * (task + 1U) / task_count;
                    const auto first_column = static_cast<std::int32_t>(first_group * 8U);
                    const auto task_columns = static_cast<std::int32_t>(
                        (last_group - first_group) * 8U);
                    inverse_columns_avx2(
                        plan, packed, input + first_column, input_row_stride,
                        output + first_column, output_row_stride, task_columns);
                })) {
            for (std::int32_t column = vector_columns;
                 column < column_count; ++column) {
                getnative::inverse_axis_f32(
                    plan, input + column, input_row_stride,
                    output + column, output_row_stride);
            }
            return;
        }
        inverse_columns_avx2(plan, packed, input, input_row_stride, output,
                             output_row_stride, column_count);
        return;
    }
#endif
    for (std::int32_t column = 0; column < column_count; ++column) {
        getnative::inverse_axis_f32(plan, input + column, input_row_stride,
                                    output + column, output_row_stride);
    }
}

} // namespace dsmvc
