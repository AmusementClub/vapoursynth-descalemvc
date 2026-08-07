#include "metal_scheduler_apple.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using dsmvc::AxisPlan;
using dsmvc::AxisRequest;
using dsmvc::BorderMode;
using dsmvc::KernelKind;
using dsmvc::metal::Client;
using dsmvc::metal::FrameJob;
using dsmvc::metal::PlaneJob;
using dsmvc::metal::RunResult;

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

[[nodiscard]] std::shared_ptr<const AxisPlan> make_plan(
    std::int32_t source, std::int32_t destination) {
    AxisRequest request;
    request.source_size = source;
    request.destination_size = destination;
    request.active_length = static_cast<double>(destination) - 0.25;
    request.shift = 0.125;
    request.kernel.kind = KernelKind::spline64;
    request.border = BorderMode::mirror;
    auto plan = std::make_shared<const AxisPlan>(dsmvc::build_axis_plan(request));
    require(plan->valid() && plan->half_bandwidth >= 5,
            "wide Metal scheduler test plan is invalid");
    return plan;
}

[[nodiscard]] FrameJob horizontal_job(
    const std::shared_ptr<const AxisPlan> &plan,
    const float *source, float *destination, std::uint32_t height) {
    PlaneJob plane;
    plane.source = source;
    plane.source_stride_bytes =
        static_cast<std::ptrdiff_t>(plan->source_size * sizeof(float));
    plane.destination = destination;
    plane.destination_stride_bytes =
        static_cast<std::ptrdiff_t>(plan->destination_size * sizeof(float));
    plane.source_width = static_cast<std::uint32_t>(plan->source_size);
    plane.source_height = height;
    plane.destination_width = static_cast<std::uint32_t>(plan->destination_size);
    plane.destination_height = height;
    plane.sample_bytes = sizeof(float);
    plane.process_horizontal = true;
    plane.horizontal = plan;

    FrameJob job;
    job.planes.push_back(std::move(plane));
    job.maximum_half_bandwidth =
        static_cast<std::uint32_t>(plan->half_bandwidth);
    return job;
}

[[nodiscard]] FrameJob two_axis_job(
    const std::shared_ptr<const AxisPlan> &horizontal,
    const std::shared_ptr<const AxisPlan> &vertical,
    const float *source, const std::vector<float *> &destinations) {
    FrameJob job;
    for (float *destination : destinations) {
        PlaneJob plane;
        plane.source = source;
        plane.source_stride_bytes =
            static_cast<std::ptrdiff_t>(horizontal->source_size * sizeof(float));
        plane.destination = destination;
        plane.destination_stride_bytes = static_cast<std::ptrdiff_t>(
            horizontal->destination_size * sizeof(float));
        plane.source_width =
            static_cast<std::uint32_t>(horizontal->source_size);
        plane.source_height = static_cast<std::uint32_t>(vertical->source_size);
        plane.destination_width =
            static_cast<std::uint32_t>(horizontal->destination_size);
        plane.destination_height =
            static_cast<std::uint32_t>(vertical->destination_size);
        plane.sample_bytes = sizeof(float);
        plane.process_horizontal = true;
        plane.process_vertical = true;
        plane.horizontal = horizontal;
        plane.vertical = vertical;
        job.planes.push_back(std::move(plane));
    }
    job.maximum_half_bandwidth = static_cast<std::uint32_t>(std::max(
        horizontal->half_bandwidth, vertical->half_bandwidth));
    return job;
}

struct Outcome {
    RunResult result;
    std::exception_ptr error;
};

void solve_rows(
    const AxisPlan &plan, const std::vector<float> &source,
    std::vector<float> &destination, std::uint32_t height);

void solve_two_axis(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::vector<float> &source, std::vector<float> &destination);

[[nodiscard]] std::vector<Outcome> run_wave(
    const std::vector<std::shared_ptr<Client>> &clients,
    std::vector<FrameJob> jobs,
    std::vector<std::function<void()>> cpu_work, bool automatic = false) {
    require(clients.size() == jobs.size() && jobs.size() == cpu_work.size(),
            "invalid scheduler test wave");
    std::vector<Outcome> outcomes(jobs.size());
    std::barrier start(static_cast<std::ptrdiff_t>(jobs.size() + 1U));
    std::vector<std::thread> threads;
    threads.reserve(jobs.size());
    for (std::size_t index = 0; index < jobs.size(); ++index) {
        threads.emplace_back([&, index] {
            start.arrive_and_wait();
            try {
                outcomes[index].result = dsmvc::metal::run(
                    clients[index], std::move(jobs[index]),
                    std::move(cpu_work[index]), automatic);
            } catch (...) {
                outcomes[index].error = std::current_exception();
            }
        });
    }
    start.arrive_and_wait();
    for (auto &thread : threads) thread.join();
    return outcomes;
}

constexpr std::uint64_t getfnative_spline36_work = 1280ULL * 720ULL * 7ULL;

void test_shared_input_automatic_admission(
    const std::shared_ptr<const AxisPlan> &plan,
    const std::vector<float> &source, std::uint32_t height) {
    constexpr std::size_t seed_count = 4U;
    constexpr std::size_t batch_count = 7U;
    const std::size_t output_size =
        static_cast<std::size_t>(plan->destination_size) * height;
    std::vector<std::vector<float>> outputs(
        batch_count, std::vector<float>(output_size));
    std::vector<std::vector<float>> expected(
        batch_count, std::vector<float>(output_size));
    std::vector<std::shared_ptr<Client>> clients;
    std::vector<FrameJob> jobs;
    std::vector<std::function<void()>> cpu_work;
    std::atomic<std::size_t> cpu_calls{0U};
    clients.reserve(seed_count + batch_count);
    for (std::size_t index = 0; index < seed_count + batch_count; ++index) {
        clients.push_back(dsmvc::metal::make_client());
    }

    std::vector<std::vector<float>> seed_outputs(
        seed_count, std::vector<float>(output_size));
    for (std::size_t index = 0; index < seed_count; ++index) {
        FrameJob job = horizontal_job(
            plan, source.data(), seed_outputs[index].data(), height);
        job.estimated_work = getfnative_spline36_work;
        bool used_cpu = false;
        const RunResult result = dsmvc::metal::run(
            clients[index], std::move(job), [&] {
                used_cpu = true;
                ++cpu_calls;
                solve_rows(*plan, source, seed_outputs[index], height);
            }, true);
        require(used_cpu && result.metal_batch_size == 0U,
                "serial shared-input census seed did not fall back to CPU");
    }

    jobs.reserve(batch_count);
    cpu_work.reserve(batch_count);
    for (std::size_t index = 0; index < batch_count; ++index) {
        FrameJob job = horizontal_job(
            plan, source.data(), outputs[index].data(), height);
        job.estimated_work = getfnative_spline36_work;
        jobs.push_back(std::move(job));
        cpu_work.emplace_back([&, index] {
            ++cpu_calls;
            solve_rows(*plan, source, outputs[index], height);
        });
        solve_rows(*plan, source, expected[index], height);
    }

    std::vector<std::shared_ptr<Client>> batch_clients(
        clients.begin() + static_cast<std::ptrdiff_t>(seed_count), clients.end());
    const auto outcomes = run_wave(
        batch_clients, std::move(jobs), std::move(cpu_work), true);
    std::size_t metal_requests = 0U;
    for (std::size_t index = 0; index < batch_count; ++index) {
        require(!outcomes[index].error,
                "shared-input automatic request reported an error");
        if (outcomes[index].result.metal_batch_size != 0U) {
            ++metal_requests;
            require(outcomes[index].result.unique_input_planes == 1U,
                    "shared-input automatic batch lost source deduplication");
        }
        float maximum = 0.0F;
        for (std::size_t sample = 0; sample < output_size; ++sample) {
            maximum = std::max(
                maximum, std::abs(outputs[index][sample] - expected[index][sample]));
        }
        require(maximum <= 3.0e-6F,
                "shared-input automatic output differs from CPU");
    }
    require(metal_requests == batch_count,
            "recent cross-client fan-out did not form a full Metal batch");
    require(cpu_calls.load() == seed_count,
            "recent shared-input admission unexpectedly fell back after seeding");
}

void test_recent_input_expiry_and_key_isolation(
    const std::shared_ptr<const AxisPlan> &plan,
    const std::vector<float> &source, std::uint32_t height) {
    constexpr std::size_t seed_count = 4U;
    constexpr std::size_t probe_count = 2U;
    const std::size_t output_size =
        static_cast<std::size_t>(plan->destination_size) * height;
    std::vector<std::shared_ptr<Client>> clients;
    clients.reserve(seed_count);
    for (std::size_t index = 0; index < seed_count; ++index) {
        clients.push_back(dsmvc::metal::make_client());
    }

    std::vector<std::vector<float>> seed_outputs(
        seed_count, std::vector<float>(output_size));
    for (std::size_t index = 0; index < seed_count; ++index) {
        FrameJob job = horizontal_job(
            plan, source.data(), seed_outputs[index].data(), height);
        job.estimated_work = getfnative_spline36_work;
        const RunResult result = dsmvc::metal::run(
            clients[index], std::move(job),
            [&, index] { solve_rows(*plan, source, seed_outputs[index], height); },
            true);
        require(result.metal_batch_size == 0U,
                "recent-input isolation seed unexpectedly entered Metal");
    }

    auto run_probe = [&](const std::vector<float> &probe_source) {
        std::vector<std::vector<float>> outputs(
            probe_count, std::vector<float>(output_size));
        std::vector<std::shared_ptr<Client>> probe_clients(
            clients.begin(),
            clients.begin() + static_cast<std::ptrdiff_t>(probe_count));
        std::vector<FrameJob> jobs;
        std::vector<std::function<void()>> cpu_work;
        std::atomic<std::size_t> cpu_calls{0U};
        for (std::size_t index = 0; index < probe_count; ++index) {
            FrameJob job = horizontal_job(
                plan, probe_source.data(), outputs[index].data(), height);
            job.estimated_work = getfnative_spline36_work;
            jobs.push_back(std::move(job));
            cpu_work.emplace_back([&, index] {
                ++cpu_calls;
                solve_rows(*plan, probe_source, outputs[index], height);
            });
        }
        const auto outcomes = run_wave(
            probe_clients, std::move(jobs), std::move(cpu_work), true);
        for (const auto &outcome : outcomes) {
            require(!outcome.error && outcome.result.metal_batch_size == 0U,
                    "isolated recent-input probe unexpectedly entered Metal");
        }
        require(cpu_calls.load() == probe_count,
                "isolated recent-input probe skipped CPU fallback");
    };

    std::vector<float> alternate_source = source;
    alternate_source.front() += 0.125F;
    run_probe(alternate_source);

    std::this_thread::sleep_for(std::chrono::milliseconds{75});
    run_probe(source);
}

void test_automatic_admission_boundaries(
    const std::shared_ptr<Client> &client,
    const std::shared_ptr<const AxisPlan> &plan,
    const std::vector<float> &source, std::uint32_t height) {
    constexpr std::size_t count = 16U;
    const std::size_t output_size =
        static_cast<std::size_t>(plan->destination_size) * height;
    std::vector<std::vector<float>> outputs(
        count, std::vector<float>(output_size));
    std::vector<std::shared_ptr<Client>> clients(count, client);
    std::vector<FrameJob> jobs;
    std::vector<std::function<void()>> cpu_work;
    std::atomic<std::size_t> cpu_calls{0U};
    for (std::size_t index = 0; index < count; ++index) {
        FrameJob job = horizontal_job(
            plan, source.data(), outputs[index].data(), height);
        job.estimated_work = getfnative_spline36_work;
        jobs.push_back(std::move(job));
        cpu_work.emplace_back([&, index] {
            ++cpu_calls;
            solve_rows(*plan, source, outputs[index], height);
        });
    }
    const auto outcomes = run_wave(
        clients, std::move(jobs), std::move(cpu_work), true);
    for (const auto &outcome : outcomes) {
        require(!outcome.error && outcome.result.metal_batch_size == 0U,
                "one client incorrectly qualified for shared-input Metal");
    }
    require(cpu_calls.load() == count,
            "one-client automatic fallback did not execute every CPU job");

    std::vector<float> destination(output_size);
    bool used_cpu = false;
    FrameJob high_work = horizontal_job(
        plan, source.data(), destination.data(), height);
    high_work.estimated_work = 1920ULL * 1080ULL * 6ULL;
    const RunResult single = dsmvc::metal::run(
        client, std::move(high_work), [&] {
            used_cpu = true;
            solve_rows(*plan, source, destination, height);
        }, true);
    require(used_cpu && single.metal_batch_size == 0U,
            "a single automatic call bypassed the low-concurrency CPU gate");
}

void solve_rows(
    const AxisPlan &plan, const std::vector<float> &source,
    std::vector<float> &destination, std::uint32_t height) {
    for (std::uint32_t row = 0; row < height; ++row) {
        dsmvc::inverse_axis_f32(
            plan,
            source.data() + static_cast<std::size_t>(row) * plan.source_size,
            1,
            destination.data()
                + static_cast<std::size_t>(row) * plan.destination_size,
            1);
    }
}

void solve_two_axis(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::vector<float> &source, std::vector<float> &destination) {
    std::vector<float> intermediate(
        static_cast<std::size_t>(horizontal.destination_size)
        * vertical.source_size);
    solve_rows(
        horizontal, source, intermediate,
        static_cast<std::uint32_t>(vertical.source_size));
    for (std::int32_t column = 0; column < horizontal.destination_size;
         ++column) {
        dsmvc::inverse_axis_f32(
            vertical, intermediate.data() + column,
            horizontal.destination_size, destination.data() + column,
            horizontal.destination_size);
    }
}

void warm_cpu_share(
    const std::shared_ptr<Client> &client,
    const std::shared_ptr<const AxisPlan> &plan,
    const std::vector<float> &source, std::uint32_t height) {
    std::vector<float> destination(
        static_cast<std::size_t>(plan->destination_size) * height);
    for (std::size_t index = 0; index < 7U; ++index) {
        const RunResult result = dsmvc::metal::run(
            client, horizontal_job(plan, source.data(), destination.data(), height),
            [&] { solve_rows(*plan, source, destination, height); }, false);
        require(result.metal_batch_size == 0U,
                "wide scheduler CPU share changed before the GPU window");
    }
}

void test_shared_source_and_recovery(
    const std::shared_ptr<Client> &first,
    const std::shared_ptr<Client> &second,
    const std::shared_ptr<const AxisPlan> &plan,
    const std::vector<float> &source, std::uint32_t height) {
    warm_cpu_share(first, plan, source, height);

    constexpr std::size_t count = 7U;
    const std::size_t output_size =
        static_cast<std::size_t>(plan->destination_size) * height;
    std::vector<std::vector<float>> outputs(
        count, std::vector<float>(output_size));
    std::vector<std::vector<float>> expected(
        count, std::vector<float>(output_size));
    std::vector<std::shared_ptr<Client>> clients;
    std::vector<FrameJob> jobs;
    std::vector<std::function<void()>> cpu_work;
    for (std::size_t index = 0; index < count; ++index) {
        clients.push_back(index % 2U == 0U ? first : second);
        jobs.push_back(horizontal_job(
            plan, source.data(), outputs[index].data(), height));
        cpu_work.emplace_back([&, index] {
            solve_rows(*plan, source, outputs[index], height);
        });
        solve_rows(*plan, source, expected[index], height);
    }

    const auto outcomes = run_wave(clients, std::move(jobs), std::move(cpu_work));
    for (std::size_t index = 0; index < count; ++index) {
        require(!outcomes[index].error, "valid Metal batch reported an error");
        require(outcomes[index].result.metal_batch_size == count,
                "wide GPU batch did not collect seven requests");
        require(outcomes[index].result.unique_input_planes == 1U,
                "cross-client source plane was uploaded more than once");
        require(outcomes[index].result.staging_memcpy_calls == count + 1U,
                "cross-client staging copy count is not deduplicated");
        require(outcomes[index].result.axis_dispatches == 1U
                    && outcomes[index].result.conversion_dispatches == 0U,
                "compatible shared-source jobs were not encoded as one batch");
        float maximum = 0.0F;
        for (std::size_t sample = 0; sample < output_size; ++sample) {
            maximum = std::max(
                maximum, std::abs(outputs[index][sample] - expected[index][sample]));
        }
        require(maximum <= 3.0e-6F,
                "direct scheduler Float32 output differs from CPU");
    }
}

void test_ring_backpressure(
    const std::shared_ptr<Client> &client,
    const std::shared_ptr<const AxisPlan> &horizontal,
    const std::shared_ptr<const AxisPlan> &vertical,
    const std::vector<float> &source) {
    constexpr std::size_t request_count = 64U;
    constexpr std::size_t plane_count = 4U;
    const std::size_t plane_samples =
        static_cast<std::size_t>(horizontal->destination_size)
        * vertical->destination_size;
    std::vector<std::vector<std::vector<float>>> outputs(
        request_count,
        std::vector<std::vector<float>>(
            plane_count, std::vector<float>(plane_samples)));
    std::vector<std::shared_ptr<Client>> clients(request_count, client);
    std::vector<FrameJob> jobs;
    std::vector<std::function<void()>> cpu_work;
    jobs.reserve(request_count);
    cpu_work.reserve(request_count);
    for (std::size_t request = 0; request < request_count; ++request) {
        std::vector<float *> destinations;
        for (auto &plane : outputs[request]) destinations.push_back(plane.data());
        jobs.push_back(two_axis_job(
            horizontal, vertical, source.data(), destinations));
        cpu_work.emplace_back([] {});
    }

    const auto before = dsmvc::metal::diagnostics();
    const auto outcomes = run_wave(clients, std::move(jobs), std::move(cpu_work));
    std::size_t metal_requests = 0;
    for (const auto &outcome : outcomes) {
        require(!outcome.error, "ring stress request failed");
        metal_requests += outcome.result.metal_batch_size != 0U;
    }
    const auto after = dsmvc::metal::diagnostics();
    require(metal_requests >= 21U,
            "ring stress did not submit enough Metal requests");
    require(after.submissions >= before.submissions + 3U,
            "ring stress did not create multiple command buffers");
    require(after.submissions == after.completions,
            "command buffer slot remained occupied after completion");
    require(after.ring_slots == 3U && after.maximum_in_flight >= 2U
                && after.maximum_in_flight <= after.ring_slots,
            "three-slot asynchronous ring/backpressure was not exercised");
    require(after.plan_cache_entries <= 128U && after.plan_cache_bytes != 0U,
            "Metal plan cache is unbounded or empty after use");
}

void test_interleaved_plan_offsets(
    const std::shared_ptr<Client> &client,
    const std::shared_ptr<const AxisPlan> &primary,
    const std::vector<float> &first_source,
    const std::vector<float> &second_source, std::uint32_t height) {
    warm_cpu_share(client, primary, first_source, height);

    constexpr std::size_t count = 7U;
    std::vector<std::shared_ptr<const AxisPlan>> plans;
    plans.reserve(count);
    plans.push_back(make_plan(primary->source_size, primary->destination_size));
    plans.push_back(primary);
    plans.push_back(primary);
    for (std::size_t index = plans.size(); index < count; ++index) {
        plans.push_back(make_plan(primary->source_size, primary->destination_size));
    }

    std::vector<std::vector<float>> outputs(
        count, std::vector<float>(
                   static_cast<std::size_t>(primary->destination_size) * height));
    std::vector<std::vector<float>> expected(
        count, std::vector<float>(
                   static_cast<std::size_t>(primary->destination_size) * height));
    std::vector<std::shared_ptr<Client>> clients(count, client);
    std::vector<FrameJob> jobs;
    std::vector<std::function<void()>> cpu_work;
    for (std::size_t index = 0; index < count; ++index) {
        const std::vector<float> &source = index == 1U
            ? second_source : first_source;
        jobs.push_back(horizontal_job(
            plans[index], source.data(), outputs[index].data(), height));
        cpu_work.emplace_back([] {});
        solve_rows(*plans[index], source, expected[index], height);
    }

    const auto outcomes = run_wave(clients, std::move(jobs), std::move(cpu_work));
    for (std::size_t index = 0; index < count; ++index) {
        require(!outcomes[index].error,
                "interleaved plan/input offsets rejected a valid submission");
        require(outcomes[index].result.metal_batch_size == count,
                "interleaved scheduler wave did not enter Metal");
        require(outcomes[index].result.axis_dispatches == 1U,
                "interleaved plans were not merged into one dispatch");
        require(outcomes[index].result.heterogeneous_axis_dispatches == 1U
                    && outcomes[index].result.heterogeneous_axis_descriptors >= 2U,
                "interleaved plans did not report heterogeneous descriptors");
        float maximum = 0.0F;
        for (std::size_t sample = 0; sample < outputs[index].size(); ++sample) {
            maximum = std::max(maximum, std::abs(
                outputs[index][sample] - expected[index][sample]));
        }
        require(maximum <= 3.0e-6F,
                "interleaved plan output differs from CPU");
    }
}

void test_heterogeneous_plan_geometry(
    const std::shared_ptr<Client> &client,
    const std::shared_ptr<const AxisPlan> &primary,
    const std::vector<float> &source, std::uint32_t height) {
    warm_cpu_share(client, primary, source, height);

    constexpr std::size_t count = 7U;
    std::vector<std::shared_ptr<const AxisPlan>> plans;
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<float>> expected;
    std::vector<FrameJob> jobs;
    std::vector<std::function<void()>> cpu_work;
    plans.reserve(count);
    outputs.reserve(count);
    expected.reserve(count);
    jobs.reserve(count);
    cpu_work.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        plans.push_back(make_plan(
            primary->source_size,
            primary->destination_size - static_cast<std::int32_t>(index)));
        const std::size_t output_size =
            static_cast<std::size_t>(plans.back()->destination_size) * height;
        outputs.emplace_back(output_size);
        expected.emplace_back(output_size);
        jobs.push_back(horizontal_job(
            plans.back(), source.data(), outputs.back().data(), height));
        cpu_work.emplace_back([] {});
        solve_rows(*plans.back(), source, expected.back(), height);
    }

    std::vector<std::shared_ptr<Client>> clients(count, client);
    const auto outcomes = run_wave(
        clients, std::move(jobs), std::move(cpu_work));
    for (std::size_t index = 0; index < count; ++index) {
        require(!outcomes[index].error,
                "heterogeneous geometry request failed");
        require(outcomes[index].result.metal_batch_size == count,
                "heterogeneous geometry wave did not enter Metal");
        require(outcomes[index].result.unique_input_planes == 1U,
                "heterogeneous geometry wave duplicated its source upload");
        require(outcomes[index].result.axis_dispatches == 1U,
                "heterogeneous geometry plans were not merged into one dispatch");
        require(outcomes[index].result.heterogeneous_axis_dispatches == 1U
                    && outcomes[index].result.heterogeneous_axis_descriptors
                        == count,
                "heterogeneous geometry descriptors were not observable");
        float maximum = 0.0F;
        for (std::size_t sample = 0; sample < outputs[index].size(); ++sample) {
            maximum = std::max(maximum, std::abs(
                outputs[index][sample] - expected[index][sample]));
        }
        require(maximum <= 3.0e-6F,
                "heterogeneous geometry output differs from CPU");
    }
}

void test_heterogeneous_two_axis_geometry(
    const std::shared_ptr<Client> &client,
    const std::shared_ptr<const AxisPlan> &primary_horizontal,
    const std::vector<float> &source, std::uint32_t source_height) {
    warm_cpu_share(client, primary_horizontal, source, source_height);

    constexpr std::size_t count = 7U;
    const auto primary_vertical = make_plan(
        static_cast<std::int32_t>(source_height),
        static_cast<std::int32_t>(source_height) - 4);
    std::vector<std::shared_ptr<const AxisPlan>> horizontal_plans;
    std::vector<std::shared_ptr<const AxisPlan>> vertical_plans;
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<float>> expected;
    std::vector<FrameJob> jobs;
    std::vector<std::function<void()>> cpu_work;
    horizontal_plans.reserve(count);
    vertical_plans.reserve(count);
    outputs.reserve(count);
    expected.reserve(count);
    jobs.reserve(count);
    cpu_work.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        horizontal_plans.push_back(make_plan(
            primary_horizontal->source_size,
            primary_horizontal->destination_size
                - static_cast<std::int32_t>(index)));
        vertical_plans.push_back(make_plan(
            primary_vertical->source_size,
            primary_vertical->destination_size
                - static_cast<std::int32_t>(index)));
        const std::size_t output_size = static_cast<std::size_t>(
            horizontal_plans.back()->destination_size)
            * vertical_plans.back()->destination_size;
        outputs.emplace_back(output_size);
        expected.emplace_back(output_size);
        jobs.push_back(two_axis_job(
            horizontal_plans.back(), vertical_plans.back(), source.data(),
            std::vector<float *>{outputs.back().data()}));
        cpu_work.emplace_back([] {});
        solve_two_axis(
            *horizontal_plans.back(), *vertical_plans.back(),
            source, expected.back());
    }

    std::vector<std::shared_ptr<Client>> clients(count, client);
    const auto outcomes = run_wave(
        clients, std::move(jobs), std::move(cpu_work));
    for (std::size_t index = 0; index < count; ++index) {
        require(!outcomes[index].error,
                "heterogeneous two-axis request failed");
        require(outcomes[index].result.metal_batch_size == count,
                "heterogeneous two-axis wave did not enter Metal");
        require(outcomes[index].result.unique_input_planes == 1U,
                "heterogeneous two-axis wave duplicated its source upload");
        require(outcomes[index].result.axis_dispatches == 2U,
                "heterogeneous two-axis plans were not merged into two dispatches");
        require(outcomes[index].result.heterogeneous_axis_dispatches == 2U
                    && outcomes[index].result.heterogeneous_axis_descriptors
                        == count * 2U,
                "heterogeneous two-axis descriptors were not observable");
        float maximum = 0.0F;
        for (std::size_t sample = 0; sample < outputs[index].size(); ++sample) {
            maximum = std::max(maximum, std::abs(
                outputs[index][sample] - expected[index][sample]));
        }
        require(maximum <= 3.0e-6F,
                "heterogeneous two-axis output differs from CPU");
    }
}

void test_error_propagation(
    const std::shared_ptr<Client> &client,
    const std::shared_ptr<const AxisPlan> &plan, std::uint32_t height) {
    constexpr std::size_t count = 16U;
    std::vector<std::shared_ptr<Client>> clients(count, client);
    std::vector<FrameJob> jobs;
    std::vector<std::function<void()>> cpu_work;
    std::vector<float> destinations(
        count * static_cast<std::size_t>(plan->destination_size) * height);
    for (std::size_t index = 0; index < count; ++index) {
        FrameJob job = horizontal_job(
            plan, nullptr,
            destinations.data()
                + index * static_cast<std::size_t>(plan->destination_size) * height,
            height);
        jobs.push_back(std::move(job));
        cpu_work.emplace_back([] {});
    }
    const auto outcomes = run_wave(clients, std::move(jobs), std::move(cpu_work));
    const auto failures = std::count_if(
        outcomes.begin(), outcomes.end(), [](const Outcome &outcome) {
            return outcome.error != nullptr;
        });
    require(failures == 7U,
            "invalid GPU submission was not propagated to its full batch");
}

} // namespace

int main() {
    try {
        require(dsmvc::metal::available(),
                "direct Metal scheduler test requires unified memory");
        auto first = dsmvc::metal::make_client();
        auto second = dsmvc::metal::make_client();

        constexpr std::uint32_t small_height = 24U;
        const auto small_plan = make_plan(64, 56);
        std::vector<float> small_source(
            static_cast<std::size_t>(small_plan->source_size) * small_height);
        for (std::size_t index = 0; index < small_source.size(); ++index) {
            small_source[index] = static_cast<float>((index * 37U + 11U) & 4095U)
                / 4095.0F;
        }
        test_shared_input_automatic_admission(
            small_plan, small_source, small_height);
        test_automatic_admission_boundaries(
            first, small_plan, small_source, small_height);
        test_recent_input_expiry_and_key_isolation(
            small_plan, small_source, small_height);
        test_shared_source_and_recovery(
            first, second, small_plan, small_source, small_height);

        std::vector<float> second_small_source = small_source;
        for (float &sample : second_small_source) sample *= 0.75F;
        test_interleaved_plan_offsets(
            first, small_plan, small_source, second_small_source, small_height);
        test_heterogeneous_plan_geometry(
            first, small_plan, small_source, small_height);
        test_heterogeneous_two_axis_geometry(
            first, small_plan, small_source, small_height);

        const auto horizontal = make_plan(384, 352);
        const auto vertical = make_plan(216, 200);
        std::vector<float> stress_source(
            static_cast<std::size_t>(horizontal->source_size)
            * vertical->source_size);
        for (std::size_t index = 0; index < stress_source.size(); ++index) {
            stress_source[index] = static_cast<float>((index * 19U + 7U) & 1023U)
                / 1023.0F;
        }
        test_ring_backpressure(first, horizontal, vertical, stress_source);
        test_error_propagation(first, small_plan, small_height);

        // The failed submission must not poison subsequent command buffers.
        test_shared_source_and_recovery(
            first, second, small_plan, small_source, small_height);

        first->close();
        std::vector<float> closed_output(
            static_cast<std::size_t>(small_plan->destination_size) * small_height);
        bool rejected = false;
        try {
            (void)dsmvc::metal::run(
                first,
                horizontal_job(
                    small_plan, small_source.data(), closed_output.data(),
                    small_height),
                [] {}, false);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        require(rejected, "closed Metal client accepted a new request");
        second->close();

        const auto diagnostics = dsmvc::metal::diagnostics();
        std::cout << "Metal scheduler tests passed: submissions="
                  << diagnostics.submissions
                  << " max_in_flight=" << diagnostics.maximum_in_flight
                  << " plan_entries=" << diagnostics.plan_cache_entries
                  << " plan_evictions=" << diagnostics.plan_cache_evictions
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Metal scheduler tests failed: " << error.what() << '\n';
        return 1;
    }
}
