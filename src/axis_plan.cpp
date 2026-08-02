#include <dsmvc/engine.hpp>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// The inverse geometry and banded LDLT strategy follow GetNative-VF's planner,
// which is independently expressed from Frechdachs/descale. This variant omits
// the zimg forward projection because dsmvc only executes the inverse solve.

namespace dsmvc {
namespace {

constexpr std::size_t plan_cache_entries = 2048U;
constexpr std::size_t plan_cache_bytes = 256U * 1024U * 1024U;
constexpr std::size_t geometry_cache_entries = 4096U;
constexpr std::size_t geometry_cache_bytes = 256U * 1024U * 1024U;

struct PlanKey {
    std::int32_t source_size;
    std::int32_t destination_size;
    std::uint64_t active_length;
    std::uint64_t shift;
    KernelKind kind;
    std::int32_t taps;
    std::uint64_t b;
    std::uint64_t c;
    getnative::BorderMode border;

    friend bool operator==(const PlanKey &, const PlanKey &) = default;
};

struct GeometryKey {
    std::int32_t source_size;
    std::int32_t destination_size;
    std::uint64_t active_length;
    std::uint64_t shift;
    std::int32_t support;
    getnative::BorderMode border;

    friend bool operator==(const GeometryKey &, const GeometryKey &) = default;
};

struct KeyHash {
    template <class Key>
    [[nodiscard]] std::size_t operator()(const Key &key) const noexcept {
        std::size_t hash = 1469598103934665603ULL;
        const auto mix = [&hash](std::uint64_t value) {
            hash ^= static_cast<std::size_t>(value);
            hash *= 1099511628211ULL;
        };
        mix(static_cast<std::uint32_t>(key.source_size));
        mix(static_cast<std::uint32_t>(key.destination_size));
        mix(key.active_length);
        mix(key.shift);
        if constexpr (std::is_same_v<Key, PlanKey>) {
            mix(static_cast<std::uint8_t>(key.kind));
            mix(static_cast<std::uint32_t>(key.taps));
            mix(key.b);
            mix(key.c);
        } else {
            mix(static_cast<std::uint32_t>(key.support));
        }
        mix(static_cast<std::uint8_t>(key.border));
        return hash;
    }
};

struct AxisGeometry {
    std::int32_t source_size = 0;
    std::int32_t destination_size = 0;
    std::int32_t support = 0;
    std::vector<double> distances;
    std::vector<std::uint32_t> row_offsets;
    std::vector<std::int32_t> unique_indices;
    std::vector<std::int32_t> tap_slots;

    [[nodiscard]] std::size_t storage_bytes() const noexcept {
        return sizeof(*this)
            + distances.capacity() * sizeof(double)
            + row_offsets.capacity() * sizeof(std::uint32_t)
            + unique_indices.capacity() * sizeof(std::int32_t)
            + tap_slots.capacity() * sizeof(std::int32_t);
    }
};

template <class Key, class Value, class Hash>
class SingleFlightLru {
    enum class State : std::uint8_t { building, ready, failed };

    struct Slot {
        State state = State::building;
        std::shared_ptr<const Value> value;
        std::exception_ptr error;
        std::size_t bytes = 0U;
        bool resident = false;
        typename std::list<Key>::iterator lru;
        std::condition_variable changed;
    };

public:
    SingleFlightLru(std::size_t maximum_entries, std::size_t maximum_bytes)
        : maximum_entries_(maximum_entries), maximum_bytes_(maximum_bytes) {}

    template <class Builder, class SizeFunction>
    [[nodiscard]] std::shared_ptr<const Value> get(
        const Key &key, Builder builder, SizeFunction size_function, bool &hit) {
        std::shared_ptr<Slot> slot;
        {
            std::unique_lock lock(mutex_);
            if (const auto found = entries_.find(key); found != entries_.end()) {
                hit = true;
                slot = found->second;
                slot->changed.wait(lock, [&] { return slot->state != State::building; });
                if (slot->state == State::failed) std::rethrow_exception(slot->error);
                touch(*slot);
                return slot->value;
            }
            hit = false;
            slot = std::make_shared<Slot>();
            entries_.emplace(key, slot);
        }

        std::shared_ptr<const Value> value;
        std::size_t bytes = 0U;
        try {
            value = builder();
            bytes = size_function(*value);
        } catch (...) {
            const auto failure = std::current_exception();
            {
                const std::scoped_lock lock(mutex_);
                slot->state = State::failed;
                slot->error = failure;
                if (const auto found = entries_.find(key);
                    found != entries_.end() && found->second == slot) {
                    entries_.erase(found);
                }
            }
            slot->changed.notify_all();
            std::rethrow_exception(failure);
        }

        {
            const std::scoped_lock lock(mutex_);
            slot->value = value;
            slot->bytes = bytes;
            slot->state = State::ready;
            const auto found = entries_.find(key);
            const bool is_current = found != entries_.end()
                && found->second == slot;
            if (is_current && maximum_entries_ != 0U
                && bytes <= maximum_bytes_) {
                lru_.push_front(key);
                slot->lru = lru_.begin();
                slot->resident = true;
                resident_bytes_ += bytes;
                evict();
            } else if (is_current) {
                entries_.erase(found);
            }
        }
        slot->changed.notify_all();
        return value;
    }

    void clear() {
        const std::scoped_lock lock(mutex_);
        entries_.clear();
        lru_.clear();
        resident_bytes_ = 0U;
    }

    [[nodiscard]] std::pair<std::size_t, std::size_t> snapshot() const {
        const std::scoped_lock lock(mutex_);
        return {lru_.size(), resident_bytes_};
    }

private:
    void touch(Slot &slot) {
        if (!slot.resident) return;
        lru_.splice(lru_.begin(), lru_, slot.lru);
        slot.lru = lru_.begin();
    }

    void evict() {
        while (lru_.size() > maximum_entries_ || resident_bytes_ > maximum_bytes_) {
            const Key victim = lru_.back();
            lru_.pop_back();
            const auto found = entries_.find(victim);
            if (found == entries_.end()) continue;
            const auto &slot = found->second;
            if (slot->state != State::ready || !slot->resident) continue;
            slot->resident = false;
            resident_bytes_ -= slot->bytes;
            entries_.erase(found);
        }
    }

    std::size_t maximum_entries_;
    std::size_t maximum_bytes_;
    mutable std::mutex mutex_;
    std::unordered_map<Key, std::shared_ptr<Slot>, Hash> entries_;
    std::list<Key> lru_;
    std::size_t resident_bytes_ = 0U;
};

struct DoubleCsr {
    std::vector<std::uint32_t> offsets;
    std::vector<std::int32_t> indices;
    std::vector<double> weights;
};

std::atomic<std::uint64_t> plan_hits{};
std::atomic<std::uint64_t> plan_builds{};
std::atomic<std::uint64_t> geometry_hits{};
std::atomic<std::uint64_t> geometry_builds{};

SingleFlightLru<PlanKey, AxisPlan, KeyHash> &plan_cache() {
    static SingleFlightLru<PlanKey, AxisPlan, KeyHash> cache{
        plan_cache_entries, plan_cache_bytes};
    return cache;
}

SingleFlightLru<GeometryKey, AxisGeometry, KeyHash> &geometry_cache() {
    static SingleFlightLru<GeometryKey, AxisGeometry, KeyHash> cache{
        geometry_cache_entries, geometry_cache_bytes};
    return cache;
}

[[nodiscard]] double round_half_up(double value) noexcept {
    return value < 0.0 ? std::floor(value + 0.5)
                       : std::floor(value + 0.49999999999999994);
}

[[nodiscard]] std::size_t checked_elements(
    std::int32_t rows, std::int32_t width, const char *name) {
    const auto row_count = static_cast<std::size_t>(rows);
    const auto row_width = static_cast<std::size_t>(width);
    if (row_width != 0U
        && row_count > std::numeric_limits<std::size_t>::max() / row_width) {
        throw std::length_error(std::string{name} + " is too large");
    }
    return row_count * row_width;
}

[[nodiscard]] std::int32_t border_index(
    double pixel_center, std::int32_t size, getnative::BorderMode border) {
    double mapped = pixel_center;
    if (pixel_center < 0.0 || pixel_center >= static_cast<double>(size)) {
        if (border == getnative::BorderMode::zero) return -1;
        if (border == getnative::BorderMode::repeat) {
            mapped = pixel_center < 0.0 ? 0.0 : static_cast<double>(size) - 0.5;
        } else {
            mapped = pixel_center < 0.0
                ? -pixel_center
                : std::min(2.0 * static_cast<double>(size) - pixel_center,
                           static_cast<double>(size) - 0.5);
        }
    }
    constexpr double minimum_index =
        static_cast<double>(std::numeric_limits<std::int32_t>::min());
    constexpr double maximum_index_exclusive =
        static_cast<double>(std::numeric_limits<std::int32_t>::max()) + 1.0;
    if (!std::isfinite(mapped) || mapped < minimum_index
        || mapped >= maximum_index_exclusive) {
        throw std::out_of_range(
            "shift places filter support outside the 32-bit pixel grid");
    }
    const auto index = static_cast<std::int32_t>(std::floor(mapped));
    return index >= 0 && index < size ? index : -1;
}

[[nodiscard]] std::int32_t filter_support(const AxisRequest &request) {
    if (request.source_size <= 0 || request.destination_size <= 0) {
        throw std::invalid_argument("axis dimensions must be positive");
    }
    if (!(request.active_length > 0.0) || !std::isfinite(request.active_length)
        || !std::isfinite(request.shift)) {
        throw std::invalid_argument(
            "active length and shift must be finite, with positive active length");
    }
    std::int32_t support = 0;
    switch (request.kernel.kind) {
    case KernelKind::bilinear: support = 1; break;
    case KernelKind::bicubic:
    case KernelKind::spline16: support = 2; break;
    case KernelKind::spline36: support = 3; break;
    case KernelKind::spline64: support = 4; break;
    case KernelKind::lanczos:
    case KernelKind::custom: support = request.kernel.taps; break;
    }
    if (support <= 0
        || support > (std::numeric_limits<std::int32_t>::max() - 1) / 2) {
        throw std::invalid_argument("filter support is invalid or too large");
    }
    return support;
}

[[nodiscard]] getnative::Filter to_filter(const KernelSpec &kernel) {
    switch (kernel.kind) {
    case KernelKind::bilinear: return getnative::Filter::bilinear();
    case KernelKind::bicubic: return getnative::Filter::bicubic(kernel.b, kernel.c);
    case KernelKind::lanczos: return getnative::Filter::lanczos(kernel.taps);
    case KernelKind::spline16: return getnative::Filter::spline16();
    case KernelKind::spline36: return getnative::Filter::spline36();
    case KernelKind::spline64: return getnative::Filter::spline64();
    case KernelKind::custom: break;
    }
    throw std::invalid_argument("custom kernels require a callback");
}

[[nodiscard]] PlanKey plan_key(const AxisRequest &request) noexcept {
    const bool bicubic = request.kernel.kind == KernelKind::bicubic;
    const bool lanczos = request.kernel.kind == KernelKind::lanczos;
    return {
        request.source_size,
        request.destination_size,
        std::bit_cast<std::uint64_t>(request.active_length),
        std::bit_cast<std::uint64_t>(request.shift),
        request.kernel.kind,
        lanczos ? request.kernel.taps : 0,
        bicubic ? std::bit_cast<std::uint64_t>(request.kernel.b) : 0U,
        bicubic ? std::bit_cast<std::uint64_t>(request.kernel.c) : 0U,
        request.border,
    };
}

[[nodiscard]] GeometryKey geometry_key(
    const AxisRequest &request, std::int32_t support) noexcept {
    return {
        request.source_size,
        request.destination_size,
        std::bit_cast<std::uint64_t>(request.active_length),
        std::bit_cast<std::uint64_t>(request.shift),
        support,
        request.border,
    };
}

[[nodiscard]] AxisGeometry build_geometry(
    const AxisRequest &request, std::int32_t support) {
    const std::int32_t tap_count = 2 * support;
    AxisGeometry geometry;
    geometry.source_size = request.source_size;
    geometry.destination_size = request.destination_size;
    geometry.support = support;
    const auto elements = checked_elements(
        request.source_size, tap_count, "inverse geometry");
    geometry.distances.resize(elements);
    geometry.tap_slots.assign(elements, -1);
    geometry.row_offsets.reserve(
        static_cast<std::size_t>(request.source_size) + 1U);
    geometry.unique_indices.reserve(elements);
    geometry.row_offsets.push_back(0U);

    const double ratio = static_cast<double>(request.source_size)
        / request.active_length;
    std::vector<std::int32_t> tap_indices(static_cast<std::size_t>(tap_count));
    std::vector<std::int32_t> unique_indices;
    unique_indices.reserve(static_cast<std::size_t>(tap_count));
    for (std::int32_t row = 0; row < request.source_size; ++row) {
        const double position = (static_cast<double>(row) + 0.5) / ratio
            + request.shift;
        const double begin = round_half_up(
            position - static_cast<double>(support)) + 0.5;
        const std::size_t row_base = static_cast<std::size_t>(row)
            * static_cast<std::size_t>(tap_count);
        unique_indices.clear();
        for (std::int32_t tap = 0; tap < tap_count; ++tap) {
            const double center = begin + static_cast<double>(tap);
            const std::size_t offset = row_base + static_cast<std::size_t>(tap);
            geometry.distances[offset] = center - position;
            const auto index = border_index(
                center, request.destination_size, request.border);
            tap_indices[static_cast<std::size_t>(tap)] = index;
            if (index >= 0
                && std::find(unique_indices.begin(), unique_indices.end(), index)
                    == unique_indices.end()) {
                unique_indices.push_back(index);
            }
        }
        std::sort(unique_indices.begin(), unique_indices.end());
        if (geometry.unique_indices.size() + unique_indices.size()
            > std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("inverse geometry coefficient table is too large");
        }
        geometry.unique_indices.insert(
            geometry.unique_indices.end(), unique_indices.begin(), unique_indices.end());
        for (std::int32_t tap = 0; tap < tap_count; ++tap) {
            const auto index = tap_indices[static_cast<std::size_t>(tap)];
            if (index < 0) continue;
            const auto slot = std::lower_bound(
                unique_indices.begin(), unique_indices.end(), index);
            geometry.tap_slots[row_base + static_cast<std::size_t>(tap)] =
                static_cast<std::int32_t>(
                    std::distance(unique_indices.begin(), slot));
        }
        geometry.row_offsets.push_back(static_cast<std::uint32_t>(
            geometry.unique_indices.size()));
    }
    return geometry;
}

[[nodiscard]] std::shared_ptr<const AxisGeometry> get_geometry(
    const AxisRequest &request, std::int32_t support) {
    bool hit = false;
    auto geometry = geometry_cache().get(
        geometry_key(request, support),
        [&] {
            geometry_builds.fetch_add(1U, std::memory_order_relaxed);
            return std::make_shared<const AxisGeometry>(
                build_geometry(request, support));
        },
        [](const AxisGeometry &value) { return value.storage_bytes(); }, hit);
    if (hit) geometry_hits.fetch_add(1U, std::memory_order_relaxed);
    return geometry;
}

[[nodiscard]] DoubleCsr make_descale_matrix(
    const AxisRequest &request, std::int32_t support,
    const AxisGeometry &geometry, const CustomKernel &custom_kernel) {
    const std::int32_t tap_count = 2 * support;
    DoubleCsr result;
    result.offsets.reserve(static_cast<std::size_t>(request.source_size) + 1U);
    result.indices.reserve(geometry.unique_indices.size());
    result.weights.reserve(geometry.unique_indices.size());
    result.offsets.push_back(0U);

    const bool custom = request.kernel.kind == KernelKind::custom;
    const auto filter = custom ? getnative::Filter{} : to_filter(request.kernel);
    std::vector<double> tap_weights(static_cast<std::size_t>(tap_count));
    std::vector<double> coalesced(static_cast<std::size_t>(tap_count));
    std::vector<bool> seen(static_cast<std::size_t>(tap_count));
    for (std::int32_t row = 0; row < request.source_size; ++row) {
        const std::size_t row_base = static_cast<std::size_t>(row)
            * static_cast<std::size_t>(tap_count);
        double total = 0.0;
        for (std::int32_t tap = 0; tap < tap_count; ++tap) {
            const double distance = geometry.distances[
                row_base + static_cast<std::size_t>(tap)];
            const double weight = custom
                ? custom_kernel(std::abs(distance)) : filter.weight(distance);
            if (!std::isfinite(weight)) {
                throw std::runtime_error("filter returned a non-finite value");
            }
            tap_weights[static_cast<std::size_t>(tap)] = weight;
            total += weight;
        }
        if (!std::isfinite(total) || total == 0.0) {
            throw std::runtime_error("filter produced a zero or non-finite weight sum");
        }

        const auto index_begin = geometry.row_offsets[static_cast<std::size_t>(row)];
        const auto index_end = geometry.row_offsets[static_cast<std::size_t>(row) + 1U];
        const auto unique_count = static_cast<std::size_t>(index_end - index_begin);
        std::fill_n(coalesced.begin(), unique_count, 0.0);
        std::fill_n(seen.begin(), unique_count, false);
        for (std::int32_t tap = 0; tap < tap_count; ++tap) {
            const auto slot = geometry.tap_slots[
                row_base + static_cast<std::size_t>(tap)];
            if (slot < 0) continue;
            const double weight = tap_weights[static_cast<std::size_t>(tap)] / total;
            if (weight == 0.0) continue;
            const auto output_slot = static_cast<std::size_t>(slot);
            coalesced[output_slot] += weight;
            seen[output_slot] = true;
        }
        for (std::size_t slot = 0; slot < unique_count; ++slot) {
            if (!seen[slot]) continue;
            result.indices.push_back(geometry.unique_indices[index_begin + slot]);
            result.weights.push_back(coalesced[slot]);
        }
        if (result.indices.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("inverse coefficient table is too large");
        }
        result.offsets.push_back(
            static_cast<std::uint32_t>(result.indices.size()));
    }
    return result;
}

[[nodiscard]] DoubleCsr transpose_csr(
    const DoubleCsr &input, std::int32_t rows, std::int32_t columns) {
    DoubleCsr result;
    result.offsets.assign(static_cast<std::size_t>(columns) + 1U, 0U);
    for (const auto index : input.indices) {
        ++result.offsets[static_cast<std::size_t>(index) + 1U];
    }
    for (std::int32_t column = 0; column < columns; ++column) {
        result.offsets[static_cast<std::size_t>(column) + 1U] +=
            result.offsets[static_cast<std::size_t>(column)];
    }
    result.indices.resize(input.indices.size());
    result.weights.resize(input.weights.size());
    auto cursor = result.offsets;
    for (std::int32_t row = 0; row < rows; ++row) {
        const auto begin = input.offsets[static_cast<std::size_t>(row)];
        const auto end = input.offsets[static_cast<std::size_t>(row) + 1U];
        for (auto offset = begin; offset < end; ++offset) {
            const auto column = static_cast<std::size_t>(input.indices[offset]);
            const auto target = cursor[column]++;
            result.indices[target] = row;
            result.weights[target] = input.weights[offset];
        }
    }
    return result;
}

[[nodiscard]] std::vector<double> form_normal_bands(
    const DoubleCsr &transpose, std::int32_t columns,
    std::int32_t half_bandwidth) {
    const auto width = static_cast<std::size_t>(columns);
    std::vector<double> bands(
        (static_cast<std::size_t>(half_bandwidth) + 1U) * width, 0.0);
    // Iterate source observations in ascending order for each native-pixel
    // pair. This preserves original descale's Float64 accumulation order while
    // storing and factorizing only the nonzero band.
    for (std::int32_t left_row = 0; left_row < columns; ++left_row) {
        const auto maximum_distance = std::min(
            half_bandwidth, columns - left_row - 1);
        for (std::int32_t distance = 0;
             distance <= maximum_distance; ++distance) {
            const auto right_row = left_row + distance;
            auto left = transpose.offsets[static_cast<std::size_t>(left_row)];
            const auto left_end =
                transpose.offsets[static_cast<std::size_t>(left_row) + 1U];
            auto right = transpose.offsets[static_cast<std::size_t>(right_row)];
            const auto right_end =
                transpose.offsets[static_cast<std::size_t>(right_row) + 1U];
            double sum = 0.0;
            while (left < left_end && right < right_end) {
                const auto left_index = transpose.indices[left];
                const auto right_index = transpose.indices[right];
                if (left_index < right_index) {
                    ++left;
                } else if (right_index < left_index) {
                    ++right;
                } else {
                    sum += transpose.weights[left] * transpose.weights[right];
                    ++left;
                    ++right;
                }
            }
            bands[static_cast<std::size_t>(distance) * width
                  + static_cast<std::size_t>(left_row)] = sum;
        }
    }
    return bands;
}

void factor_banded_ldlt(
    std::vector<double> &bands, std::int32_t n,
    std::int32_t half_bandwidth) noexcept {
    const auto width = static_cast<std::size_t>(n);
    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    for (std::int32_t i = 0; i < n; ++i) {
        const auto end = std::min(half_bandwidth + 1, n - i);
        const double pivot = bands[static_cast<std::size_t>(i)] + epsilon;
        for (std::int32_t distance = 1; distance < end; ++distance) {
            const auto upper = static_cast<std::size_t>(distance) * width
                + static_cast<std::size_t>(i);
            const double multiplier = bands[upper] / pivot;
            for (std::int32_t offset = 0; offset < end - distance; ++offset) {
                bands[static_cast<std::size_t>(offset) * width
                      + static_cast<std::size_t>(i + distance)] -= multiplier
                    * bands[static_cast<std::size_t>(distance + offset) * width
                            + static_cast<std::size_t>(i)];
            }
        }
        const double inverse_pivot = 1.0 / pivot;
        for (std::int32_t distance = 1; distance < end; ++distance) {
            bands[static_cast<std::size_t>(distance) * width
                  + static_cast<std::size_t>(i)] *= inverse_pivot;
        }
    }
}

[[nodiscard]] AxisPlan build_axis_plan_impl(
    const AxisRequest &request, const CustomKernel &custom_kernel) {
    const auto support = filter_support(request);
    if (request.kernel.kind == KernelKind::custom && !custom_kernel) {
        throw std::invalid_argument("custom kernel callback is missing");
    }
    const auto geometry = get_geometry(request, support);
    auto matrix = make_descale_matrix(
        request, support, *geometry, custom_kernel);
    auto transpose = transpose_csr(
        matrix, request.source_size, request.destination_size);
    const auto half_bandwidth = std::min(
        2 * support - 1, request.destination_size - 1);
    auto factors = form_normal_bands(
        transpose, request.destination_size, half_bandwidth);
    factor_banded_ldlt(factors, request.destination_size, half_bandwidth);

    AxisPlan plan;
    plan.source_size = request.source_size;
    plan.destination_size = request.destination_size;
    plan.support = support;
    plan.half_bandwidth = half_bandwidth;
    plan.active_length = request.active_length;
    plan.shift = request.shift;
    plan.transpose_offsets = std::move(transpose.offsets);
    plan.transpose_indices = std::move(transpose.indices);
    plan.transpose_weights.resize(transpose.weights.size());
    std::transform(
        transpose.weights.begin(), transpose.weights.end(),
        plan.transpose_weights.begin(),
        [](double weight) { return static_cast<float>(weight); });

    const auto width = static_cast<std::size_t>(request.destination_size);
    const auto factor_count = static_cast<std::size_t>(half_bandwidth) * width;
    plan.lower_ld.assign(factor_count, 0.0F);
    plan.upper_l.assign(factor_count, 0.0F);
    plan.inverse_diagonal.resize(width);
    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    for (std::int32_t i = 0; i < request.destination_size; ++i) {
        const double diagonal = factors[static_cast<std::size_t>(i)];
        plan.inverse_diagonal[static_cast<std::size_t>(i)] =
            static_cast<float>(1.0 / (diagonal + epsilon));
        const auto available = std::min(
            half_bandwidth, request.destination_size - i - 1);
        for (std::int32_t distance = 1; distance <= available; ++distance) {
            const float l = static_cast<float>(
                factors[static_cast<std::size_t>(distance) * width
                        + static_cast<std::size_t>(i)]);
            plan.upper_l[static_cast<std::size_t>(distance - 1) * width
                         + static_cast<std::size_t>(i)] = l;
            const auto row = i + distance;
            plan.lower_ld[static_cast<std::size_t>(distance - 1) * width
                          + static_cast<std::size_t>(row)] =
                static_cast<float>(
                    factors[static_cast<std::size_t>(distance) * width
                            + static_cast<std::size_t>(i)] * diagonal);
        }
    }
    if (!plan.valid()) throw std::runtime_error("failed to build inverse axis plan");
    return plan;
}

template <std::int32_t FixedHalfBandwidth>
void inverse_axis_impl(
    const AxisPlan &plan, const float *input, std::ptrdiff_t input_stride,
    float *output, std::ptrdiff_t output_stride) noexcept {
    const auto n = plan.destination_size;
    const auto band = FixedHalfBandwidth == 0
        ? plan.half_bandwidth : FixedHalfBandwidth;
    const auto width = static_cast<std::size_t>(n);
    for (std::int32_t i = 0; i < n; ++i) {
        float sum = 0.0F;
        for (auto offset = plan.transpose_offsets[static_cast<std::size_t>(i)];
             offset < plan.transpose_offsets[static_cast<std::size_t>(i) + 1U];
             ++offset) {
            sum += plan.transpose_weights[offset]
                * input[static_cast<std::ptrdiff_t>(
                    plan.transpose_indices[offset]) * input_stride];
        }
        const auto available = std::min(band, i);
        for (std::int32_t distance = available; distance >= 1; --distance) {
            sum -= plan.lower_ld[static_cast<std::size_t>(distance - 1) * width
                                 + static_cast<std::size_t>(i)]
                * output[static_cast<std::ptrdiff_t>(i - distance) * output_stride];
        }
        output[static_cast<std::ptrdiff_t>(i) * output_stride] =
            sum * plan.inverse_diagonal[static_cast<std::size_t>(i)];
    }
    for (std::int32_t i = n - 2; i >= 0; --i) {
        float sum = 0.0F;
        const auto available = std::min(band, n - i - 1);
        if constexpr (FixedHalfBandwidth == 3) {
            for (std::int32_t distance = 1; distance <= available; ++distance) {
                sum += plan.upper_l[static_cast<std::size_t>(distance - 1) * width
                                    + static_cast<std::size_t>(i)]
                    * output[static_cast<std::ptrdiff_t>(i + distance)
                             * output_stride];
            }
        } else {
            for (std::int32_t distance = available; distance >= 1; --distance) {
                sum += plan.upper_l[static_cast<std::size_t>(distance - 1) * width
                                    + static_cast<std::size_t>(i)]
                    * output[static_cast<std::ptrdiff_t>(i + distance)
                             * output_stride];
            }
        }
        output[static_cast<std::ptrdiff_t>(i) * output_stride] -= sum;
    }
}

} // namespace

bool AxisPlan::valid() const noexcept {
    if (source_size <= 0 || destination_size <= 0 || support <= 0
        || half_bandwidth < 0 || half_bandwidth >= destination_size
        || !(active_length > 0.0) || !std::isfinite(active_length)
        || !std::isfinite(shift)) {
        return false;
    }
    const auto destination = static_cast<std::size_t>(destination_size);
    const auto factors = static_cast<std::size_t>(half_bandwidth) * destination;
    return transpose_offsets.size() == destination + 1U
        && transpose_indices.size() == transpose_weights.size()
        && !transpose_offsets.empty()
        && transpose_offsets.back() == transpose_indices.size()
        && lower_ld.size() == factors
        && upper_l.size() == factors
        && inverse_diagonal.size() == destination;
}

std::size_t AxisPlan::storage_bytes() const noexcept {
    return sizeof(*this)
        + transpose_offsets.capacity() * sizeof(std::uint32_t)
        + transpose_indices.capacity() * sizeof(std::int32_t)
        + transpose_weights.capacity() * sizeof(float)
        + lower_ld.capacity() * sizeof(float)
        + upper_l.capacity() * sizeof(float)
        + inverse_diagonal.capacity() * sizeof(float);
}

AxisPlan build_axis_plan(
    const AxisRequest &request, const CustomKernel &custom_kernel) {
    plan_builds.fetch_add(1U, std::memory_order_relaxed);
    return build_axis_plan_impl(request, custom_kernel);
}

std::shared_ptr<const AxisPlan> get_or_build_axis_plan(
    const AxisRequest &request, const CustomKernel &custom_kernel) {
    if (request.kernel.kind == KernelKind::custom) {
        return std::make_shared<const AxisPlan>(
            build_axis_plan(request, custom_kernel));
    }
    bool hit = false;
    auto plan = plan_cache().get(
        plan_key(request),
        [&] {
            return std::make_shared<const AxisPlan>(
                build_axis_plan(request, custom_kernel));
        },
        [](const AxisPlan &value) { return value.storage_bytes(); }, hit);
    if (hit) plan_hits.fetch_add(1U, std::memory_order_relaxed);
    return plan;
}

PlannerCacheStats planner_cache_stats() {
    const auto [plan_entry_count, plan_bytes] = plan_cache().snapshot();
    const auto [geometry_entry_count, geometry_bytes] = geometry_cache().snapshot();
    return {
        plan_hits.load(std::memory_order_relaxed),
        plan_builds.load(std::memory_order_relaxed),
        geometry_hits.load(std::memory_order_relaxed),
        geometry_builds.load(std::memory_order_relaxed),
        plan_entry_count,
        plan_bytes,
        geometry_entry_count,
        geometry_bytes,
    };
}

void clear_planner_caches() {
    plan_cache().clear();
    geometry_cache().clear();
    plan_hits.store(0U, std::memory_order_relaxed);
    plan_builds.store(0U, std::memory_order_relaxed);
    geometry_hits.store(0U, std::memory_order_relaxed);
    geometry_builds.store(0U, std::memory_order_relaxed);
}

void inverse_axis_f32(
    const AxisPlan &plan, const float *input, std::ptrdiff_t input_stride,
    float *output, std::ptrdiff_t output_stride) {
    if (!plan.valid() || input == nullptr || output == nullptr
        || input_stride == 0 || output_stride == 0) {
        throw std::invalid_argument("invalid inverse axis arguments");
    }
    if (plan.half_bandwidth == 1) {
        inverse_axis_impl<1>(plan, input, input_stride, output, output_stride);
    } else if (plan.half_bandwidth == 3) {
        inverse_axis_impl<3>(plan, input, input_stride, output, output_stride);
    } else {
        inverse_axis_impl<0>(plan, input, input_stride, output, output_stride);
    }
}

} // namespace dsmvc
