#include <dsmvc/engine.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dsmvc {

namespace {

using SparseRow = std::vector<std::pair<std::int32_t, double>>;

double round_half_up(double value) noexcept {
    return value < 0.0 ? std::floor(value + 0.5)
                       : std::floor(value + 0.49999999999999994);
}

std::int32_t map_border(double center, std::int32_t size,
                        getnative::BorderMode border) {
    double mapped = center;
    if (center < 0.0 || center >= static_cast<double>(size)) {
        if (border == getnative::BorderMode::zero) return -1;
        if (border == getnative::BorderMode::repeat) {
            mapped = center < 0.0 ? 0.0 : static_cast<double>(size) - 0.5;
        } else {
            mapped = center < 0.0
                ? -center
                : std::min(2.0 * static_cast<double>(size) - center,
                           static_cast<double>(size) - 0.5);
        }
    }
    if (!std::isfinite(mapped)) {
        throw std::invalid_argument("custom kernel maps outside the pixel grid");
    }
    const auto index = static_cast<std::int32_t>(std::floor(mapped));
    return index >= 0 && index < size ? index : -1;
}

std::vector<SparseRow> build_rows(const AxisRequest &request,
                                  const CustomKernel &kernel) {
    if (request.source_size <= 0 || request.destination_size <= 0
        || request.kernel.taps <= 0 || request.kernel.taps > 64
        || !(request.active_length > 0.0)
        || !std::isfinite(request.active_length)
        || !std::isfinite(request.shift)) {
        throw std::invalid_argument("invalid custom axis request");
    }

    const auto support = request.kernel.taps;
    const auto tap_count = support * 2;
    const double ratio = static_cast<double>(request.source_size)
        / request.active_length;
    std::vector<SparseRow> rows(static_cast<std::size_t>(request.source_size));
    std::vector<double> tap_weights(static_cast<std::size_t>(tap_count));

    for (std::int32_t row_index = 0; row_index < request.source_size; ++row_index) {
        const double position = (static_cast<double>(row_index) + 0.5) / ratio
            + request.shift;
        const double begin = round_half_up(position - support) + 0.5;
        double total = 0.0;
        for (std::int32_t tap = 0; tap < tap_count; ++tap) {
            const double weight = kernel(std::abs(begin + tap - position));
            if (!std::isfinite(weight)) {
                throw std::runtime_error("custom kernel returned a non-finite value");
            }
            tap_weights[static_cast<std::size_t>(tap)] = weight;
            total += weight;
        }
        if (!std::isfinite(total) || total == 0.0) {
            throw std::runtime_error("custom kernel produced a zero weight sum");
        }

        auto &row = rows[static_cast<std::size_t>(row_index)];
        row.reserve(static_cast<std::size_t>(tap_count));
        for (std::int32_t tap = 0; tap < tap_count; ++tap) {
            const auto column = map_border(begin + tap, request.destination_size,
                                           request.border);
            if (column < 0) continue;
            const double weight = tap_weights[static_cast<std::size_t>(tap)] / total;
            const auto duplicate = std::find_if(
                row.begin(), row.end(), [column](const auto &entry) {
                    return entry.first == column;
                });
            if (duplicate == row.end()) row.emplace_back(column, weight);
            else duplicate->second += weight;
        }
        std::sort(row.begin(), row.end(), [](const auto &left, const auto &right) {
            return left.first < right.first;
        });
    }
    return rows;
}

void factor_banded_ldlt(std::vector<double> &bands, std::int32_t n,
                        std::int32_t half_bandwidth) noexcept {
    const auto width = static_cast<std::size_t>(n);
    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    for (std::int32_t i = 0; i < n; ++i) {
        const std::int32_t end = std::min(half_bandwidth + 1, n - i);
        const double pivot = bands[static_cast<std::size_t>(i)] + epsilon;
        for (std::int32_t distance = 1; distance < end; ++distance) {
            const std::size_t upper = static_cast<std::size_t>(distance) * width
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

} // namespace

getnative::AxisPlan build_custom_axis_plan(const AxisRequest &request,
                                           const CustomKernel &custom_kernel) {
    const auto rows = build_rows(request, custom_kernel);
    const auto source_size = request.source_size;
    const auto destination_size = request.destination_size;
    const auto half_bandwidth = std::min(
        2 * request.kernel.taps - 1, destination_size - 1);
    const auto n = static_cast<std::size_t>(destination_size);

    std::vector<double> bands(
        static_cast<std::size_t>(half_bandwidth + 1) * n, 0.0);
    for (const auto &row : rows) {
        for (std::size_t left = 0; left < row.size(); ++left) {
            for (std::size_t right = left; right < row.size(); ++right) {
                const auto a = row[left].first;
                const auto b = row[right].first;
                const auto low = std::min(a, b);
                const auto distance = std::abs(a - b);
                if (distance <= half_bandwidth) {
                    bands[static_cast<std::size_t>(distance) * n
                          + static_cast<std::size_t>(low)] +=
                        row[left].second * row[right].second;
                }
            }
        }
    }
    factor_banded_ldlt(bands, destination_size, half_bandwidth);

    getnative::AxisPlan plan;
    plan.source_size = source_size;
    plan.destination_size = destination_size;
    plan.support = request.kernel.taps;
    plan.half_bandwidth = half_bandwidth;
    plan.active_length = request.active_length;
    plan.shift = request.shift;

    std::vector<std::uint32_t> counts(n + 1U, 0U);
    for (const auto &row : rows) {
        for (const auto &[column, weight] : row) {
            (void)weight;
            ++counts[static_cast<std::size_t>(column) + 1U];
        }
    }
    for (std::size_t index = 1; index < counts.size(); ++index) {
        counts[index] += counts[index - 1U];
    }
    plan.transpose_offsets = counts;
    plan.transpose_indices.resize(counts.back());
    plan.transpose_weights.resize(counts.back());
    auto cursor = counts;
    for (std::int32_t source = 0; source < source_size; ++source) {
        for (const auto &[column, weight] : rows[static_cast<std::size_t>(source)]) {
            const auto offset = cursor[static_cast<std::size_t>(column)]++;
            plan.transpose_indices[offset] = source;
            plan.transpose_weights[offset] = static_cast<float>(weight);
        }
    }

    const auto factor_count = static_cast<std::size_t>(half_bandwidth) * n;
    plan.lower_ld.assign(factor_count, 0.0F);
    plan.upper_l.assign(factor_count, 0.0F);
    plan.inverse_diagonal.resize(n);
    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    for (std::int32_t i = 0; i < destination_size; ++i) {
        const double diagonal = bands[static_cast<std::size_t>(i)];
        plan.inverse_diagonal[static_cast<std::size_t>(i)] =
            static_cast<float>(1.0 / (diagonal + epsilon));
        const auto available = std::min(half_bandwidth, destination_size - i - 1);
        for (std::int32_t distance = 1; distance <= available; ++distance) {
            const float l = static_cast<float>(
                bands[static_cast<std::size_t>(distance) * n
                      + static_cast<std::size_t>(i)]);
            plan.upper_l[static_cast<std::size_t>(distance - 1) * n
                         + static_cast<std::size_t>(i)] = l;
            const auto row = i + distance;
            plan.lower_ld[static_cast<std::size_t>(distance - 1) * n
                          + static_cast<std::size_t>(row)] =
                static_cast<float>(
                    bands[static_cast<std::size_t>(distance) * n
                          + static_cast<std::size_t>(i)] * diagonal);
        }
    }

    // inverse_axis_f32 does not use the forward table, but AxisPlan::valid does.
    plan.forward_width = 1;
    plan.forward_offsets.resize(static_cast<std::size_t>(source_size) + 1U);
    plan.forward_indices.resize(static_cast<std::size_t>(source_size));
    plan.forward_weights.assign(static_cast<std::size_t>(source_size), 1.0F);
    for (std::int32_t row = 0; row < source_size; ++row) {
        plan.forward_offsets[static_cast<std::size_t>(row)] =
            static_cast<std::uint32_t>(row);
        plan.forward_indices[static_cast<std::size_t>(row)] =
            std::min(row, destination_size - 1);
    }
    plan.forward_offsets[static_cast<std::size_t>(source_size)] =
        static_cast<std::uint32_t>(source_size);

    if (!plan.valid()) throw std::runtime_error("failed to build custom axis plan");
    return plan;
}

} // namespace dsmvc
