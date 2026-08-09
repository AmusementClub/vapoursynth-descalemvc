// Shared production kernels for the plugin and standalone diagnostics.
#include <metal_stdlib>

using namespace metal;

enum : uint {
    horizontal_axis = 0,
};

struct AxisJob {
    uint source_size;
    uint destination_size;
    uint vector_count;
    uint input_stride;
    uint output_stride;
    uint direction;
    uint half_bandwidth;
    uint reserved;
    uint batch_count;
    uint input_frame_stride;
    uint output_frame_stride;
    uint reserved_2;
};

struct AxisBatchJob {
    AxisJob axis;
    uint input_offset;
    uint output_offset;
    uint transpose_offsets_offset;
    uint transpose_indices_offset;
    uint transpose_weights_offset;
    uint lower_ld_offset;
    uint upper_l_offset;
    uint inverse_diagonal_offset;
};

struct IntegerConversion {
    float input_offset;
    float input_scale;
    float output_scale;
    float output_offset;
    uint output_maximum;
};

struct ConvertJob {
    uint width;
    uint height;
    uint input_stride;
    uint output_stride;
    uint batch_count;
    uint input_frame_stride;
    uint output_frame_stride;
    uint reserved;
};

static inline uint image_index(uint direction, uint vector, uint axis_index,
                               uint stride, uint flags) {
    if ((flags & 1u) != 0u) return axis_index * stride + vector;
    return direction == horizontal_axis ? vector * stride + axis_index
                                        : axis_index * stride + vector;
}

template <uint HalfBandwidth>
static inline float apply_lower_window(
    float sum, device const float *lower_ld, uint destination_size, uint i,
    float lag1, float lag2, float lag3, float lag4,
    float lag5, float lag6, float lag7) {
    if (HalfBandwidth >= 7u && i >= 7u) {
        sum = fma(-lower_ld[6u * destination_size + i], lag7, sum);
    }
    if (HalfBandwidth >= 6u && i >= 6u) {
        sum = fma(-lower_ld[5u * destination_size + i], lag6, sum);
    }
    if (HalfBandwidth >= 5u && i >= 5u) {
        sum = fma(-lower_ld[4u * destination_size + i], lag5, sum);
    }
    if (HalfBandwidth >= 4u && i >= 4u) {
        sum = fma(-lower_ld[3u * destination_size + i], lag4, sum);
    }
    if (HalfBandwidth >= 3u && i >= 3u) {
        sum = fma(-lower_ld[2u * destination_size + i], lag3, sum);
    }
    if (HalfBandwidth >= 2u && i >= 2u) {
        sum = fma(-lower_ld[destination_size + i], lag2, sum);
    }
    if (i >= 1u) sum = fma(-lower_ld[i], lag1, sum);
    return sum;
}

template <uint HalfBandwidth>
static inline float apply_upper_window(
    float sum, device const float *upper_l, uint destination_size, uint i,
    uint available, float lag1, float lag2, float lag3, float lag4,
    float lag5, float lag6, float lag7) {
    if (HalfBandwidth >= 7u && available >= 7u) {
        sum = fma(-upper_l[6u * destination_size + i], lag7, sum);
    }
    if (HalfBandwidth >= 6u && available >= 6u) {
        sum = fma(-upper_l[5u * destination_size + i], lag6, sum);
    }
    if (HalfBandwidth >= 5u && available >= 5u) {
        sum = fma(-upper_l[4u * destination_size + i], lag5, sum);
    }
    if (HalfBandwidth >= 4u && available >= 4u) {
        sum = fma(-upper_l[3u * destination_size + i], lag4, sum);
    }
    if (HalfBandwidth >= 3u && available >= 3u) {
        sum = fma(-upper_l[2u * destination_size + i], lag3, sum);
    }
    if (HalfBandwidth >= 2u && available >= 2u) {
        sum = fma(-upper_l[destination_size + i], lag2, sum);
    }
    if (available >= 1u) sum = fma(-upper_l[i], lag1, sum);
    return sum;
}

template <uint HalfBandwidth>
static inline void push_window(
    float value, thread float &lag1, thread float &lag2,
    thread float &lag3, thread float &lag4, thread float &lag5,
    thread float &lag6, thread float &lag7) {
    if (HalfBandwidth >= 7u) lag7 = lag6;
    if (HalfBandwidth >= 6u) lag6 = lag5;
    if (HalfBandwidth >= 5u) lag5 = lag4;
    if (HalfBandwidth >= 4u) lag4 = lag3;
    if (HalfBandwidth >= 3u) lag3 = lag2;
    if (HalfBandwidth >= 2u) lag2 = lag1;
    lag1 = value;
}

template <uint HalfBandwidth>
static inline void inverse_axis_fixed_wide_impl(
    device const float *source,
    constant AxisJob &job,
    device const uint *transpose_offsets,
    device const int *transpose_indices,
    device const float *transpose_weights,
    device const float *lower_ld,
    device const float *upper_l,
    device const float *inverse_diagonal,
    device float *output,
    uint global_vector) {
    const uint batch = global_vector / job.vector_count;
    const uint vector = global_vector - batch * job.vector_count;
    if (batch >= job.batch_count || vector >= job.vector_count) return;

    source += batch * job.input_frame_stride;
    output += batch * job.output_frame_stride;

    const uint output_base = job.direction == horizontal_axis
        ? vector * job.output_stride : vector;
    const uint output_step = job.direction == horizontal_axis
        ? 1u : job.output_stride;
    float lag1 = 0.0f;
    float lag2 = 0.0f;
    float lag3 = 0.0f;
    float lag4 = 0.0f;
    float lag5 = 0.0f;
    float lag6 = 0.0f;
    float lag7 = 0.0f;

    for (uint i = 0; i < job.destination_size; ++i) {
        float sum = 0.0f;
        const uint begin = transpose_offsets[i];
        const uint end = transpose_offsets[i + 1u];
        for (uint entry = begin; entry < end; ++entry) {
            const uint source_axis = uint(transpose_indices[entry]);
            sum = fma(
                transpose_weights[entry],
                source[image_index(job.direction, vector, source_axis,
                                   job.input_stride, job.reserved)],
                sum);
        }
        sum = apply_lower_window<HalfBandwidth>(
            sum, lower_ld, job.destination_size, i,
            lag1, lag2, lag3, lag4, lag5, lag6, lag7);
        const float value = sum * inverse_diagonal[i];
        output[output_base + i * output_step] = value;
        push_window<HalfBandwidth>(
            value, lag1, lag2, lag3, lag4, lag5, lag6, lag7);
    }

    if (job.destination_size < 2u) return;
    for (uint i = job.destination_size - 1u; i-- > 0u;) {
        const uint index = output_base + i * output_step;
        float value = output[index];
        const uint available = min(
            HalfBandwidth, job.destination_size - i - 1u);
        value = apply_upper_window<HalfBandwidth>(
            value, upper_l, job.destination_size, i, available,
            lag1, lag2, lag3, lag4, lag5, lag6, lag7);
        output[index] = value;
        push_window<HalfBandwidth>(
            value, lag1, lag2, lag3, lag4, lag5, lag6, lag7);
    }
}

static inline void inverse_axis_impl(
    device const float *source,
    constant AxisJob &job,
    device const uint *transpose_offsets,
    device const int *transpose_indices,
    device const float *transpose_weights,
    device const float *lower_ld,
    device const float *upper_l,
    device const float *inverse_diagonal,
    device float *output,
    uint global_vector,
    uint fixed_half_bandwidth) {
    const uint batch = global_vector / job.vector_count;
    const uint vector = global_vector - batch * job.vector_count;
    if (batch >= job.batch_count) return;
    if (vector >= job.vector_count) return;

    source += batch * job.input_frame_stride;
    output += batch * job.output_frame_stride;

    const uint half_bandwidth = fixed_half_bandwidth == 0u
        ? job.half_bandwidth : fixed_half_bandwidth;
    const uint output_base = job.direction == horizontal_axis
        ? vector * job.output_stride : vector;
    const uint output_step = job.direction == horizontal_axis
        ? 1u : job.output_stride;

    for (uint i = 0; i < job.destination_size; ++i) {
        float sum = 0.0f;
        const uint begin = transpose_offsets[i];
        const uint end = transpose_offsets[i + 1u];
        for (uint entry = begin; entry < end; ++entry) {
            const uint source_axis = uint(transpose_indices[entry]);
            sum = fma(
                transpose_weights[entry],
                source[image_index(job.direction, vector, source_axis,
                                   job.input_stride, job.reserved)],
                sum);
        }
        const uint available = min(half_bandwidth, i);
        for (uint distance = available; distance >= 1u; --distance) {
            sum = fma(
                -lower_ld[(distance - 1u) * job.destination_size + i],
                output[output_base + (i - distance) * output_step],
                sum);
        }
        output[output_base + i * output_step] =
            sum * inverse_diagonal[i];
    }

    if (job.destination_size < 2u) return;
    for (uint i = job.destination_size - 1u; i-- > 0u;) {
        const uint output_index = output_base + i * output_step;
        float value = output[output_index];
        const uint available = min(
            half_bandwidth, job.destination_size - i - 1u);
        if (half_bandwidth == 3u) {
            for (uint distance = 1u; distance <= available; ++distance) {
                value = fma(
                    -upper_l[(distance - 1u) * job.destination_size + i],
                    output[output_base + (i + distance) * output_step],
                    value);
            }
        } else {
            for (uint distance = available; distance >= 1u; --distance) {
                value = fma(
                    -upper_l[(distance - 1u) * job.destination_size + i],
                    output[output_base + (i + distance) * output_step],
                    value);
            }
        }
        output[output_index] = value;
    }
}

#define DEFINE_INVERSE_AXIS(NAME, HALF_BANDWIDTH) \
kernel void NAME( \
    device const float *source [[buffer(0)]], \
    constant AxisJob &job [[buffer(1)]], \
    device const uint *transpose_offsets [[buffer(2)]], \
    device const int *transpose_indices [[buffer(3)]], \
    device const float *transpose_weights [[buffer(4)]], \
    device const float *lower_ld [[buffer(5)]], \
    device const float *upper_l [[buffer(6)]], \
    device const float *inverse_diagonal [[buffer(7)]], \
    device float *output [[buffer(8)]], \
    uint vector [[thread_position_in_grid]]) { \
    inverse_axis_impl(source, job, transpose_offsets, transpose_indices, \
        transpose_weights, lower_ld, upper_l, inverse_diagonal, output, \
        vector, HALF_BANDWIDTH); \
}

#define DEFINE_INVERSE_AXIS_WIDE(NAME, HALF_BANDWIDTH) \
kernel void NAME( \
    device const float *source [[buffer(0)]], \
    constant AxisJob &job [[buffer(1)]], \
    device const uint *transpose_offsets [[buffer(2)]], \
    device const int *transpose_indices [[buffer(3)]], \
    device const float *transpose_weights [[buffer(4)]], \
    device const float *lower_ld [[buffer(5)]], \
    device const float *upper_l [[buffer(6)]], \
    device const float *inverse_diagonal [[buffer(7)]], \
    device float *output [[buffer(8)]], \
    uint vector [[thread_position_in_grid]]) { \
    inverse_axis_fixed_wide_impl<HALF_BANDWIDTH>(source, job, \
        transpose_offsets, transpose_indices, transpose_weights, lower_ld, \
        upper_l, inverse_diagonal, output, vector); \
}

DEFINE_INVERSE_AXIS(inverse_axis_h1, 1u)
DEFINE_INVERSE_AXIS(inverse_axis_h3, 3u)
DEFINE_INVERSE_AXIS_WIDE(inverse_axis_h5, 5u)
DEFINE_INVERSE_AXIS_WIDE(inverse_axis_h7, 7u)
DEFINE_INVERSE_AXIS(inverse_axis_generic, 0u)

DEFINE_INVERSE_AXIS(inverse_axis_transposed_h1, 1u)
DEFINE_INVERSE_AXIS(inverse_axis_transposed_h3, 3u)
DEFINE_INVERSE_AXIS_WIDE(inverse_axis_transposed_h5, 5u)
DEFINE_INVERSE_AXIS_WIDE(inverse_axis_transposed_h7, 7u)
DEFINE_INVERSE_AXIS(inverse_axis_transposed_generic, 0u)

#define DEFINE_INVERSE_AXIS_BATCH(NAME, HALF_BANDWIDTH) \
kernel void NAME( \
    device const float *source [[buffer(0)]], \
    constant AxisBatchJob *jobs [[buffer(1)]], \
    device const uint *transpose_offsets [[buffer(2)]], \
    device const int *transpose_indices [[buffer(3)]], \
    device const float *transpose_weights [[buffer(4)]], \
    device const float *lower_ld [[buffer(5)]], \
    device const float *upper_l [[buffer(6)]], \
    device const float *inverse_diagonal [[buffer(7)]], \
    device float *output [[buffer(8)]], \
    uint2 position [[thread_position_in_grid]]) { \
    constant AxisBatchJob &batch = jobs[position.y]; \
    inverse_axis_impl(source + batch.input_offset, batch.axis, \
        transpose_offsets + batch.transpose_offsets_offset, \
        transpose_indices + batch.transpose_indices_offset, \
        transpose_weights + batch.transpose_weights_offset, \
        lower_ld + batch.lower_ld_offset, upper_l + batch.upper_l_offset, \
        inverse_diagonal + batch.inverse_diagonal_offset, \
        output + batch.output_offset, position.x, HALF_BANDWIDTH); \
}

#define DEFINE_INVERSE_AXIS_BATCH_WIDE(NAME, HALF_BANDWIDTH) \
kernel void NAME( \
    device const float *source [[buffer(0)]], \
    constant AxisBatchJob *jobs [[buffer(1)]], \
    device const uint *transpose_offsets [[buffer(2)]], \
    device const int *transpose_indices [[buffer(3)]], \
    device const float *transpose_weights [[buffer(4)]], \
    device const float *lower_ld [[buffer(5)]], \
    device const float *upper_l [[buffer(6)]], \
    device const float *inverse_diagonal [[buffer(7)]], \
    device float *output [[buffer(8)]], \
    uint2 position [[thread_position_in_grid]]) { \
    constant AxisBatchJob &batch = jobs[position.y]; \
    inverse_axis_fixed_wide_impl<HALF_BANDWIDTH>( \
        source + batch.input_offset, batch.axis, \
        transpose_offsets + batch.transpose_offsets_offset, \
        transpose_indices + batch.transpose_indices_offset, \
        transpose_weights + batch.transpose_weights_offset, \
        lower_ld + batch.lower_ld_offset, upper_l + batch.upper_l_offset, \
        inverse_diagonal + batch.inverse_diagonal_offset, \
        output + batch.output_offset, position.x); \
}

DEFINE_INVERSE_AXIS_BATCH(inverse_axis_batch_h1, 1u)
DEFINE_INVERSE_AXIS_BATCH(inverse_axis_batch_h3, 3u)
DEFINE_INVERSE_AXIS_BATCH_WIDE(inverse_axis_batch_h5, 5u)
DEFINE_INVERSE_AXIS_BATCH_WIDE(inverse_axis_batch_h7, 7u)
DEFINE_INVERSE_AXIS_BATCH(inverse_axis_batch_generic, 0u)

DEFINE_INVERSE_AXIS_BATCH(inverse_axis_transposed_batch_h1, 1u)
DEFINE_INVERSE_AXIS_BATCH(inverse_axis_transposed_batch_h3, 3u)
DEFINE_INVERSE_AXIS_BATCH_WIDE(inverse_axis_transposed_batch_h5, 5u)
DEFINE_INVERSE_AXIS_BATCH_WIDE(inverse_axis_transposed_batch_h7, 7u)
DEFINE_INVERSE_AXIS_BATCH(inverse_axis_transposed_batch_generic, 0u)

template <typename Sample>
static inline void inverse_axis_integer_input_impl(
    device const Sample *source,
    constant AxisJob &job,
    device const uint *transpose_offsets,
    device const int *transpose_indices,
    device const float *transpose_weights,
    device const float *lower_ld,
    device const float *upper_l,
    device const float *inverse_diagonal,
    device float *output,
    constant IntegerConversion *conversions,
    uint global_vector,
    uint fixed_half_bandwidth) {
    const uint batch = global_vector / job.vector_count;
    const uint vector = global_vector - batch * job.vector_count;
    if (batch >= job.batch_count || vector >= job.vector_count) return;
    constant IntegerConversion &conversion = conversions[batch];

    source += batch * job.input_frame_stride;
    output += batch * job.output_frame_stride;

    const uint half_bandwidth = fixed_half_bandwidth == 0u
        ? job.half_bandwidth : fixed_half_bandwidth;
    const uint output_base = job.direction == horizontal_axis
        ? vector * job.output_stride : vector;
    const uint output_step = job.direction == horizontal_axis
        ? 1u : job.output_stride;

    for (uint i = 0; i < job.destination_size; ++i) {
        float sum = 0.0f;
        const uint begin = transpose_offsets[i];
        const uint end = transpose_offsets[i + 1u];
        for (uint entry = begin; entry < end; ++entry) {
            const uint source_axis = uint(transpose_indices[entry]);
            const float sample = float(source[
                image_index(job.direction, vector, source_axis,
                            job.input_stride, job.reserved)]);
            const float normalized =
                (sample - conversion.input_offset) * conversion.input_scale;
            sum = fma(transpose_weights[entry], normalized, sum);
        }
        const uint available = min(half_bandwidth, i);
        for (uint distance = available; distance >= 1u; --distance) {
            sum = fma(
                -lower_ld[(distance - 1u) * job.destination_size + i],
                output[output_base + (i - distance) * output_step],
                sum);
        }
        output[output_base + i * output_step] =
            sum * inverse_diagonal[i];
    }

    if (job.destination_size < 2u) return;
    for (uint i = job.destination_size - 1u; i-- > 0u;) {
        const uint output_index = output_base + i * output_step;
        float value = output[output_index];
        const uint available = min(
            half_bandwidth, job.destination_size - i - 1u);
        if (half_bandwidth == 3u) {
            for (uint distance = 1u; distance <= available; ++distance) {
                value = fma(
                    -upper_l[(distance - 1u) * job.destination_size + i],
                    output[output_base + (i + distance) * output_step],
                    value);
            }
        } else {
            for (uint distance = available; distance >= 1u; --distance) {
                value = fma(
                    -upper_l[(distance - 1u) * job.destination_size + i],
                    output[output_base + (i + distance) * output_step],
                    value);
            }
        }
        output[output_index] = value;
    }
}

template <typename Sample, uint HalfBandwidth>
static inline void inverse_axis_integer_input_fixed_wide_impl(
    device const Sample *source,
    constant AxisJob &job,
    device const uint *transpose_offsets,
    device const int *transpose_indices,
    device const float *transpose_weights,
    device const float *lower_ld,
    device const float *upper_l,
    device const float *inverse_diagonal,
    device float *output,
    constant IntegerConversion *conversions,
    uint global_vector) {
    const uint batch = global_vector / job.vector_count;
    const uint vector = global_vector - batch * job.vector_count;
    if (batch >= job.batch_count || vector >= job.vector_count) return;
    constant IntegerConversion &conversion = conversions[batch];

    source += batch * job.input_frame_stride;
    output += batch * job.output_frame_stride;

    const uint output_base = job.direction == horizontal_axis
        ? vector * job.output_stride : vector;
    const uint output_step = job.direction == horizontal_axis
        ? 1u : job.output_stride;
    float lag1 = 0.0f;
    float lag2 = 0.0f;
    float lag3 = 0.0f;
    float lag4 = 0.0f;
    float lag5 = 0.0f;
    float lag6 = 0.0f;
    float lag7 = 0.0f;

    for (uint i = 0; i < job.destination_size; ++i) {
        float sum = 0.0f;
        const uint begin = transpose_offsets[i];
        const uint end = transpose_offsets[i + 1u];
        for (uint entry = begin; entry < end; ++entry) {
            const uint source_axis = uint(transpose_indices[entry]);
            const float sample = float(source[
                image_index(job.direction, vector, source_axis,
                            job.input_stride, job.reserved)]);
            const float normalized =
                (sample - conversion.input_offset) * conversion.input_scale;
            sum = fma(transpose_weights[entry], normalized, sum);
        }
        sum = apply_lower_window<HalfBandwidth>(
            sum, lower_ld, job.destination_size, i,
            lag1, lag2, lag3, lag4, lag5, lag6, lag7);
        const float value = sum * inverse_diagonal[i];
        output[output_base + i * output_step] = value;
        push_window<HalfBandwidth>(
            value, lag1, lag2, lag3, lag4, lag5, lag6, lag7);
    }

    if (job.destination_size < 2u) return;
    for (uint i = job.destination_size - 1u; i-- > 0u;) {
        const uint index = output_base + i * output_step;
        float value = output[index];
        const uint available = min(
            HalfBandwidth, job.destination_size - i - 1u);
        value = apply_upper_window<HalfBandwidth>(
            value, upper_l, job.destination_size, i, available,
            lag1, lag2, lag3, lag4, lag5, lag6, lag7);
        output[index] = value;
        push_window<HalfBandwidth>(
            value, lag1, lag2, lag3, lag4, lag5, lag6, lag7);
    }
}

#define DEFINE_INTEGER_INPUT_AXIS(NAME, SAMPLE, HALF_BANDWIDTH) \
kernel void NAME( \
    device const SAMPLE *source [[buffer(0)]], \
    constant AxisJob &job [[buffer(1)]], \
    device const uint *transpose_offsets [[buffer(2)]], \
    device const int *transpose_indices [[buffer(3)]], \
    device const float *transpose_weights [[buffer(4)]], \
    device const float *lower_ld [[buffer(5)]], \
    device const float *upper_l [[buffer(6)]], \
    device const float *inverse_diagonal [[buffer(7)]], \
    device float *output [[buffer(8)]], \
    constant IntegerConversion *conversions [[buffer(9)]], \
    uint vector [[thread_position_in_grid]]) { \
    inverse_axis_integer_input_impl(source, job, transpose_offsets, \
        transpose_indices, transpose_weights, lower_ld, upper_l, \
        inverse_diagonal, output, conversions, vector, HALF_BANDWIDTH); \
}

#define DEFINE_INTEGER_INPUT_AXIS_WIDE(NAME, SAMPLE, HALF_BANDWIDTH) \
kernel void NAME( \
    device const SAMPLE *source [[buffer(0)]], \
    constant AxisJob &job [[buffer(1)]], \
    device const uint *transpose_offsets [[buffer(2)]], \
    device const int *transpose_indices [[buffer(3)]], \
    device const float *transpose_weights [[buffer(4)]], \
    device const float *lower_ld [[buffer(5)]], \
    device const float *upper_l [[buffer(6)]], \
    device const float *inverse_diagonal [[buffer(7)]], \
    device float *output [[buffer(8)]], \
    constant IntegerConversion *conversions [[buffer(9)]], \
    uint vector [[thread_position_in_grid]]) { \
    inverse_axis_integer_input_fixed_wide_impl<SAMPLE, HALF_BANDWIDTH>( \
        source, job, transpose_offsets, transpose_indices, \
        transpose_weights, lower_ld, upper_l, inverse_diagonal, output, \
        conversions, vector); \
}

DEFINE_INTEGER_INPUT_AXIS(inverse_axis_u8_h1, uchar, 1u)
DEFINE_INTEGER_INPUT_AXIS(inverse_axis_u8_h3, uchar, 3u)
DEFINE_INTEGER_INPUT_AXIS_WIDE(inverse_axis_u8_h5, uchar, 5u)
DEFINE_INTEGER_INPUT_AXIS_WIDE(inverse_axis_u8_h7, uchar, 7u)
DEFINE_INTEGER_INPUT_AXIS(inverse_axis_u8_generic, uchar, 0u)
DEFINE_INTEGER_INPUT_AXIS(inverse_axis_u16_h1, ushort, 1u)
DEFINE_INTEGER_INPUT_AXIS(inverse_axis_u16_h3, ushort, 3u)
DEFINE_INTEGER_INPUT_AXIS_WIDE(inverse_axis_u16_h5, ushort, 5u)
DEFINE_INTEGER_INPUT_AXIS_WIDE(inverse_axis_u16_h7, ushort, 7u)
DEFINE_INTEGER_INPUT_AXIS(inverse_axis_u16_generic, ushort, 0u)

template <typename Sample>
static inline void convert_f32_to_integer_impl(
    device const float *source,
    constant ConvertJob &job,
    constant IntegerConversion *conversions,
    device Sample *output,
    uint global_index) {
    const uint pixels_per_frame = job.width * job.height;
    const uint batch = global_index / pixels_per_frame;
    const uint pixel = global_index - batch * pixels_per_frame;
    if (batch >= job.batch_count) return;
    constant IntegerConversion &conversion = conversions[batch];
    const uint row = pixel / job.width;
    const uint column = pixel - row * job.width;
    const uint source_index = batch * job.input_frame_stride
        + row * job.input_stride + column;
    const uint output_index = batch * job.output_frame_stride
        + row * job.output_stride + column;
    const float scaled = clamp(
        fma(source[source_index], conversion.output_scale,
            conversion.output_offset),
        0.0f, float(conversion.output_maximum));
    output[output_index] = Sample(uint(rint(scaled)));
}

kernel void convert_f32_to_u8(
    device const float *source [[buffer(0)]],
    constant ConvertJob &job [[buffer(1)]],
    constant IntegerConversion *conversions [[buffer(2)]],
    device uchar *output [[buffer(3)]],
    uint global_index [[thread_position_in_grid]]) {
    convert_f32_to_integer_impl(
        source, job, conversions, output, global_index);
}

kernel void convert_f32_to_u16(
    device const float *source [[buffer(0)]],
    constant ConvertJob &job [[buffer(1)]],
    constant IntegerConversion *conversions [[buffer(2)]],
    device ushort *output [[buffer(3)]],
    uint global_index [[thread_position_in_grid]]) {
    convert_f32_to_integer_impl(
        source, job, conversions, output, global_index);
}
