"""In-memory BlankClip graph for absolute dsmvc pipeline benchmarks."""

import vapoursynth as vs


KERNELS = {
    "b1": ("Debilinear", {}),
    "b3": ("Debicubic", {"b": 0.0, "c": 0.5}),
    "b5": ("Delanczos", {"taps": 3}),
    "b7": ("Despline64", {}),
}

FORMATS = {
    "float32": (vs.GRAYS, [0.375]),
    "gray8": (vs.GRAY8, [97]),
    "gray16": (vs.GRAY16, [24832]),
    "yuv420p10": (vs.YUV420P10, [400, 470, 550]),
    "rgb24": (vs.RGB24, [63, 127, 211]),
}


def fixed_geometry(width, height, native_height, base_height):
    base_height = int(round(base_height))
    base_width = round(width / height * base_height)
    native_width = width / height * native_height
    output_width = base_width - 2 * int((base_width - native_width) / 2)
    output_height = base_height - 2 * int(
        (base_height - native_height) / 2)
    return {
        "width": output_width,
        "height": int(output_height),
        "src_left": (output_width - native_width) / 2,
        "src_top": (output_height - native_height) / 2,
        "src_width": native_width,
        "src_height": native_height,
    }


core = vs.core
core.num_threads = int(threads)
core.std.LoadPlugin(path=plugin)

format_id, color = FORMATS[format_name]
source = core.std.BlankClip(
    width=int(source_width),
    height=int(source_height),
    length=int(frames),
    format=format_id,
    color=color,
)
arguments = fixed_geometry(
    source.width, source.height, float(native_height), float(base_height))
function, kernel_arguments = KERNELS[kernel]
output = getattr(core.dsmvc, function)(
    source, backend="cpu", **arguments, **kernel_arguments)
output.set_output(0)
