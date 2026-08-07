"""CUDA saturation graph with a reused, low-traffic source frame."""

import vapoursynth as vs


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
frame_count = int(frames)
if frame_count < 1:
    raise ValueError("frames must be positive")

sample_format_name = globals().get("sample_format", "GRAY8")
sample_formats = {
    "GRAY8": vs.GRAY8,
    "GRAY16": vs.GRAY16,
    "GRAYS": vs.GRAYS,
}
if sample_format_name not in sample_formats:
    raise ValueError("unsupported sample_format: " + str(sample_format_name))

if not hasattr(core, "dsmvc"):
    core.std.LoadPlugin(path=plugin)

single_frame = core.std.BlankClip(
    width=int(width),
    height=int(height),
    length=1,
    format=sample_formats[sample_format_name],
    color=0,
)
source = core.std.Loop(single_frame, times=frame_count)
arguments = fixed_geometry(
    source.width, source.height, float(src_height), float(base_height))
arguments.update(taps=int(taps), backend="cuda")
core.dsmvc.Delanczos(source, **arguments).set_output(0)
