"""Fixed-kernel graph backed by an in-memory VapourSynth blank clip."""

import vapoursynth as vs


FUNCTIONS = {
    "bilinear": "Debilinear",
    "bicubic_b0_c0_5": "Debicubic",
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

if implementation == "new":
    if not hasattr(core, "dsmvc"):
        core.std.LoadPlugin(path=plugin)
    namespace = core.dsmvc
else:
    if not hasattr(core, "descale"):
        core.std.LoadPlugin(path=old_plugin)
    namespace = core.descale

source = core.std.BlankClip(
    width=1920,
    height=1080,
    length=int(frames),
    format=vs.GRAYS,
    color=0,
)
arguments = fixed_geometry(
    source.width, source.height, float(src_height), float(base_height))
if kernel == "bicubic_b0_c0_5":
    arguments.update(b=0.0, c=0.5)
output = getattr(namespace, FUNCTIONS[kernel])(source, **arguments)
output.set_output(0)
