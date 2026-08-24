"""Fixed GRAYS BlankClip graph for control/candidate plugin measurements."""

import vapoursynth as vs


FUNCTIONS = {
    "bilinear": ("Debilinear", {}),
    "bicubic_b0_c0_5": ("Debicubic", {"b": 0.0, "c": 0.5}),
    "lanczos2": ("Delanczos", {"taps": 2}),
    "lanczos3": ("Delanczos", {"taps": 3}),
    "lanczos4": ("Delanczos", {"taps": 4}),
    "spline16": ("Despline16", {}),
    "spline36": ("Despline36", {}),
    "spline64": ("Despline64", {}),
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
source = core.std.BlankClip(
    width=1920,
    height=1080,
    length=int(frames),
    format=vs.GRAYS,
    color=0,
)
arguments = fixed_geometry(
    source.width, source.height, float(src_height), float(base_height))
function_name, kernel_arguments = FUNCTIONS[kernel]
arguments.update(kernel_arguments)
arguments["backend"] = str(backend)
opt_value = int(opt or 0)
if opt_value:
    arguments["opt"] = opt_value
output = getattr(core.dsmvc, function_name)(source, **arguments)
output.set_output(0)
