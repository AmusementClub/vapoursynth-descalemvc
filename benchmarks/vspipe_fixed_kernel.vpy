"""Fixed-kernel Digimon benchmark graph.

The graph evaluates one descale kernel over the first ``frames`` source
frames.  The runner supplies old/new plugin paths and the thread count as
VSPipe arguments.
"""

import vapoursynth as vs


FUNCTIONS = {
    "bilinear": "Debilinear",
    "bicubic_b0_c0_5": "Debicubic",
    "lanczos2": "Delanczos",
    "lanczos3": "Delanczos",
    "lanczos4": "Delanczos",
    "lanczos5": "Delanczos",
    "lanczos6": "Delanczos",
    "spline16": "Despline16",
    "spline36": "Despline36",
    "spline64": "Despline64",
}


def open_source(path):
    if source_filter == "lsmas":
        kwargs = {}
        if source_decoder:
            kwargs["decoder"] = source_decoder
        if source_prefer_hw:
            kwargs["prefer_hw"] = int(source_prefer_hw)
        if source_ff_loglevel:
            kwargs["ff_loglevel"] = int(source_ff_loglevel)
        if source_rap_verification >= 0:
            kwargs["rap_verification"] = int(source_rap_verification)
        return core.lsmas.LWLibavSource(path, **kwargs)
    if source_filter == "ffms2":
        return core.ffms2.Source(path)
    if source_filter == "bestsource":
        return core.bs.VideoSource(path)
    raise ValueError("unknown source filter: " + str(source_filter))


def fixed_geometry(width, height, native_height, base_height):
    base_height = int(round(base_height))
    base_width = round(width / height * base_height)
    native_width = width / height * native_height
    output_width = base_width - 2 * int((base_width - native_width) / 2)
    output_height = base_height - 2 * int((base_height - native_height) / 2)
    return {
        "width": output_width,
        "height": int(output_height),
        "src_left": (output_width - native_width) / 2,
        "src_top": (output_height - native_height) / 2,
        "src_width": native_width,
        "src_height": native_height,
    }


def descale(source, kernel_name, arguments):
    namespace = core.dsmvc if implementation == "new" else core.descale
    kwargs = dict(arguments)
    if kernel_name == "bicubic_b0_c0_5":
        kwargs.update(b=0.0, c=0.5)
    elif kernel_name.startswith("lanczos"):
        kwargs["taps"] = int(kernel_name.removeprefix("lanczos"))
    if implementation == "new":
        kwargs["backend"] = "cpu"
    return getattr(namespace, FUNCTIONS[kernel_name])(
        source, **kwargs)


core = vs.core
core.num_threads = int(threads)
source_decoder = globals().get("source_decoder", "")
source_prefer_hw = int(globals().get("source_prefer_hw", "0") or 0)
source_ff_loglevel = int(globals().get("source_ff_loglevel", "0") or 0)
source_rap_verification = int(
    globals().get("source_rap_verification", "-1") or -1)

source_namespace = {"lsmas": "lsmas", "ffms2": "ffms2",
                    "bestsource": "bs"}[source_filter]
if source_plugin and not hasattr(core, source_namespace):
    core.std.LoadPlugin(path=source_plugin)
if implementation == "new":
    core.std.LoadPlugin(path=plugin)
elif not hasattr(core, "descale"):
    core.std.LoadPlugin(path=old_plugin)

source = open_source(source)
source = source.std.Trim(0, int(frames) - 1)
source = core.std.ShufflePlanes(source, 0, vs.GRAY)
source = source.resize.Point(format=vs.GRAYS)
arguments = fixed_geometry(
    source.width, source.height, float(src_height), float(base_height))
output = descale(source, kernel, arguments)
output.set_output(0)
