import math

import vapoursynth as vs


def repeated_arange(start, stop, step):
    values = []
    current = start
    while current < stop:
        values.append(current)
        current += step
    return values


def scaler(kernel, b=0.0, c=0.5, taps=3):
    return {"kernel": kernel, "b": b, "c": c, "taps": taps}


GETFNATIVE_SCALERS = [
    scaler("bilinear"),
    scaler("bicubic", 1 / 3, 1 / 3),
    scaler("bicubic", 0.0, 0.5),
    scaler("bicubic", 0.0, 1.0),
    scaler("bicubic", 1.0, 0.0),
    scaler("bicubic", 0.0, 0.75),
    scaler("lanczos", taps=2),
    scaler("lanczos", taps=3),
    scaler("lanczos", taps=4),
    scaler("spline16"),
    scaler("spline36"),
]
GETFNATIVE_V2_SCALERS = GETFNATIVE_SCALERS[:6] + [
    scaler("spline16"), scaler("spline36")]
SELECTKERNEL_PARAMETERS = repeated_arange(0, 1, 0.1)[:10]
SELECTKERNEL_SCALERS = [scaler("bilinear")] + [
    scaler("bicubic", b, c)
    for b in SELECTKERNEL_PARAMETERS
    for c in SELECTKERNEL_PARAMETERS
]


def descale_args(width, height, src_width, src_height, base_height):
    base_width = round(width / height * base_height)
    output_width = base_width - 2 * int((base_width - src_width) / 2)
    output_height = base_height - 2 * int((base_height - src_height) / 2)
    return {
        "width": output_width,
        "height": output_height,
        "src_left": (output_width - src_width) / 2,
        "src_top": (output_height - src_height) / 2,
        "src_width": src_width,
        "src_height": src_height,
    }


def vertical_descale_args(width, src_height, base_height):
    output_height = base_height - 2 * int((base_height - src_height) / 2)
    return {
        "width": width,
        "height": output_height,
        "src_left": 0.0,
        "src_top": (output_height - src_height) / 2,
        "src_width": float(width),
        "src_height": src_height,
    }


def reconstruct(source, scaler_spec, src_height, vertical_only):
    if vertical_only:
        arguments = vertical_descale_args(source.width, src_height, 1000)
    else:
        src_width = source.width / source.height * src_height
        arguments = descale_args(
            source.width, source.height, src_width, src_height, 1000)

    kwargs = {"kernel": scaler_spec["kernel"]}
    if scaler_spec["kernel"] == "bicubic":
        kwargs.update(b=scaler_spec["b"], c=scaler_spec["c"])
    elif scaler_spec["kernel"] == "lanczos":
        kwargs["taps"] = scaler_spec["taps"]
    if implementation == "new":
        kwargs["backend"] = "cpu"
    descaled = namespace.Descale(source, **arguments, **kwargs)

    resize = getattr(core.resize, scaler_spec["kernel"].capitalize())
    resize_kwargs = {
        "src_left": arguments["src_left"],
        "src_top": arguments["src_top"],
        "src_width": arguments["src_width"],
        "src_height": arguments["src_height"],
    }
    if scaler_spec["kernel"] == "bicubic":
        resize_kwargs["filter_param_a"] = scaler_spec["b"]
        resize_kwargs["filter_param_b"] = scaler_spec["c"]
    elif scaler_spec["kernel"] == "lanczos":
        resize_kwargs["filter_param_a"] = scaler_spec["taps"]
    return resize(descaled, source.width, source.height, **resize_kwargs)


def statistics_segment(source, scaler_spec, heights, vertical_only, ex_thr):
    looped = core.std.Loop(source, times=len(heights))
    reconstructed = core.std.FrameEval(
        looped,
        lambda n: reconstruct(
            looped, scaler_spec, heights[n], vertical_only))
    difference = core.std.Expr(
        [looped, reconstructed],
        [f"x y - abs dup {ex_thr:.17g} > swap 0 ?"])
    return core.std.CropRel(difference, 5, 5, 5, 5).std.PlaneStats()


core = vs.core
core.num_threads = int(threads)
if implementation == "new":
    core.std.LoadPlugin(path=plugin)
elif not hasattr(core, "descale"):
    core.std.LoadPlugin(path=old_plugin)

namespace = core.dsmvc if implementation == "new" else core.descale
if globals().get("synthetic", "0") == "1":
    source = core.std.BlankClip(
        width=1920, height=1080, format=vs.GRAYS, color=[0.5], keep=True)
else:
    source = core.imwri.Read(image).resize.Point(format=vs.GRAYS, matrix_s="709")

if case == "getfnative":
    heights = repeated_arange(700, 980, 0.1)
    segments = [
        statistics_segment(source, item, heights, False, 0.015)
        for item in GETFNATIVE_SCALERS
    ]
    output = core.std.Splice(segments)
elif case == "getfnative_v2":
    heights = repeated_arange(840, 880, 0.1)
    segments = [
        statistics_segment(source, item, heights, True, 0.015)
        for item in GETFNATIVE_V2_SCALERS
    ]
    output = core.std.Splice(segments)
elif case == "selectkernel":
    # Preserve muvsfunc's runtime multi-kernel behavior by selecting the
    # rescaler from the frame index rather than building all kernels eagerly.
    looped = core.std.Loop(source, times=len(SELECTKERNEL_SCALERS))
    reconstructed = core.std.FrameEval(
        looped,
        lambda n: reconstruct(
            looped, SELECTKERNEL_SCALERS[n], 719.8, False))
    difference = core.std.Expr(
        [looped, reconstructed], ["x y - abs dup 0.012 > swap 0 ?"])
    output = core.std.CropRel(difference, 5, 5, 5, 5).std.PlaneStats()
    output = core.std.Loop(
        output, times=math.ceil(int(frames) / output.num_frames))
else:
    raise ValueError(f"unknown benchmark case: {case}")

if output.num_frames < int(frames):
    output = core.std.Loop(
        output, times=math.ceil(int(frames) / output.num_frames))
output.set_output(0)
