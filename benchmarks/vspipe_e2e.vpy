"""VapourSynth graph for the real-video descale E2E benchmark.

The graph deliberately mirrors the three training scripts.  The Python runner
owns process isolation and reporting; this file only builds the graph that
VSPipe measures.
"""

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
STRATIFIED_GETFNATIVE_SCALERS = [
    GETFNATIVE_SCALERS[index] for index in (0, 2, 7, 10)
]
SELECTKERNEL_PARAMETERS = repeated_arange(0, 1, 0.1)[:10]
SELECTKERNEL_SCALERS = [scaler("bilinear")] + [
    scaler("bicubic", b, c)
    for b in SELECTKERNEL_PARAMETERS
    for c in SELECTKERNEL_PARAMETERS
]
MINIMUM_AUTOMATIC_SOURCE_FRAMES = 64

SMOKE_HEIGHTS = {
    "getfnative": [700.0, 719.8, 840.0, 900.0, 951.4, 951.5, 951.6, 979.9],
    "getfnative_v2": [840.0, 859.9, 860.0, 860.1, 879.9],
}

FUNCTIONS = {
    "bilinear": "Debilinear",
    "bicubic": "Debicubic",
    "lanczos": "Delanczos",
    "spline16": "Despline16",
    "spline36": "Despline36",
}


def candidates(case, profile):
    repetitions = 1
    if case == "getfnative":
        if profile == "full":
            heights = repeated_arange(700.0, 980.0, 0.1)
            scalers = GETFNATIVE_SCALERS
        elif profile == "stratified32x4":
            full_heights = [value / 10.0 for value in range(7000, 9800)]
            heights = [
                full_heights[(2 * index + 1) * len(full_heights) // 16]
                for index in range(8)
            ]
            scalers = STRATIFIED_GETFNATIVE_SCALERS
            repetitions = 4
        elif profile == "stratified256":
            full_heights = [value / 10.0 for value in range(7000, 9800)]
            heights = [
                full_heights[(2 * index + 1) * len(full_heights) // 128]
                for index in range(64)
            ]
            scalers = STRATIFIED_GETFNATIVE_SCALERS
        else:
            heights = SMOKE_HEIGHTS[case]
            scalers = GETFNATIVE_SCALERS
    elif case == "getfnative_v2":
        if profile in ("stratified32x4", "stratified256"):
            raise ValueError(
                "stratified profiles are defined only for getfnative")
        heights = (repeated_arange(840.0, 880.0, 0.1)
                   if profile == "full" else SMOKE_HEIGHTS[case])
        scalers = GETFNATIVE_V2_SCALERS
    elif case == "selectkernel":
        if profile in ("stratified32x4", "stratified256"):
            raise ValueError(
                "stratified profiles are defined only for getfnative")
        heights = [719.8]
        scalers = SELECTKERNEL_SCALERS
    else:
        raise ValueError("unknown benchmark case: " + str(case))
    return [(item, height)
            for _ in range(repetitions)
            for item in scalers for height in heights]


def selected_candidates(case, profile, start, end):
    all_items = candidates(case, profile)
    if start < 0 or end < start or end > len(all_items):
        raise ValueError(
            f"candidate range [{start}, {end}) is outside 0..{len(all_items)}")
    return all_items[start:end]


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


def build_geometry(source, height, vertical_only):
    if vertical_only:
        return vertical_descale_args(source.width, height, 1000)
    return descale_args(
        source.width, source.height,
        source.width / source.height * height, height, 1000)


def build_descale(source, scaler_spec, height, vertical_only):
    arguments = build_geometry(source, height, vertical_only)
    namespace = core.dsmvc if implementation == "new" else core.descale
    kwargs = dict(arguments)
    if scaler_spec["kernel"] == "bicubic":
        kwargs.update(b=scaler_spec["b"], c=scaler_spec["c"])
    elif scaler_spec["kernel"] == "lanczos":
        kwargs["taps"] = scaler_spec["taps"]
    if implementation == "new":
        kwargs["backend"] = backend
        if opt:
            kwargs["opt"] = opt
    return getattr(namespace, FUNCTIONS[scaler_spec["kernel"]])(
        source, **kwargs), arguments


def resize_with_scaler(clip, scaler_spec, arguments, width, height):
    resize = getattr(core.resize, scaler_spec["kernel"].capitalize())
    kwargs = {
        "src_left": arguments["src_left"],
        "src_top": arguments["src_top"],
        "src_width": arguments["src_width"],
        "src_height": arguments["src_height"],
    }
    if scaler_spec["kernel"] == "bicubic":
        kwargs["filter_param_a"] = scaler_spec["b"]
        kwargs["filter_param_b"] = scaler_spec["c"]
    elif scaler_spec["kernel"] == "lanczos":
        kwargs["filter_param_a"] = scaler_spec["taps"]
    return resize(clip, width, height, **kwargs)


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


def reconstruct(source, scaler_spec, height, vertical_only):
    descaled, arguments = build_descale(
        source, scaler_spec, height, vertical_only)
    return resize_with_scaler(
        descaled, scaler_spec, arguments, source.width, source.height)


def statistics_segment(source, scaler_spec, heights, vertical_only, ex_thr):
    shared_source = core.std.Loop(
        source, times=max(MINIMUM_AUTOMATIC_SOURCE_FRAMES, len(heights)))
    if shared_source.num_frames < MINIMUM_AUTOMATIC_SOURCE_FRAMES:
        raise RuntimeError(
            "GetFnative dsmvc source is shorter than automatic Metal admission")
    driver = core.std.Trim(shared_source, 0, len(heights) - 1)
    reconstructed = core.std.FrameEval(
        driver,
        lambda n: reconstruct(
            shared_source, scaler_spec, heights[n], vertical_only))
    difference = core.std.Expr(
        [reconstructed, driver],
        [f"x y - abs dup {ex_thr:.17g} > swap 0 ?"])
    return core.std.CropRel(difference, 5, 5, 5, 5).std.PlaneStats()


core = vs.core
core.num_threads = int(threads)
backend = globals().get("backend", "cpu")
opt = int(globals().get("opt", "0") or 0)
cache_mb = int(globals().get("cache_mb", "512") or 512)
if cache_mb < 16:
    raise ValueError("cache_mb must be at least 16")
core.max_cache_size = cache_mb
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

src8 = open_source(source)
source_frame = src8[int(frame)]
source_frame = core.std.ShufflePlanes(source_frame, 0, vs.GRAY)
source_frame = source_frame.resize.Point(format=vs.GRAYS)

candidate_start = int(globals().get("candidate_start", "0") or 0)
candidate_end = int(globals().get(
    "candidate_end", str(len(candidates(case, profile)))
) or 0)
selected = selected_candidates(case, profile, candidate_start, candidate_end)


def grouped_segments(items, vertical_only, ex_thr):
    segments = []
    current_scaler = None
    current_heights = []
    for scaler_spec, height in items:
        if current_scaler is not None and scaler_spec != current_scaler:
            segments.append(statistics_segment(
                source_frame, current_scaler, current_heights,
                vertical_only, ex_thr))
            current_heights = []
        current_scaler = scaler_spec
        current_heights.append(height)
    if current_scaler is not None:
        segments.append(statistics_segment(
            source_frame, current_scaler, current_heights,
            vertical_only, ex_thr))
    return segments

if case == "getfnative":
    segments = grouped_segments(selected, False, 0.015)
    output = core.std.Splice(segments)
elif case == "getfnative_v2":
    segments = grouped_segments(selected, True, 0.015)
    output = core.std.Splice(segments)
elif case == "selectkernel":
    selected_scalers = [item[0] for item in selected]
    looped = core.std.Loop(source_frame, times=len(selected_scalers))
    reconstructed = core.std.FrameEval(
        looped,
        lambda n: reconstruct(
            looped, selected_scalers[n], 719.8, False))
    difference = core.std.Expr(
        [reconstructed, looped], ["x y - abs dup 0.012 > swap 0 ?"])
    output = core.std.CropRel(difference, 5, 5, 5, 5).std.PlaneStats()
else:
    raise ValueError("unknown benchmark case: " + str(case))

output.set_output(0)
