"""Fixed GRAYS/YUV420 graph for experimental plugin-level Metal measurements."""

import vapoursynth as vs


GEOMETRY = {
    "width": 1692,
    "height": 952,
    "src_left": 0.2222222222221717,
    "src_top": 0.25,
    "src_width": 1691.5555555555557,
    "src_height": 951.5,
}

FILTERS = {
    "bilinear": ("Debilinear", {}),
    "spline16": ("Despline16", {}),
    "bicubic": ("Debicubic", {"b": 0.0, "c": 0.5}),
    "spline36": ("Despline36", {}),
    "lanczos3": ("Delanczos", {"taps": 3}),
    "spline64": ("Despline64", {}),
}


core = vs.core
core.num_threads = int(threads)
core.std.LoadPlugin(path=plugin)

frame_count = int(frames)
if format == "grays":
    source = core.std.BlankClip(
        width=1920, height=1080, length=frame_count,
        format=vs.GRAYS, color=0.25)
else:
    format_id = {
        "p8": vs.YUV420P8,
        "p10": vs.YUV420P10,
    }[format]
    depth_scale = 1 if format == "p8" else 4
    source_frames = (frame_count + 1) // 2
    limited = core.std.BlankClip(
        width=1920, height=1080, length=source_frames, format=format_id,
        color=[64 * depth_scale, 96 * depth_scale, 160 * depth_scale])
    limited = core.std.SetFrameProps(limited, _Range=0)
    full = core.std.BlankClip(
        width=1920, height=1080, length=source_frames, format=format_id,
        color=[160 * depth_scale, 80 * depth_scale, 192 * depth_scale])
    full = core.std.SetFrameProps(full, _Range=1)
    source = core.std.Interleave([limited, full])[:frame_count]

function_name, arguments = FILTERS[kernel]
output = getattr(core.dsmvc, function_name)(
    source, backend=backend, **GEOMETRY, **arguments)
output.set_output(0)
