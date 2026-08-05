"""VapourSynth source-only graph for decoder ceiling measurements."""

import vapoursynth as vs


core = vs.core


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


core.num_threads = int(threads)
source_decoder = globals().get("source_decoder", "")
source_prefer_hw = int(globals().get("source_prefer_hw", "0") or 0)
source_ff_loglevel = int(globals().get("source_ff_loglevel", "0") or 0)
source_rap_verification = int(
    globals().get("source_rap_verification", "-1") or -1)
source_filter = globals().get("source_filter", "ffms2")
source_plugin = globals().get("source_plugin", "")

source_namespace = {"lsmas": "lsmas", "ffms2": "ffms2",
                    "bestsource": "bs"}[source_filter]
if source_plugin and not hasattr(core, source_namespace):
    core.std.LoadPlugin(path=source_plugin)

source = open_source(source)
source = source.std.Trim(0, int(frames) - 1)
source = core.std.ShufflePlanes(source, 0, vs.GRAY)
source = source.resize.Point(format=vs.GRAYS)
source.set_output(0)
