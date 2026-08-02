import vapoursynth as vs


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


core = vs.core
core.num_threads = int(threads)
if implementation == "new":
    core.std.LoadPlugin(path=plugin)

rgb = core.imwri.Read(image)
source = rgb.resize.Point(format=vs.GRAYS, matrix_s="709")
source = core.std.Loop(source, times=int(frames))
arguments = descale_args(1920, 1080, 951.5 * 16 / 9, 951.5, 1000)
namespace = core.dsmvc if implementation == "new" else core.descale
if case == "bicubic_0_7_0_6":
    output = namespace.Debicubic(
        source, b=0.7, c=0.6, **arguments,
        **({"backend": "cpu"} if implementation == "new" else {}))
else:
    output = namespace.Debicubic(
        source, b=0.0, c=1.0, **arguments,
        **({"backend": "cpu"} if implementation == "new" else {}))

output.set_output(0)
