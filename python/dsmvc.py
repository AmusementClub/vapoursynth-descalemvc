from enum import IntEnum

from vapoursynth import GRAY, GRAYS, RGB, RGBS, YUV, core


class BorderHandling(IntEnum):
    MIRROR = 0
    ZERO = 1
    REPEAT = 2


class Opt(IntEnum):
    AUTO = 0
    NONE = 1
    AVX2 = 2


def _plugin_args(opt, backend):
    result = {}
    if opt is not None:
        result["opt"] = int(opt)
    if backend is not None:
        result["backend"] = backend
    return result


def Debilinear(src, width, height, border_handling=None, yuv444=False,
               gray=False, chromaloc=None, opt=None, backend=None):
    return Descale(src, width, height, kernel="bilinear",
                   border_handling=border_handling, yuv444=yuv444,
                   gray=gray, chromaloc=chromaloc, opt=opt, backend=backend)


def Debicubic(src, width, height, b=0.0, c=0.5, border_handling=None,
              yuv444=False, gray=False, chromaloc=None, opt=None, backend=None):
    return Descale(src, width, height, kernel="bicubic", b=b, c=c,
                   border_handling=border_handling, yuv444=yuv444,
                   gray=gray, chromaloc=chromaloc, opt=opt, backend=backend)


def Delanczos(src, width, height, taps=3, border_handling=None,
              yuv444=False, gray=False, chromaloc=None, opt=None, backend=None):
    return Descale(src, width, height, kernel="lanczos", taps=taps,
                   border_handling=border_handling, yuv444=yuv444,
                   gray=gray, chromaloc=chromaloc, opt=opt, backend=backend)


def Despline16(src, width, height, border_handling=None, yuv444=False,
               gray=False, chromaloc=None, opt=None, backend=None):
    return Descale(src, width, height, kernel="spline16",
                   border_handling=border_handling, yuv444=yuv444,
                   gray=gray, chromaloc=chromaloc, opt=opt, backend=backend)


def Despline36(src, width, height, border_handling=None, yuv444=False,
               gray=False, chromaloc=None, opt=None, backend=None):
    return Descale(src, width, height, kernel="spline36",
                   border_handling=border_handling, yuv444=yuv444,
                   gray=gray, chromaloc=chromaloc, opt=opt, backend=backend)


def Despline64(src, width, height, border_handling=None, yuv444=False,
               gray=False, chromaloc=None, opt=None, backend=None):
    return Descale(src, width, height, kernel="spline64",
                   border_handling=border_handling, yuv444=yuv444,
                   gray=gray, chromaloc=chromaloc, opt=opt, backend=backend)


def Decustom(src, width, height, custom_kernel, taps,
             border_handling=None, yuv444=False, gray=False,
             chromaloc=None, opt=None, backend=None):
    return Descale(src, width, height, custom_kernel=custom_kernel, taps=taps,
                   border_handling=border_handling, yuv444=yuv444,
                   gray=gray, chromaloc=chromaloc, opt=opt, backend=backend)


def Descale(src, width, height, kernel=None, custom_kernel=None, taps=None,
            b=None, c=None, border_handling=None, yuv444=False, gray=False,
            chromaloc=None, opt=None, backend=None):
    src_f = src.format
    src_cf = src_f.color_family
    src_st = src_f.sample_type
    src_bits = src_f.bits_per_sample
    src_sw = src_f.subsampling_w
    src_sh = src_f.subsampling_h
    call_args = dict(kernel=kernel, taps=taps, b=b, c=c,
                     custom_kernel=custom_kernel,
                     border_handling=border_handling)
    call_args.update(_plugin_args(opt, backend))

    if src_cf == RGB and not gray:
        rgb = core.dsmvc.Descale(to_rgbs(src), width, height, **call_args)
        return rgb.resize.Point(format=src_f.id)

    y = core.dsmvc.Descale(to_grays(src), width, height, **call_args)
    y_f = core.register_format(GRAY, src_st, src_bits, 0, 0)
    y = y.resize.Point(format=y_f.id)

    if src_cf == GRAY or gray:
        return y

    if not yuv444 and ((width % 2 and src_sw) or (height % 2 and src_sh)):
        raise ValueError("Descale: output dimensions and subsampling are incompatible")

    uv_f = core.query_video_format(src_cf, src_st, src_bits,
                                   0 if yuv444 else src_sw,
                                   0 if yuv444 else src_sh)
    uv = src.resize.Spline36(width, height, format=uv_f.id,
                             chromaloc_s=chromaloc)
    return core.std.ShufflePlanes([y, uv], [0, 1, 2], YUV)


def to_grays(src):
    return src.resize.Point(format=GRAYS)


def to_rgbs(src):
    return src.resize.Point(format=RGBS)


def get_plane(src, plane):
    return core.std.ShufflePlanes(src, plane, GRAY)
