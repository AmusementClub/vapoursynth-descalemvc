#!/usr/bin/env python3
"""Compare dsmvc blur output with JET vapoursynth-descale v12."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import vapoursynth as vs


GEOMETRY = {
    "width": 16,
    "height": 64,
    "src_left": 0.0,
    "src_top": 0.0,
    "src_width": 16.0,
    "src_height": 64.0,
    "border_handling": 0,
}
CASES = (
    ("bilinear", "Debilinear", {}),
    ("bicubic", "Debicubic", {"b": 0.0, "c": 0.5}),
    ("lanczos2", "Delanczos", {"taps": 2}),
    ("lanczos3", "Delanczos", {"taps": 3}),
    ("spline16", "Despline16", {}),
    ("spline36", "Despline36", {}),
    ("spline64", "Despline64", {}),
)
BLURS = (0.75, 1.01, 1.25)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def patterned_source(core: vs.Core, format_id: int) -> vs.VideoNode:
    blank = core.std.BlankClip(
        width=96, height=64, length=1, format=format_id)

    def fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        del n
        output = f.copy()
        for plane_index in range(output.format.num_planes):
            plane = np.asarray(output[plane_index])
            rows = np.arange(plane.shape[0], dtype=np.uint32)[:, None]
            columns = np.arange(plane.shape[1], dtype=np.uint32)[None, :]
            values = (columns * 193 + rows * 389 + columns * rows * 17
                      + plane_index * 521)
            if output.format.sample_type == vs.FLOAT:
                np.copyto(
                    plane,
                    (values & 4095).astype(np.float32) / np.float32(4095.0))
            else:
                maximum = (1 << output.format.bits_per_sample) - 1
                np.copyto(plane, values & maximum, casting="unsafe")
        output.props["_Range"] = 1
        return output

    return core.std.ModifyFrame(blank, blank, fill)


def maximum_difference(reference: vs.VideoFrame,
                       candidate: vs.VideoFrame) -> float:
    require(reference.format.id == candidate.format.id,
            "reference and candidate formats differ")
    maximum = 0.0
    for plane_index in range(reference.format.num_planes):
        left = np.asarray(reference[plane_index]).astype(np.float64)
        right = np.asarray(candidate[plane_index]).astype(np.float64)
        require(left.shape == right.shape, "reference and candidate shapes differ")
        maximum = max(maximum, float(np.max(np.abs(left - right))))
    return maximum


def compare_float(core: vs.Core) -> float:
    source = patterned_source(core, vs.GRAYS)
    maximum = 0.0
    for blur in BLURS:
        for label, function_name, kernel_arguments in CASES:
            arguments = dict(GEOMETRY, blur=blur, **kernel_arguments)
            reference = getattr(core.descale, function_name)(
                source, **arguments).get_frame(0)
            candidate = getattr(core.dsmvc, function_name)(
                source, f64mode=1, opt=1,
                backend="cpu", **arguments).get_frame(0)
            difference = maximum_difference(reference, candidate)
            require(difference <= 1.0e-6,
                    f"{label}/blur={blur:g}: float max error {difference}")
            maximum = max(maximum, difference)

        custom_kernel = lambda x: max(1.0 - abs(x), 0.0)
        reference = core.descale.Decustom(
            source, custom_kernel=custom_kernel, taps=1,
            blur=blur, **GEOMETRY).get_frame(0)
        candidate = core.dsmvc.Descale(
            source, custom_kernel=custom_kernel, taps=1,
            blur=blur, f64mode=1, opt=1,
            backend="cpu", **GEOMETRY).get_frame(0)
        difference = maximum_difference(reference, candidate)
        require(difference <= 1.0e-6,
                f"custom/blur={blur:g}: float max error {difference}")
        maximum = max(maximum, difference)
    return maximum


def float_format(core: vs.Core, source: vs.VideoNode) -> int:
    return core.query_video_format(
        source.format.color_family, vs.FLOAT, 32,
        source.format.subsampling_w, source.format.subsampling_h).id


def compare_integer(core: vs.Core) -> float:
    maximum = 0.0
    for format_id in (vs.GRAY16, vs.YUV420P10):
        integer_source = patterned_source(core, format_id)
        float_source = integer_source.resize.Point(
            format=float_format(core, integer_source), dither_type="none",
            range_in_s="full", range_s="full")
        for blur in BLURS:
            for label, function_name, kernel_arguments in CASES:
                arguments = dict(GEOMETRY, blur=blur, **kernel_arguments)
                reference = getattr(core.descale, function_name)(
                    float_source, **arguments).resize.Point(
                        format=format_id, dither_type="none",
                        range_in_s="full", range_s="full").get_frame(0)
                candidate = getattr(core.dsmvc, function_name)(
                    integer_source, f64mode=1, opt=1, backend="cpu",
                    **arguments).get_frame(0)
                difference = maximum_difference(reference, candidate)
                require(difference <= 1.0,
                        f"{integer_source.format.name}/{label}/blur={blur:g}: "
                        f"integer max error {difference}")
                maximum = max(maximum, difference)
    return maximum


def run(options: argparse.Namespace) -> None:
    core = vs.core
    core.num_threads = options.threads
    core.std.LoadPlugin(path=str(options.jet_plugin.resolve()))
    core.std.LoadPlugin(path=str(options.plugin.resolve()))
    float_error = compare_float(core)
    integer_error = compare_integer(core)
    print(
        "dsmvc JET v12 blur conformance passed: "
        f"float_max={float_error} integer_max={integer_error}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plugin", type=Path, required=True)
    result.add_argument("--jet-plugin", type=Path, required=True)
    result.add_argument("--threads", type=int, default=8)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
