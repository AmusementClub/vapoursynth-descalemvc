#!/usr/bin/env python3
"""Render deterministic dsmvc frames and dump active plane samples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import vapoursynth as vs


KERNELS = {
    "b1": ("Debilinear", {}),
    "b3": ("Debicubic", {"b": 0.0, "c": 0.5}),
    "b5": ("Delanczos", {"taps": 3}),
    "b7": ("Despline64", {}),
}

FORMATS = {
    "float32": (vs.GRAYS, None),
    "gray8": (vs.GRAY8, None),
    "gray16": (vs.GRAY16, 1),
    "yuv420p10": (vs.YUV420P10, None),
    "rgb24": (vs.RGB24, None),
}


def patterned_clip(core, format_id: int, range_value: int | None):
    blank = core.std.BlankClip(
        width=96, height=64, length=1, format=format_id)

    def fill(n, f):
        del n
        output = f.copy()
        maximum = (1 << output.format.bits_per_sample) - 1
        for plane_index in range(output.format.num_planes):
            plane = output[plane_index]
            for y in range(plane.shape[0]):
                for x in range(plane.shape[1]):
                    code = x * 37 + y * 73 + plane_index * 109 + (x * y) % 97
                    if output.format.sample_type == vs.FLOAT:
                        plane[y, x] = (code % 1200) / 1000.0 - 0.1
                    else:
                        plane[y, x] = code % (maximum + 1)
        if range_value is not None:
            output.props["_Range"] = range_value
        return output

    return core.std.ModifyFrame(blank, blank, fill)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--formats", nargs="+", choices=tuple(FORMATS),
                        default=list(FORMATS))
    parser.add_argument("--kernels", nargs="+", choices=tuple(KERNELS),
                        default=list(KERNELS))
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)

    core = vs.core
    core.num_threads = 1
    core.std.LoadPlugin(path=str(options.plugin.expanduser().resolve()))
    cases = []
    for format_name in options.formats:
        format_id, range_value = FORMATS[format_name]
        source = patterned_clip(core, format_id, range_value)
        for kernel in options.kernels:
            function, kernel_arguments = KERNELS[kernel]
            output = getattr(core.dsmvc, function)(
                source,
                width=80,
                height=48,
                src_left=0.125,
                src_top=0.25,
                src_width=79.75,
                src_height=47.5,
                backend="cpu",
                **kernel_arguments,
            )
            frame = output.get_frame(0)
            planes = []
            for plane_index in range(frame.format.num_planes):
                payload = frame[plane_index].tobytes()
                filename = f"{format_name}-{kernel}-p{plane_index}.bin"
                (options.output / filename).write_bytes(payload)
                planes.append({
                    "plane": plane_index,
                    "file": filename,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "width": frame[plane_index].shape[1],
                    "height": frame[plane_index].shape[0],
                })
            cases.append({
                "format": format_name,
                "kernel": kernel,
                "sample_type": "float" if frame.format.sample_type == vs.FLOAT
                               else "integer",
                "bits_per_sample": frame.format.bits_per_sample,
                "range": dict(frame.props).get("_Range"),
                "planes": planes,
            })
    manifest = {"plugin": str(options.plugin.resolve()), "cases": cases}
    (options.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
