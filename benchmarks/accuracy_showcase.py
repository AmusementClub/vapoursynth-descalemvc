#!/usr/bin/env python3
"""Render the release benchmark's ill-conditioned-plan accuracy showcase."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np


FRAME = 12493
SOURCE_SHA256 = "864d552f8e2ead057ebd2c202c7580442a5f22c8acecd08167eb8a07110d1bf4"
BASE_GEOMETRY = {
    "width": 1920,
    "height": 980,
    "src_width": 1920.0,
    "src_height": 978.1,
}
CASES = (
    {
        "name": "lanczos2-catastrophic",
        "function": "Delanczos",
        "kernel": {"taps": 2},
        "geometry": {**BASE_GEOMETRY, "src_top": 0.0},
    },
    {
        "name": "lanczos2-fractional",
        "function": "Delanczos",
        "kernel": {"taps": 2},
        "geometry": {**BASE_GEOMETRY, "src_top": 0.95, "force_h": 1},
    },
    {
        "name": "bicubic-c060-fractional",
        "function": "Debicubic",
        "kernel": {"b": 0.0, "c": 0.60},
        "geometry": {**BASE_GEOMETRY, "src_top": 0.95, "force_h": 1},
    },
    {
        "name": "bicubic-c075-fractional",
        "function": "Debicubic",
        "kernel": {"b": 0.0, "c": 0.75},
        "geometry": {**BASE_GEOMETRY, "src_top": 0.95, "force_h": 1},
    },
    {
        "name": "bicubic-c100-fractional",
        "function": "Debicubic",
        "kernel": {"b": 0.0, "c": 1.0},
        "geometry": {**BASE_GEOMETRY, "src_top": 0.95, "force_h": 1},
    },
    {
        "name": "spline64-near-unity-control",
        "function": "Despline64",
        "kernel": {},
        "geometry": {**BASE_GEOMETRY, "src_top": 0.95, "force_h": 1},
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, str | int]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def frame_array(clip) -> np.ndarray:
    return np.asarray(clip.get_frame(0)[0]).copy()


def worker(options: argparse.Namespace) -> int:
    import vapoursynth as vs

    core = vs.core
    core.num_threads = 1
    core.std.LoadPlugin(path=str(options.source_plugin.resolve()))
    core.std.LoadPlugin(path=str(options.reference_plugin.resolve()))
    core.std.LoadPlugin(path=str(options.new_plugin.resolve()))
    source = core.ffms2.Source(source=str(options.source.resolve()))[FRAME]
    source = core.std.ShufflePlanes(source, 0, vs.GRAY)
    source = source.resize.Point(format=vs.GRAYS)

    arrays: dict[str, np.ndarray] = {}
    for case in CASES:
        arguments = case["geometry"] | case["kernel"]
        function = case["function"]
        name = case["name"]
        arrays[f"{name}.reference"] = frame_array(
            getattr(core.descale, function)(source, **arguments))
        for precision, f64mode in (("f32", 1), ("f64", 2), ("auto", 0)):
            arrays[f"{name}.{precision}"] = frame_array(
                getattr(core.dsmvc, function)(
                    source, **arguments, backend="cpu", f64mode=f64mode))
    options.worker_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(options.worker_output, **arrays)
    return 0


def difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(left.astype(np.float64) - right.astype(np.float64))


def metrics(delta: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(delta, dtype=np.float64)),
        "max_abs": float(np.max(delta)),
    }


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def write_diff_png(path: Path, delta: np.ndarray) -> None:
    peak = float(np.max(delta))
    normalized = np.zeros(delta.shape, dtype=np.float64)
    if peak > 0.0:
        normalized = np.clip(delta / peak, 0.0, 1.0)
    red = np.clip(normalized * 3.0, 0.0, 1.0)
    green = np.clip(normalized * 3.0 - 1.0, 0.0, 1.0)
    blue = np.clip(normalized * 3.0 - 2.0, 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=2)
    pixels = np.rint(rgb * 255.0).astype(np.uint8)
    height, width, _ = pixels.shape
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def polyline(values: np.ndarray, left: float, top: float,
             width: float, height: float, maximum: float) -> str:
    floor = max(maximum * 1.0e-12, 1.0e-15)
    log_min = math.log10(floor)
    log_max = math.log10(max(maximum, floor * 10.0))
    points = []
    denominator = max(len(values) - 1, 1)
    for index, value in enumerate(values):
        x = left + width * index / denominator
        y_value = math.log10(max(float(value), floor))
        y = top + height * (log_max - y_value) / (log_max - log_min)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def write_curves_svg(path: Path, case_name: str,
                     curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    width, height = 1200, 470
    colors = {"old": "#dc2626", "jet": "#2563eb", "dsmvc-f32": "#059669"}
    maximum = max(float(np.max(values)) for pair in curves.values() for values in pair)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1200" height="470" fill="#ffffff"/>',
        f'<text x="600" y="30" text-anchor="middle" font-family="sans-serif" font-size="21" font-weight="700" fill="#111827">{html.escape(case_name)} absolute error</text>',
        '<text x="600" y="52" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#475569">Mean absolute error by output row and column; logarithmic scale</text>',
    ]
    for index, (label, color) in enumerate(colors.items()):
        x = 390 + index * 160
        parts.extend((
            f'<line x1="{x}" y1="72" x2="{x + 24}" y2="72" stroke="{color}" stroke-width="2.5"/>',
            f'<text x="{x + 31}" y="76" font-family="sans-serif" font-size="12" fill="#374151">{label}</text>',
        ))
    for panel, title in enumerate(("Rows", "Columns")):
        left = 62 + panel * 590
        top, plot_width, plot_height = 105, 520, 310
        parts.extend((
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#f8fafc" stroke="#cbd5e1"/>',
            f'<text x="{left + plot_width / 2:.1f}" y="{top + 22}" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="700" fill="#111827">{title}</text>',
        ))
        curve_top = top + 34
        curve_height = plot_height - 48
        for label, pair in curves.items():
            values = pair[panel]
            points = polyline(values, left + 8, curve_top, plot_width - 16,
                              curve_height, maximum)
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{colors[label]}" stroke-width="1.7"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def validate_paths(options: argparse.Namespace) -> None:
    for path in (options.source, options.source_plugin, options.old_plugin,
                 options.jet_plugin, options.new_plugin):
        if path is None or not path.expanduser().resolve().is_file():
            raise FileNotFoundError(path)
    actual_sha = sha256_file(options.source.expanduser().resolve())
    if actual_sha != SOURCE_SHA256:
        raise RuntimeError(
            f"source SHA-256 mismatch: expected {SOURCE_SHA256}, got {actual_sha}")


def main(options: argparse.Namespace) -> int:
    if options.worker:
        return worker(options)
    if options.output is None:
        raise ValueError("--output is required")
    validate_paths(options)
    output = options.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    top_level_command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    commands = [" ".join(json.dumps(item) for item in top_level_command)]
    worker_files = {}
    for label, plugin in (("old", options.old_plugin), ("jet", options.jet_plugin)):
        worker_output = output / f".{label}-showcase-arrays.npz"
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--source", str(options.source.resolve()),
            "--source-plugin", str(options.source_plugin.resolve()),
            "--reference-plugin", str(plugin.resolve()),
            "--new-plugin", str(options.new_plugin.resolve()),
            "--worker-output", str(worker_output),
        ]
        commands.append(" ".join(json.dumps(item) for item in command))
        subprocess.run(command, check=True)
        worker_files[label] = worker_output
    (output / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")

    loaded = {label: np.load(path) for label, path in worker_files.items()}
    rows = []
    case_results = []
    try:
        for case in CASES:
            name = case["name"]
            f64 = loaded["old"][f"{name}.f64"]
            for key in ("f32", "f64", "auto"):
                if not np.array_equal(loaded["old"][f"{name}.{key}"],
                                      loaded["jet"][f"{name}.{key}"]):
                    raise RuntimeError(f"dsmvc {key} control drifted between workers: {name}")
            implementations = {
                "old": loaded["old"][f"{name}.reference"],
                "jet": loaded["jet"][f"{name}.reference"],
                "dsmvc-f32": loaded["old"][f"{name}.f32"],
                "dsmvc-auto": loaded["old"][f"{name}.auto"],
            }
            case_metrics = {label: metrics(difference(array, f64))
                            for label, array in implementations.items()}
            auto_routed_f64 = np.array_equal(implementations["dsmvc-auto"], f64)
            for metric_name in ("mae", "max_abs"):
                rows.append({
                    "case": name,
                    "metric": metric_name,
                    **{label: values[metric_name]
                       for label, values in case_metrics.items()},
                    "auto_routed_f64": auto_routed_f64,
                })
            curves = {}
            for label in ("old", "jet", "dsmvc-f32"):
                delta = difference(implementations[label], f64)
                curves[label] = (np.mean(delta, axis=1), np.mean(delta, axis=0))
            write_curves_svg(output / f"{name}.curves.svg", name, curves)
            for label in ("old", "jet"):
                write_diff_png(
                    output / f"{name}.{label}.diff.png",
                    difference(implementations[label], f64),
                )
            case_results.append({
                "name": name,
                "function": case["function"],
                "kernel": case["kernel"],
                "geometry": case["geometry"],
                "metrics": case_metrics,
                "auto_routed_f64": auto_routed_f64,
            })
    finally:
        for archive in loaded.values():
            archive.close()
        for path in worker_files.values():
            path.unlink(missing_ok=True)

    result = {
        "schema_version": 1,
        "source": str(options.source.resolve()),
        "source_sha256": SOURCE_SHA256,
        "source_plugin": file_record(options.source_plugin),
        "old_plugin": file_record(options.old_plugin),
        "jet_plugin": file_record(options.jet_plugin),
        "new_plugin": file_record(options.new_plugin),
        "runner": file_record(Path(__file__)),
        "command": top_level_command,
        "frame": FRAME,
        "reference": "dsmvc forced-F64 CPU ordered semantics",
        "cases": case_results,
        "rows": rows,
    }
    (output / "benchmark.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    lines = [
        "# Ill-conditioned plan accuracy showcase",
        "",
        "| Case | Metric | old | JET | dsmvc F32 | dsmvc auto | auto routed F64 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['case']}` | {row['metric']} | {row['old']:.6g} | "
            f"{row['jet']:.6g} | {row['dsmvc-f32']:.6g} | "
            f"{row['dsmvc-auto']:.6g} | {row['auto_routed_f64']} |")
    (output / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(case_results), "output": str(output)}))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source", required=True, type=Path)
    result.add_argument("--source-plugin", type=Path, default=Path(
        "/opt/homebrew/opt/ffms2/lib/python3.14/site-packages/"
        "vapoursynth/plugins/libffms2.dylib"))
    result.add_argument("--old-plugin", type=Path)
    result.add_argument("--jet-plugin", type=Path)
    result.add_argument("--new-plugin", required=True, type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--reference-plugin", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return result


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
