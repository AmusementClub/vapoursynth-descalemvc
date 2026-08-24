#!/usr/bin/env python3
"""Numerical gate for two scalar descale references and dsmvc on ARM."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


KERNELS = {
    "bilinear": ("Debilinear", {}),
    "bicubic_b0_c0_5": ("Debicubic", {"b": 0.0, "c": 0.5}),
    "lanczos2": ("Delanczos", {"taps": 2}),
    "lanczos3": ("Delanczos", {"taps": 3}),
    "lanczos4": ("Delanczos", {"taps": 4}),
    "spline16": ("Despline16", {}),
    "spline36": ("Despline36", {}),
    "spline64": ("Despline64", {}),
}
SOURCE_SHA256 = "864d552f8e2ead057ebd2c202c7580442a5f22c8acecd08167eb8a07110d1bf4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_array(clip) -> np.ndarray:
    frame = clip.get_frame(0)
    return np.asarray(frame[0]).copy()


def worker(options: argparse.Namespace) -> int:
    import vapoursynth as vs

    core = vs.core
    core.num_threads = 1
    core.std.LoadPlugin(path=str(options.source_plugin.resolve()))
    core.std.LoadPlugin(path=str(options.reference_plugin.resolve()))
    core.std.LoadPlugin(path=str(options.new_plugin.resolve()))
    source = core.ffms2.Source(source=str(options.source.resolve()))[12493]
    source = core.std.ShufflePlanes(source, 0, vs.GRAY)
    source = source.resize.Point(format=vs.GRAYS)
    geometry = {
        "width": 1440,
        "height": 810,
        "src_width": 1440.0,
        "src_height": 810.0,
    }
    arrays: dict[str, np.ndarray] = {}
    for name, (function, kernel_args) in KERNELS.items():
        args = geometry | kernel_args
        arrays[f"{name}.reference"] = frame_array(
            getattr(core.descale, function)(source, **args))
        arrays[f"{name}.dsmvc_f32"] = frame_array(
            getattr(core.dsmvc, function)(
                source, **args, backend="cpu", f64mode=1))
        arrays[f"{name}.dsmvc_f64"] = frame_array(
            getattr(core.dsmvc, function)(
                source, **args, backend="cpu", f64mode=2))
    options.worker_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(options.worker_output, **arrays)
    return 0


def metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {
        "max_abs": float(np.max(difference)),
        "mae": float(np.mean(difference, dtype=np.float64)),
    }


def main(options: argparse.Namespace) -> int:
    if options.worker:
        return worker(options)

    required = (
        options.source,
        options.source_plugin,
        options.old_plugin,
        options.jet_plugin,
        options.new_plugin,
    )
    for path in required:
        if path is None or not path.expanduser().resolve().is_file():
            raise FileNotFoundError(path)
    source_sha256 = sha256_file(options.source.expanduser().resolve())
    if source_sha256 != SOURCE_SHA256:
        raise RuntimeError(
            f"source SHA-256 mismatch: expected {SOURCE_SHA256}, "
            f"got {source_sha256}")
    output = options.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    worker_files = {}
    for label, plugin in (("old", options.old_plugin),
                          ("jet", options.jet_plugin)):
        worker_output = output / f"{label}-arrays.npz"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--source", str(options.source.resolve()),
            "--source-plugin", str(options.source_plugin.resolve()),
            "--reference-plugin", str(plugin.resolve()),
            "--new-plugin", str(options.new_plugin.resolve()),
            "--worker-output", str(worker_output),
        ]
        subprocess.run(command, check=True)
        worker_files[label] = worker_output

    old = np.load(worker_files["old"])
    jet = np.load(worker_files["jet"])
    rows = []
    passed = True
    for kernel in KERNELS:
        f64_key = f"{kernel}.dsmvc_f64"
        if not np.array_equal(old[f64_key], jet[f64_key]):
            raise RuntimeError(f"dsmvc F64 control drifted between workers: {kernel}")
        comparisons = {
            "old_vs_f64": metrics(old[f"{kernel}.reference"], old[f64_key]),
            "jet_vs_f64": metrics(jet[f"{kernel}.reference"], old[f64_key]),
            "dsmvc_f32_vs_f64": metrics(old[f"{kernel}.dsmvc_f32"], old[f64_key]),
            "old_vs_jet": metrics(
                old[f"{kernel}.reference"], jet[f"{kernel}.reference"]),
        }
        healthy = max(item["max_abs"] for item in comparisons.values()) <= 1.0e-4
        passed = passed and healthy
        rows.append({"kernel": kernel, "healthy": healthy, **comparisons})

    result = {
        "schema_version": 1,
        "source": str(options.source.resolve()),
        "source_sha256": source_sha256,
        "frame": 12493,
        "geometry": "1920x1080 GRAYS -> 1440x810",
        "threshold": 1.0e-4,
        "passed": passed,
        "rows": rows,
    }
    (output / "three-way-diff.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ARM three-way numerical gate",
        "",
        "| Kernel | old vs F64 max | JET vs F64 max | dsmvc F32 vs F64 max | old vs JET max | Healthy |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['kernel']}` | {row['old_vs_f64']['max_abs']:.9g} | "
            f"{row['jet_vs_f64']['max_abs']:.9g} | "
            f"{row['dsmvc_f32_vs_f64']['max_abs']:.9g} | "
            f"{row['old_vs_jet']['max_abs']:.9g} | {row['healthy']} |")
    (output / "three-way-diff.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "kernels": len(rows)}))
    return 0 if passed else 2


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
