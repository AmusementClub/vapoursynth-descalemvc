#!/usr/bin/env python3
"""Run paired end-to-end VSPipe CPU/Metal plugin measurements."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from analyze_metal_routes import paired_bootstrap_speedup
from paired_benchmark_support import (
    ResourceMonitor,
    ResourcePressureError,
    command_text,
    load_frame_properties,
    paired_analysis,
    parse_vspipe_timing,
    sha256_file,
    summarize,
    summarize_frame_properties,
)


KERNELS = (
    "bilinear", "spline16", "bicubic",
    "spline36", "lanczos3", "spline64",
)
SUPPORTED_KERNELS_BY_FORMAT = {
    "grays": ("bilinear", "bicubic", "spline36", "lanczos3", "spline64"),
    "p8": KERNELS,
    "p10": KERNELS,
}


def build_command(options: argparse.Namespace, format_name: str,
                  kernel: str, backend: str, requests: int,
                  properties_path: Path) -> list[str]:
    script = Path(__file__).with_name("vspipe_metal_plugin.vpy").resolve()
    values = {
        "plugin": str(options.plugin),
        "format": format_name,
        "kernel": kernel,
        "backend": backend,
        "frames": str(options.frames),
        "threads": str(requests),
    }
    command = [str(options.vspipe)]
    for key, value in values.items():
        command.extend(["--arg", f"{key}={value}"])
    command.extend([
        "--requests", str(requests),
        "--start", "0",
        "--end", str(options.frames - 1),
        "--json", str(properties_path),
        "--filter-time",
        str(script),
        "--",
    ])
    return command


def run_once(options: argparse.Namespace, format_name: str, kernel: str,
             variant: str, backend: str, requests: int, sample: int,
             warmup: bool, pair_order: str,
             monitor: ResourceMonitor) -> dict:
    sample_label = f"warmup{-sample}" if warmup else f"sample{sample + 1}"
    properties_path = options.telemetry_dir / (
        f"{format_name}-{kernel}-r{requests}-{variant}-{sample_label}.json")
    command = build_command(
        options, format_name, kernel, backend, requests, properties_path)
    before = monitor.capture(
        f"{format_name}/{kernel}/R{requests}/{variant}/{sample_label}/before")
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace", check=False)
    elapsed_seconds = (time.perf_counter_ns() - started) / 1e9
    after = monitor.capture(
        f"{format_name}/{kernel}/R{requests}/{variant}/{sample_label}/after")
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        termination = (
            f"signal {-completed.returncode}" if completed.returncode < 0
            else f"exit {completed.returncode}")
        raise RuntimeError(
            f"VSPipe failed ({termination}) for "
            f"{format_name}/{kernel}/{backend}/R{requests}:\n{output[-8000:]}")
    timing = parse_vspipe_timing(output, options.frames)
    frame_properties = load_frame_properties(properties_path, options.frames)
    telemetry = summarize_frame_properties(frame_properties)
    return {
        "format": format_name,
        "kernel": kernel,
        "variant": variant,
        "backend": backend,
        "requests": requests,
        "sample": sample,
        "observation": None if warmup else sample + 1,
        "pair_order": pair_order,
        "warmup": warmup,
        "frames": options.frames,
        "elapsed_seconds": elapsed_seconds,
        "fps": options.frames / elapsed_seconds,
        "vspipe_seconds": timing["seconds"],
        "vspipe_fps": timing["fps"],
        "process_overhead_seconds": max(0.0, elapsed_seconds - timing["seconds"]),
        "telemetry": telemetry,
        "frame_properties": {
            "path": str(properties_path),
            "sha256": sha256_file(properties_path),
        },
        "system_before": before,
        "system_after": after,
        "command": command_text(command),
        "vspipe_output_tail": output[-4000:],
    }


def aggregate(samples: list[dict], options: argparse.Namespace) -> tuple[
        list[dict], list[dict]]:
    grouped: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[(sample["format"], sample["kernel"], sample["variant"],
                 sample["requests"])].append(sample)
    summaries = []
    for key, items in sorted(grouped.items()):
        summaries.append({
            "format": key[0],
            "kernel": key[1],
            "variant": key[2],
            "requests": key[3],
            "runs": len(items),
            "elapsed_seconds": summarize(
                [item["elapsed_seconds"] for item in items]),
            "vspipe_seconds": summarize(
                [item["vspipe_seconds"] for item in items]),
            "fps": summarize([item["fps"] for item in items]),
            "vspipe_fps": summarize([item["vspipe_fps"] for item in items]),
        })

    by_key = {
        (item["format"], item["kernel"], item["variant"], item["requests"]): item
        for item in summaries
    }
    comparisons = []
    cases = sorted({
        (item["format"], item["kernel"], item["requests"])
        for item in summaries
    })
    for format_name, kernel, requests in cases:
        control = by_key[(format_name, kernel, "control", requests)]
        candidate = by_key[(format_name, kernel, "candidate", requests)]
        case_samples = [
            item for item in samples
            if (item["format"], item["kernel"], item["requests"])
            == (format_name, kernel, requests)
        ]
        sample_numbers = sorted({item["sample"] for item in case_samples})
        control_elapsed = []
        candidate_elapsed = []
        paired_speedups = []
        for number in sample_numbers:
            elapsed = {
                item["variant"]: item["elapsed_seconds"]
                for item in case_samples if item["sample"] == number
            }
            if set(elapsed) != {"control", "candidate"}:
                raise RuntimeError(
                    f"unpaired sample {number} for "
                    f"{format_name}/{kernel}/R{requests}")
            control_elapsed.append(elapsed["control"])
            candidate_elapsed.append(elapsed["candidate"])
            paired_speedups.append(elapsed["control"] / elapsed["candidate"])
        comparison = {
            "format": format_name,
            "kernel": kernel,
            "requests": requests,
            "control_backend": "cpu",
            "candidate_backend": options.candidate_backend,
            "speedup": (
                control["elapsed_seconds"]["median"]
                / candidate["elapsed_seconds"]["median"]),
            "paired_speedup": summarize(paired_speedups),
            "paired_bootstrap": paired_bootstrap_speedup(
                control_elapsed,
                candidate_elapsed,
                resamples=options.bootstrap_resamples,
                seed=options.bootstrap_seed + len(comparisons) * 1009,
            ),
            "control_fps": control["fps"]["median"],
            "candidate_fps": candidate["fps"]["median"],
        }
        if len(sample_numbers) == 6:
            comparison["paired_analysis"] = paired_analysis(case_samples)
        comparisons.append(comparison)
    return summaries, comparisons


def run(options: argparse.Namespace) -> dict:
    monitor = ResourceMonitor(options.minimum_free_percent)
    samples = []
    warmups = []
    for format_name in options.formats:
        supported = SUPPORTED_KERNELS_BY_FORMAT[format_name]
        kernels = [kernel for kernel in options.kernels if kernel in supported]
        skipped = [kernel for kernel in options.kernels if kernel not in supported]
        if skipped:
            print(
                f"skip {format_name}: unsupported Metal kernels "
                f"{', '.join(skipped)}", flush=True)
        for kernel in kernels:
            for requests in options.requests:
                run_numbers = list(range(-options.warmups, 0)) \
                    + list(range(options.samples))
                for sample in run_numbers:
                    warmup = sample < 0
                    if warmup:
                        order = ("control", "candidate")
                    else:
                        order = ("control", "candidate") if sample % 2 == 0 \
                            else ("candidate", "control")
                    pair_order = "C-A" if order[0] == "control" else "A-C"
                    for variant in order:
                        backend = "cpu" if variant == "control" \
                            else options.candidate_backend
                        result = run_once(
                            options, format_name, kernel, variant, backend,
                            requests, sample, warmup, pair_order, monitor)
                        (warmups if warmup else samples).append(result)
                        print(
                            f"{format_name}/{kernel}/R{requests}/{variant}/"
                            f"{sample}: {result['fps']:.3f} wall fps, "
                            f"{result['vspipe_fps']:.3f} filter fps",
                            flush=True)
    summaries, comparisons = aggregate(samples, options)
    manifest = None
    if options.build_manifest:
        manifest = json.loads(options.build_manifest.read_text(encoding="utf-8"))
    return {
        "schema": "dsmvc-plugin-backend-benchmark-v4",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "plugin": str(options.plugin),
            "plugin_sha256": sha256_file(options.plugin),
            "vspipe": str(options.vspipe),
            "frames": options.frames,
            "samples": options.samples,
            "warmups": options.warmups,
            "formats": options.formats,
            "kernels": options.kernels,
            "requests": options.requests,
            "control_backend": "cpu",
            "candidate_backend": options.candidate_backend,
            "minimum_free_percent": options.minimum_free_percent,
            "bootstrap_resamples": options.bootstrap_resamples,
            "bootstrap_seed": options.bootstrap_seed,
            "build_manifest": str(options.build_manifest)
            if options.build_manifest else None,
            "build_manifest_sha256": sha256_file(options.build_manifest)
            if options.build_manifest else None,
            "source_fingerprint": manifest.get("source_fingerprint")
            if isinstance(manifest, dict) else None,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "vpy_sha256": sha256_file(
                Path(__file__).with_name("vspipe_metal_plugin.vpy")),
        },
        "warmup_samples": warmups,
        "samples": samples,
        "summaries": summaries,
        "comparisons": comparisons,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plugin", required=True, type=Path)
    result.add_argument("--vspipe", required=True, type=Path)
    result.add_argument("--json-out", required=True, type=Path)
    result.add_argument("--telemetry-dir", type=Path)
    result.add_argument("--build-manifest", type=Path)
    result.add_argument("--formats", nargs="+", choices=("grays", "p8", "p10"),
                        default=("p8", "p10"))
    result.add_argument(
        "--kernels", nargs="+", choices=KERNELS,
        default=("bilinear", "bicubic", "lanczos3", "spline64"))
    result.add_argument("--requests", nargs="+", type=int,
                        default=(1, 4, 16, 32))
    result.add_argument("--candidate-backend", choices=("metal", "auto"),
                        default="metal")
    result.add_argument("--frames", type=int, default=96)
    result.add_argument("--samples", type=int, default=6)
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument("--minimum-free-percent", type=int, default=10)
    result.add_argument("--bootstrap-resamples", type=int, default=50000)
    result.add_argument(
        "--bootstrap-seed", type=lambda value: int(value, 0),
        default=0x44534D56)
    return result


def main() -> int:
    options = parser().parse_args()
    for name in ("plugin", "vspipe", "json_out"):
        setattr(options, name, getattr(options, name).expanduser().resolve())
    if options.build_manifest:
        options.build_manifest = options.build_manifest.expanduser().resolve()
    options.telemetry_dir = (
        options.telemetry_dir.expanduser().resolve()
        if options.telemetry_dir else
        options.json_out.parent / f"{options.json_out.stem}-frame-properties")
    for path in (options.plugin, options.vspipe):
        if not path.is_file():
            raise FileNotFoundError(path)
    if options.build_manifest and not options.build_manifest.is_file():
        raise FileNotFoundError(options.build_manifest)
    if options.frames < 1 or options.samples < 1 or options.warmups < 0:
        raise ValueError("frames and samples must be positive; warmups nonnegative")
    if any(request < 1 for request in options.requests):
        raise ValueError("requests must be positive")
    if options.bootstrap_resamples < 1000:
        raise ValueError("bootstrap resamples must be at least 1000")
    options.telemetry_dir.mkdir(parents=True, exist_ok=False)
    options.json_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        output = run(options)
    except ResourcePressureError as error:
        print(f"RESOURCE_STOP: {error}", flush=True)
        return 125
    options.json_out.write_text(
        json.dumps(output, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
