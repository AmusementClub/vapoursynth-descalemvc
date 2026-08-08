#!/usr/bin/env python3
"""Run alternating control/candidate measurements on a GRAYS BlankClip."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from analyze_metal_routes import paired_bootstrap_speedup


KERNELS = (
    "bilinear",
    "bicubic_b0_c0_5",
    "lanczos2",
    "lanczos3",
    "lanczos4",
    "spline16",
    "spline36",
    "spline64",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    median = statistics.median(ordered)
    return {
        "minimum": ordered[0],
        "median": median,
        "maximum": ordered[-1],
        "mad": statistics.median(abs(value - median) for value in ordered),
    }


def command_text(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([item]) for item in command)


def build_command(options: argparse.Namespace, plugin: Path, backend: str,
                  opt: int, kernel: str, requests: int) -> list[str]:
    script = Path(__file__).with_name("vspipe_blank_plugin_ab.vpy").resolve()
    values = {
        "plugin": str(plugin),
        "backend": backend,
        "opt": str(opt),
        "kernel": kernel,
        "frames": str(options.frames),
        "threads": str(requests),
        "src_height": str(options.src_height),
        "base_height": str(options.base_height),
    }
    command = [str(options.vspipe)]
    for key, value in values.items():
        command.extend(["--arg", f"{key}={value}"])
    command.extend([
        "--requests", str(requests),
        "--start", "0",
        "--end", str(options.frames - 1),
        "--filter-time", str(script),
        "--",
    ])
    return command


def process_environment(options: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    if options.memory_concurrency is not None:
        environment["DSMVC_MEMORY_CONCURRENCY"] = str(
            options.memory_concurrency)
    return environment


def run_once(options: argparse.Namespace, variant: str, plugin: Path,
             backend: str, opt: int, kernel: str, requests: int,
             sample: int, warmup: bool) -> dict:
    command = build_command(options, plugin, backend, opt, kernel, requests)
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command,
        env=process_environment(options),
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    elapsed_seconds = (time.perf_counter_ns() - started) / 1e9
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"VSPipe failed for {variant}/{kernel}/R{requests}/sample-{sample}:\n"
            f"{output[-8000:]}")
    return {
        "variant": variant,
        "plugin": str(plugin),
        "backend": backend,
        "opt": opt,
        "kernel": kernel,
        "requests": requests,
        "sample": sample,
        "warmup": warmup,
        "frames": options.frames,
        "elapsed_seconds": elapsed_seconds,
        "fps": options.frames / elapsed_seconds,
        "command": command_text(command),
        "vspipe_output_tail": output[-2000:],
    }


def aggregate(samples: list[dict], options: argparse.Namespace) -> tuple[
        list[dict], list[dict]]:
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[(sample["kernel"], sample["requests"], sample["variant"])]\
            .append(sample)

    summaries = []
    for key, items in sorted(grouped.items()):
        summaries.append({
            "kernel": key[0],
            "requests": key[1],
            "variant": key[2],
            "elapsed_seconds": summarize(
                [item["elapsed_seconds"] for item in items]),
            "fps": summarize([item["fps"] for item in items]),
        })

    by_key = {
        (item["kernel"], item["requests"], item["variant"]): item
        for item in summaries
    }
    comparisons = []
    cases = sorted({(item["kernel"], item["requests"]) for item in samples})
    for kernel, requests in cases:
        control = by_key[(kernel, requests, "control")]
        candidate = by_key[(kernel, requests, "candidate")]
        case_samples = [
            item for item in samples
            if (item["kernel"], item["requests"]) == (kernel, requests)
        ]
        control_elapsed = []
        candidate_elapsed = []
        paired_speedups = []
        for sample_number in sorted({item["sample"] for item in case_samples}):
            elapsed = {
                item["variant"]: item["elapsed_seconds"]
                for item in case_samples if item["sample"] == sample_number
            }
            control_elapsed.append(elapsed["control"])
            candidate_elapsed.append(elapsed["candidate"])
            paired_speedups.append(elapsed["control"] / elapsed["candidate"])
        comparisons.append({
            "kernel": kernel,
            "requests": requests,
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
        })
    return summaries, comparisons


def run(options: argparse.Namespace) -> dict:
    variants = {
        "control": (options.control_plugin, options.control_backend,
                     options.control_opt),
        "candidate": (options.candidate_plugin, options.candidate_backend,
                       options.candidate_opt),
    }
    samples = []
    for kernel in options.kernels:
        for requests in options.requests:
            for sample in range(-options.warmups, options.samples):
                order = ("control", "candidate") if sample % 2 == 0 else (
                    "candidate", "control")
                for variant in order:
                    plugin, backend, opt = variants[variant]
                    result = run_once(
                        options, variant, plugin, backend, opt, kernel,
                        requests, sample, sample < 0)
                    if sample >= 0:
                        samples.append(result)
                    print(
                        f"{kernel}/R{requests}/{variant}/{sample}: "
                        f"{result['fps']:.3f} fps", flush=True)
    summaries, comparisons = aggregate(samples, options)
    return {
        "schema": "dsmvc-grays-plugin-ab-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "vspipe": str(options.vspipe),
            "frames": options.frames,
            "src_height": options.src_height,
            "base_height": options.base_height,
            "samples": options.samples,
            "warmups": options.warmups,
            "kernels": options.kernels,
            "requests": options.requests,
            "memory_concurrency": os.environ.get("DSMVC_MEMORY_CONCURRENCY")
            if options.memory_concurrency is None
            else str(options.memory_concurrency),
            "metal_float_cpu_frames": os.environ.get(
                "DSMVC_METAL_FLOAT_CPU_FRAMES"),
            "metal_float_batch_size": os.environ.get(
                "DSMVC_METAL_FLOAT_BATCH_SIZE"),
            "control_plugin": str(options.control_plugin),
            "control_plugin_sha256": sha256_file(options.control_plugin),
            "control_backend": options.control_backend,
            "control_opt": options.control_opt,
            "candidate_plugin": str(options.candidate_plugin),
            "candidate_plugin_sha256": sha256_file(options.candidate_plugin),
            "candidate_backend": options.candidate_backend,
            "candidate_opt": options.candidate_opt,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "vpy_sha256": sha256_file(
                Path(__file__).with_name("vspipe_blank_plugin_ab.vpy")),
            "input": {
                "type": "VapourSynth std.BlankClip",
                "width": 1920,
                "height": 1080,
                "format": "GRAYS",
                "color": 0,
            },
        },
        "summaries": summaries,
        "comparisons": comparisons,
        "samples": samples,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--control-plugin", required=True, type=Path)
    result.add_argument("--candidate-plugin", required=True, type=Path)
    result.add_argument("--control-backend", choices=("cpu", "auto", "metal"),
                        default="cpu")
    result.add_argument("--candidate-backend", choices=("cpu", "auto", "metal"),
                        default="cpu")
    result.add_argument("--control-opt", choices=(0, 1, 2), type=int, default=0)
    result.add_argument("--candidate-opt", choices=(0, 1, 2), type=int, default=0)
    result.add_argument("--vspipe", required=True, type=Path)
    result.add_argument("--json-out", required=True, type=Path)
    result.add_argument("--kernels", nargs="+", choices=KERNELS,
                        default=("bilinear", "spline36", "spline64"))
    result.add_argument("--requests", nargs="+", type=int,
                        default=(8, 16, 32))
    result.add_argument("--frames", type=int, default=512)
    result.add_argument("--src-height", type=float, default=810.0)
    result.add_argument("--base-height", type=float, default=1000.0)
    result.add_argument("--samples", type=int, default=21)
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument("--bootstrap-resamples", type=int, default=50000)
    result.add_argument("--bootstrap-seed", type=lambda value: int(value, 0),
                        default=0x44534D56)
    result.add_argument("--memory-concurrency", type=int, default=None,
                        help="Set DSMVC_MEMORY_CONCURRENCY for every sample.")
    return result


if __name__ == "__main__":
    options = parser().parse_args()
    for name in ("control_plugin", "candidate_plugin", "vspipe", "json_out"):
        setattr(options, name, getattr(options, name).expanduser().resolve())
    for plugin in (options.control_plugin, options.candidate_plugin):
        if not plugin.is_file():
            raise FileNotFoundError(plugin)
    if not options.vspipe.is_file():
        raise FileNotFoundError(options.vspipe)
    if options.frames < 1 or options.samples < 1 or options.warmups < 0:
        raise ValueError("frames/samples must be positive; warmups nonnegative")
    if any(request < 1 for request in options.requests):
        raise ValueError("requests must be positive")
    if options.memory_concurrency is not None and options.memory_concurrency < 0:
        raise ValueError("memory concurrency cannot be negative")
    if options.bootstrap_resamples < 1000:
        raise ValueError("bootstrap resamples must be at least 1000")
    output = run(options)
    options.json_out.parent.mkdir(parents=True, exist_ok=True)
    options.json_out.write_text(
        json.dumps(output, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8")
