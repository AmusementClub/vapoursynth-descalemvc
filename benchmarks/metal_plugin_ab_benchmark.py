#!/usr/bin/env python3
"""Run alternating plugin-to-plugin VSPipe backend measurements."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from analyze_metal_routes import paired_bootstrap_speedup
from metal_plugin_benchmark import KERNELS, sha256_file, summarize


def build_command(options: argparse.Namespace, plugin: Path,
                  format_name: str, kernel: str, backend: str,
                  requests: int) -> list[str]:
    script = Path(__file__).with_name("vspipe_metal_plugin.vpy").resolve()
    values = {
        "plugin": str(plugin),
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
        "--filter-time",
        str(script),
        "--",
    ])
    return command


def run_once(options: argparse.Namespace, variant: str, plugin: Path,
             backend: str, format_name: str, kernel: str, requests: int,
             sample: int, warmup: bool) -> dict:
    command = build_command(
        options, plugin, format_name, kernel, backend, requests)
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace", check=False)
    elapsed_seconds = (time.perf_counter_ns() - started) / 1e9
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"VSPipe failed for {variant}/{format_name}/{kernel}/R{requests}:\n"
            f"{output[-8000:]}")
    return {
        "variant": variant,
        "plugin": str(plugin),
        "backend": backend,
        "format": format_name,
        "kernel": kernel,
        "requests": requests,
        "sample": sample,
        "warmup": warmup,
        "frames": options.frames,
        "elapsed_seconds": elapsed_seconds,
        "fps": options.frames / elapsed_seconds,
        "command": command,
        "vspipe_output_tail": output[-4000:],
    }


def aggregate(samples: list[dict], *, resamples: int,
              seed: int) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str, int, str], list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[(sample["format"], sample["kernel"], sample["requests"],
                 sample["variant"])].append(sample)

    summaries = []
    for key, items in sorted(grouped.items()):
        summaries.append({
            "format": key[0],
            "kernel": key[1],
            "requests": key[2],
            "variant": key[3],
            "elapsed_seconds": summarize(
                [item["elapsed_seconds"] for item in items]),
            "fps": summarize([item["fps"] for item in items]),
        })

    by_key = {
        (item["format"], item["kernel"], item["requests"],
         item["variant"]): item
        for item in summaries
    }
    comparisons = []
    cases = sorted({
        (item["format"], item["kernel"], item["requests"])
        for item in samples
    })
    for format_name, kernel, requests in cases:
        control = by_key[(format_name, kernel, requests, "control")]
        candidate = by_key[(format_name, kernel, requests, "candidate")]
        case_samples = [
            item for item in samples
            if (item["format"], item["kernel"], item["requests"])
            == (format_name, kernel, requests)
        ]
        paired_speedups = []
        control_elapsed = []
        candidate_elapsed = []
        for sample_number in sorted({item["sample"] for item in case_samples}):
            elapsed = {
                item["variant"]: item["elapsed_seconds"]
                for item in case_samples if item["sample"] == sample_number
            }
            paired_speedups.append(elapsed["control"] / elapsed["candidate"])
            control_elapsed.append(elapsed["control"])
            candidate_elapsed.append(elapsed["candidate"])
        case_seed = seed + len(comparisons) * 1009
        comparisons.append({
            "format": format_name,
            "kernel": kernel,
            "requests": requests,
            "speedup": (
                control["elapsed_seconds"]["median"]
                / candidate["elapsed_seconds"]["median"]),
            "paired_speedup": summarize(paired_speedups),
            "paired_bootstrap": paired_bootstrap_speedup(
                control_elapsed, candidate_elapsed,
                resamples=resamples, seed=case_seed),
            "control_fps": control["fps"]["median"],
            "candidate_fps": candidate["fps"]["median"],
        })
    return summaries, comparisons


def run(options: argparse.Namespace) -> dict:
    variants = {
        "control": (options.control_plugin, options.control_backend),
        "candidate": (options.candidate_plugin, options.candidate_backend),
    }
    samples = []
    for format_name in options.formats:
        for kernel in options.kernels:
            for requests in options.requests:
                for sample in range(-options.warmups, options.samples):
                    warmup = sample < 0
                    order = ("control", "candidate") \
                        if sample % 2 == 0 else ("candidate", "control")
                    for variant in order:
                        plugin, backend = variants[variant]
                        result = run_once(
                            options, variant, plugin, backend, format_name,
                            kernel, requests, sample, warmup)
                        if not warmup:
                            samples.append(result)
                        print(
                            f"{format_name}/{kernel}/R{requests}/{variant}/"
                            f"{sample}: {result['fps']:.3f} fps",
                            flush=True)
    summaries, comparisons = aggregate(
        samples, resamples=options.bootstrap_resamples,
        seed=options.bootstrap_seed)
    return {
        "schema": "dsmvc-plugin-ab-benchmark-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "control_plugin": str(options.control_plugin),
            "control_plugin_sha256": sha256_file(options.control_plugin),
            "control_backend": options.control_backend,
            "candidate_plugin": str(options.candidate_plugin),
            "candidate_plugin_sha256": sha256_file(options.candidate_plugin),
            "candidate_backend": options.candidate_backend,
            "vspipe": str(options.vspipe),
            "frames": options.frames,
            "samples": options.samples,
            "warmups": options.warmups,
            "formats": options.formats,
            "kernels": options.kernels,
            "requests": options.requests,
            "bootstrap_resamples": options.bootstrap_resamples,
            "bootstrap_seed": options.bootstrap_seed,
        },
        "samples": samples,
        "summaries": summaries,
        "comparisons": comparisons,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--control-plugin", required=True, type=Path)
    result.add_argument("--candidate-plugin", required=True, type=Path)
    result.add_argument("--control-backend", default="metal",
                        choices=("cpu", "metal", "auto"))
    result.add_argument("--candidate-backend", default="metal",
                        choices=("cpu", "metal", "auto"))
    result.add_argument("--vspipe", required=True, type=Path)
    result.add_argument("--json-out", required=True, type=Path)
    result.add_argument("--formats", nargs="+", choices=("grays", "p8", "p10"),
                        default=("p8", "p10"))
    result.add_argument("--kernels", nargs="+", choices=KERNELS,
                        default=("spline36", "lanczos3", "spline64"))
    result.add_argument("--requests", nargs="+", type=int,
                        default=(16, 32))
    result.add_argument("--frames", type=int, default=512)
    result.add_argument("--samples", type=int, default=7)
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument("--bootstrap-resamples", type=int, default=50000)
    result.add_argument(
        "--bootstrap-seed", type=lambda value: int(value, 0),
        default=0x44534D56)
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
        raise ValueError("frames and samples must be positive; warmups nonnegative")
    if any(request < 1 for request in options.requests):
        raise ValueError("requests must be positive")
    if options.bootstrap_resamples < 1000:
        raise ValueError("bootstrap resamples must be at least 1000")
    output = run(options)
    options.json_out.parent.mkdir(parents=True, exist_ok=True)
    options.json_out.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
