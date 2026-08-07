#!/usr/bin/env python3
"""Run paired end-to-end VSPipe CPU/Metal plugin measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from analyze_metal_routes import paired_bootstrap_speedup


KERNELS = (
    "bilinear", "spline16", "bicubic",
    "spline36", "lanczos3", "spline64",
)
SUPPORTED_KERNELS_BY_FORMAT = {
    # The float Metal path intentionally omits Spline16; YUV supports it.
    "grays": ("bilinear", "bicubic", "spline36", "lanczos3", "spline64"),
    "p8": KERNELS,
    "p10": KERNELS,
}


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


def build_command(options: argparse.Namespace, format_name: str,
                  kernel: str, backend: str, requests: int) -> list[str]:
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
        "--filter-time",
        str(script),
        "--",
    ])
    return command


def run_once(options: argparse.Namespace, format_name: str, kernel: str,
             backend: str, requests: int, sample: int,
             warmup: bool) -> dict:
    command = build_command(options, format_name, kernel, backend, requests)
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace", check=False)
    elapsed_seconds = (time.perf_counter_ns() - started) / 1e9
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"VSPipe failed for {format_name}/{kernel}/{backend}/R{requests}:\n"
            f"{output[-8000:]}")
    return {
        "format": format_name,
        "kernel": kernel,
        "backend": backend,
        "requests": requests,
        "sample": sample,
        "warmup": warmup,
        "frames": options.frames,
        "elapsed_seconds": elapsed_seconds,
        "fps": options.frames / elapsed_seconds,
        "command": command,
        "vspipe_output_tail": output[-4000:],
    }


def aggregate(samples: list[dict], candidate_backend: str, *,
              bootstrap_resamples: int,
              bootstrap_seed: int) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[(sample["format"], sample["kernel"], sample["backend"],
                 sample["requests"])].append(sample)
    summaries = []
    for key, items in sorted(grouped.items()):
        summaries.append({
            "format": key[0],
            "kernel": key[1],
            "backend": key[2],
            "requests": key[3],
            "elapsed_seconds": summarize(
                [item["elapsed_seconds"] for item in items]),
            "fps": summarize([item["fps"] for item in items]),
        })

    by_key = {
        (item["format"], item["kernel"], item["backend"], item["requests"]): item
        for item in summaries
    }
    comparisons = []
    cases = sorted({
        (item["format"], item["kernel"], item["requests"])
        for item in summaries
    })
    for format_name, kernel, requests in cases:
        cpu = by_key[(format_name, kernel, "cpu", requests)]
        candidate = by_key[
            (format_name, kernel, candidate_backend, requests)]
        case_samples = [
            item for item in samples
            if (item["format"], item["kernel"], item["requests"])
            == (format_name, kernel, requests)
        ]
        sample_numbers = sorted({item["sample"] for item in case_samples})
        paired_speedups = []
        cpu_elapsed = []
        candidate_elapsed = []
        for sample_number in sample_numbers:
            elapsed = {
                item["backend"]: item["elapsed_seconds"]
                for item in case_samples if item["sample"] == sample_number
            }
            if set(elapsed) != {"cpu", candidate_backend}:
                raise RuntimeError(
                    f"unpaired sample {sample_number} for "
                    f"{format_name}/{kernel}/R{requests}")
            cpu_elapsed.append(elapsed["cpu"])
            candidate_elapsed.append(elapsed[candidate_backend])
            paired_speedups.append(
                elapsed["cpu"] / elapsed[candidate_backend])
        comparisons.append({
            "format": format_name,
            "kernel": kernel,
            "requests": requests,
            "candidate_backend": candidate_backend,
            "speedup": (
                cpu["elapsed_seconds"]["median"]
                / candidate["elapsed_seconds"]["median"]),
            "paired_speedup": summarize(paired_speedups),
            "paired_bootstrap": paired_bootstrap_speedup(
                cpu_elapsed,
                candidate_elapsed,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + len(comparisons) * 1009,
            ),
            "cpu_fps": cpu["fps"]["median"],
            "candidate_fps": candidate["fps"]["median"],
        })
    return summaries, comparisons


def run(options: argparse.Namespace) -> dict:
    samples = []
    for format_name in options.formats:
        supported = SUPPORTED_KERNELS_BY_FORMAT[format_name]
        kernels = [kernel for kernel in options.kernels if kernel in supported]
        skipped = [kernel for kernel in options.kernels if kernel not in supported]
        if skipped:
            print(
                f"skip {format_name}: unsupported Metal kernels "
                f"{', '.join(skipped)}",
                flush=True,
            )
        for kernel in kernels:
            for requests in options.requests:
                for sample in range(-options.warmups, options.samples):
                    warmup = sample < 0
                    order = ("cpu", options.candidate_backend) \
                        if sample % 2 == 0 else (
                            options.candidate_backend, "cpu")
                    for backend in order:
                        result = run_once(
                            options, format_name, kernel, backend, requests,
                            sample, warmup)
                        if not warmup:
                            samples.append(result)
                        print(
                            f"{format_name}/{kernel}/R{requests}/{backend}/"
                            f"{sample}: {result['fps']:.3f} fps",
                            flush=True)
    summaries, comparisons = aggregate(
        samples,
        options.candidate_backend,
        bootstrap_resamples=options.bootstrap_resamples,
        bootstrap_seed=options.bootstrap_seed,
    )
    return {
        "schema": "dsmvc-plugin-backend-benchmark-v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "plugin": str(options.plugin),
            "plugin_sha256": sha256_file(options.plugin),
            "vspipe": str(options.vspipe),
            "frames": options.frames,
            "samples": options.samples,
            "warmups": options.warmups,
            "formats": options.formats,
            "kernels": options.kernels,
            "kernels_by_format": {
                format_name: [
                    kernel for kernel in options.kernels
                    if kernel in SUPPORTED_KERNELS_BY_FORMAT[format_name]
                ]
                for format_name in options.formats
            },
            "requests": options.requests,
            "candidate_backend": options.candidate_backend,
            "bootstrap_resamples": options.bootstrap_resamples,
            "bootstrap_seed": options.bootstrap_seed,
        },
        "samples": samples,
        "summaries": summaries,
        "comparisons": comparisons,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--plugin", required=True, type=Path)
    result.add_argument("--vspipe", required=True, type=Path)
    result.add_argument("--json-out", required=True, type=Path)
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
    result.add_argument("--samples", type=int, default=5)
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument("--bootstrap-resamples", type=int, default=50000)
    result.add_argument(
        "--bootstrap-seed", type=lambda value: int(value, 0),
        default=0x44534D56)
    return result


if __name__ == "__main__":
    options = parser().parse_args()
    options.plugin = options.plugin.expanduser().resolve()
    options.vspipe = options.vspipe.expanduser().resolve()
    options.json_out = options.json_out.expanduser().resolve()
    if not options.plugin.is_file():
        raise FileNotFoundError(options.plugin)
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
