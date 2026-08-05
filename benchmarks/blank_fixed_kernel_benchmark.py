#!/usr/bin/env python3
"""Compare old and current descale on an in-memory blank clip."""

from __future__ import annotations

import argparse
import csv
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


KERNELS = {
    "bilinear": "bilinear",
    "bicubic_b0_c0_5": "bicubic (0, 0.5)",
    "lanczos2": "lanczos2",
    "lanczos3": "lanczos3",
    "lanczos4": "lanczos4",
    "lanczos5": "lanczos5",
    "lanczos6": "lanczos6",
    "spline16": "spline16",
    "spline36": "spline36",
    "spline64": "spline64",
}
DESCALE_FILTERS = {
    "bilinear": "Debilinear",
    "bicubic_b0_c0_5": "Debicubic",
    "lanczos2": "Delanczos",
    "lanczos3": "Delanczos",
    "lanczos4": "Delanczos",
    "lanczos5": "Delanczos",
    "lanczos6": "Delanczos",
    "spline16": "Despline16",
    "spline36": "Despline36",
    "spline64": "Despline64",
}
DEFAULT_THREADS = (1, 8, 16, 32)
IMPLEMENTATIONS = ("old", "new")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_info(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "size": resolved.stat().st_size if resolved.is_file() else None,
        "sha256": sha256_file(resolved) if resolved.is_file() else None,
    }


def command_text(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([item]) for item in command)


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    median = statistics.median(ordered)
    deviations = [abs(value - median) for value in ordered]
    return {
        "median": median,
        "mad": statistics.median(deviations),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def parse_filter_times(stderr: str) -> dict[str, dict[str, float]]:
    result = {}
    table_started = False
    for line in stderr.splitlines():
        if line.strip().startswith("Filtername"):
            table_started = True
            continue
        if not table_started:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            percentage = float(parts[-2])
            seconds = float(parts[-1])
        except ValueError:
            continue
        result[" ".join(parts[:-3])] = {
            "mode": parts[-3],
            "percent": percentage,
            "seconds": seconds,
        }
    return result


def build_command(options, kernel: str, implementation: str,
                  threads: int) -> list[str]:
    script = Path(__file__).with_name("vspipe_blank_fixed_kernel.vpy").resolve()
    values = {
        "implementation": implementation,
        "kernel": kernel,
        "plugin": str(options.new_plugin),
        "old_plugin": str(options.old_plugin),
        "frames": str(options.frames),
        "threads": str(threads),
        "src_height": str(options.src_height),
        "base_height": str(options.base_height),
    }
    command = [str(options.vspipe)]
    for key, value in values.items():
        command.extend(["--arg", f"{key}={value}"])
    command.extend([
        "--requests", str(threads),
        "--start", "0",
        "--end", str(options.frames - 1),
        "--filter-time", str(script),
        "--",
    ])
    return command


def run_sample(options, kernel: str, implementation: str,
               threads: int, run: int) -> dict:
    command = build_command(options, kernel, implementation, threads)
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace",
        check=False)
    elapsed_ns = time.perf_counter_ns() - started
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"VSPipe failed for {kernel}/{implementation}/R{threads}T{threads}/"
            f"run-{run}:\n{output[-8000:]}")
    elapsed_seconds = elapsed_ns / 1e9
    return {
        "kernel": kernel,
        "label": KERNELS[kernel],
        "implementation": implementation,
        "run": run,
        "frames": options.frames,
        "threads": threads,
        "requests": threads,
        "elapsed_seconds": elapsed_seconds,
        "fps": options.frames / elapsed_seconds,
        "filter_time": parse_filter_times(output),
        "command": command_text(command),
        "vspipe_output_tail": output[-4000:],
    }


def summarize_cases(samples: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for sample in samples:
        grouped[(sample["kernel"], sample["implementation"],
                 sample["threads"])].append(sample)
    result = []
    for key, items in sorted(grouped.items()):
        result.append({
            "kernel": key[0],
            "label": items[0]["label"],
            "implementation": key[1],
            "threads": key[2],
            "runs": len(items),
            "elapsed_seconds": summarize([
                item["elapsed_seconds"] for item in items]),
            "fps": summarize([item["fps"] for item in items]),
            "filter_time": {
                name: {
                    metric: statistics.median([
                        item["filter_time"].get(name, {}).get(metric, 0.0)
                        for item in items])
                    for metric in ("percent", "seconds")
                }
                for name in sorted({
                    name for item in items for name in item["filter_time"]
                })
            },
        })
    return result


def filter_value(case: dict, *names: str) -> dict:
    for name in names:
        if name in case["filter_time"]:
            return case["filter_time"][name]
    return {"seconds": 0.0, "percent": 0.0}


def write_csv(samples: list[dict], path: Path) -> None:
    fields = [
        "kernel", "implementation", "threads", "run", "frames",
        "elapsed_seconds", "fps",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({field: sample[field] for field in fields})


def write_markdown(result: dict, path: Path) -> None:
    cases = result["cases"]
    by_key = {(case["kernel"], case["implementation"], case["threads"]): case
              for case in cases}
    threads = result["environment"]["threads"]
    throughput_header = ["Kernel"]
    throughput_separator = ["---"]
    for thread in threads:
        throughput_header.extend([
            f"R{thread} old", f"R{thread} new", f"R{thread} new/old",
        ])
        throughput_separator.extend(["---:", "---:", "---:"])
    lines = [
        "# Blank fixed-kernel benchmark",
        "",
        "This compares old descale and current dsmvc without a decoder or "
        f"source clip. Each run processes {result['environment']['frames']:,} "
        "frames from an in-memory "
        "1920x1080 GRAYS `std.BlankClip` at fixed 810p geometry.",
        "",
        "## Throughput",
        "",
        "| " + " | ".join(throughput_header) + " |",
        "| " + " | ".join(throughput_separator) + " |",
    ]
    for kernel, label in KERNELS.items():
        row = [f"`{label}`"]
        for thread in threads:
            old = by_key[(kernel, "old", thread)]["fps"]["median"]
            new = by_key[(kernel, "new", thread)]["fps"]["median"]
            row.extend([f"{old:.2f}", f"{new:.2f}", f"{new / old:.3f}x"])
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## Filter Time",
        "",
        "There is no LSMASH, decoder, or Point conversion in this graph. "
        "The remaining filter time is the blank producer and descale node; "
        "R8 percentages are accumulated across worker threads.",
        "",
        "| Kernel | Impl | Threads | BlankClip s / % | dsmvc s / % |",
        "|---|---|---:|---:|---:|",
    ])
    for case in cases:
        descale_name = DESCALE_FILTERS[case["kernel"]]
        blank = filter_value(case, "BlankClip")
        descale = filter_value(case, descale_name)
        lines.append(
            f"| `{case['label']}` | `{case['implementation']}` | "
            f"R{case['threads']}T{case['threads']} | "
            f"{blank['seconds']:.2f} / {blank['percent']:.1f}% | "
            f"{descale['seconds']:.2f} / {descale['percent']:.1f}% |")

    lines.extend([
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(result["environment"], indent=2, ensure_ascii=True),
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-plugin", required=True, type=Path)
    parser.add_argument("--new-plugin", required=True, type=Path)
    parser.add_argument("--vspipe", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=root / "benchmark-results" /
                        "blank-fixed-kernel-digimon-810p")
    parser.add_argument("--frames", type=int, default=8000)
    parser.add_argument("--src-height", type=float, default=810.0)
    parser.add_argument("--base-height", type=float, default=1000.0)
    parser.add_argument("--threads", nargs="*", type=int,
                        default=list(DEFAULT_THREADS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--kernels", nargs="*", choices=tuple(KERNELS),
                        default=list(KERNELS))
    options = parser.parse_args()
    options.old_plugin = options.old_plugin.expanduser().resolve()
    options.new_plugin = options.new_plugin.expanduser().resolve()
    options.vspipe = options.vspipe.expanduser().resolve()
    options.output = options.output.expanduser().resolve()
    if options.frames < 1 or options.runs < 1:
        raise ValueError("--frames and --runs must be positive")
    if not options.threads or any(value < 1 for value in options.threads):
        raise ValueError("thread counts must be positive")
    if len(set(options.threads)) != len(options.threads):
        raise ValueError("thread counts must be unique")
    for required in (options.old_plugin, options.new_plugin, options.vspipe):
        if not required.is_file():
            raise FileNotFoundError(required)

    script = Path(__file__).with_name("vspipe_blank_fixed_kernel.vpy")
    samples = []
    options.output.mkdir(parents=True, exist_ok=True)
    for threads in options.threads:
        for kernel in options.kernels:
            for run in range(1, options.runs + 1):
                for implementation in IMPLEMENTATIONS:
                    sample = run_sample(options, kernel, implementation,
                                        threads, run)
                    samples.append(sample)
                    print(
                        f"{kernel} {implementation} R{threads}T{threads} "
                        f"run {run}: {sample['fps']:.3f} fps", flush=True)

    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "vspipe": str(options.vspipe),
        "input": {
            "type": "VapourSynth std.BlankClip",
            "width": 1920,
            "height": 1080,
            "format": "GRAYS",
            "color": 0,
        },
        "old_plugin": file_info(options.old_plugin),
        "new_plugin": file_info(options.new_plugin),
        "vpy": file_info(script),
        "frames": options.frames,
        "src_height": options.src_height,
        "base_height": options.base_height,
        "threads": options.threads,
        "runs": options.runs,
        "kernels": options.kernels,
        "implementations": list(IMPLEMENTATIONS),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    result = {
        "schema_version": 1,
        "environment": environment,
        "cases": summarize_cases(samples),
        "raw_samples": samples,
    }
    (options.output / "benchmark.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8")
    write_csv(samples, options.output / "benchmark.csv")
    write_markdown(result, options.output / "benchmark.md")
    (options.output / "commands.txt").write_text(
        "\n".join(sample["command"] for sample in samples) + "\n",
        encoding="utf-8")
    print(options.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
