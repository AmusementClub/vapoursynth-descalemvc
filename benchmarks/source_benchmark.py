#!/usr/bin/env python3
"""Measure the VapourSynth source and pre-filter decoder ceiling."""

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
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_THREADS = (1, 8, 16, 32)


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
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return " ".join(subprocess.list2cmdline([item]) for item in command)


def version(command: list[str]) -> str:
    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace", check=False)
    return (completed.stdout + completed.stderr).strip()


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


def run_sample(options, threads: int, run: int) -> dict:
    script = Path(__file__).with_name("vspipe_source_only.vpy").resolve()
    source = Path(options.source).expanduser().resolve()
    args = {
        "source": str(source),
        "source_plugin": (str(Path(options.source_plugin).expanduser().resolve())
                          if options.source_plugin else ""),
        "source_filter": options.source_filter,
        "source_decoder": options.source_decoder,
        "source_prefer_hw": str(options.source_prefer_hw),
        "source_ff_loglevel": str(options.source_ff_loglevel),
        "source_rap_verification": str(options.source_rap_verification),
        "frames": str(options.frames),
        "threads": str(threads),
    }
    command = [options.vspipe]
    for key, value in args.items():
        command.extend(["--arg", f"{key}={value}"])
    command.extend([
        "--requests", str(threads),
        "--start", "0",
        "--end", str(options.frames - 1),
        "--filter-time", str(script),
        "--",
    ])
    start_ns = time.perf_counter_ns()
    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace", check=False)
    elapsed_ns = time.perf_counter_ns() - start_ns
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"VSPipe failed for {options.source_filter}/R{threads}T{threads}/"
            f"run-{run}:\n{output[-8000:]}")
    elapsed_seconds = elapsed_ns / 1e9
    return {
        "source_filter": options.source_filter,
        "source_decoder": options.source_decoder,
        "source_prefer_hw": options.source_prefer_hw,
        "source_ff_loglevel": options.source_ff_loglevel,
        "source_rap_verification": options.source_rap_verification,
        "run": run,
        "frames": options.frames,
        "requests": threads,
        "threads": threads,
        "elapsed_ns": elapsed_ns,
        "elapsed_seconds": elapsed_seconds,
        "fps": options.frames / elapsed_seconds,
        "command": command_text(command),
        "vspipe_output_tail": output[-4000:],
    }


def write_csv(samples: list[dict], path: Path) -> None:
    fields = [
        "source_filter", "source_decoder", "source_prefer_hw",
        "source_ff_loglevel", "source_rap_verification", "run", "frames",
        "requests", "threads", "elapsed_seconds", "fps",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({field: sample[field] for field in fields})


def summarize_cases(samples: list[dict]) -> list[dict]:
    result = []
    for threads in sorted({sample["threads"] for sample in samples}):
        selected = [sample for sample in samples
                    if sample["threads"] == threads]
        result.append({
            "source_filter": selected[0]["source_filter"],
            "threads": threads,
            "requests": threads,
            "frames": selected[0]["frames"],
            "runs": len(selected),
            "elapsed_seconds": summarize([
                sample["elapsed_seconds"] for sample in selected]),
            "fps": summarize([sample["fps"] for sample in selected]),
        })
    return result


def write_markdown(result: dict, path: Path) -> None:
    environment = result["environment"]
    lines = [
        "# Source-only benchmark",
        "",
        "This measures source decode, luma extraction, and Point conversion "
        "without dsmvc.",
        "",
        f"- Source filter: `{environment['source_filter']}`",
        f"- Frames: `{environment['frames']}`",
        f"- Runs: `{environment['runs']}`",
        "",
        "| Config | Median FPS | Median wall (s) | MAD FPS |",
        "|---:|---:|---:|---:|",
    ]
    for case in result["cases"]:
        lines.append(
            f"| R{case['threads']}T{case['threads']} | "
            f"{case['fps']['median']:.3f} | "
            f"{case['elapsed_seconds']['median']:.4f} | "
            f"{case['fps']['mad']:.3f} |")
    lines.extend(["", "## Environment", "", "```json",
                  json.dumps(environment, indent=2, ensure_ascii=True),
                  "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Measure the source-only VapourSynth decoder ceiling.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--vspipe", default="vspipe")
    parser.add_argument("--source-plugin")
    parser.add_argument("--source-filter", choices=("lsmas", "ffms2", "bestsource"),
                        default="ffms2")
    parser.add_argument("--source-decoder", default="")
    parser.add_argument("--source-prefer-hw", type=int, default=0)
    parser.add_argument("--source-ff-loglevel", type=int, default=0)
    parser.add_argument("--source-rap-verification", type=int, default=-1)
    parser.add_argument("--output", default=str(
        root / "benchmark-results" / "source-ceiling-digimon"))
    parser.add_argument("--frames", type=int, default=4000)
    parser.add_argument("--threads", nargs="*", type=int,
                        default=list(DEFAULT_THREADS))
    parser.add_argument("--runs", type=int, default=3)
    options = parser.parse_args()

    source = Path(options.source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source does not exist: {source}")
    if options.source_plugin and not Path(options.source_plugin).is_file():
        raise FileNotFoundError(
            f"source plugin does not exist: {options.source_plugin}")
    if options.frames < 1 or options.runs < 1:
        raise ValueError("--frames and --runs must be positive")
    if not options.threads or any(value < 1 for value in options.threads):
        raise ValueError("thread counts must be positive")

    output = Path(options.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    samples = []
    for threads in options.threads:
        for run in range(1, options.runs + 1):
            sample = run_sample(options, threads, run)
            samples.append(sample)
            print(
                f"{options.source_filter} source-only R{threads}T{threads} "
                f"run {run}: {sample['fps']:.3f} fps", flush=True)

    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "vspipe": version([options.vspipe, "--version"]),
        "source_filter": options.source_filter,
        "source_decoder": options.source_decoder,
        "source_prefer_hw": options.source_prefer_hw,
        "source_ff_loglevel": options.source_ff_loglevel,
        "source_rap_verification": options.source_rap_verification,
        "source": file_info(source),
        "source_plugin": (file_info(Path(options.source_plugin))
                          if options.source_plugin else None),
        "frames": options.frames,
        "threads": options.threads,
        "runs": options.runs,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "vpy_sha256": sha256_file(
            Path(__file__).with_name("vspipe_source_only.vpy")),
    }
    result = {
        "schema_version": 1,
        "environment": environment,
        "cases": summarize_cases(samples),
        "raw_samples": samples,
    }
    (output / "benchmark.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8")
    write_csv(samples, output / "benchmark.csv")
    write_markdown(result, output / "benchmark.md")
    (output / "commands.txt").write_text(
        "\n".join(sample["command"] for sample in samples) + "\n",
        encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
