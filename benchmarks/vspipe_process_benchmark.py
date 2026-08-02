#!/usr/bin/env python3
"""Whole-process VSPipe benchmark for the three recorded getnative cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_IMAGE_SHA256 = "61f9ee1ac858bbadd6a959ba35f5eceb077b8452b91e97a5ce3d39ebc69e20c6"
EXPECTED_OLD_SHA256 = "b02e4a2fbaaf6ba3f7e3cf2ad8a08d8eefab9e5d634e1d829764671d49933000"
CASES = ("getfnative", "getfnative_v2", "selectkernel")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def median_absolute_deviation(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def run_once(options, case: str, implementation: str, run: int) -> dict:
    script = Path(options.repo_root) / "benchmarks" / "vspipe_getnative.vpy"
    command = [
        options.vspipe,
        "--arg", f"implementation={implementation}",
        "--arg", f"image={Path(options.image).resolve()}",
        "--arg", f"plugin={Path(options.new_plugin).resolve()}",
        "--arg", f"old_plugin={Path(options.old_plugin).resolve()}",
        "--arg", f"case={case}",
        "--arg", f"frames={options.frames}",
        "--arg", f"threads={options.threads}",
        "--requests", str(options.requests),
        "--end", str(options.frames - 1),
        "--filter-time",
        str(script.resolve()),
        ".",
    ]
    start = time.perf_counter_ns()
    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace", check=False)
    elapsed_ns = time.perf_counter_ns() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"VSPipe failed for {case}/{implementation}/run-{run}:\n"
            + completed.stdout + "\n" + completed.stderr)
    return {
        "case": case,
        "implementation": implementation,
        "run": run,
        "frames": options.frames,
        "requests": options.requests,
        "elapsed_ns": elapsed_ns,
        "fps": options.frames / (elapsed_ns / 1e9),
        "command": subprocess.list2cmdline(command),
        "vspipe_output": (completed.stdout + completed.stderr).strip(),
    }


def summarize(samples: list[dict], implementation: str) -> dict:
    selected = [item for item in samples
                if item["implementation"] == implementation]
    elapsed = [item["elapsed_ns"] for item in selected]
    fps = [item["fps"] for item in selected]
    median_elapsed = statistics.median(elapsed)
    return {
        "runs": len(selected),
        "elapsed_ns": {
            "median": median_elapsed,
            "mad": median_absolute_deviation(elapsed),
            "minimum": min(elapsed),
            "maximum": max(elapsed),
        },
        "fps": {
            "median": selected[0]["frames"] / (median_elapsed / 1e9),
            "mad": median_absolute_deviation(fps),
            "minimum": min(fps),
            "maximum": max(fps),
            "samples": fps,
        },
    }


def version(command: list[str]) -> str:
    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace", check=False)
    return (completed.stdout + completed.stderr).strip()


def write_outputs(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    with (output / "benchmark.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "type", "case", "implementation", "run", "frames", "requests",
            "elapsed_seconds", "fps", "median_fps", "speedup"])
        for sample in result["raw_samples"]:
            writer.writerow([
                "raw", sample["case"], sample["implementation"], sample["run"],
                sample["frames"], sample["requests"],
                sample["elapsed_ns"] / 1e9, sample["fps"], "", ""])
        for item in result["cases"]:
            for implementation in ("old", "new"):
                writer.writerow([
                    "summary", item["case"], implementation, "",
                    result["environment"]["frames"],
                    result["environment"]["requests"],
                    item[implementation]["elapsed_ns"]["median"] / 1e9,
                    "", item[implementation]["fps"]["median"], item["speedup"]])

    lines = [
        "# Whole-process VSPipe throughput",
        "",
        f"Generated: `{result['environment']['timestamp_utc']}`",
        "",
        "Every sample is a fresh VSPipe process. FPS uses external wall time from",
        "process creation through process exit, including Python, VS core, plugin",
        "loading, PNG decoding, graph construction, cold planning, filtering, and",
        "shutdown. VSPipe uses 32 requests and writes 500 frames to the null sink.",
        "",
        "| Case | Old fps (3 runs) | New fps (3 runs) | Old median | New median | Speedup |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in result["cases"]:
        old_values = ", ".join(f"{value:.3f}" for value in item["old"]["fps"]["samples"])
        new_values = ", ".join(f"{value:.3f}" for value in item["new"]["fps"]["samples"])
        lines.append(
            f"| {item['case']} | {old_values} | {new_values} | "
            f"{item['old']['fps']['median']:.3f} | "
            f"{item['new']['fps']['median']:.3f} | {item['speedup']:.3f}x |")
    lines.extend([
        "",
        "## Case coverage",
        "",
        "- `getfnative`: original 11 x 2800 runtime-FrameEval graph; frames 0..499",
        "  cover Bilinear heights 700.0..749.9 in original output order.",
        "- `getfnative_v2`: original 8 x 400 vertical-only graph; frames 0..399",
        "  cover Bilinear 840.0..879.9 and frames 400..499 cover the first 100",
        "  Bicubic(1/3,1/3) heights.",
        "- `selectkernel`: all 101 original Bilinear/Bicubic candidates at 719.8",
        "  are looped in original order until 500 frames are emitted.",
        "",
        "All cases retain the source scripts' reconstruction kernel, threshold,",
        "5-pixel crop, and PlaneStats path. The unavailable MKV is replaced by",
        "the fixed 6.2-1.png input.",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(result["environment"], indent=2, ensure_ascii=True),
        "```",
        "",
        "Complete commands: [`commands.txt`](commands.txt)",
        "",
    ])
    (output / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "commands.txt").write_text(
        "\n".join(sample["command"] for sample in result["raw_samples"]) + "\n",
        encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    vs_root = Path(r"D:\okegui\OKEGui\tools\vapoursynth")
    parser.add_argument("--image", default=str(Path.home() / "Downloads" / "6.2-1.png"))
    parser.add_argument("--old-plugin", default=str(
        vs_root / "vapoursynth64" / "plugins" / "descale.dll"))
    parser.add_argument("--new-plugin", default=str(root / "build" / "Release" / "dsmvc.dll"))
    parser.add_argument("--vspipe", default=str(vs_root / "VSPipe.exe"))
    parser.add_argument("--repo-root", default=str(root))
    parser.add_argument("--output", default=str(
        root / "benchmark-results" / "planner-frontend-vspipe-500"))
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cases", nargs="*", choices=CASES, default=list(CASES))
    options = parser.parse_args()

    image = Path(options.image)
    old_plugin = Path(options.old_plugin)
    new_plugin = Path(options.new_plugin)
    if sha256_file(image) != EXPECTED_IMAGE_SHA256:
        raise RuntimeError("input image SHA-256 does not match the fixed benchmark input")
    if sha256_file(old_plugin) != EXPECTED_OLD_SHA256:
        raise RuntimeError("baseline descale DLL SHA-256 does not match")
    if not new_plugin.is_file():
        raise RuntimeError(f"new plugin does not exist: {new_plugin}")

    raw_samples = []
    cases = []
    for case in options.cases:
        samples = []
        for run in range(1, options.runs + 1):
            for implementation in ("old", "new"):
                sample = run_once(options, case, implementation, run)
                samples.append(sample)
                raw_samples.append(sample)
                print(
                    f"{case} {implementation} run {run}: "
                    f"{sample['fps']:.3f} fps", flush=True)
        old = summarize(samples, "old")
        new = summarize(samples, "new")
        cases.append({
            "case": case,
            "old": old,
            "new": new,
            "speedup": old["elapsed_ns"]["median"]
                / new["elapsed_ns"]["median"],
        })

    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER"),
        "logical_cpu_count": os.cpu_count(),
        "vspipe": version([options.vspipe, "--version"]),
        "frames": options.frames,
        "requests": options.requests,
        "threads": options.threads,
        "input": str(image.resolve()),
        "input_sha256": sha256_file(image),
        "old_plugin": str(old_plugin.resolve()),
        "old_plugin_sha256": sha256_file(old_plugin),
        "new_plugin": str(new_plugin.resolve()),
        "new_plugin_sha256": sha256_file(new_plugin),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "vpy_sha256": sha256_file(
            Path(options.repo_root) / "benchmarks" / "vspipe_getnative.vpy"),
    }
    result = {
        "schema_version": 1,
        "environment": environment,
        "cases": cases,
        "raw_samples": raw_samples,
    }
    output = Path(options.output)
    write_outputs(result, output)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
