#!/usr/bin/env python3
"""Benchmark baseline and candidate dsmvc plugins on in-memory BlankClip."""

from __future__ import annotations

import argparse
import array
import csv
import hashlib
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


KERNELS = ("b1", "b3", "b5", "b7")
FORMATS = ("float32", "gray16")
IMPLEMENTATIONS = ("baseline", "candidate")
DEFAULT_THREADS = (1, 8, 32)
OUTPUT_RE = re.compile(
    r"Output\s+(?P<frames>\d+)\s+frames\s+in\s+"
    r"(?P<seconds>[0-9.]+)\s+seconds\s+\((?P<fps>[0-9.]+)\s+fps\)")


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
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    median = statistics.median(ordered)
    return {
        "median": median,
        "mad": statistics.median(abs(value - median) for value in ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def parse_filter_times(output: str) -> dict[str, dict[str, float | str]]:
    result = {}
    started = False
    for line in output.splitlines():
        if line.strip().startswith("Filtername"):
            started = True
            continue
        if not started:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            percent = float(parts[-2])
            seconds = float(parts[-1])
        except ValueError:
            continue
        result[" ".join(parts[:-3])] = {
            "mode": parts[-3], "percent": percent, "seconds": seconds}
    return result


def build_command(options, implementation: str, kernel: str,
                  format_name: str, threads: int, frames: int) -> list[str]:
    plugin = (options.baseline_plugin if implementation == "baseline"
              else options.candidate_plugin)
    values = {
        "plugin": plugin,
        "kernel": kernel,
        "format_name": format_name,
        "frames": frames,
        "threads": threads,
        "source_width": options.source_width,
        "source_height": options.source_height,
        "native_height": options.native_height,
        "base_height": options.base_height,
    }
    command = [str(options.vspipe)]
    for key, value in values.items():
        command.extend(["--arg", f"{key}={value}"])
    command.extend([
        "--requests", str(threads),
        "--start", "0",
        "--end", str(frames - 1),
        "--filter-time",
        str(Path(__file__).with_name("vspipe_blank_pipeline.vpy").resolve()),
        "--",
    ])
    return command


def run_vspipe(options, implementation: str, kernel: str,
               format_name: str, threads: int, frames: int,
               run: int) -> dict:
    command = build_command(
        options, implementation, kernel, format_name, threads, frames)
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace", check=False)
    elapsed_seconds = (time.perf_counter_ns() - started) / 1e9
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"VSPipe failed for {implementation}/{format_name}/{kernel}/"
            f"R{threads}/run-{run}:\n{output[-8000:]}")
    match = OUTPUT_RE.search(output)
    if not match:
        raise RuntimeError(f"VSPipe did not report FPS:\n{output[-4000:]}")
    completed_frames = int(match.group("frames"))
    cpu_seconds = (
        usage_after.ru_utime - usage_before.ru_utime
        + usage_after.ru_stime - usage_before.ru_stime)
    return {
        "implementation": implementation,
        "format": format_name,
        "kernel": kernel,
        "threads": threads,
        "requests": threads,
        "run": run,
        "frames": completed_frames,
        "vspipe_seconds": float(match.group("seconds")),
        "vspipe_fps": float(match.group("fps")),
        "wall_seconds": elapsed_seconds,
        "wall_fps": completed_frames / elapsed_seconds,
        "cpu_seconds": cpu_seconds,
        "cpu_ms_per_frame": cpu_seconds * 1000.0 / completed_frames,
        "minor_faults": usage_after.ru_minflt - usage_before.ru_minflt,
        "major_faults": usage_after.ru_majflt - usage_before.ru_majflt,
        "filter_time": parse_filter_times(output),
        "command": subprocess.list2cmdline(command),
        "output_tail": output[-4000:],
    }


def summarize_samples(samples: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for sample in samples:
        key = (sample["format"], sample["kernel"],
               sample["threads"], sample["implementation"])
        grouped[key].append(sample)
    cases = []
    for key, items in sorted(grouped.items()):
        cases.append({
            "format": key[0],
            "kernel": key[1],
            "threads": key[2],
            "implementation": key[3],
            "runs": len(items),
            "vspipe_fps": summarize([item["vspipe_fps"] for item in items]),
            "wall_fps": summarize([item["wall_fps"] for item in items]),
            "cpu_ms_per_frame": summarize([
                item["cpu_ms_per_frame"] for item in items]),
        })
    return cases


def run_pixel_probe(options, implementation: str, output: Path) -> dict:
    plugin = (options.baseline_plugin if implementation == "baseline"
              else options.candidate_plugin)
    command = [
        str(options.vs_python),
        str(Path(__file__).with_name("blank_pipeline_pixels.py").resolve()),
        "--plugin", str(plugin),
        "--output", str(output),
        "--formats", *options.formats,
        "--kernels", *options.kernels,
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace", check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"pixel probe failed for {implementation}:\n"
            f"{completed.stdout}{completed.stderr}")
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def sample_values(payload: bytes, sample_type: str, bits: int):
    if sample_type == "float":
        values = array.array("f")
    elif bits <= 8:
        values = array.array("B")
    else:
        values = array.array("H")
    values.frombytes(payload)
    return values


def compare_pixels(options, output: Path) -> list[dict]:
    baseline_dir = output / "pixels" / "baseline"
    candidate_dir = output / "pixels" / "candidate"
    baseline = run_pixel_probe(options, "baseline", baseline_dir)
    candidate = run_pixel_probe(options, "candidate", candidate_dir)
    baseline_cases = {
        (case["format"], case["kernel"]): case for case in baseline["cases"]}
    candidate_cases = {
        (case["format"], case["kernel"]): case for case in candidate["cases"]}
    comparisons = []
    for key in sorted(baseline_cases):
        left = baseline_cases[key]
        right = candidate_cases[key]
        if (left["sample_type"], left["bits_per_sample"], left["range"]) != (
                right["sample_type"], right["bits_per_sample"], right["range"]):
            raise RuntimeError(f"pixel metadata differs for {key}")
        maximum = 0.0
        total = 0.0
        count = 0
        differing = 0
        hashes_equal = True
        for left_plane, right_plane in zip(left["planes"], right["planes"]):
            left_payload = (baseline_dir / left_plane["file"]).read_bytes()
            right_payload = (candidate_dir / right_plane["file"]).read_bytes()
            hashes_equal &= left_plane["sha256"] == right_plane["sha256"]
            left_values = sample_values(
                left_payload, left["sample_type"], left["bits_per_sample"])
            right_values = sample_values(
                right_payload, right["sample_type"], right["bits_per_sample"])
            if len(left_values) != len(right_values):
                raise RuntimeError(f"pixel plane sizes differ for {key}")
            for left_value, right_value in zip(left_values, right_values):
                difference = abs(float(left_value) - float(right_value))
                maximum = max(maximum, difference)
                total += difference
                count += 1
                differing += difference != 0.0
        comparisons.append({
            "format": key[0],
            "kernel": key[1],
            "hashes_equal": hashes_equal,
            "differing_samples": differing,
            "maximum_absolute_error": maximum,
            "mean_absolute_error": total / count if count else 0.0,
            "sample_count": count,
            "range": left["range"],
        })
    return comparisons


def write_csv(samples: list[dict], path: Path) -> None:
    fields = [
        "format", "kernel", "threads", "implementation", "run", "frames",
        "vspipe_seconds", "vspipe_fps", "wall_seconds", "wall_fps",
        "cpu_seconds", "cpu_ms_per_frame", "minor_faults", "major_faults",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({field: sample[field] for field in fields})


def write_markdown(result: dict, path: Path) -> None:
    cases = result["cases"]
    by_key = {
        (case["format"], case["kernel"], case["threads"],
         case["implementation"]): case for case in cases}
    lines = [
        "# Blank pipeline benchmark",
        "",
        "All throughput cases use an in-memory `std.BlankClip`; decoder and "
        "source I/O costs are absent.",
        "",
    ]
    for format_name in result["environment"]["formats"]:
        lines.extend([
            f"## {format_name}",
            "",
            "| Kernel | Requests | Baseline FPS | Candidate FPS | Ratio | "
            "Baseline CPU ms/frame | Candidate CPU ms/frame |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for kernel in result["environment"]["kernels"]:
            for threads in result["environment"]["threads"]:
                baseline = by_key[(format_name, kernel, threads, "baseline")]
                candidate = by_key[(format_name, kernel, threads, "candidate")]
                baseline_fps = baseline["vspipe_fps"]["median"]
                candidate_fps = candidate["vspipe_fps"]["median"]
                lines.append(
                    f"| `{kernel}` | R{threads} | {baseline_fps:.2f} | "
                    f"{candidate_fps:.2f} | {candidate_fps / baseline_fps:.4f}x | "
                    f"{baseline['cpu_ms_per_frame']['median']:.3f} | "
                    f"{candidate['cpu_ms_per_frame']['median']:.3f} |")
        lines.append("")
    lines.extend([
        "## Pixel consistency",
        "",
        "| Format | Kernel | SHA-256 equal | Different samples | Max error | "
        "Mean error |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for comparison in result["pixel_comparisons"]:
        lines.append(
            f"| `{comparison['format']}` | `{comparison['kernel']}` | "
            f"{comparison['hashes_equal']} | "
            f"{comparison['differing_samples']} | "
            f"{comparison['maximum_absolute_error']:.9g} | "
            f"{comparison['mean_absolute_error']:.9g} |")
    lines.extend([
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(result["environment"], indent=2, sort_keys=True),
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-plugin", required=True, type=Path)
    parser.add_argument("--candidate-plugin", required=True, type=Path)
    parser.add_argument("--vspipe", required=True, type=Path)
    parser.add_argument("--vs-python", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=root / "benchmark-results" /
                        "blank-pipeline-streamed-rhs")
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--warmup-frames", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", nargs="+", type=int,
                        default=list(DEFAULT_THREADS))
    parser.add_argument("--kernels", nargs="+", choices=KERNELS,
                        default=list(KERNELS))
    parser.add_argument("--formats", nargs="+", choices=tuple(
        ["float32", "gray8", "gray16", "yuv420p10", "rgb24"]),
        default=list(FORMATS))
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--native-height", type=float, default=810.0)
    parser.add_argument("--base-height", type=float, default=1000.0)
    options = parser.parse_args()
    for name in ("baseline_plugin", "candidate_plugin", "vspipe", "vs_python"):
        value = getattr(options, name).expanduser().absolute()
        if not value.is_file():
            raise FileNotFoundError(value)
        setattr(options, name, value)
    options.output = options.output.expanduser().resolve()
    if options.frames < 1 or options.runs < 1 or options.warmup_frames < 1:
        raise ValueError("frame and run counts must be positive")
    if any(value < 1 for value in options.threads):
        raise ValueError("thread counts must be positive")
    options.output.mkdir(parents=True, exist_ok=True)

    samples = []
    case_index = 0
    for format_name in options.formats:
        for kernel in options.kernels:
            for threads in options.threads:
                for implementation in IMPLEMENTATIONS:
                    run_vspipe(
                        options, implementation, kernel, format_name,
                        threads, options.warmup_frames, 0)
                for run in range(1, options.runs + 1):
                    order = list(IMPLEMENTATIONS)
                    if (case_index + run) % 2:
                        order.reverse()
                    for implementation in order:
                        sample = run_vspipe(
                            options, implementation, kernel, format_name,
                            threads, options.frames, run)
                        samples.append(sample)
                        print(
                            f"{format_name} {kernel} R{threads} "
                            f"{implementation} run {run}: "
                            f"{sample['vspipe_fps']:.2f} FPS",
                            flush=True)
                case_index += 1

    pixel_comparisons = compare_pixels(options, options.output)
    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "dsmvc_memory_concurrency": os.environ.get(
            "DSMVC_MEMORY_CONCURRENCY", "auto"),
        "source": {
            "type": "VapourSynth std.BlankClip",
            "width": options.source_width,
            "height": options.source_height,
            "native_height": options.native_height,
            "base_height": options.base_height,
        },
        "frames": options.frames,
        "warmup_frames": options.warmup_frames,
        "runs": options.runs,
        "threads": options.threads,
        "kernels": options.kernels,
        "formats": options.formats,
        "baseline_plugin": file_info(options.baseline_plugin),
        "candidate_plugin": file_info(options.candidate_plugin),
        "vspipe": file_info(options.vspipe),
    }
    result = {
        "environment": environment,
        "samples": samples,
        "cases": summarize_samples(samples),
        "pixel_comparisons": pixel_comparisons,
    }
    (options.output / "benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(samples, options.output / "samples.csv")
    write_markdown(result, options.output / "report.md")
    print(f"wrote {options.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
