#!/usr/bin/env python3
"""Compare old and current descale on an in-memory blank clip."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
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
KERNEL_BASE_SUPPORT = {
    "bilinear": 1,
    "bicubic_b0_c0_5": 2,
    "lanczos2": 2,
    "lanczos3": 3,
    "lanczos4": 4,
    "lanczos5": 5,
    "lanczos6": 6,
    "spline16": 2,
    "spline36": 3,
    "spline64": 4,
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
IMPLEMENTATIONS = ("old", "jet", "baseline", "new")
DEFAULT_IMPLEMENTATIONS = ("old", "new")
OUTPUT_RE = re.compile(
    r"Output (?P<frames>\d+) frames in (?P<seconds>[0-9.]+) seconds "
    r"\((?P<fps>[0-9.]+) fps\)"
)


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


def parse_vspipe_timing(output: str, expected_frames: int) -> dict:
    match = OUTPUT_RE.search(output)
    if match is None:
        raise RuntimeError("VSPipe output did not contain an Output timing line")
    emitted_frames = int(match.group("frames"))
    if emitted_frames != expected_frames:
        raise RuntimeError(
            f"VSPipe emitted {emitted_frames} frames; expected {expected_frames}")
    return {
        "frames": emitted_frames,
        "seconds": float(match.group("seconds")),
        "fps": float(match.group("fps")),
    }


def blur_metadata(kernel: str, blur: float) -> dict[str, int | float]:
    support = math.ceil(KERNEL_BASE_SUPPORT[kernel] * blur)
    return {
        "blur": blur,
        "effective_support": support,
        "half_bandwidth": 2 * support - 1,
    }


def passes_blur(options, implementation: str, blur: float) -> bool:
    if implementation == "old":
        return False
    if (implementation == "baseline"
            and not options.baseline_supports_blur):
        return False
    return not (options.omit_unity_blur and blur == 1.0)


def build_command(options, kernel: str, implementation: str, blur: float,
                  threads: int, frames: int) -> list[str]:
    script = Path(__file__).with_name("vspipe_blank_fixed_kernel.vpy").resolve()
    values = {
        "implementation": implementation,
        "kernel": kernel,
        "plugin": str(options.new_plugin),
        "old_plugin": str(options.old_plugin or ""),
        "jet_plugin": str(getattr(options, "jet_plugin", None) or ""),
        "baseline_plugin": str(getattr(options, "baseline_plugin", None) or ""),
        "frames": str(frames),
        "threads": str(threads),
        "backend": options.backend,
        "opt": str(options.opt),
        "src_height": str(options.src_height),
        "base_height": str(options.base_height),
        "blur": repr(blur),
        "pass_blur": str(int(passes_blur(
            options, implementation, blur))),
    }
    command = [str(options.vspipe)]
    for key, value in values.items():
        command.extend(["--arg", f"{key}={value}"])
    command.extend([
        "--requests", str(threads),
        "--start", "0",
        "--end", str(frames - 1),
        "--filter-time", str(script),
        "--",
    ])
    return command


def run_sample(options, kernel: str, implementation: str, blur: float,
               threads: int, run: int, frames: int,
               warmup: bool = False) -> dict:
    command = build_command(
        options, kernel, implementation, blur, threads, frames)
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace",
        check=False)
    elapsed_ns = time.perf_counter_ns() - started
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"VSPipe failed for {kernel}/blur-{blur:g}/{implementation}/"
            f"R{threads}T{threads}/"
            f"run-{run}:\n{output[-8000:]}")
    vspipe_timing = parse_vspipe_timing(output, frames)
    elapsed_seconds = elapsed_ns / 1e9
    result = {
        "kernel": kernel,
        "label": KERNELS[kernel],
        "implementation": implementation,
        "blur_argument": (
            "explicit" if passes_blur(options, implementation, blur)
            else "omitted"),
        "run": run,
        "warmup": warmup,
        "frames": frames,
        "threads": threads,
        "requests": threads,
        "backend": (
            options.backend if implementation in ("baseline", "new")
            else "reference"),
        "opt": options.opt if implementation in ("baseline", "new") else 0,
        "elapsed_seconds": elapsed_seconds,
        "fps": frames / elapsed_seconds,
        "vspipe_seconds": vspipe_timing["seconds"],
        "vspipe_fps": vspipe_timing["fps"],
        "process_overhead_seconds": max(
            0.0, elapsed_seconds - vspipe_timing["seconds"]),
        "filter_time": parse_filter_times(output),
        "command": command_text(command),
        "vspipe_output_tail": output[-4000:],
    }
    result.update(blur_metadata(kernel, blur))
    return result


def summarize_cases(samples: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for sample in samples:
        grouped[(sample["kernel"], sample["implementation"],
                 sample["threads"], sample["blur"])].append(sample)
    result = []
    for key, items in sorted(grouped.items()):
        result.append({
            "kernel": key[0],
            "label": items[0]["label"],
            "implementation": key[1],
            "threads": key[2],
            "blur": key[3],
            "blur_argument": items[0]["blur_argument"],
            "effective_support": items[0]["effective_support"],
            "half_bandwidth": items[0]["half_bandwidth"],
            "runs": len(items),
            "elapsed_seconds": summarize([
                item["elapsed_seconds"] for item in items]),
            "fps": summarize([item["fps"] for item in items]),
            "vspipe_seconds": summarize([
                item["vspipe_seconds"] for item in items]),
            "vspipe_fps": summarize([
                item["vspipe_fps"] for item in items]),
            "process_overhead_seconds": summarize([
                item["process_overhead_seconds"] for item in items]),
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
        "kernel", "implementation", "blur", "blur_argument",
        "effective_support", "half_bandwidth", "threads", "run", "frames",
        "elapsed_seconds", "fps", "vspipe_seconds", "vspipe_fps",
        "process_overhead_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({field: sample[field] for field in fields})


def write_markdown(result: dict, path: Path) -> None:
    cases = result["cases"]
    by_key = {(case["kernel"], case["implementation"], case["threads"],
               case["blur"]): case
              for case in cases}
    threads = result["environment"]["threads"]
    blurs = result["environment"]["blurs"]
    implementations = result["environment"]["implementations"]
    throughput_header = ["Kernel"]
    throughput_separator = ["---"]
    for thread in threads:
        if set(implementations) == set(DEFAULT_IMPLEMENTATIONS):
            throughput_header.extend([
                f"R{thread} old", f"R{thread} new", f"R{thread} new/old",
            ])
            throughput_separator.extend(["---:", "---:", "---:"])
        else:
            for implementation in implementations:
                throughput_header.append(f"R{thread} {implementation}")
                throughput_separator.append("---:")
    lines = [
        "# Blank fixed-kernel benchmark",
        "",
        "This compares the selected descale implementations without a decoder or "
        f"source clip. Each run processes {result['environment']['frames']:,} "
        "frames from an in-memory "
        "1920x1080 GRAYS `std.BlankClip` at fixed 810p geometry.",
        f"Each cell has {result['environment']['warmup_runs']} untimed warm-up "
        f"run(s) of {result['environment']['warmup_frames']} frames. Reported "
        "throughput is external process wall time; VSPipe's internal timing is "
        "also retained in the JSON and CSV; use that internal value for kernel "
        "A/B decisions and wall time for end-to-end cost. Warm-up processes "
        "warm driver, "
        "module, page caches, and GPU clocks, but each measured fresh process "
        "still creates its own CUDA context.",
        "",
        "## Throughput",
        "",
        "| " + " | ".join(throughput_header) + " |",
        "| " + " | ".join(throughput_separator) + " |",
    ]
    for kernel in result["environment"]["kernels"]:
        for blur in blurs:
            metadata = blur_metadata(kernel, blur)
            label = (f"{KERNELS[kernel]}; blur={blur:g}; "
                     f"support={metadata['effective_support']}; "
                     f"H{metadata['half_bandwidth']}")
            row = [f"`{label}`"]
            for thread in threads:
                if set(implementations) == set(DEFAULT_IMPLEMENTATIONS):
                    old = by_key[(kernel, "old", thread, blur)]["fps"]["median"]
                    new = by_key[(kernel, "new", thread, blur)]["fps"]["median"]
                    row.extend([f"{old:.2f}", f"{new:.2f}", f"{new / old:.3f}x"])
                else:
                    for implementation in implementations:
                        value = by_key[
                            (kernel, implementation, thread, blur)]["fps"]["median"]
                        row.append(f"{value:.2f}")
            lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## Filter Time",
        "",
        "There is no LSMASH, decoder, or Point conversion in this graph. "
        "The remaining filter time is the blank producer and descale node; "
        "R8 percentages are accumulated across worker threads.",
        "",
        "| Kernel | Impl | Blur | Support | H | Threads | BlankClip s / % | dsmvc s / % |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for case in cases:
        descale_name = DESCALE_FILTERS[case["kernel"]]
        blank = filter_value(case, "BlankClip")
        descale = filter_value(case, descale_name)
        lines.append(
            f"| `{case['label']}` | `{case['implementation']}` | "
            f"{case['blur']:g} | {case['effective_support']} | "
            f"{case['half_bandwidth']} | "
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
    parser.add_argument("--old-plugin", type=Path, default=None)
    parser.add_argument("--jet-plugin", type=Path, default=None,
                        help="JET vapoursynth-descale plugin (descale namespace; "
                        "always loaded into its own process).")
    parser.add_argument(
        "--baseline-plugin", type=Path, default=None,
        help="Preserved pre-change dsmvc plugin. It is loaded in a separate "
        "process and receives blur only with --baseline-supports-blur.")
    parser.add_argument("--new-plugin", required=True, type=Path)
    parser.add_argument("--vspipe", required=True, type=Path)
    parser.add_argument("--implementations", nargs="+", choices=IMPLEMENTATIONS,
                        default=list(DEFAULT_IMPLEMENTATIONS))
    parser.add_argument("--output", type=Path, default=root / "benchmark-results" /
                        "blank-fixed-kernel-digimon-810p")
    parser.add_argument("--frames", type=int, default=8000)
    parser.add_argument("--src-height", type=float, default=810.0)
    parser.add_argument("--base-height", type=float, default=1000.0)
    parser.add_argument("--threads", nargs="*", type=int,
                        default=list(DEFAULT_THREADS))
    parser.add_argument("--blurs", nargs="+", type=float, default=[1.0],
                        help="Kernel stretch factors. Values other than 1 "
                        "require jet/new or --baseline-supports-blur.")
    parser.add_argument(
        "--omit-unity-blur", action="store_true",
        help="Do not pass blur when its value is 1.0; used to benchmark the "
        "default API path separately from explicit blur=1.0.")
    parser.add_argument(
        "--baseline-supports-blur", action="store_true",
        help="Allow the preserved dsmvc baseline to receive blur. Use only "
        "when that baseline is known to implement the same blur API.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1,
                        help="Untimed throwaway VSPipe runs per benchmark cell.")
    parser.add_argument("--warmup-frames", type=int, default=256,
                        help="Frames in each warm-up run; 0 disables warm-up.")
    parser.add_argument("--backend",
                        choices=("auto", "cpu", "metal", "vulkan", "cuda"),
                        default="cpu")
    parser.add_argument("--opt", choices=(0, 1, 2), type=int, default=0,
                        help="Optional dsmvc CPU path selector; 0 uses default.")
    parser.add_argument("--kernels", nargs="*", choices=tuple(KERNELS),
                        default=list(KERNELS))
    options = parser.parse_args()
    options.old_plugin = (options.old_plugin.expanduser().resolve()
                          if options.old_plugin else None)
    options.jet_plugin = (options.jet_plugin.expanduser().resolve()
                          if options.jet_plugin else None)
    options.baseline_plugin = (
        options.baseline_plugin.expanduser().resolve()
        if options.baseline_plugin else None)
    options.new_plugin = options.new_plugin.expanduser().resolve()
    options.vspipe = options.vspipe.expanduser().resolve()
    options.output = options.output.expanduser().resolve()
    if (options.frames < 1 or options.runs < 1
            or options.warmup_runs < 0 or options.warmup_frames < 0):
        raise ValueError(
            "--frames and --runs must be positive; warm-up values cannot be negative")
    if not options.threads or any(value < 1 for value in options.threads):
        raise ValueError("thread counts must be positive")
    if len(set(options.threads)) != len(options.threads):
        raise ValueError("thread counts must be unique")
    if (not options.blurs
            or any(not math.isfinite(value) or value <= 0.0
                   for value in options.blurs)):
        raise ValueError("blur values must be finite and greater than zero")
    if any(value >= 1080.0 for value in options.blurs):
        raise ValueError("blur values must be smaller than the source-plane extent")
    if len(set(options.blurs)) != len(options.blurs):
        raise ValueError("blur values must be unique")
    blur_implementations = {"jet", "new"}
    if options.baseline_supports_blur:
        blur_implementations.add("baseline")
    if (any(value != 1.0 for value in options.blurs)
            and any(implementation not in blur_implementations
                    for implementation in options.implementations)):
        raise ValueError(
            "blur values other than 1 require jet/new or a blur-capable "
            "baseline")
    if "old" in options.implementations and (
            options.old_plugin is None or not options.old_plugin.is_file()):
        raise FileNotFoundError(
            f"old plugin does not exist: {options.old_plugin}")
    if "jet" in options.implementations and (
            options.jet_plugin is None or not options.jet_plugin.is_file()):
        raise FileNotFoundError(
            f"jet plugin does not exist: {options.jet_plugin}")
    if "baseline" in options.implementations and (
            options.baseline_plugin is None
            or not options.baseline_plugin.is_file()):
        raise FileNotFoundError(
            f"baseline plugin does not exist: {options.baseline_plugin}")
    for required in (options.new_plugin, options.vspipe):
        if not required.is_file():
            raise FileNotFoundError(required)

    script = Path(__file__).with_name("vspipe_blank_fixed_kernel.vpy")
    samples = []
    warmups = []
    options.output.mkdir(parents=True, exist_ok=True)
    for threads in options.threads:
        for kernel in options.kernels:
            for blur in options.blurs:
                if options.warmup_frames > 0:
                    for implementation in options.implementations:
                        for warmup_run in range(1, options.warmup_runs + 1):
                            warmup = run_sample(
                                options, kernel, implementation, blur, threads,
                                warmup_run, options.warmup_frames, warmup=True)
                            warmups.append(warmup)
                            print(
                                f"warmup {kernel} blur={blur:g} {implementation} "
                                f"R{threads}T{threads} {warmup_run}/"
                                f"{options.warmup_runs}: "
                                f"{warmup['vspipe_fps']:.3f} VSPipe fps",
                                flush=True)
                for run in range(1, options.runs + 1):
                    measured_order = list(options.implementations)
                    if run % 2 == 0:
                        measured_order.reverse()
                    for implementation in measured_order:
                        sample = run_sample(
                            options, kernel, implementation, blur,
                            threads, run, options.frames)
                        samples.append(sample)
                        print(
                            f"{kernel} blur={blur:g} {implementation} "
                            f"R{threads}T{threads} run {run}: "
                            f"{sample['fps']:.3f} fps", flush=True)

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
        "old_plugin": (file_info(options.old_plugin)
                       if options.old_plugin else None),
        "jet_plugin": (file_info(options.jet_plugin)
                       if options.jet_plugin else None),
        "baseline_plugin": (file_info(options.baseline_plugin)
                            if options.baseline_plugin else None),
        "new_plugin": file_info(options.new_plugin),
        "vpy": file_info(script),
        "frames": options.frames,
        "src_height": options.src_height,
        "base_height": options.base_height,
        "threads": options.threads,
        "blurs": options.blurs,
        "omit_unity_blur": options.omit_unity_blur,
        "baseline_supports_blur": options.baseline_supports_blur,
        "runs": options.runs,
        "warmup_runs": options.warmup_runs,
        "warmup_frames": options.warmup_frames,
        "backend": options.backend,
        "opt": options.opt,
        "kernels": options.kernels,
        "implementations": list(options.implementations),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    result = {
        "schema_version": 2,
        "environment": environment,
        "cases": summarize_cases(samples),
        "raw_samples": samples,
        "warmup_samples": warmups,
    }
    (options.output / "benchmark.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8")
    write_csv(samples, options.output / "benchmark.csv")
    write_markdown(result, options.output / "benchmark.md")
    (options.output / "commands.txt").write_text(
        "\n".join(
            sample["command"] for sample in [*warmups, *samples]) + "\n",
        encoding="utf-8")
    print(options.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
