#!/usr/bin/env python3
"""Measure CPU throughput before and after the VapourSynth API4 migration."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import blank_fixed_kernel_benchmark as blank


VARIANTS = ("api3", "api4")
DEFAULT_KERNELS = ("bilinear", "bicubic_b0_c0_5", "spline64")
DEFAULT_THREADS = (1, 8, 16, 32)


def read_optional_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def sample_options(options, variant: str):
    plugin = options.api3_plugin if variant == "api3" else options.api4_plugin
    return SimpleNamespace(
        new_plugin=plugin,
        old_plugin=options.api3_plugin,
        vspipe=options.vspipe,
        backend="cpu",
        src_height=options.src_height,
        base_height=options.base_height,
    )


def run_sample(options, kernel: str, variant: str, threads: int,
               run: int, frames: int, warmup: bool) -> dict:
    result = blank.run_sample(
        sample_options(options, variant), kernel, "new", threads,
        run, frames, warmup=warmup)
    result["variant"] = variant
    result.pop("implementation", None)
    return result


def summarize(samples: list[dict], threshold: float) -> list[dict]:
    grouped: dict[tuple[str, int, str], list[dict]] = {}
    for sample in samples:
        key = (sample["kernel"], sample["threads"], sample["variant"])
        grouped.setdefault(key, []).append(sample)

    result = []
    for threads in sorted({sample["threads"] for sample in samples}):
        for kernel in dict.fromkeys(sample["kernel"] for sample in samples):
            api3 = grouped[(kernel, threads, "api3")]
            api4 = grouped[(kernel, threads, "api4")]
            api3_by_run = {sample["run"]: sample for sample in api3}
            api4_by_run = {sample["run"]: sample for sample in api4}
            paired_ratios = [
                api4_by_run[run]["vspipe_fps"]
                / api3_by_run[run]["vspipe_fps"]
                for run in sorted(api3_by_run)
            ]
            api3_vspipe = statistics.median(
                sample["vspipe_fps"] for sample in api3)
            api4_vspipe = statistics.median(
                sample["vspipe_fps"] for sample in api4)
            api3_wall = statistics.median(sample["fps"] for sample in api3)
            api4_wall = statistics.median(sample["fps"] for sample in api4)
            ratio = api4_vspipe / api3_vspipe
            result.append({
                "kernel": kernel,
                "label": blank.KERNELS[kernel],
                "threads": threads,
                "runs": len(api3),
                "api3_vspipe_fps": api3_vspipe,
                "api4_vspipe_fps": api4_vspipe,
                "vspipe_ratio": ratio,
                "vspipe_change_percent": (ratio - 1.0) * 100.0,
                "paired_ratio_median": statistics.median(paired_ratios),
                "paired_ratio_minimum": min(paired_ratios),
                "paired_ratio_maximum": max(paired_ratios),
                "api3_wall_fps": api3_wall,
                "api4_wall_fps": api4_wall,
                "wall_ratio": api4_wall / api3_wall,
                "regression": ratio < 1.0 - threshold,
            })
    return result


def write_samples_csv(samples: list[dict], path: Path) -> None:
    fields = [
        "kernel", "variant", "threads", "run", "frames",
        "elapsed_seconds", "fps", "vspipe_seconds", "vspipe_fps",
        "process_overhead_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({field: sample[field] for field in fields})


def write_summary_csv(summary: list[dict], path: Path) -> None:
    fields = list(summary[0]) if summary else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def write_markdown(result: dict, path: Path) -> None:
    environment = result["environment"]
    summary = result["summary"]
    regressions = [case for case in summary if case["regression"]]
    lines = [
        "# CPU API4 Regression Benchmark",
        "",
        "This isolates the VapourSynth API3-to-API4 migration by comparing "
        "two dsmvc builds from the same CUDA/CPU implementation state. Each "
        "sample runs in a fresh VSPipe process with `backend=cpu`.",
        "",
        f"Each cell processes {environment['frames']:,} 1920x1080 GRAYS "
        "BlankClip frames at fixed 810p geometry. Every measured process is "
        f"immediately preceded by an independent {environment['warmup_frames']}-"
        "frame warm-up process. API3/API4 order alternates by cell and run.",
        "",
        "VSPipe internal throughput is the primary metric. A cell is flagged "
        f"only when its median drops by more than "
        f"{environment['regression_threshold_percent']:.1f}%.",
        "",
        "## Result",
        "",
        f"**{'PASS' if not regressions else 'REGRESSION'}:** "
        f"{len(regressions)} of {len(summary)} cells exceeded the threshold.",
        "",
        "| Kernel | Threads | API3 fps | API4 fps | Change | Paired range | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for case in summary:
        status = "regression" if case["regression"] else "pass"
        paired_min = (case["paired_ratio_minimum"] - 1.0) * 100.0
        paired_max = (case["paired_ratio_maximum"] - 1.0) * 100.0
        lines.append(
            f"| `{case['label']}` | R{case['threads']}T{case['threads']} | "
            f"{case['api3_vspipe_fps']:.2f} | "
            f"{case['api4_vspipe_fps']:.2f} | "
            f"{case['vspipe_change_percent']:+.2f}% | "
            f"{paired_min:+.2f}%..{paired_max:+.2f}% | {status} |")

    changes = [case["vspipe_change_percent"] for case in summary]
    lines.extend([
        "",
        "## Summary",
        "",
        f"- Median cell change: {statistics.median(changes):+.2f}%",
        f"- Worst cell change: {min(changes):+.2f}%",
        f"- Best cell change: {max(changes):+.2f}%",
        "- External wall-clock results, raw filter timings, and exact commands "
        "are retained in the JSON/CSV artifacts.",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(environment, indent=2, ensure_ascii=True),
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--api3-plugin", required=True, type=Path)
    result.add_argument("--api4-plugin", required=True, type=Path)
    result.add_argument("--vspipe", required=True, type=Path)
    result.add_argument(
        "--output", type=Path,
        default=root / "benchmark-results" / "cpu-api4-regression")
    result.add_argument("--frames", type=int, default=5000)
    result.add_argument("--warmup-frames", type=int, default=128)
    result.add_argument("--runs", type=int, default=3)
    result.add_argument("--threads", nargs="+", type=int,
                        default=list(DEFAULT_THREADS))
    result.add_argument("--kernels", nargs="+", choices=tuple(blank.KERNELS),
                        default=list(DEFAULT_KERNELS))
    result.add_argument("--src-height", type=float, default=810.0)
    result.add_argument("--base-height", type=float, default=1000.0)
    result.add_argument(
        "--regression-threshold", type=float, default=0.03,
        help="Fractional median throughput drop that flags a regression.")
    return result


def main() -> int:
    options = parser().parse_args()
    for name in ("api3_plugin", "api4_plugin", "vspipe", "output"):
        value = getattr(options, name).expanduser().resolve()
        setattr(options, name, value)
    if options.frames < 1 or options.warmup_frames < 0 or options.runs < 1:
        raise ValueError("frame counts and runs must be valid positive values")
    if not 0.0 <= options.regression_threshold < 1.0:
        raise ValueError("--regression-threshold must be in [0, 1)")
    if len(set(options.threads)) != len(options.threads):
        raise ValueError("thread counts must be unique")
    if any(threads < 1 for threads in options.threads):
        raise ValueError("thread counts must be positive")
    for path in (options.api3_plugin, options.api4_plugin, options.vspipe):
        if not path.is_file():
            raise FileNotFoundError(path)

    options.output.mkdir(parents=True, exist_ok=True)
    samples = []
    warmups = []
    cell_index = 0
    for threads in options.threads:
        for kernel in options.kernels:
            for run in range(1, options.runs + 1):
                order = VARIANTS if (cell_index + run) % 2 else VARIANTS[::-1]
                for variant in order:
                    if options.warmup_frames:
                        warmup = run_sample(
                            options, kernel, variant, threads, run,
                            options.warmup_frames, warmup=True)
                        warmups.append(warmup)
                    sample = run_sample(
                        options, kernel, variant, threads, run,
                        options.frames, warmup=False)
                    samples.append(sample)
                    print(
                        f"{kernel} {variant} R{threads}T{threads} run {run}: "
                        f"{sample['vspipe_fps']:.2f} VSPipe fps, "
                        f"{sample['fps']:.2f} wall fps",
                        flush=True)
            cell_index += 1

    summary = summarize(samples, options.regression_threshold)
    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_governor": read_optional_text(Path(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")),
        "vspipe": str(options.vspipe),
        "api3_plugin": blank.file_info(options.api3_plugin),
        "api4_plugin": blank.file_info(options.api4_plugin),
        "vpy": blank.file_info(
            Path(__file__).with_name("vspipe_blank_fixed_kernel.vpy")),
        "frames": options.frames,
        "warmup_frames": options.warmup_frames,
        "runs": options.runs,
        "threads": options.threads,
        "kernels": options.kernels,
        "backend": "cpu",
        "src_height": options.src_height,
        "base_height": options.base_height,
        "regression_threshold_percent": options.regression_threshold * 100.0,
        "relevant_environment": {
            name: os.environ.get(name)
            for name in ("DSMVC_MEMORY_CONCURRENCY", "OMP_NUM_THREADS")
        },
        "runner": blank.file_info(Path(__file__).resolve()),
    }
    result = {
        "schema_version": 1,
        "environment": environment,
        "summary": summary,
        "raw_samples": samples,
        "warmup_samples": warmups,
    }
    (options.output / "benchmark.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8")
    write_samples_csv(samples, options.output / "benchmark.csv")
    write_summary_csv(summary, options.output / "summary.csv")
    write_markdown(result, options.output / "benchmark.md")
    (options.output / "commands.txt").write_text(
        "\n".join(
            sample["command"] for sample in [*warmups, *samples]) + "\n",
        encoding="utf-8")

    regressions = [case for case in summary if case["regression"]]
    print(
        f"{options.output}: {len(regressions)}/{len(summary)} cells "
        "exceeded the regression threshold",
        flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
