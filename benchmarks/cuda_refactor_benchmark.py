#!/usr/bin/env python3
"""Compare CUDA steady-state saturation and plugin lifecycle costs.

The steady-state cases deliberately reuse one input frame so host DRAM and
PCIe traffic do not hide CUDA execution throughput. Lifecycle cases use a
small bilinear filter and fixed stream count to keep kernel work and slot
policy out of the create/first-frame/destroy comparison.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SATURATION_GRAPH = Path(__file__).with_name("vspipe_cuda_saturation.vpy")
VSPIPE_RESULT = re.compile(
    r"Output\s+(?P<frames>\d+)\s+frames\s+in\s+"
    r"(?P<seconds>[0-9.]+)\s+seconds\s+\((?P<fps>[0-9.]+)\s+fps\)"
)

CUDA_TUNING_ENV = (
    "DSMVC_CUDA_STREAMS",
    "DSMVC_CUDA_BATCH_FRAMES",
    "DSMVC_CUDA_HORIZONTAL_THREADS",
    "DSMVC_CUDA_VERTICAL_THREADS",
    "DSMVC_CUDA_SPLIT_HORIZONTAL_THREADS",
    "DSMVC_CUDA_SPLIT_VERTICAL_THREADS",
    "DSMVC_CUDA_HORIZONTAL_GLOBAL_TRANSPOSE",
    "DSMVC_CUDA_SPLIT_RHS",
    "DSMVC_CUDA_INPUT_CACHE_MB",
    "DSMVC_CUDA_PLAN_CACHE_MB",
)


def parse_variant(text: str) -> tuple[str, Path]:
    label, separator, raw_path = text.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("variant must be LABEL=/path/to/dsmvc.so")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"plugin does not exist: {path}")
    return label, path


def clean_cuda_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in CUDA_TUNING_ENV:
        environment.pop(name, None)
    return environment


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "p95": percentile(values, 0.95),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def timed_lifecycle(core: Any, source: Any) -> dict[str, int]:
    started = time.perf_counter_ns()
    node = core.dsmvc.Debilinear(
        source,
        width=480,
        height=270,
        src_left=0.0,
        src_top=0.0,
        src_width=480.0,
        src_height=270.0,
        backend="cuda",
    )
    created = time.perf_counter_ns()
    frame = node.get_frame(0)
    rendered = time.perf_counter_ns()
    frame = None
    released = time.perf_counter_ns()
    node = None
    destroyed = time.perf_counter_ns()
    return {
        "filter_create_ns": created - started,
        "first_frame_ns": rendered - created,
        "frame_release_ns": released - rendered,
        "filter_destroy_ns": destroyed - released,
    }


def worker_main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker-mode", choices=("cold", "warm"), required=True)
    parser.add_argument("--worker-plugin", type=Path, required=True)
    parser.add_argument("--worker-iterations", type=int, default=1)
    parser.add_argument("--worker-warmups", type=int, default=2)
    options = parser.parse_args(arguments)
    if options.worker_iterations < 1 or options.worker_warmups < 0:
        parser.error("worker iteration counts are invalid")

    total_started = time.perf_counter_ns()
    import_started = time.perf_counter_ns()
    vs = importlib.import_module("vapoursynth")
    imported = time.perf_counter_ns()
    core = vs.core
    core.num_threads = 1
    core_ready = time.perf_counter_ns()
    if hasattr(core, "dsmvc"):
        raise RuntimeError("a dsmvc plugin was autoloaded before the requested plugin")
    core.std.LoadPlugin(path=str(options.worker_plugin.resolve()))
    plugin_loaded = time.perf_counter_ns()
    source = core.std.BlankClip(
        width=640,
        height=360,
        length=1,
        format=vs.GRAY8,
        color=0,
    )
    source_created = time.perf_counter_ns()
    common = {
        "vapoursynth_import_ns": imported - import_started,
        "core_create_ns": core_ready - imported,
        "plugin_load_ns": plugin_loaded - core_ready,
        "source_create_ns": source_created - plugin_loaded,
    }

    if options.worker_mode == "cold":
        sample = timed_lifecycle(core, source)
        measured = time.perf_counter_ns()
        core.clear_cache()
        source = None
        gc.collect()
        payload = {
            "mode": "cold",
            "common": common,
            "sample": sample,
            "measured_internal_ns": measured - total_started,
            "cleanup_ns": time.perf_counter_ns() - measured,
        }
    else:
        warmups = []
        for _ in range(options.worker_warmups):
            warmups.append(timed_lifecycle(core, source))
        samples = []
        for repeat in range(options.worker_iterations):
            sample = timed_lifecycle(core, source)
            sample["repeat"] = repeat
            samples.append(sample)
            if repeat % 16 == 15:
                gc.collect()
        measured = time.perf_counter_ns()
        core.clear_cache()
        source = None
        gc.collect()
        payload = {
            "mode": "warm",
            "common": common,
            "warmups": warmups,
            "samples": samples,
            "measured_internal_ns": measured - total_started,
            "cleanup_ns": time.perf_counter_ns() - measured,
        }

    print(json.dumps(payload, separators=(",", ":")), flush=True)
    return 0


def run_worker(
    python: Path,
    plugin: Path,
    mode: str,
    iterations: int,
    warmups: int,
    timeout: float,
) -> tuple[dict[str, Any], list[str]]:
    command = [
        str(python),
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--worker-plugin",
        str(plugin),
        "--worker-iterations",
        str(iterations),
        "--worker-warmups",
        str(warmups),
    ]
    environment = clean_cuda_environment()
    environment.update(
        DSMVC_CUDA_STREAMS="4",
        DSMVC_CUDA_INPUT_CACHE_MB="64",
        DSMVC_CUDA_PLAN_CACHE_MB="16",
    )
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    process_wall_ns = time.perf_counter_ns() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed with exit code {completed.returncode}:\n"
            f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("worker produced no JSON output")
    payload = json.loads(lines[-1])
    payload["process_wall_ns"] = process_wall_ns
    if completed.stderr:
        payload["stderr_tail"] = completed.stderr[-4000:]
    return payload, command


def run_steady_sample(
    vspipe: Path,
    plugin: Path,
    frames: int,
    streams: int | None,
    timeout: float,
) -> tuple[dict[str, Any], list[str]]:
    command = [
        str(vspipe),
        "--arg", f"plugin={plugin}",
        "--arg", f"frames={frames}",
        "--arg", "threads=32",
        "--arg", "width=7680",
        "--arg", "height=4320",
        "--arg", "src_height=3240",
        "--arg", "base_height=4000",
        "--arg", "taps=32",
        "--requests", "32",
        "--start", "0",
        "--end", str(frames - 1),
        "--filter-time",
        str(SATURATION_GRAPH),
        "--",
    ]
    environment = clean_cuda_environment()
    environment.update(
        DSMVC_CUDA_INPUT_CACHE_MB="256",
        DSMVC_CUDA_PLAN_CACHE_MB="16",
    )
    if streams is not None:
        environment["DSMVC_CUDA_STREAMS"] = str(streams)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    wall_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"VSPipe failed with exit code {completed.returncode}:\n"
            f"{completed.stdout[-6000:]}"
        )
    matches = list(VSPIPE_RESULT.finditer(completed.stdout))
    if not matches:
        raise RuntimeError(f"could not parse VSPipe result:\n{completed.stdout[-6000:]}")
    match = matches[-1]
    return {
        "frames": int(match.group("frames")),
        "reported_seconds": float(match.group("seconds")),
        "fps": float(match.group("fps")),
        "wall_seconds": wall_seconds,
        "output_tail": completed.stdout[-4000:],
    }, command


def collect_machine_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
    }
    for key, command in (
        ("git_head", ["git", "rev-parse", "HEAD"]),
        (
            "gpu",
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,power.limit",
                "--format=csv,noheader",
            ],
        ),
    ):
        try:
            metadata[key] = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            metadata[key] = None
    return metadata


def build_summaries(
    steady_samples: list[dict[str, Any]],
    cold_samples: list[dict[str, Any]],
    warm_samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []

    def append_groups(
        samples: list[dict[str, Any]],
        phase: str,
        metrics: tuple[tuple[str, str], ...],
    ) -> None:
        keys = sorted({(sample["case"], sample["variant"]) for sample in samples})
        for case, variant in keys:
            selected = [
                sample for sample in samples
                if sample["case"] == case and sample["variant"] == variant
            ]
            for metric, unit in metrics:
                values = [float(sample[metric]) for sample in selected]
                summaries.append({
                    "phase": phase,
                    "case": case,
                    "variant": variant,
                    "metric": metric,
                    "unit": unit,
                    **summary(values),
                })

    append_groups(
        steady_samples,
        "steady",
        (("fps", "frames/s"), ("wall_seconds", "s")),
    )
    lifecycle_metrics = (
        ("plugin_load_ns", "ns"),
        ("filter_create_ns", "ns"),
        ("first_frame_ns", "ns"),
        ("filter_destroy_ns", "ns"),
    )
    append_groups(
        cold_samples,
        "cold",
        lifecycle_metrics + (("process_wall_ns", "ns"),),
    )
    append_groups(warm_samples, "warm", lifecycle_metrics[1:])
    return summaries


def write_csv(
    path: Path,
    steady_samples: list[dict[str, Any]],
    cold_samples: list[dict[str, Any]],
    warm_samples: list[dict[str, Any]],
) -> None:
    rows = []
    for phase, samples in (
        ("steady", steady_samples),
        ("cold", cold_samples),
        ("warm", warm_samples),
    ):
        for sample in samples:
            for key, value in sample.items():
                if key in {"case", "variant", "repeat", "block"}:
                    continue
                if not isinstance(value, (int, float)):
                    continue
                unit = "ns" if key.endswith("_ns") else (
                    "frames/s" if key == "fps" else "s" if key.endswith("seconds") else "count"
                )
                rows.append({
                    "phase": phase,
                    "case": sample["case"],
                    "variant": sample["variant"],
                    "block": sample.get("block", ""),
                    "repeat": sample.get("repeat", ""),
                    "metric": key,
                    "value": value,
                    "unit": unit,
                })
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "phase", "case", "variant", "block", "repeat",
                "metric", "value", "unit",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def find_summary(
    summaries: list[dict[str, Any]],
    phase: str,
    case: str,
    variant: str,
    metric: str,
) -> dict[str, Any]:
    return next(
        item for item in summaries
        if item["phase"] == phase and item["case"] == case
        and item["variant"] == variant and item["metric"] == metric
    )


def write_markdown(
    path: Path,
    variants: list[tuple[str, Path]],
    summaries: list[dict[str, Any]],
    steady_enabled: bool,
    cold_enabled: bool,
    warm_enabled: bool,
) -> None:
    labels = [label for label, _ in variants]
    baseline = labels[0]
    lines = [
        "# CUDA refactor benchmark",
        "",
        "The steady workload loops one 8K GRAY8 frame through Lanczos-32. "
        "Its input cache removes repeated upload/transpose traffic; the output "
        "is discarded by VSPipe. The `default` case intentionally leaves "
        "`DSMVC_CUDA_STREAMS` unset, while the other cases normalize it to 8 "
        "or 16.",
        "",
        "Cold lifecycle launches a fresh Python/VapourSynth process per sample. "
        "Warm lifecycle repeatedly creates, renders, and releases a small "
        "bilinear CUDA filter in one process. Both lifecycle modes force four "
        "streams so slot policy is not part of the measurement.",
        "",
    ]
    if steady_enabled:
        lines.extend([
            "## Steady GPU throughput",
            "",
            f"Ratios use `{baseline}` as 1.000; higher FPS is better.",
            "",
            "| Case | Variant | Median FPS | P95 FPS | Ratio |",
            "|---|---|---:|---:|---:|",
        ])
        for case in ("gpu_default", "gpu_streams8", "gpu_streams16"):
            base = find_summary(summaries, "steady", case, baseline, "fps")["median"]
            for label in labels:
                item = find_summary(summaries, "steady", case, label, "fps")
                lines.append(
                    f"| `{case}` | `{label}` | {item['median']:.2f} | "
                    f"{item['p95']:.2f} | {item['median'] / base:.3f}x |"
                )
        lines.append("")

    def lifecycle_table(phase: str, title: str, metrics: tuple[str, ...]) -> None:
        lines.extend([
            f"## {title}",
            "",
            f"Ratios use `{baseline}` as 1.000; lower time is better.",
            "",
            "| Metric | Variant | Median (ms) | P95 (ms) | Ratio |",
            "|---|---|---:|---:|---:|",
        ])
        for metric in metrics:
            base = find_summary(
                summaries, phase, "lifecycle", baseline, metric)["median"]
            for label in labels:
                item = find_summary(
                    summaries, phase, "lifecycle", label, metric)
                ratio = item["median"] / base if base else float("nan")
                lines.append(
                    f"| `{metric.removesuffix('_ns')}` | `{label}` | "
                    f"{item['median'] / 1e6:.3f} | {item['p95'] / 1e6:.3f} | "
                    f"{ratio:.3f}x |"
                )
        lines.append("")

    if cold_enabled:
        lifecycle_table(
            "cold",
            "Cold process lifecycle",
            (
                "plugin_load_ns", "filter_create_ns", "first_frame_ns",
                "filter_destroy_ns", "process_wall_ns",
            ),
        )
    if warm_enabled:
        lifecycle_table(
            "warm",
            "Warm filter churn",
            ("filter_create_ns", "first_frame_ns", "filter_destroy_ns"),
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_options(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        type=parse_variant,
        required=True,
        help="variant as LABEL=/path/to/dsmvc.so; repeat for paired runs",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable from the VapourSynth environment",
    )
    parser.add_argument(
        "--vspipe",
        type=Path,
        default=Path("vspipe"),
        help="VSPipe executable",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark-results" / "cuda-refactor",
    )
    parser.add_argument("--steady-frames", type=int, default=320)
    parser.add_argument("--steady-warmups", type=int, default=1)
    parser.add_argument("--steady-repeats", type=int, default=3)
    parser.add_argument("--cold-repeats", type=int, default=15)
    parser.add_argument("--warm-repeats", type=int, default=60)
    parser.add_argument("--warm-blocks", type=int, default=2)
    parser.add_argument("--warm-worker-warmups", type=int, default=2)
    parser.add_argument("--lifecycle-timeout", type=float, default=120.0)
    parser.add_argument("--steady-timeout", type=float, default=600.0)
    parser.add_argument("--skip-steady", action="store_true")
    parser.add_argument("--skip-cold", action="store_true")
    parser.add_argument("--skip-warm", action="store_true")
    options = parser.parse_args(arguments)
    if len(options.variant) < 2:
        parser.error("at least two --variant values are required")
    labels = [label for label, _ in options.variant]
    if len(labels) != len(set(labels)):
        parser.error("variant labels must be unique")
    for path, label in ((options.python, "Python"), (options.vspipe, "VSPipe")):
        resolved = Path(shutil_which(path)) if not path.is_absolute() else path
        if not resolved.is_file():
            parser.error(f"{label} executable does not exist: {path}")
        setattr(options, label.lower(), resolved.resolve())
    for value, label in (
        (options.steady_frames, "steady frames"),
        (options.steady_repeats, "steady repeats"),
        (options.cold_repeats, "cold repeats"),
        (options.warm_repeats, "warm repeats"),
        (options.warm_blocks, "warm blocks"),
    ):
        if value < 1:
            parser.error(f"{label} must be positive")
    if options.steady_warmups < 0 or options.warm_worker_warmups < 0:
        parser.error("warmup counts cannot be negative")
    return options


def shutil_which(path: Path) -> str:
    if path.parent != Path("."):
        return str(path)
    import shutil
    resolved = shutil.which(str(path))
    return resolved or str(path)


def main(arguments: list[str]) -> int:
    if "--worker-mode" in arguments:
        return worker_main(arguments)
    options = parse_options(arguments)
    options.output.mkdir(parents=True, exist_ok=True)
    commands: list[str] = []
    steady_samples: list[dict[str, Any]] = []
    cold_samples: list[dict[str, Any]] = []
    warm_samples: list[dict[str, Any]] = []
    variants = options.variant

    if not options.skip_cold:
        for repeat in range(options.cold_repeats):
            ordered = variants if repeat % 2 == 0 else list(reversed(variants))
            for label, plugin in ordered:
                payload, command = run_worker(
                    options.python,
                    plugin,
                    "cold",
                    1,
                    0,
                    options.lifecycle_timeout,
                )
                commands.append(shlex.join(command))
                sample = {
                    "case": "lifecycle",
                    "variant": label,
                    "repeat": repeat,
                    **payload["common"],
                    **payload["sample"],
                    "measured_internal_ns": payload["measured_internal_ns"],
                    "cleanup_ns": payload["cleanup_ns"],
                    "process_wall_ns": payload["process_wall_ns"],
                }
                cold_samples.append(sample)
                print(
                    f"cold {repeat + 1}/{options.cold_repeats} {label}: "
                    f"create={sample['filter_create_ns'] / 1e6:.2f} ms "
                    f"first={sample['first_frame_ns'] / 1e6:.2f} ms",
                    flush=True,
                )

    if not options.skip_warm:
        per_block = math.ceil(options.warm_repeats / options.warm_blocks)
        remaining = {label: options.warm_repeats for label, _ in variants}
        for block in range(options.warm_blocks):
            ordered = variants if block % 2 == 0 else list(reversed(variants))
            for label, plugin in ordered:
                count = min(per_block, remaining[label])
                if count == 0:
                    continue
                payload, command = run_worker(
                    options.python,
                    plugin,
                    "warm",
                    count,
                    options.warm_worker_warmups,
                    options.lifecycle_timeout,
                )
                commands.append(shlex.join(command))
                first_repeat = options.warm_repeats - remaining[label]
                for offset, worker_sample in enumerate(payload["samples"]):
                    warm_samples.append({
                        "case": "lifecycle",
                        "variant": label,
                        "block": block,
                        "repeat": first_repeat + offset,
                        **worker_sample,
                    })
                remaining[label] -= count
                print(
                    f"warm block {block + 1}/{options.warm_blocks} {label}: "
                    f"{count} samples",
                    flush=True,
                )

    if not options.skip_steady:
        steady_cases = (
            ("gpu_default", None),
            ("gpu_streams8", 8),
            ("gpu_streams16", 16),
        )
        for case, streams in steady_cases:
            for warmup in range(options.steady_warmups):
                ordered = variants if warmup % 2 == 0 else list(reversed(variants))
                for label, plugin in ordered:
                    _, command = run_steady_sample(
                        options.vspipe,
                        plugin,
                        options.steady_frames,
                        streams,
                        options.steady_timeout,
                    )
                    commands.append(shlex.join(command))
                    print(f"steady warmup {case} {label}", flush=True)
            for repeat in range(options.steady_repeats):
                ordered = variants if repeat % 2 == 0 else list(reversed(variants))
                for label, plugin in ordered:
                    sample, command = run_steady_sample(
                        options.vspipe,
                        plugin,
                        options.steady_frames,
                        streams,
                        options.steady_timeout,
                    )
                    commands.append(shlex.join(command))
                    sample.update(
                        case=case,
                        variant=label,
                        repeat=repeat,
                        streams=streams if streams is not None else "default",
                    )
                    steady_samples.append(sample)
                    print(
                        f"steady {case} {repeat + 1}/{options.steady_repeats} "
                        f"{label}: {sample['fps']:.2f} fps",
                        flush=True,
                    )

    summaries = build_summaries(steady_samples, cold_samples, warm_samples)
    result = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "machine": collect_machine_metadata(),
        "variants": [
            {"label": label, "plugin": str(plugin)}
            for label, plugin in variants
        ],
        "configuration": {
            "steady_frames": options.steady_frames,
            "steady_warmups": options.steady_warmups,
            "steady_repeats": options.steady_repeats,
            "cold_repeats": options.cold_repeats,
            "warm_repeats": options.warm_repeats,
            "warm_blocks": options.warm_blocks,
            "lifecycle_streams": 4,
            "steady_input_cache_mb": 256,
        },
        "steady_samples": steady_samples,
        "cold_samples": cold_samples,
        "warm_samples": warm_samples,
        "summaries": summaries,
        "commands": commands,
    }
    (options.output / "benchmark.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(
        options.output / "benchmark.csv",
        steady_samples,
        cold_samples,
        warm_samples,
    )
    write_markdown(
        options.output / "benchmark.md",
        variants,
        summaries,
        not options.skip_steady,
        not options.skip_cold,
        not options.skip_warm,
    )
    (options.output / "commands.txt").write_text(
        "\n".join(commands) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {options.output / 'benchmark.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
