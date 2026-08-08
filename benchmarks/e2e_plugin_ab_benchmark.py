#!/usr/bin/env python3
"""Run balanced control/candidate measurements on the GetFnative graph."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from analyze_metal_routes import paired_bootstrap_speedup
from e2e_benchmark import CASES, PROFILES, RECIPE_FACTS, recipe_candidates
from paired_benchmark_support import (
    ResourceMonitor,
    ResourcePressureError,
    command_text,
    load_frame_properties,
    paired_analysis,
    parse_vspipe_timing,
    sha256_file,
    summarize,
    summarize_frame_properties,
)


def build_command(options: argparse.Namespace, plugin: Path, backend: str,
                  case: str, properties_path: Path) -> list[str]:
    script = Path(__file__).with_name("vspipe_e2e.vpy").resolve()
    candidate_count = len(recipe_candidates(case, options.profile))
    values = {
        "implementation": "new",
        "case": case,
        "profile": options.profile,
        "source": str(options.source),
        "plugin": str(plugin),
        "old_plugin": "",
        "source_plugin": str(options.source_plugin),
        "source_filter": options.source_filter,
        "source_decoder": options.source_decoder,
        "source_prefer_hw": str(options.source_prefer_hw),
        "source_ff_loglevel": str(options.source_ff_loglevel),
        "source_rap_verification": str(options.source_rap_verification),
        "backend": backend,
        "opt": str(options.opt),
        "frame": str(RECIPE_FACTS[case]["frame"]),
        "threads": str(options.threads),
    }
    command = [str(options.vspipe)]
    for key, value in values.items():
        command.extend(["--arg", f"{key}={value}"])
    command.extend([
        "--requests", str(options.requests),
        "--start", "0",
        "--end", str(candidate_count - 1),
        "--json", str(properties_path),
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
             backend: str, case: str, sample: int, warmup: bool,
             pair_order: str, monitor: ResourceMonitor) -> dict:
    candidates = recipe_candidates(case, options.profile)
    candidate_count = len(candidates)
    unique_cells = len({item["id"] for item in candidates})
    sample_label = f"warmup{-sample}" if warmup else f"sample{sample + 1}"
    properties_path = options.telemetry_dir / (
        f"{case}-{variant}-{sample_label}.json")
    command = build_command(options, plugin, backend, case, properties_path)
    before = monitor.capture(f"{case}/{variant}/{sample_label}/before")
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
    after = monitor.capture(f"{case}/{variant}/{sample_label}/after")
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        termination = (
            f"signal {-completed.returncode}" if completed.returncode < 0
            else f"exit {completed.returncode}")
        raise RuntimeError(
            f"VSPipe failed ({termination}) for "
            f"{case}/{variant}/sample-{sample}:\n{output[-12000:]}")
    timing = parse_vspipe_timing(output, candidate_count)
    frames = load_frame_properties(properties_path, candidate_count)
    return {
        "case": case,
        "variant": variant,
        "backend": backend,
        "sample": sample,
        "observation": None if warmup else sample + 1,
        "pair_order": pair_order,
        "warmup": warmup,
        "candidate_count": candidate_count,
        "unique_cell_count": unique_cells,
        "elapsed_seconds": elapsed_seconds,
        "candidates_per_second": candidate_count / elapsed_seconds,
        "vspipe_seconds": timing["seconds"],
        "vspipe_candidates_per_second": candidate_count / timing["seconds"],
        "process_overhead_seconds": max(0.0, elapsed_seconds - timing["seconds"]),
        "telemetry": summarize_frame_properties(frames),
        "frame_properties": {
            "path": str(properties_path),
            "sha256": sha256_file(properties_path),
        },
        "system_before": before,
        "system_after": after,
        "command": command_text(command),
        "vspipe_output_tail": output[-4000:],
    }


def aggregate(samples: list[dict], options: argparse.Namespace) -> tuple[
        list[dict], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[(sample["case"], sample["variant"])].append(sample)
    summaries = []
    for key, items in sorted(grouped.items()):
        summaries.append({
            "case": key[0],
            "variant": key[1],
            "runs": len(items),
            "elapsed_seconds": summarize(
                [item["elapsed_seconds"] for item in items]),
            "vspipe_seconds": summarize(
                [item["vspipe_seconds"] for item in items]),
            "candidates_per_second": summarize(
                [item["candidates_per_second"] for item in items]),
            "vspipe_candidates_per_second": summarize([
                item["vspipe_candidates_per_second"] for item in items]),
        })

    by_key = {(item["case"], item["variant"]): item for item in summaries}
    comparisons = []
    for case in options.cases:
        control = by_key[(case, "control")]
        candidate = by_key[(case, "candidate")]
        case_samples = [item for item in samples if item["case"] == case]
        control_elapsed = []
        candidate_elapsed = []
        paired_speedups = []
        for number in sorted({item["sample"] for item in case_samples}):
            elapsed = {
                item["variant"]: item["elapsed_seconds"]
                for item in case_samples if item["sample"] == number
            }
            if set(elapsed) != {"control", "candidate"}:
                raise RuntimeError(f"unpaired GetFnative sample {number}")
            control_elapsed.append(elapsed["control"])
            candidate_elapsed.append(elapsed["candidate"])
            paired_speedups.append(elapsed["control"] / elapsed["candidate"])
        recipe = recipe_candidates(case, options.profile)
        comparison = {
            "case": case,
            "candidate_count": len(recipe),
            "unique_cell_count": len({item["id"] for item in recipe}),
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
            "control_candidates_per_second": control[
                "candidates_per_second"]["median"],
            "candidate_candidates_per_second": candidate[
                "candidates_per_second"]["median"],
        }
        if len(control_elapsed) == 6:
            comparison["paired_analysis"] = paired_analysis(case_samples)
        comparisons.append(comparison)
    return summaries, comparisons


def run(options: argparse.Namespace) -> dict:
    variants = {
        "control": (options.control_plugin, options.control_backend),
        "candidate": (options.candidate_plugin, options.candidate_backend),
    }
    monitor = ResourceMonitor(options.minimum_free_percent)
    samples = []
    warmups = []
    for case in options.cases:
        run_numbers = list(range(-options.warmups, 0)) \
            + list(range(options.samples))
        for sample in run_numbers:
            warmup = sample < 0
            if warmup:
                order = ("control", "candidate")
            else:
                order = ("control", "candidate") if sample % 2 == 0 \
                    else ("candidate", "control")
            pair_order = "C-A" if order[0] == "control" else "A-C"
            for variant in order:
                plugin, backend = variants[variant]
                result = run_once(
                    options, variant, plugin, backend, case, sample, warmup,
                    pair_order, monitor)
                (warmups if warmup else samples).append(result)
                print(
                    f"{case}/{variant}/{sample}: "
                    f"{result['candidates_per_second']:.3f} wall candidates/s, "
                    f"{result['vspipe_candidates_per_second']:.3f} filter candidates/s",
                    flush=True)
    summaries, comparisons = aggregate(samples, options)
    manifest = None
    if options.build_manifest:
        manifest = json.loads(options.build_manifest.read_text(encoding="utf-8"))
    return {
        "schema": "dsmvc-getfnative-plugin-ab-v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "vspipe": str(options.vspipe),
            "source": str(options.source),
            "source_sha256": sha256_file(options.source),
            "source_filter": options.source_filter,
            "source_plugin": str(options.source_plugin),
            "source_plugin_sha256": sha256_file(options.source_plugin),
            "profile": options.profile,
            "cases": options.cases,
            "requests": options.requests,
            "threads": options.threads,
            "control_backend": options.control_backend,
            "candidate_backend": options.candidate_backend,
            "opt": options.opt,
            "samples": options.samples,
            "warmups": options.warmups,
            "minimum_free_percent": options.minimum_free_percent,
            "memory_concurrency": os.environ.get("DSMVC_MEMORY_CONCURRENCY")
            if options.memory_concurrency is None
            else str(options.memory_concurrency),
            "control_plugin": str(options.control_plugin),
            "control_plugin_sha256": sha256_file(options.control_plugin),
            "candidate_plugin": str(options.candidate_plugin),
            "candidate_plugin_sha256": sha256_file(options.candidate_plugin),
            "build_manifest": str(options.build_manifest)
            if options.build_manifest else None,
            "build_manifest_sha256": sha256_file(options.build_manifest)
            if options.build_manifest else None,
            "source_fingerprint": manifest.get("source_fingerprint")
            if isinstance(manifest, dict) else None,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "vpy_sha256": sha256_file(
                Path(__file__).with_name("vspipe_e2e.vpy")),
        },
        "warmup_samples": warmups,
        "samples": samples,
        "summaries": summaries,
        "comparisons": comparisons,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--control-plugin", required=True, type=Path)
    result.add_argument("--candidate-plugin", required=True, type=Path)
    result.add_argument("--vspipe", required=True, type=Path)
    result.add_argument("--source", required=True, type=Path)
    result.add_argument("--source-plugin", required=True, type=Path)
    result.add_argument("--source-filter", choices=("lsmas", "ffms2",
                                                      "bestsource"),
                        default="ffms2")
    result.add_argument("--source-decoder", default="")
    result.add_argument("--source-prefer-hw", type=int, default=0)
    result.add_argument("--source-ff-loglevel", type=int, default=0)
    result.add_argument("--source-rap-verification", type=int, default=-1)
    result.add_argument("--profile", choices=PROFILES, default="smoke")
    result.add_argument("--cases", nargs="+", choices=CASES,
                        default=("getfnative",))
    result.add_argument("--requests", type=int, default=16)
    result.add_argument("--threads", type=int, default=16)
    result.add_argument("--backend", choices=("cpu", "auto"), default="cpu")
    result.add_argument(
        "--control-backend", choices=("cpu", "auto", "metal"))
    result.add_argument(
        "--candidate-backend", choices=("cpu", "auto", "metal"))
    result.add_argument("--opt", choices=(0, 1, 2), type=int, default=0)
    result.add_argument("--samples", type=int, default=6)
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument("--minimum-free-percent", type=int, default=10)
    result.add_argument("--bootstrap-resamples", type=int, default=50000)
    result.add_argument("--bootstrap-seed", type=lambda value: int(value, 0),
                        default=0x44534D56)
    result.add_argument("--memory-concurrency", type=int, default=None)
    result.add_argument("--build-manifest", type=Path)
    result.add_argument("--telemetry-dir", type=Path)
    result.add_argument("--json-out", required=True, type=Path)
    return result


def main() -> int:
    options = parser().parse_args()
    options.control_backend = options.control_backend or options.backend
    options.candidate_backend = options.candidate_backend or options.backend
    for name in ("control_plugin", "candidate_plugin", "vspipe", "source",
                 "source_plugin", "json_out"):
        setattr(options, name, getattr(options, name).expanduser().resolve())
    if options.build_manifest:
        options.build_manifest = options.build_manifest.expanduser().resolve()
    options.telemetry_dir = (
        options.telemetry_dir.expanduser().resolve()
        if options.telemetry_dir else
        options.json_out.parent / f"{options.json_out.stem}-frame-properties")
    for path in (options.control_plugin, options.candidate_plugin,
                 options.vspipe, options.source, options.source_plugin):
        if not path.is_file():
            raise FileNotFoundError(path)
    if options.build_manifest and not options.build_manifest.is_file():
        raise FileNotFoundError(options.build_manifest)
    if options.samples < 1 or options.warmups < 0:
        raise ValueError("samples must be positive; warmups nonnegative")
    if options.requests < 1 or options.threads < 1:
        raise ValueError("requests and threads must be positive")
    if options.bootstrap_resamples < 1000:
        raise ValueError("bootstrap resamples must be at least 1000")
    if options.memory_concurrency is not None and options.memory_concurrency < 0:
        raise ValueError("memory concurrency cannot be negative")
    options.telemetry_dir.mkdir(parents=True, exist_ok=False)
    options.json_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        output = run(options)
    except ResourcePressureError as error:
        print(f"RESOURCE_STOP: {error}", flush=True)
        return 125
    options.json_out.write_text(
        json.dumps(output, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
