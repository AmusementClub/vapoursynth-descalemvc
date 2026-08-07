#!/usr/bin/env python3
"""Run alternating control/candidate measurements on the getfnative graph."""

from __future__ import annotations

import argparse
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

from analyze_metal_routes import paired_bootstrap_speedup
from e2e_benchmark import CASES, PROFILES, RECIPE_FACTS, recipe_candidates


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


def command_text(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([item]) for item in command)


def build_command(options: argparse.Namespace, plugin: Path, backend: str,
                  case: str, requests: int) -> list[str]:
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
             backend: str, case: str, sample: int, warmup: bool) -> dict:
    command = build_command(
        options, plugin, backend, case, options.requests)
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
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"VSPipe failed for {case}/{variant}/sample-{sample}:\n"
            f"{output[-12000:]}")
    candidate_count = len(recipe_candidates(case, options.profile))
    return {
        "case": case,
        "variant": variant,
        "backend": backend,
        "sample": sample,
        "warmup": warmup,
        "candidate_count": candidate_count,
        "elapsed_seconds": elapsed_seconds,
        "candidates_per_second": candidate_count / elapsed_seconds,
        "command": command_text(command),
        "vspipe_output_tail": output[-3000:],
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
            "candidates_per_second": summarize(
                [item["candidates_per_second"] for item in items]),
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
        for sample_number in sorted({item["sample"] for item in case_samples}):
            elapsed = {
                item["variant"]: item["elapsed_seconds"]
                for item in case_samples if item["sample"] == sample_number
            }
            control_elapsed.append(elapsed["control"])
            candidate_elapsed.append(elapsed["candidate"])
            paired_speedups.append(elapsed["control"] / elapsed["candidate"])
        comparisons.append({
            "case": case,
            "candidate_count": len(recipe_candidates(case, options.profile)),
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
        })
    return summaries, comparisons


def run(options: argparse.Namespace) -> dict:
    variants = {
        "control": (options.control_plugin, options.control_backend),
        "candidate": (options.candidate_plugin, options.candidate_backend),
    }
    samples = []
    for case in options.cases:
        for sample in range(-options.warmups, options.samples):
            order = ("control", "candidate") if sample % 2 == 0 else (
                "candidate", "control")
            for variant in order:
                plugin, backend = variants[variant]
                result = run_once(
                    options, variant, plugin, backend, case, sample,
                    sample < 0)
                if sample >= 0:
                    samples.append(result)
                print(
                    f"{case}/{variant}/{sample}: "
                    f"{result['candidates_per_second']:.3f} candidates/s",
                    flush=True)
    summaries, comparisons = aggregate(samples, options)
    return {
        "schema": "dsmvc-getfnative-plugin-ab-v2",
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
            "memory_concurrency": os.environ.get("DSMVC_MEMORY_CONCURRENCY")
            if options.memory_concurrency is None
            else str(options.memory_concurrency),
            "control_plugin": str(options.control_plugin),
            "control_plugin_sha256": sha256_file(options.control_plugin),
            "candidate_plugin": str(options.candidate_plugin),
            "candidate_plugin_sha256": sha256_file(options.candidate_plugin),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "vpy_sha256": sha256_file(
                Path(__file__).with_name("vspipe_e2e.vpy")),
        },
        "summaries": summaries,
        "comparisons": comparisons,
        "samples": samples,
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
    result.add_argument("--samples", type=int, default=21)
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument("--bootstrap-resamples", type=int, default=50000)
    result.add_argument("--bootstrap-seed", type=lambda value: int(value, 0),
                        default=0x44534D56)
    result.add_argument("--memory-concurrency", type=int, default=None)
    result.add_argument("--json-out", required=True, type=Path)
    return result


if __name__ == "__main__":
    options = parser().parse_args()
    options.control_backend = options.control_backend or options.backend
    options.candidate_backend = options.candidate_backend or options.backend
    for name in ("control_plugin", "candidate_plugin", "vspipe", "source",
                 "source_plugin", "json_out"):
        setattr(options, name, getattr(options, name).expanduser().resolve())
    for path in (options.control_plugin, options.candidate_plugin,
                 options.vspipe, options.source, options.source_plugin):
        if not path.is_file():
            raise FileNotFoundError(path)
    if options.samples < 1 or options.warmups < 0:
        raise ValueError("samples must be positive; warmups nonnegative")
    if options.requests < 1 or options.threads < 1:
        raise ValueError("requests and threads must be positive")
    if options.bootstrap_resamples < 1000:
        raise ValueError("bootstrap resamples must be at least 1000")
    if options.memory_concurrency is not None and options.memory_concurrency < 0:
        raise ValueError("memory concurrency cannot be negative")
    output = run(options)
    options.json_out.parent.mkdir(parents=True, exist_ok=True)
    options.json_out.write_text(
        json.dumps(output, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8")
