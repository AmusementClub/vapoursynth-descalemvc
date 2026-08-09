#!/usr/bin/env python3
"""Run the paired Vulkan F32 VapourSynth plugin regression gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

import blank_plugin_ab_benchmark as shared  # noqa: E402


MINIMUM_SAMPLES = 30
GRAPH = Path(__file__).with_suffix(".vpy")
DEFAULT_KERNELS = (
    "bilinear", "bicubic_b0_c0_5", "lanczos3", "spline64")


def summarize(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    median = statistics.median(ordered)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": median,
        "maximum": ordered[-1],
        "mad": statistics.median(abs(value - median) for value in ordered),
    }


def build_command(options: argparse.Namespace, plugin: Path, backend: str,
                  opt: int, kernel: str, requests: int) -> list[str]:
    values = {
        "plugin": str(plugin),
        "backend": backend,
        "opt": str(opt),
        "kernel": kernel,
        "frames": str(options.frames),
        "threads": str(requests),
        "src_height": str(options.src_height),
        "base_height": str(options.base_height),
    }
    command = [str(options.vspipe)]
    for key, value in values.items():
        command.extend(["--arg", f"{key}={value}"])
    command.extend([
        "--requests", str(requests),
        "--start", "0",
        "--end", str(options.frames - 1),
        "--filter-time", str(GRAPH),
        "--",
    ])
    return command


def hardware_preflight() -> tuple[str | None, str]:
    icd = os.environ.get("VK_ICD_FILENAMES", "")
    if not any(Path(item).name == "nvidia_icd.json" for item in icd.split(":")):
        return "VK_ICD_FILENAMES does not select nvidia_icd.json", ""
    if not Path("/dev/nvidia0").exists():
        return "/dev/nvidia0 is unavailable", ""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True,
            errors="replace", check=False, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return f"nvidia-smi preflight failed: {error}", ""
    if result.returncode != 0:
        return ("nvidia-smi could not query a GPU: "
                + (result.stdout + result.stderr).strip()), ""
    return None, (result.stdout + result.stderr).strip()


def add_gate(output: dict, maximum_regression: float) -> bool:
    samples = output["samples"]
    cases = sorted({(item["kernel"], item["requests"]) for item in samples})
    case_summaries = []
    suite_totals = {
        "control": {},
        "candidate": {},
    }
    for kernel, requests in cases:
        ratios = []
        case_samples = [
            item for item in samples
            if (item["kernel"], item["requests"]) == (kernel, requests)
        ]
        for sample_number in sorted({item["sample"] for item in case_samples}):
            elapsed = {
                item["variant"]: item["elapsed_seconds"]
                for item in case_samples if item["sample"] == sample_number
            }
            if set(elapsed) != {"control", "candidate"}:
                raise RuntimeError("plugin samples are not paired")
            ratios.append(elapsed["candidate"] / elapsed["control"])
            for variant in ("control", "candidate"):
                suite_totals[variant][sample_number] = (
                    suite_totals[variant].get(sample_number, 0.0)
                    + elapsed[variant])
        case_summaries.append({
            "kernel": kernel,
            "requests": requests,
            "candidate_over_control": summarize(ratios),
        })

    if set(suite_totals["control"]) != set(suite_totals["candidate"]):
        raise RuntimeError("plugin suite totals are not paired")
    suite_ratios = [
        suite_totals["candidate"][sample]
        / suite_totals["control"][sample]
        for sample in sorted(suite_totals["control"])
    ]
    suite_summary = summarize(suite_ratios)
    passed = float(suite_summary["median"]) <= 1.0 + maximum_regression
    output["vulkan_f32_regression_gate"] = {
        "status": "passed" if passed else "failed",
        "maximum_regression": maximum_regression,
        "suite_candidate_over_control": suite_summary,
        "cases": case_summaries,
    }
    return passed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--control-plugin", required=True, type=Path)
    result.add_argument("--candidate-plugin", required=True, type=Path)
    result.add_argument("--control-source-sha", required=True)
    result.add_argument("--candidate-source-sha", required=True)
    result.add_argument("--vspipe", required=True, type=Path)
    result.add_argument("--json-out", required=True, type=Path)
    result.add_argument("--kernels", nargs="+", choices=shared.KERNELS,
                        default=DEFAULT_KERNELS)
    result.add_argument("--requests", nargs="+", type=int, default=(16,))
    result.add_argument("--frames", type=int, default=256)
    result.add_argument("--src-height", type=float, default=810.0)
    result.add_argument("--base-height", type=float, default=1000.0)
    result.add_argument("--samples", type=int, default=MINIMUM_SAMPLES)
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument("--maximum-regression", type=float, default=0.03)
    result.add_argument("--bootstrap-resamples", type=int, default=50000)
    result.add_argument("--bootstrap-seed", type=lambda value: int(value, 0),
                        default=0x564B4633)
    result.add_argument("--memory-concurrency", type=int)
    return result


def main() -> int:
    options = parser().parse_args()
    for name in ("control_plugin", "candidate_plugin", "vspipe", "json_out"):
        setattr(options, name, getattr(options, name).expanduser().resolve())
    for plugin in (options.control_plugin, options.candidate_plugin):
        if not plugin.is_file():
            raise FileNotFoundError(plugin)
    if not options.vspipe.is_file():
        raise FileNotFoundError(options.vspipe)
    if options.samples < MINIMUM_SAMPLES:
        raise ValueError("--samples must be at least 30")
    if options.frames < 1 or options.warmups < 0:
        raise ValueError("frames must be positive and warmups nonnegative")
    if any(request < 1 for request in options.requests):
        raise ValueError("requests must be positive")
    if not 0.0 <= options.maximum_regression < 1.0:
        raise ValueError("--maximum-regression must be in [0, 1)")
    if options.bootstrap_resamples < 1000:
        raise ValueError("--bootstrap-resamples must be at least 1000")

    reason, nvidia_smi = hardware_preflight()
    if reason is not None:
        output = {
            "schema": "dsmvc-vulkan-f32-plugin-ab-v1",
            "status": "skipped",
            "reason": reason,
        }
        returncode = 77
    else:
        options.control_backend = "vulkan"
        options.candidate_backend = "vulkan"
        options.control_opt = 0
        options.candidate_opt = 0
        shared.build_command = build_command
        output = shared.run(options)
        output["schema"] = "dsmvc-vulkan-f32-plugin-ab-v1"
        output["environment"].update({
            "control_source_sha": options.control_source_sha,
            "candidate_source_sha": options.candidate_source_sha,
            "VK_ICD_FILENAMES": os.environ.get("VK_ICD_FILENAMES"),
            "DSMVC_VULKAN_DEVICE": os.environ.get("DSMVC_VULKAN_DEVICE"),
            "nvidia_smi": nvidia_smi,
            "precision": "f32",
            "f64mode": 1,
            "wrapper_sha256": shared.sha256_file(Path(__file__).resolve()),
            "vpy": str(GRAPH),
            "vpy_sha256": shared.sha256_file(GRAPH),
        })
        returncode = 0 if add_gate(
            output, options.maximum_regression) else 2

    options.json_out.parent.mkdir(parents=True, exist_ok=True)
    options.json_out.write_text(
        json.dumps(output, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=True))
    return returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
