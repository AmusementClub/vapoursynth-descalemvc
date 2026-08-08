"""Shared evidence helpers for bounded dsmvc paired E2E benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_RE = re.compile(
    r"Output (?P<frames>\d+) frames in (?P<seconds>[0-9.]+) seconds "
    r"\((?P<fps>[0-9.]+) fps\)"
)

SUBMISSION_METRICS = {
    "staging_memcpy_calls": "_DSMVCMetalStagingCopies",
    "staging_copied_bytes": "_DSMVCMetalStagingBytes",
    "unique_input_planes": "_DSMVCMetalUniqueInputs",
    "resident_producers": "_DSMVCMetalResidentProducers",
    "resident_hits": "_DSMVCMetalResidentHits",
    "resident_evictions": "_DSMVCMetalResidentEvictions",
    "eliminated_staging_bytes": "_DSMVCMetalEliminatedStagingBytes",
}

CPU_PACKING_METRICS = {
    "pack_executions": "_DSMVCCpuPlanPackExecutions",
    "single_flight_waits": "_DSMVCCpuPlanPackWaits",
    "single_flight_wait_nanoseconds": "_DSMVCCpuPlanPackWaitNs",
    "lazy_requests": "_DSMVCCpuPlanLazyRequests",
    "lazy_hits": "_DSMVCCpuPlanLazyHits",
    "maximum_concurrent_packs": "_DSMVCCpuPlanMaxConcurrentPacks",
}


class ResourcePressureError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def command_text(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([item]) for item in command)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)
    median = statistics.median(ordered)
    return {
        "minimum": ordered[0],
        "median": median,
        "maximum": ordered[-1],
        "mad": statistics.median(abs(value - median) for value in ordered),
    }


def parse_vspipe_timing(output: str, expected_frames: int) -> dict:
    match = OUTPUT_RE.search(output)
    if match is None:
        raise RuntimeError("VSPipe output did not contain an Output timing line")
    frames = int(match.group("frames"))
    if frames != expected_frames:
        raise RuntimeError(
            f"VSPipe emitted {frames} frames; expected {expected_frames}")
    return {
        "frames": frames,
        "seconds": float(match.group("seconds")),
        "fps": float(match.group("fps")),
    }


def _command_output(command: list[str]) -> dict:
    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace", check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "output": (completed.stdout + completed.stderr).strip(),
    }


def _top_processes() -> list[dict]:
    result = _command_output(["ps", "-A", "-o", "pid=,pcpu=,pmem=,comm="])
    processes = []
    if result["returncode"] == 0:
        for line in result["output"].splitlines():
            parts = line.strip().split(None, 3)
            if len(parts) != 4:
                continue
            try:
                processes.append({
                    "pid": int(parts[0]),
                    "cpu_percent": float(parts[1]),
                    "memory_percent": float(parts[2]),
                    "command": parts[3],
                })
            except ValueError:
                continue
    return sorted(
        processes, key=lambda item: item["cpu_percent"], reverse=True,
    )[:12]


def capture_system_state(label: str) -> dict:
    pressure = _command_output(["memory_pressure", "-Q"])
    vm_stat = _command_output(["vm_stat"])
    therm = _command_output(["pmset", "-g", "therm"])
    swap_usage = _command_output(["sysctl", "-n", "vm.swapusage"])
    free_match = re.search(
        r"System-wide memory free percentage:\s*(\d+)%", pressure["output"],
    )
    swap_match = re.search(r"(?m)^Swapouts:\s*(\d+)\.?", vm_stat["output"])
    load_average = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    return {
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "memory_free_percent": int(free_match.group(1)) if free_match else None,
        "swapouts": int(swap_match.group(1)) if swap_match else None,
        "load_average": load_average,
        "memory_pressure": pressure,
        "vm_stat": vm_stat,
        "swap_usage": swap_usage,
        "pmset_therm": therm,
        "top_processes": _top_processes(),
    }


class ResourceMonitor:
    def __init__(self, minimum_free_percent: int = 10) -> None:
        if not 0 <= minimum_free_percent <= 100:
            raise ValueError("minimum free percentage must be in [0, 100]")
        self.minimum_free_percent = minimum_free_percent
        self.initial_swapouts: int | None = None

    def capture(self, label: str) -> dict:
        state = capture_system_state(label)
        free_percent = state["memory_free_percent"]
        swapouts = state["swapouts"]
        if free_percent is None or swapouts is None:
            raise ResourcePressureError(
                f"resource telemetry is unreadable at {label}")
        if self.initial_swapouts is None:
            self.initial_swapouts = swapouts
        if free_percent < self.minimum_free_percent:
            raise ResourcePressureError(
                f"memory free percentage {free_percent} is below "
                f"{self.minimum_free_percent} at {label}")
        if swapouts > self.initial_swapouts:
            raise ResourcePressureError(
                f"swapouts grew from {self.initial_swapouts} to {swapouts} "
                f"at {label}")
        return state


def load_frame_properties(path: Path, expected_frames: int) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, list) or len(document) != expected_frames:
        raise RuntimeError(
            f"VSPipe property JSON has {len(document) if isinstance(document, list) else 'invalid'} "
            f"frames; expected {expected_frames}")
    if not all(isinstance(item, dict) for item in document):
        raise RuntimeError("VSPipe property JSON contains a non-object frame")
    return document


def summarize_frame_properties(frames: list[dict]) -> dict:
    batch_frames: dict[str, int] = {}
    batch_submissions: dict[str, float] = {}
    normalized = {name: 0.0 for name in SUBMISSION_METRICS}
    cpu_sum = {name: 0 for name in CPU_PACKING_METRICS}
    cpu_max = {name: 0 for name in CPU_PACKING_METRICS}
    metal_frames = 0
    metal_property_frames = 0
    submission_equivalents = 0.0
    gpu_interval_weighted = 0.0
    submission_gap_weighted = 0.0
    maximum_gpu_interval = 0
    maximum_submission_gap = 0
    maximum_resident_bytes = 0
    diagnostic_maxima = {
        "resident_pinned_eviction_blocks": 0,
        "metal_errors": 0,
        "consecutive_metal_errors": 0,
        "maximum_consecutive_metal_errors": 0,
    }
    diagnostic_properties = {
        "resident_pinned_eviction_blocks": "_DSMVCMetalResidentPinnedBlocks",
        "metal_errors": "_DSMVCMetalErrors",
        "consecutive_metal_errors": "_DSMVCMetalConsecutiveErrors",
        "maximum_consecutive_metal_errors": "_DSMVCMetalMaxConsecutiveErrors",
    }

    for frame in frames:
        for name, prop in CPU_PACKING_METRICS.items():
            value = int(frame.get(prop, 0))
            cpu_sum[name] += value
            cpu_max[name] = max(cpu_max[name], value)

        if "_DSMVCMetal" not in frame:
            continue
        metal_property_frames += 1
        batch = int(frame.get("_DSMVCMetalBatch", -1))
        marker = int(frame.get("_DSMVCMetal", -1))
        if batch < 0 or marker not in (0, 1) or marker != (1 if batch else 0):
            raise RuntimeError("inconsistent Metal route properties")
        key = str(batch)
        batch_frames[key] = batch_frames.get(key, 0) + 1
        if batch == 0:
            continue
        metal_frames += 1
        weight = 1.0 / batch
        submission_equivalents += weight
        batch_submissions[key] = batch_submissions.get(key, 0.0) + weight
        for name, prop in SUBMISSION_METRICS.items():
            value = int(frame.get(prop, -1))
            if value < 0:
                raise RuntimeError(f"missing Metal telemetry property {prop}")
            normalized[name] += value * weight
        resident_bytes = int(frame.get("_DSMVCMetalResidentBytes", -1))
        gpu_interval = int(frame.get("_DSMVCMetalGpuIntervalNs", -1))
        submission_gap = int(frame.get("_DSMVCMetalSubmissionGapNs", -1))
        if resident_bytes < 0 or gpu_interval < 0 or submission_gap < 0:
            raise RuntimeError("missing resident or timing telemetry properties")
        maximum_resident_bytes = max(maximum_resident_bytes, resident_bytes)
        maximum_gpu_interval = max(maximum_gpu_interval, gpu_interval)
        maximum_submission_gap = max(maximum_submission_gap, submission_gap)
        gpu_interval_weighted += gpu_interval * weight
        submission_gap_weighted += submission_gap * weight
        for name, prop in diagnostic_properties.items():
            value = int(frame.get(prop, -1))
            if value < 0:
                raise RuntimeError(f"missing Metal diagnostic property {prop}")
            diagnostic_maxima[name] = max(diagnostic_maxima[name], value)

    divisor = submission_equivalents if submission_equivalents else 1.0
    return {
        "frame_count": len(frames),
        "metal_property_frames": metal_property_frames,
        "metal_frames": metal_frames,
        "cpu_frames": len(frames) - metal_frames,
        "metal_route_fraction": metal_frames / len(frames) if frames else 0.0,
        "batch_distribution_frames": batch_frames,
        "batch_distribution_submission_equivalents": batch_submissions,
        "submission_equivalents": submission_equivalents,
        "submission_normalized_totals": normalized,
        "maximum_resident_bytes": maximum_resident_bytes,
        "gpu_interval_nanoseconds": {
            "submission_weighted_mean": gpu_interval_weighted / divisor,
            "maximum": maximum_gpu_interval,
        },
        "submission_gap_nanoseconds": {
            "submission_weighted_mean": submission_gap_weighted / divisor,
            "maximum": maximum_submission_gap,
        },
        "cpu_plan_packing": {
            "sum_of_frame_snapshots": cpu_sum,
            "maximum_frame_snapshot": cpu_max,
        },
        "diagnostic_maxima": diagnostic_maxima,
    }


def _linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    xs = list(range(1, len(values) + 1))
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(values)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    return sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, values)
    ) / denominator


def _metric_analysis(pairs: list[dict], prefix: str) -> dict:
    ratios = [item[f"{prefix}_ratio"] for item in pairs]
    controls = [item[f"control_{prefix}_seconds"] for item in pairs]
    candidates = [item[f"candidate_{prefix}_seconds"] for item in pairs]
    by_order = {
        order: [item[f"{prefix}_ratio"] for item in pairs
                if item["order"] == order]
        for order in ("C-A", "A-C")
    }
    phases = {
        "cold": ratios[:1],
        "warming": ratios[1:3],
        "sustained": ratios[3:],
    }
    return {
        "paired_ratio": summarize(ratios),
        "ratio_of_medians": statistics.median(controls)
        / statistics.median(candidates),
        "observations": [
            {"observation": item["observation"],
             "ratio": item[f"{prefix}_ratio"]}
            for item in pairs
        ],
        "later_observations": summarize(ratios[3:]),
        "phase": {name: summarize(values) for name, values in phases.items()},
        "order_strata": {
            name: summarize(values) for name, values in by_order.items()
        },
        "run_index_ratio_slope": _linear_slope(ratios),
        "order_effect_median_difference": (
            statistics.median(by_order["C-A"])
            - statistics.median(by_order["A-C"])
        ),
    }


def paired_analysis(samples: list[dict]) -> dict:
    pairs = []
    sample_numbers = sorted({int(item["sample"]) for item in samples})
    for sample_number in sample_numbers:
        items = [item for item in samples if int(item["sample"]) == sample_number]
        by_variant = {item["variant"]: item for item in items}
        if set(by_variant) != {"control", "candidate"}:
            raise RuntimeError(f"sample {sample_number} is not a complete pair")
        control = by_variant["control"]
        candidate = by_variant["candidate"]
        orders = {item["pair_order"] for item in items}
        if len(orders) != 1:
            raise RuntimeError(f"sample {sample_number} has inconsistent order")
        pairs.append({
            "observation": sample_number + 1,
            "order": orders.pop(),
            "control_wall_seconds": control["elapsed_seconds"],
            "candidate_wall_seconds": candidate["elapsed_seconds"],
            "wall_ratio": control["elapsed_seconds"]
            / candidate["elapsed_seconds"],
            "control_filter_seconds": control["vspipe_seconds"],
            "candidate_filter_seconds": candidate["vspipe_seconds"],
            "filter_ratio": control["vspipe_seconds"]
            / candidate["vspipe_seconds"],
        })
    if len(pairs) != 6:
        raise RuntimeError(f"paired gate requires exactly six pairs, got {len(pairs)}")
    expected = ["C-A", "A-C", "C-A", "A-C", "C-A", "A-C"]
    if [item["order"] for item in pairs] != expected:
        raise RuntimeError("paired gate order is not the required balanced sequence")
    return {
        "pairs": pairs,
        "wall": _metric_analysis(pairs, "wall"),
        "filter": _metric_analysis(pairs, "filter"),
        "outlier_policy": "all successful runs retained",
    }
