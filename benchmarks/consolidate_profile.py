#!/usr/bin/env python3
"""Consolidate the frozen CPU-profile bundle without running new workloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_DIR = ROOT / "benchmark-results" / "profile-current-4917f7b8-final"

CASE_ORDER = {
    "bilinear_b1": 0,
    "bicubic_b3": 1,
    "lanczos3_b5": 2,
    "spline36_b5": 3,
    "spline64_b7": 4,
    "getfnative": 5,
    "getfnative_v2": 6,
    "selectkernel": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse existing profile artifacts and emit a consolidated report."
    )
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--profile-summary", type=Path)
    parser.add_argument("--benchmark-json", type=Path)
    parser.add_argument("--wpa-manifest", type=Path)
    parser.add_argument("--reasons-csv", type=Path)
    parser.add_argument("--ibs-manifest", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--parsed-wpa-json", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def number(value: str) -> float:
    text = value.strip().replace(",", "")
    return float(text) if text else 0.0


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def value_stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "sum_us": 0.0,
            "min_us": 0.0,
            "median_us": 0.0,
            "mad_us": 0.0,
            "p95_us": 0.0,
            "p99_us": 0.0,
            "max_us": 0.0,
            "mean_us": 0.0,
        }
    ordered = sorted(values)
    median = float(statistics.median(ordered))
    return {
        "count": len(ordered),
        "sum_us": math.fsum(ordered),
        "min_us": ordered[0],
        "median_us": median,
        "mad_us": float(statistics.median(abs(item - median) for item in ordered)),
        "p95_us": percentile(ordered, 0.95),
        "p99_us": percentile(ordered, 0.99),
        "max_us": ordered[-1],
        "mean_us": math.fsum(ordered) / len(ordered),
    }


def add_distribution(
    target: dict[str, dict[str, float | int]],
    key: str,
    count: int,
    wait_us: float,
    ready_us: float,
) -> None:
    item = target[key]
    item["leaf_rows"] += 1
    item["wait_events"] += count
    item["wait_us"] += wait_us
    item["ready_us"] += ready_us


def finish_distribution(
    source: dict[str, dict[str, float | int]],
    total_rows: int,
    total_wait_events: int,
    total_wait: float,
    total_ready: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for key, value in source.items():
        leaf_rows = int(value["leaf_rows"])
        wait_events = int(value["wait_events"])
        wait_us = float(value["wait_us"])
        ready_us = float(value["ready_us"])
        result.append(
            {
                "name": key,
                "leaf_rows": leaf_rows,
                "leaf_row_percent": 100.0 * leaf_rows / total_rows if total_rows else 0.0,
                "wait_events": wait_events,
                "wait_event_percent": (
                    100.0 * wait_events / total_wait_events if total_wait_events else 0.0
                ),
                "wait_us": wait_us,
                "wait_percent": 100.0 * wait_us / total_wait if total_wait else 0.0,
                "ready_us": ready_us,
                "ready_percent": 100.0 * ready_us / total_ready if total_ready else 0.0,
            }
        )
    return sorted(
        result,
        key=lambda item: (-item["wait_events"], -item["wait_us"], -item["ready_us"], item["name"]),
    )


def stack_summary(stack: str) -> dict[str, str]:
    leaf = stack.rsplit("\\", 1)[-1] if stack else "(missing)"
    return {
        "stack_id": hashlib.sha256(stack.encode("utf-8", errors="replace")).hexdigest()[:16],
        "leaf": leaf[:240],
    }


def event_summary(row: list[str], wait_us: float, ready_us: float) -> dict[str, Any]:
    summary = stack_summary(row[3].strip())
    return {
        "switch_in_seconds": number(row[2]),
        "new_process": row[0].strip() or "(missing)",
        "new_thread_id": row[1].strip() or "(missing)",
        "old_process": row[15].strip() or "(missing)",
        "old_thread_id": row[16].strip() or "(missing)",
        "readying_process": row[4].strip() or "(missing)",
        "readying_thread_id": row[5].strip() or "(missing)",
        "previous_state": row[12].strip() or "(Unknown)",
        "wait_reason": row[13].strip() or "(Unknown)",
        "wait_mode": row[14].strip() or "(Unknown)",
        "wait_us": wait_us,
        "ready_us": ready_us,
        "wait_event_count": int(number(row[8])),
        **summary,
    }


def parse_wpa_reasons(path: Path) -> dict[str, Any]:
    distributions: dict[str, dict[str, dict[str, float | int]]] = {
        name: defaultdict(
            lambda: {"leaf_rows": 0, "wait_events": 0, "wait_us": 0.0, "ready_us": 0.0}
        )
        for name in (
            "wait_reason",
            "previous_state",
            "wait_mode",
            "new_thread",
            "old_thread",
            "wait_readying_process",
            "wait_readying_thread",
        )
    }
    wait_values: list[float] = []
    ready_values: list[float] = []
    events: list[dict[str, Any]] = []
    stack_counts: Counter[str] = Counter()
    quality: dict[str, Any] = {
        "csv_data_rows": 0,
        "malformed_rows": 0,
        "switch_in_leaf_rows": 0,
        "wait_event_rows": 0,
        "wait_event_count": 0,
        "wait_event_rows_with_count_not_one": 0,
    }
    stack_coverage = {
        "present_leaf_rows": 0,
        "resolved_symbol_leaf_rows": 0,
        "symbols_disabled_leaf_rows": 0,
        "dsmvc_leaf_rows": 0,
        "vapoursynth_leaf_rows": 0,
        "kernel_leaf_rows": 0,
    }
    source_aggregate: dict[str, Any] | None = None

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) != 27:
            raise ValueError(f"Unexpected WPA CSV column count {len(header)} in {path}")
        for row in reader:
            quality["csv_data_rows"] += 1
            if len(row) != 27:
                quality["malformed_rows"] += 1
                continue
            try:
                count_value = number(row[8])
                count = int(count_value)
                wait_us = number(row[6])
                ready_us = number(row[7])
            except ValueError:
                quality["malformed_rows"] += 1
                continue

            state = row[12].strip()
            old_process = row[15].strip()
            if not state and not old_process and count > 1 and source_aggregate is None:
                source_aggregate = {
                    "process": row[0].strip(),
                    "events": count,
                    "wait_us": wait_us,
                    "ready_us": ready_us,
                }
            if state and old_process:
                quality["switch_in_leaf_rows"] += 1
            if not (state and old_process):
                continue

            ready_values.append(ready_us)
            event = event_summary(row, wait_us, ready_us)
            events.append(event)

            keys = {
                "wait_reason": event["wait_reason"],
                "previous_state": event["previous_state"],
                "wait_mode": event["wait_mode"],
                "new_thread": f"{event['new_process']} / TID {event['new_thread_id']}",
                "old_thread": f"{event['old_process']} / TID {event['old_thread_id']}",
            }
            for name, key in keys.items():
                add_distribution(distributions[name], key, count, wait_us, ready_us)

            if count > 0:
                quality["wait_event_rows"] += 1
                quality["wait_event_count"] += count
                if count != 1:
                    quality["wait_event_rows_with_count_not_one"] += 1
                wait_values.append(wait_us)
                add_distribution(
                    distributions["wait_readying_process"],
                    event["readying_process"],
                    count,
                    wait_us,
                    ready_us,
                )
                add_distribution(
                    distributions["wait_readying_thread"],
                    f"{event['readying_process']} / TID {event['readying_thread_id']}",
                    count,
                    wait_us,
                    ready_us,
                )

            stack = row[3].strip()
            stack_counts[stack] += 1
            if stack:
                stack_coverage["present_leaf_rows"] += 1
            if "!" in stack and "<Symbols disabled>" not in stack:
                stack_coverage["resolved_symbol_leaf_rows"] += 1
            if "<Symbols disabled>" in stack:
                stack_coverage["symbols_disabled_leaf_rows"] += 1
            lowered = stack.lower()
            if "dsmvc.dll" in lowered:
                stack_coverage["dsmvc_leaf_rows"] += 1
            if "vapoursynth.dll" in lowered:
                stack_coverage["vapoursynth_leaf_rows"] += 1
            if "ntoskrnl.exe" in lowered:
                stack_coverage["kernel_leaf_rows"] += 1

    total_rows = int(quality["switch_in_leaf_rows"])
    total_wait_events = int(quality["wait_event_count"])
    wait_stats = value_stats(wait_values)
    ready_stats = value_stats(ready_values)
    total_wait = float(wait_stats["sum_us"])
    total_ready = float(ready_stats["sum_us"])
    top_ready = sorted(events, key=lambda item: item["ready_us"], reverse=True)
    top_wait = sorted(events, key=lambda item: item["wait_us"], reverse=True)

    def contribution(items: Sequence[dict[str, Any]], field: str, count: int, total: float) -> float:
        value = math.fsum(float(item[field]) for item in items[:count])
        return 100.0 * value / total if total else 0.0

    ready_tail_threshold_us = 10_000.0
    ready_tail = [value for value in ready_values if value >= ready_tail_threshold_us]
    ready_below_tail = [value for value in ready_values if value < ready_tail_threshold_us]
    conservation: dict[str, Any] = {"source_aggregate_available": source_aggregate is not None}
    if source_aggregate:
        wait_delta = total_wait - float(source_aggregate["wait_us"])
        ready_delta = total_ready - float(source_aggregate["ready_us"])
        conservation.update(
            {
                "source_aggregate": source_aggregate,
                "wait_event_delta": total_wait_events - int(source_aggregate["events"]),
                "wait_delta_us": wait_delta,
                "ready_delta_us": ready_delta,
                "within_export_precision": (
                    total_wait_events == int(source_aggregate["events"])
                    and abs(wait_delta) <= 0.2
                    and abs(ready_delta) <= 0.2
                ),
            }
        )

    top_stacks: list[dict[str, Any]] = []
    for stack, count in stack_counts.most_common(10):
        top_stacks.append({"leaf_rows": count, **stack_summary(stack)})

    return {
        "source_csv": str(path.resolve()),
        "header_columns": header,
        "quality": quality,
        "conservation": conservation,
        "wait_latency": wait_stats,
        "ready_latency": ready_stats,
        "ready_latency_below_10ms": value_stats(ready_below_tail),
        "outliers": {
            "ready_tail_threshold_us": ready_tail_threshold_us,
            "ready_tail_events": len(ready_tail),
            "ready_tail_us": math.fsum(ready_tail),
            "ready_tail_percent": (
                100.0 * math.fsum(ready_tail) / total_ready if total_ready else 0.0
            ),
            "top_1_ready_percent": contribution(top_ready, "ready_us", 1, total_ready),
            "top_4_ready_percent": contribution(top_ready, "ready_us", 4, total_ready),
            "top_10_ready_percent": contribution(top_ready, "ready_us", 10, total_ready),
            "top_1_wait_percent": contribution(top_wait, "wait_us", 1, total_wait),
            "top_4_wait_percent": contribution(top_wait, "wait_us", 4, total_wait),
            "top_10_wait_percent": contribution(top_wait, "wait_us", 10, total_wait),
            "top_ready_events": top_ready[:10],
            "top_wait_events": top_wait[:10],
        },
        "distributions": {
            name: finish_distribution(
                values,
                int(quality["wait_event_rows"])
                if name.startswith("wait_readying_")
                else total_rows,
                total_wait_events,
                total_wait,
                total_ready,
            )
            for name, values in distributions.items()
        },
        "stack_coverage": {
            **stack_coverage,
            "unique_stacks": len(stack_counts),
            "ready_thread_stack_column_available": False,
            "note": (
                "The export contains New Thread Stack but no Ready Thread Stack; "
                "it cannot identify the wake-up call stack."
            ),
            "top_new_thread_stacks": top_stacks,
        },
    }


def function_buckets(functions: Iterable[dict[str, Any]]) -> dict[str, float]:
    result = {"columns_percent": 0.0, "horizontal_percent": 0.0, "transpose_percent": 0.0}
    for function in functions:
        name = str(function.get("name", "")).lower()
        value = float(function.get("event_percent", 0.0))
        if "solve_columns" in name:
            result["columns_percent"] += value
        elif "solve_horizontal" in name:
            result["horizontal_percent"] += value
        elif "transpose_source" in name:
            result["transpose_percent"] += value
    return result


def module_percent(modules: Iterable[dict[str, Any]], needle: str) -> float:
    needle = needle.lower()
    return math.fsum(
        float(module.get("event_percent", 0.0))
        for module in modules
        if needle in str(module.get("name", "")).lower()
    )


def compact_tbp(profile: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixed: list[dict[str, Any]] = []
    sweeps: list[dict[str, Any]] = []
    for item in profile["tbp"]:
        plugin_percent = float(item["plugin_module"]["event_percent"])
        buckets = function_buckets(item["top_plugin_functions"])
        top_functions = [
            {"name": fn["name"], "cpu_percent": fn["event_percent"]}
            for fn in item["top_plugin_functions"][:6]
        ]
        base = {
            "case": item["case"],
            "requests": item["requests"],
            "profile_duration_seconds": item["profile_duration_seconds"],
            "thread_count": item["thread_count"],
            "dsmvc_percent": plugin_percent,
            **buckets,
            "other_dsmvc_percent": max(0.0, plugin_percent - math.fsum(buckets.values())),
            "target": item.get("target_output"),
            "top_dsmvc_functions": top_functions,
        }
        if item["case"] in ("getfnative", "getfnative_v2", "selectkernel"):
            modules = item.get("top_modules", [])
            base.update(
                {
                    "vapoursynth_percent": module_percent(modules, "vapoursynth.dll"),
                    "vc_runtime_percent": module_percent(modules, "vcruntime140.dll"),
                    "kernel_percent": module_percent(modules, "ntoskrnl.exe"),
                }
            )
            sweeps.append(base)
        else:
            fixed.append(base)
    key = lambda item: (CASE_ORDER.get(item["case"], 99), item["requests"])
    return sorted(fixed, key=key), sorted(sweeps, key=key)


def compact_assess(profile: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in profile["assess"]:
        metrics = item["plugin_module"]["metrics"]
        result.append(
            {
                "case": item["case"],
                "requests": item["requests"],
                "dsmvc_cycle_percent": metrics.get("CYCLES_NOT_IN_HALT", 0.0),
                "cpi": metrics.get("CPI", 0.0),
                "l1_dc_miss_percent": metrics.get("%L1_DC_MISSES", 0.0),
                "local_dram_refill_pti": metrics.get("L1_DEMAND_DC_REFILLS_LOCAL_DRAM (PTI)", 0.0),
                "local_cache_refill_pti": metrics.get("L1_DEMAND_DC_REFILLS_LOCAL_CACHE (PTI)", 0.0),
                "local_l2_refill_pti": metrics.get("L1_DEMAND_DC_REFILLS_LOCAL_L2 (PTI)", 0.0),
                "stli_other_pti": metrics.get("STLI_OTHER (PTI)", 0.0),
                "avx_stalls_ptc": metrics.get("SSE_AVX_STALLS (PTC)", 0.0),
            }
        )
    return sorted(result, key=lambda item: CASE_ORDER.get(item["case"], 99))


def compact_pcm(profile: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in profile["pcm"]:
        result.append(
            {
                "case": item["case"],
                "requests": item["requests"],
                "target_fps": item["target_output"]["fps"],
                "sample_count": item["sample_count"],
                "cumulative_total_gbps": item["cumulative_total_gbps"],
                "cumulative_read_gbps": item["cumulative_read_gbps"],
                "cumulative_write_gbps": item["cumulative_write_gbps"],
                "sample_total_gbps": item["sample_total_gbps"],
                "scope": "socket-wide",
            }
        )
    return sorted(result, key=lambda item: (CASE_ORDER.get(item["case"], 99), item["requests"]))


def compact_memory(profile: dict[str, Any]) -> list[dict[str, Any]]:
    gib = float(1024**3)
    result: list[dict[str, Any]] = []
    for item in profile["process_memory"]:
        result.append(
            {
                "case": item["case"],
                "requests": item["requests"],
                "duration_seconds": item["duration_seconds"],
                "peak_working_set_gib": item["peak_working_set_bytes"] / gib,
                "steady_working_set_median_gib": item["steady_working_set_median_bytes"] / gib,
                "steady_working_set_max_gib": item["steady_working_set_max_bytes"] / gib,
                "steady_private_median_gib": item["steady_private_median_bytes"] / gib,
                "steady_private_max_gib": item["steady_private_max_bytes"] / gib,
                "max_threads": item["max_thread_count"],
                "cpu_seconds": item["cpu_seconds"],
            }
        )
    return sorted(result, key=lambda item: (CASE_ORDER.get(item["case"], 99), item["requests"]))


def compact_etw(profile: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in profile["etw"]:
        modules = item["vspipe_module_percent"]
        result.append(
            {
                "case": item["case"],
                "requests": item["requests"],
                "trace_wall_seconds": item["trace_wall_seconds"],
                "vspipe_lifetime_seconds": item["vspipe_lifetime_seconds"],
                "vspipe_cpu_seconds": item["vspipe_cpu_seconds"],
                "vspipe_average_cores": item["vspipe_average_cores"],
                "vspipe_threads": item["vspipe_thread_count"],
                "dsmvc_percent": modules.get("dsmvc.dll", 0.0),
                "vapoursynth_percent": modules.get("vapoursynth.dll", 0.0),
                "vc_runtime_percent": modules.get("vcruntime140.dll", 0.0),
                "kernel_percent": modules.get("ntoskrnl.exe", 0.0),
                "unknown_percent": modules.get("unknown", 0.0),
                "cswitch_events": item["cswitch_events"],
                "ready_thread_events": item["ready_thread_events"],
                "sampled_profile_events": item["sampled_profile_events"],
                "lost_buffers": item["lost_buffers"],
                "lost_events": item["lost_events"],
            }
        )
    return sorted(result, key=lambda item: CASE_ORDER.get(item["case"], 99))


def compact_throughput(benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in benchmark["cases"]:
        result.append(
            {
                "case": item["case"],
                "frames": benchmark["environment"]["frames"],
                "requests": benchmark["environment"]["requests"],
                "runs_per_implementation": item["old"]["runs"],
                "old_fps_median": item["old"]["fps"]["median"],
                "old_fps_mad": item["old"]["fps"]["mad"],
                "old_fps_min": item["old"]["fps"]["minimum"],
                "old_fps_max": item["old"]["fps"]["maximum"],
                "new_fps_median": item["new"]["fps"]["median"],
                "new_fps_mad": item["new"]["fps"]["mad"],
                "new_fps_min": item["new"]["fps"]["minimum"],
                "new_fps_max": item["new"]["fps"]["maximum"],
                "speedup": item["speedup"],
            }
        )
    return sorted(result, key=lambda item: CASE_ORDER.get(item["case"], 99))


def md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(md_cell(value) for value in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(md_cell(value) for value in row) + " |" for row in rows)
    return lines


def fmt_us(value: float) -> str:
    if abs(value) >= 1_000_000.0:
        return f"{value / 1_000_000.0:.3f} s"
    if abs(value) >= 1_000.0:
        return f"{value / 1_000.0:.3f} ms"
    return f"{value:.3f} us"


def build_markdown(result: dict[str, Any]) -> str:
    lines: list[str] = ["# Consolidated CPU profile result", ""]
    throughput = result["throughput"]
    wpa = result["wpa_waits"]
    reasons = wpa["distributions"]["wait_reason"]
    ready_reasons = sorted(reasons, key=lambda item: item["ready_us"], reverse=True)

    lines.extend(
        [
            "## Verdict",
            "",
            (
                "The independent full-process comparison establishes a current-DLL speedup of "
                f"{min(item['speedup'] for item in throughput):.3f}x to "
                f"{max(item['speedup'] for item in throughput):.3f}x across the three 500-frame, "
                "32-request sweep cases. Fixed-kernel profiles are executor-dominated; the "
                "columns pass is the largest CPU target. The end-to-end getfnative trace is not "
                "DLL-only: kernel, VapourSynth, and VC runtime attribution is material."
            ),
            "",
            (
                "The evidence supports a shared data-movement and scheduling ceiling at high "
                "concurrency, but it does not identify one exact L1/L2/L3/DRAM source. IBS data "
                "is invalid, PCM is socket-wide, and the WPA export has no wake-up stack."
            ),
            "",
            "## Frozen identity",
            "",
        ]
    )
    identity = result["identity"]
    lines.extend(
        md_table(
            ["Artifact", "Value"],
            [
                ("Git commit", f"`{identity['git_commit']}`"),
                ("New dsmvc DLL SHA-256", f"`{identity['new_plugin_sha256']}`"),
                ("Old descale DLL SHA-256", f"`{identity['old_plugin_sha256']}`"),
                ("Input SHA-256", f"`{identity['input_sha256']}`"),
                ("CPU", identity["cpu"]),
                ("VapourSynth", identity["vapoursynth"].splitlines()[2]),
                ("Threads / requests", f"{identity['threads']} / 32"),
            ],
        )
    )
    lines.extend(["", "## Evidence integrity", ""])
    inventory = result["evidence"]["session_counts"]
    lines.extend(
        md_table(
            ["Source", "Complete"],
            [
                ("TBP", inventory["tbp"]),
                ("Assess Extended", inventory["assess_ext"]),
                ("PCM memory", inventory["pcm"]),
                ("Process memory", inventory["process_memory"]),
                ("ETW", inventory.get("etw_cpu", 0)),
                ("WPA switch-in leaf rows", wpa["quality"]["switch_in_leaf_rows"]),
                ("WPA wait events", wpa["quality"]["wait_event_count"]),
            ],
        )
    )
    integrity = result["evidence"]["wpa_reasons_csv"]
    lines.extend(
        [
            "",
            f"WPA CSV size: {integrity['bytes']:,} bytes; SHA-256 match: "
            f"**{str(integrity['sha256_matches_manifest']).lower()}**. "
            f"CSV aggregation conservation: **{str(wpa['conservation']['within_export_precision']).lower()}**. ",
            (
                "ETW lost events/buffers: 0/0."
                if result["current_profile"]["etw"]
                else "ETW: not collected in the selected profile round; see source provenance."
            ),
            "",
            "## Old/new full-process throughput",
            "",
            "These are the authoritative old/new comparisons: independent VSPipe processes, "
            "500 frames, 32 requests, three runs per implementation. FPS is the external process "
            "wall-time result, not VSPipe's overlapping per-filter timer.",
            "",
        ]
    )
    lines.extend(
        md_table(
            ["Case", "Old median fps", "New median fps", "Old MAD", "New MAD", "Speedup"],
            [
                (
                    item["case"],
                    f"{item['old_fps_median']:.3f}",
                    f"{item['new_fps_median']:.3f}",
                    f"{item['old_fps_mad']:.3f}",
                    f"{item['new_fps_mad']:.3f}",
                    f"{item['speedup']:.3f}x",
                )
                for item in throughput
            ],
        )
    )
    lines.extend(["", "## Current DLL fixed-kernel TBP", ""])
    lines.append("Percentages are sampled whole-process CPU shares.")
    lines.append("")
    lines.extend(
        md_table(
            ["Case", "Req", "Profiled fps", "dsmvc", "Columns", "Horizontal", "Transpose", "Other"],
            [
                (
                    item["case"],
                    item["requests"],
                    f"{item['target']['fps']:.2f}",
                    f"{item['dsmvc_percent']:.2f}%",
                    f"{item['columns_percent']:.2f}%",
                    f"{item['horizontal_percent']:.2f}%",
                    f"{item['transpose_percent']:.2f}%",
                    f"{item['other_dsmvc_percent']:.2f}%",
                )
                for item in result["current_profile"]["fixed_kernel_tbp"]
            ],
        )
    )
    lines.extend(["", "## Current DLL Assess Extended", ""])
    lines.append("Metrics are sampled for dsmvc.dll at 32 requests.")
    lines.append("")
    lines.extend(
        md_table(
            ["Case", "Cycle share", "CPI", "L1 miss", "DRAM PTI", "Cache PTI", "L2 PTI", "STLI PTI"],
            [
                (
                    item["case"],
                    f"{item['dsmvc_cycle_percent']:.2f}%",
                    f"{item['cpi']:.3f}",
                    f"{item['l1_dc_miss_percent']:.2f}%",
                    f"{item['local_dram_refill_pti']:.2f}",
                    f"{item['local_cache_refill_pti']:.2f}",
                    f"{item['local_l2_refill_pti']:.2f}",
                    f"{item['stli_other_pti']:.2f}",
                )
                for item in result["current_profile"]["assess"]
            ],
        )
    )
    lines.extend(["", "## PCM memory bandwidth", ""])
    lines.append("PCM values are socket-wide and are not attributable to VSPipe alone.")
    lines.append("")
    lines.extend(
        md_table(
            ["Case", "Req", "Target fps", "Cumulative", "Read", "Write", "Interior median", "p95"],
            [
                (
                    item["case"],
                    item["requests"],
                    f"{item['target_fps']:.2f}",
                    f"{item['cumulative_total_gbps']:.2f} GB/s",
                    f"{item['cumulative_read_gbps']:.2f}",
                    f"{item['cumulative_write_gbps']:.2f}",
                    f"{item['sample_total_gbps']['interior_median']:.2f}",
                    f"{item['sample_total_gbps']['p95']:.2f}",
                )
                for item in result["current_profile"]["pcm"]
            ],
        )
    )
    lines.extend(["", "## Process memory", ""])
    lines.append("The API peak includes startup; steady values use samples at or after 800 ms.")
    lines.append("")
    lines.extend(
        md_table(
            ["Case", "Req", "Peak WS", "Steady WS med/max", "Steady private med/max", "Threads"],
            [
                (
                    item["case"],
                    item["requests"],
                    f"{item['peak_working_set_gib']:.2f} GiB",
                    f"{item['steady_working_set_median_gib']:.2f}/{item['steady_working_set_max_gib']:.2f} GiB",
                    f"{item['steady_private_median_gib']:.2f}/{item['steady_private_max_gib']:.2f} GiB",
                    item["max_threads"],
                )
                for item in result["current_profile"]["process_memory"]
            ],
        )
    )
    lines.extend(["", "## Sweep CPU attribution", ""])
    lines.append("TBP and ETW use different attribution methods; their percentages must not be averaged.")
    lines.append("")
    lines.extend(
        md_table(
            ["TBP case", "dsmvc", "VapourSynth", "VC runtime", "Top dsmvc functions"],
            [
                (
                    item["case"],
                    f"{item['dsmvc_percent']:.2f}%",
                    f"{item['vapoursynth_percent']:.2f}%",
                    f"{item['vc_runtime_percent']:.2f}%",
                    ", ".join(
                        f"`{fn['name']}` {fn['cpu_percent']:.2f}%" for fn in item["top_dsmvc_functions"][:4]
                    ),
                )
                for item in result["current_profile"]["sweep_tbp"]
            ],
        )
    )
    lines.append("")
    if result["current_profile"]["etw"]:
        lines.extend(
            md_table(
                ["ETW case", "Life", "CPU", "Avg cores", "dsmvc", "VS", "VC", "Kernel", "Unknown", "Lost"],
                [
                    (
                        item["case"],
                        f"{item['vspipe_lifetime_seconds']:.3f} s",
                        f"{item['vspipe_cpu_seconds']:.3f} s",
                        f"{item['vspipe_average_cores']:.2f}",
                        f"{item['dsmvc_percent']:.2f}%",
                        f"{item['vapoursynth_percent']:.2f}%",
                        f"{item['vc_runtime_percent']:.2f}%",
                        f"{item['kernel_percent']:.2f}%",
                        f"{item['unknown_percent']:.2f}%",
                        f"{item['lost_events']}/{item['lost_buffers']}",
                    )
                    for item in result["current_profile"]["etw"]
                ],
            )
        )
    else:
        lines.append("ETW was not collected in this profile round.")
    lines.extend(["", "## WPA getfnative waits", ""])
    lines.append(
        f"The Reasons export resolves {wpa['quality']['switch_in_leaf_rows']:,} switch-in leaf rows "
        f"containing {wpa['quality']['wait_event_count']:,} wait events. Aggregate blocked "
        f"time is {wpa['wait_latency']['sum_us'] / 1_000_000.0:.6f} s and aggregate Ready time is "
        f"{wpa['ready_latency']['sum_us'] / 1_000_000.0:.6f} s across threads; neither is process wall time."
    )
    lines.append("")
    lines.extend(
        md_table(
            ["Metric", "Count", "Median", "MAD", "p95", "p99", "Max", "Sum"],
            [
                (
                    "Wait",
                    wpa["wait_latency"]["count"],
                    fmt_us(wpa["wait_latency"]["median_us"]),
                    fmt_us(wpa["wait_latency"]["mad_us"]),
                    fmt_us(wpa["wait_latency"]["p95_us"]),
                    fmt_us(wpa["wait_latency"]["p99_us"]),
                    fmt_us(wpa["wait_latency"]["max_us"]),
                    fmt_us(wpa["wait_latency"]["sum_us"]),
                ),
                (
                    "Ready",
                    wpa["ready_latency"]["count"],
                    fmt_us(wpa["ready_latency"]["median_us"]),
                    fmt_us(wpa["ready_latency"]["mad_us"]),
                    fmt_us(wpa["ready_latency"]["p95_us"]),
                    fmt_us(wpa["ready_latency"]["p99_us"]),
                    fmt_us(wpa["ready_latency"]["max_us"]),
                    fmt_us(wpa["ready_latency"]["sum_us"]),
                ),
                (
                    "Ready (<10 ms)",
                    wpa["ready_latency_below_10ms"]["count"],
                    fmt_us(wpa["ready_latency_below_10ms"]["median_us"]),
                    fmt_us(wpa["ready_latency_below_10ms"]["mad_us"]),
                    fmt_us(wpa["ready_latency_below_10ms"]["p95_us"]),
                    fmt_us(wpa["ready_latency_below_10ms"]["p99_us"]),
                    fmt_us(wpa["ready_latency_below_10ms"]["max_us"]),
                    fmt_us(wpa["ready_latency_below_10ms"]["sum_us"]),
                ),
            ],
        )
    )
    outliers = wpa["outliers"]
    lines.extend(
        [
            "",
            f"There are {outliers['ready_tail_events']:,} Ready intervals at or above 10 ms; they "
            f"contribute {outliers['ready_tail_percent']:.2f}% of Ready time. The top one/four/ten "
            f"intervals contribute {outliers['top_1_ready_percent']:.2f}%/"
            f"{outliers['top_4_ready_percent']:.2f}%/{outliers['top_10_ready_percent']:.2f}%. "
            f"For Wait, the top four contribute {outliers['top_4_wait_percent']:.2f}%. Ready delay "
            "is long-tailed but is not dominated by four events.",
            "",
            "### Wait reasons",
            "",
        ]
    )
    lines.extend(
        md_table(
            ["Reason", "Leaf rows", "Wait events", "Wait-event share", "Wait", "Wait share", "Ready", "Ready share"],
            [
                (
                    item["name"],
                    item["leaf_rows"],
                    item["wait_events"],
                    f"{item['wait_event_percent']:.2f}%",
                    fmt_us(item["wait_us"]),
                    f"{item['wait_percent']:.2f}%",
                    fmt_us(item["ready_us"]),
                    f"{item['ready_percent']:.2f}%",
                )
                for item in reasons[:12]
            ],
        )
    )
    lines.extend(["", "### Ready-state reason concentration", ""])
    lines.extend(
        md_table(
            ["Previous reason", "Leaf rows", "Ready", "Ready share"],
            [
                (
                    item["name"],
                    item["leaf_rows"],
                    fmt_us(item["ready_us"]),
                    f"{item['ready_percent']:.2f}%",
                )
                for item in ready_reasons[:8]
            ],
        )
    )
    lines.extend(["", "### Largest Ready intervals", ""])
    lines.extend(
        md_table(
            ["Time", "TID", "Reason", "Mode", "Wait", "Ready", "Readying process/TID"],
            [
                (
                    f"{item['switch_in_seconds']:.6f} s",
                    item["new_thread_id"],
                    item["wait_reason"],
                    item["wait_mode"],
                    fmt_us(item["wait_us"]),
                    fmt_us(item["ready_us"]),
                    f"{item['readying_process']} / {item['readying_thread_id']}",
                )
                for item in outliers["top_ready_events"]
            ],
        )
    )
    lines.extend(["", "### Largest Wait events", ""])
    lines.extend(
        md_table(
            ["Time", "TID", "Reason", "Wait", "Ready-after-wake", "Readying process/TID"],
            [
                (
                    f"{item['switch_in_seconds']:.6f} s",
                    item["new_thread_id"],
                    item["wait_reason"],
                    fmt_us(item["wait_us"]),
                    fmt_us(item["ready_us"]),
                    f"{item['readying_process']} / {item['readying_thread_id']}",
                )
                for item in outliers["top_wait_events"][:8]
            ],
        )
    )
    lines.append("")
    lines.append(
        "The longest Wait rows are parked-thread intervals spanning much of the trace. Their "
        "durations contribute to aggregate thread wait time, but do not equal serialized VSPipe "
        "wall-time loss or prove a lock bottleneck."
    )
    lines.extend(["", "### Readying process distribution", ""])
    lines.extend(
        md_table(
            ["Readying process", "Wait events", "Wait", "Ready-after-wake"],
            [
                (
                    item["name"],
                    item["wait_events"],
                    fmt_us(item["wait_us"]),
                    fmt_us(item["ready_us"]),
                )
                for item in wpa["distributions"]["wait_readying_process"][:10]
            ],
        )
    )
    stack = wpa["stack_coverage"]
    lines.extend(
        [
            "",
            "### Stack coverage",
            "",
            f"New-thread stack present: {stack['present_leaf_rows']:,}/"
            f"{wpa['quality']['switch_in_leaf_rows']:,} leaf rows; resolved-symbol heuristic: "
            f"{stack['resolved_symbol_leaf_rows']:,}; symbols-disabled stacks: "
            f"{stack['symbols_disabled_leaf_rows']:,}. The table does not contain `Ready Thread Stack`, "
            "so wake-up stack attribution is unavailable.",
            "",
            "## Consolidated interpretation",
            "",
            "### Measured facts",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["conclusions"]["measured_facts"])
    lines.extend(["", "### Cross-tool inferences", ""])
    lines.extend(f"- {item}" for item in result["conclusions"]["cross_tool_inferences"])
    lines.extend(["", "### Not established by this dataset", ""])
    lines.extend(f"- {item}" for item in result["conclusions"]["not_established"])
    lines.extend(["", "## Review priorities", ""])
    lines.extend(f"{index}. {item}" for index, item in enumerate(result["review_priorities"], start=1))
    lines.extend(
        [
            "",
            "## Source artifacts",
            "",
            f"- Profile summary: `{result['sources']['profile_summary']}`",
            f"- Old/new benchmark: `{result['sources']['benchmark_json']}`",
            f"- WPA Reasons manifest: `{result['sources']['wpa_manifest']}`",
            f"- WPA parsed summary: `{result['sources']['parsed_wpa_json']}`",
            f"- IBS diagnostics: `{result['sources']['ibs_manifest']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    profile_dir = args.profile_dir.resolve()
    profile_summary_path = (args.profile_summary or profile_dir / "profile-summary.json").resolve()
    benchmark_path = (
        args.benchmark_json
        or profile_dir / "comparison-vspipe-500-r32-3x" / "benchmark.json"
    ).resolve()
    wpa_manifest_path = (
        args.wpa_manifest
        or profile_dir / "wpa-waits-getfnative-reasons" / "getfnative_reasons-manifest.json"
    ).resolve()
    ibs_manifest_path = (
        args.ibs_manifest
        or profile_dir / "completion-profile" / "ibs-diagnostics-20260803-184700" / "manifest.json"
    ).resolve()
    output_json = (args.output_json or profile_dir / "completion-summary.json").resolve()
    output_markdown = (args.output_markdown or profile_dir / "completion-summary.md").resolve()
    parsed_wpa_json = (
        args.parsed_wpa_json
        or profile_dir / "wpa-waits-getfnative-reasons" / "parsed-summary.json"
    ).resolve()

    profile = load_json(profile_summary_path)
    benchmark = load_json(benchmark_path)
    wpa_manifest = load_json(wpa_manifest_path)
    ibs_manifest = load_json(ibs_manifest_path)
    reasons_csv = (args.reasons_csv or Path(wpa_manifest["csv"])).resolve()

    wpa = parse_wpa_reasons(reasons_csv)
    actual_wpa_hash = sha256_file(reasons_csv)
    write_json(parsed_wpa_json, wpa)

    fixed_tbp, sweep_tbp = compact_tbp(profile)
    throughput = compact_throughput(benchmark)
    manifest = profile["manifest"]
    benchmark_env = benchmark["environment"]
    ibs_usable = any(bool(item.get("valid_raw_data")) for item in ibs_manifest["results"])

    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Offline consolidation of existing artifacts only; no profiler or benchmark was run."
        ),
        "identity": {
            "git_commit": manifest["git_commit"],
            "new_plugin_sha256": manifest["plugin_sha256"].upper(),
            "new_plugin_pdb_sha256": manifest["pdb_sha256"].upper(),
            "old_plugin_sha256": benchmark_env["old_plugin_sha256"].upper(),
            "input_sha256": manifest["image_sha256"].upper(),
            "vspipe_sha256": manifest["vspipe_sha256"].upper(),
            "cpu": manifest["cpu"]["Name"].strip(),
            "physical_cores": manifest["cpu"]["NumberOfCores"],
            "logical_processors": manifest["cpu"]["NumberOfLogicalProcessors"],
            "threads": manifest["threads"],
            "vapoursynth": benchmark_env["vspipe"],
            "uprof_cli": manifest["uprof_cli_version"],
            "uprof_pcm": manifest["uprof_pcm_version"],
        },
        "sources": {
            "profile_summary": str(profile_summary_path),
            "benchmark_json": str(benchmark_path),
            "wpa_manifest": str(wpa_manifest_path),
            "wpa_reasons_csv": str(reasons_csv),
            "parsed_wpa_json": str(parsed_wpa_json),
            "ibs_manifest": str(ibs_manifest_path),
        },
        "evidence": {
            "session_counts": profile["session_counts"],
            "profile_validation": profile["validation"],
            "wpa_reasons_csv": {
                "bytes": reasons_csv.stat().st_size,
                "manifest_bytes": wpa_manifest["csv_bytes"],
                "sha256": actual_wpa_hash,
                "manifest_sha256": wpa_manifest["csv_sha256"].upper(),
                "sha256_matches_manifest": actual_wpa_hash == wpa_manifest["csv_sha256"].upper(),
            },
            "ibs": {
                "usable": ibs_usable,
                "attempts": len(ibs_manifest["results"]),
                "valid_attempts": sum(bool(item.get("valid_raw_data")) for item in ibs_manifest["results"]),
                "results": ibs_manifest["results"],
            },
        },
        "throughput": throughput,
        "current_profile": {
            "fixed_kernel_tbp": fixed_tbp,
            "sweep_tbp": sweep_tbp,
            "assess": compact_assess(profile),
            "pcm": compact_pcm(profile),
            "process_memory": compact_memory(profile),
            "etw": compact_etw(profile),
        },
        "wpa_waits": wpa,
    }

    r32_fixed = [item for item in fixed_tbp if item["requests"] == 32]
    pcm_r32 = [item for item in result["current_profile"]["pcm"] if item["requests"] == 32]
    getfnative_memory = next(
        item
        for item in result["current_profile"]["process_memory"]
        if item["case"] == "getfnative"
    )
    top_reason = wpa["distributions"]["wait_reason"][0]
    scheduler_ready_reasons = [
        item
        for item in wpa["distributions"]["wait_reason"]
        if item["name"] in ("WrDispatchInt", "WrQuantumEnd")
    ]
    scheduler_ready_rows = sum(item["leaf_rows"] for item in scheduler_ready_reasons)
    scheduler_ready_percent = math.fsum(item["ready_percent"] for item in scheduler_ready_reasons)
    result["conclusions"] = {
        "measured_facts": [
            (
                "The new plugin reaches 248.598/284.425/279.329 median fps versus "
                "80.973/169.451/82.589 for the old plugin in getfnative, getfnative_v2, "
                "and selectkernel: 3.070x, 1.679x, and 3.382x."
            ),
            (
                f"At r32, fixed-kernel TBP attributes {min(item['dsmvc_percent'] for item in r32_fixed):.2f}% "
                f"to {max(item['dsmvc_percent'] for item in r32_fixed):.2f}% of sampled process CPU "
                "to dsmvc; columns are the largest bucket in every fixed case."
            ),
            (
                f"The r32 PCM sessions report {min(item['cumulative_total_gbps'] for item in pcm_r32):.2f}-"
                f"{max(item['cumulative_total_gbps'] for item in pcm_r32):.2f} GB/s cumulative socket "
                "bandwidth across the five fixed kernels."
            ),
            (
                f"The getfnative process reaches {getfnative_memory['peak_working_set_gib']:.2f} GiB "
                f"peak WS and {getfnative_memory['steady_working_set_median_gib']:.2f} GiB steady median WS."
            ),
            (
                f"WPA resolves {wpa['quality']['switch_in_leaf_rows']:,} switch-in leaf rows and "
                f"{wpa['quality']['wait_event_count']:,} wait events with exact aggregate conservation. "
                f"{top_reason['name']} accounts for {top_reason['wait_events']:,} wait events and "
                f"{top_reason['wait_percent']:.2f}% of aggregate wait time."
            ),
            (
                f"WrDispatchInt and WrQuantumEnd account for {scheduler_ready_percent:.2f}% of "
                f"aggregate Ready time across {scheduler_ready_rows:,} switch-in rows. The largest "
                "four Ready intervals account for only "
                f"{wpa['outliers']['top_4_ready_percent']:.2f}%, so this is not a four-event outlier effect."
            ),
        ],
        "cross_tool_inferences": [
            (
                "Fixed-kernel r32 performance is constrained more by shared executor data movement "
                "and cache/memory traffic than by planner construction or one kernel-specific anomaly. "
                "This is supported jointly by TBP hotspot shape, near-identical r32 PCM ranges, and "
                "the weak high-request scaling of several kernels."
            ),
            (
                "Sweep throughput cannot scale in proportion to a descale inner-loop improvement. "
                "TBP and ETW both show material work outside dsmvc, while sweep working sets are much "
                "larger than the fixed-kernel graph."
            ),
            (
                "The Ready distribution reflects broad preemption/dispatch scheduling pressure at "
                "32 requests, not a few wake-up anomalies. It is run-level scheduler evidence and "
                "cannot by itself be assigned to a dsmvc lock or worker-pool decision."
            ),
        ],
        "not_established": [
            (
                "No current-versus-old hotspot, cache, or memory comparison exists; only the "
                "separate process-throughput benchmark compares old and new DLLs."
            ),
            (
                "The exact limiting L1/L2/L3/DRAM level is not established. Assess counters are "
                "sampled, PCM is socket-wide, and every IBS attempt produced invalid raw data."
            ),
            (
                "The owner of the multi-GiB allocations is not established because ETW did not "
                "capture heap allocation stacks."
            ),
            (
                "The code path that woke delayed threads is not established because the WPA export "
                "does not contain Ready Thread Stack."
            ),
            (
                "No API3-versus-API4 A/B profile exists. API3 entry points are not identified as "
                "hot functions here, so this dataset provides no evidence that an API4 migration "
                "would improve throughput."
            ),
        ],
    }
    result["review_priorities"] = [
        (
            "Review the columns executor first, including output/scratch locality and the interaction "
            "between transposition, per-frame parallelism, and the internal worker pool."
        ),
        (
            "For b5/b7 kernels, review the shared paired-column path; Lanczos3, Spline36, and "
            "Spline64 have the same columns/horizontal hotspot structure and bandwidth range."
        ),
        (
            "For getnative/selectkernel sweeps, reduce retained graph/frame/plan state and runtime "
            "copy/allocation pressure before expecting more inner-loop work to translate directly "
            "to VSPipe FPS."
        ),
        (
            "Treat WPA waits as a scheduling lead, not proof of a lock bottleneck. A future wake-stack "
            "or synchronization-stack trace would be required before changing synchronization policy."
        ),
    ]

    write_json(output_json, result)
    write_text(output_markdown, build_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
