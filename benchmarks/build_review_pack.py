#!/usr/bin/env python3
"""Build a portable, fact-only review pack from completed local artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRENT = ROOT / "benchmark-results" / "profile-current-4917f7b8-20260803-203321"
DEFAULT_PREVIOUS = ROOT / "benchmark-results" / "profile-current-4917f7b8-final"
DEFAULT_PACK = ROOT / "benchmark-results" / "review-pack-20260803"
DEFAULT_ARCHIVE = ROOT / "benchmark-results" / "dsmvc-review-pack-20260803.zip"
SESSION_ID = "019fc18a-8517-73b3-943d-6c9d58f5bcd3"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from consolidate_profile import (  # noqa: E402
    compact_assess,
    compact_etw,
    compact_memory,
    compact_pcm,
    compact_tbp,
    compact_throughput,
    load_json,
    sha256_file,
    write_json,
    write_text,
)


ALLOWED_CODE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".md",
    ".py",
    ".ps1",
    ".txt",
    ".vpy",
}
DISALLOWED_ARCHIVE_SUFFIXES = {".etl", ".wpaprofile", ".db", ".sqlite", ".sqlite3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-dir", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--previous-dir", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n").encode("utf-8")


def write_json_sorted(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def relpath(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def clean_target(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        key: value[key]
        for key in ("frames", "seconds", "fps")
        if key in value
    }


def clean_tbp(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in items:
        copied = dict(item)
        copied["target"] = clean_target(copied.get("target"))
        result.append(copied)
    return result


def compact_wpa(summary: dict[str, Any], source_manifest: dict[str, Any]) -> dict[str, Any]:
    source = summary["wpa_waits"]
    outliers = source["outliers"]

    def event_fact(event: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "switch_in_seconds",
            "new_process",
            "new_thread_id",
            "previous_state",
            "wait_reason",
            "wait_mode",
            "wait_us",
            "ready_us",
            "wait_event_count",
            "readying_process",
            "readying_thread_id",
        )
        return {key: event[key] for key in keys if key in event}

    distributions = {}
    for name, values in source["distributions"].items():
        distributions[name] = [
            {
                key: item[key]
                for key in (
                    "name",
                    "leaf_rows",
                    "leaf_row_percent",
                    "wait_events",
                    "wait_event_percent",
                    "wait_us",
                    "wait_percent",
                    "ready_us",
                    "ready_percent",
                )
                if key in item
            }
            for item in values[:20]
        ]

    conservation = dict(source["conservation"])
    conservation.pop("source_aggregate", None)
    stack = dict(source["stack_coverage"])
    stack.pop("top_new_thread_stacks", None)
    return {
        "source": {
            "round": "previous-etw-wpa",
            "manifest": relpath(Path(source_manifest["_path"])),
            "etl_sha256": source_manifest.get("etl_sha256", "").upper(),
            "csv_sha256": source_manifest.get("csv_sha256", "").upper(),
            "csv_bytes": source_manifest.get("csv_bytes"),
            "exporter_version": source_manifest.get("exporter_version"),
            "mode": source_manifest.get("mode"),
        },
        "quality": source["quality"],
        "conservation": conservation,
        "wait_latency": source["wait_latency"],
        "ready_latency": source["ready_latency"],
        "ready_latency_below_10ms": source["ready_latency_below_10ms"],
        "outliers": {
            key: outliers[key]
            for key in (
                "ready_tail_threshold_us",
                "ready_tail_events",
                "ready_tail_us",
                "ready_tail_percent",
                "top_1_ready_percent",
                "top_4_ready_percent",
                "top_10_ready_percent",
                "top_1_wait_percent",
                "top_4_wait_percent",
                "top_10_wait_percent",
            )
            if key in outliers
        },
        "largest_ready_intervals": [event_fact(item) for item in outliers["top_ready_events"][:8]],
        "largest_wait_events": [event_fact(item) for item in outliers["top_wait_events"][:8]],
        "distributions": distributions,
        "stack_coverage": stack,
        "interpretation": [
            "Ready time is aggregate thread delay, not VSPipe wall time.",
            "The export has New Thread Stack but no Ready Thread Stack, so wake-up call-site attribution is unavailable.",
            "WrDispatchInt and WrQuantumEnd show broad scheduler pressure; this does not prove a dsmvc lock bottleneck.",
        ],
    }


def sanitize_environment(environment: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "timestamp_utc",
        "platform",
        "processor",
        "processor_identifier",
        "logical_cpu_count",
        "vspipe",
        "frames",
        "requests",
        "threads",
        "input_sha256",
        "old_plugin_sha256",
        "new_plugin_sha256",
        "runner_sha256",
        "vpy_sha256",
    )
    result = {key: environment[key] for key in keys if key in environment}
    result["input_sha256"] = str(result.get("input_sha256", "")).upper()
    result["old_plugin_sha256"] = str(result.get("old_plugin_sha256", "")).upper()
    result["new_plugin_sha256"] = str(result.get("new_plugin_sha256", "")).upper()
    return result


def manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    cpu = manifest.get("cpu", {})
    return {
        "git_commit": manifest.get("git_commit"),
        "started_utc": manifest.get("started_utc"),
        "completed_utc": manifest.get("completed_utc", manifest.get("last_pass_completed_utc")),
        "plugin_sha256": str(manifest.get("plugin_sha256", "")).upper(),
        "pdb_sha256": str(manifest.get("pdb_sha256", "")).upper(),
        "image_sha256": str(manifest.get("image_sha256", "")).upper(),
        "vspipe_sha256": str(manifest.get("vspipe_sha256", "")).upper(),
        "cpu": cpu.get("Name", "").strip(),
        "physical_cores": cpu.get("NumberOfCores"),
        "logical_processors": cpu.get("NumberOfLogicalProcessors"),
        "threads": manifest.get("threads"),
        "uprof_cli": manifest.get("uprof_cli_version"),
        "uprof_pcm": manifest.get("uprof_pcm_version"),
        "profile_backend": manifest.get("profile_backend"),
        "hardware_pmu_config": manifest.get("hardware_pmu_config"),
        "iba_used": manifest.get("iba_used"),
        "skip_etw": manifest.get("skip_etw", False),
    }


def build_profile_facts(profile: dict[str, Any], round_name: str, note: str) -> dict[str, Any]:
    fixed, sweeps = compact_tbp(profile)
    return {
        "round": round_name,
        "manifest": manifest_identity(profile["manifest"]),
        "session_counts": profile["session_counts"],
        "expected_counts": profile.get("expected_counts", {}),
        "validation": profile.get("validation", {}),
        "tbp_fixed_kernel": clean_tbp(fixed),
        "tbp_sweeps": clean_tbp(sweeps),
        "assess_ext": compact_assess(profile),
        "pcm_memory": compact_pcm(profile),
        "process_memory": compact_memory(profile),
        "etw": [],
        "note": note,
    }


def build_current_facts(profile: dict[str, Any]) -> dict[str, Any]:
    return build_profile_facts(
        profile,
        "current-pmu-pcm",
        "ETW was explicitly skipped in this round; these are the latest complete PMU/PCM/TBP/process-memory facts.",
    )


def build_previous_pmu_pcm_facts(profile: dict[str, Any]) -> dict[str, Any]:
    return build_profile_facts(
        profile,
        "previous-final-pmu-pcm",
        "This is the earlier final profile round's PMU/PCM/TBP/process-memory fact set; ETW is published separately in previous-etw-facts.",
    )


def build_previous_etw_facts(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "round": "previous-etw",
        "manifest": manifest_identity(profile["manifest"]),
        "session_counts": profile["session_counts"],
        "etw": compact_etw(profile),
        "note": "ETW facts are from the earlier final profile round and are kept separate from the current PMU/PCM round.",
    }


def build_provenance(
    current_dir: Path,
    previous_dir: Path,
    current_summary_path: Path,
    previous_summary_path: Path,
    benchmark_path: Path,
    previous_manifest_path: Path,
    wpa_manifest_path: Path,
    ibs_manifest_path: Path,
) -> dict[str, Any]:
    wpa_manifest = load_json(wpa_manifest_path)
    wpa_manifest["_path"] = str(wpa_manifest_path)
    return {
        "requested_session_id": SESSION_ID,
        "session_visualization_directory": "C:/Users/lsy39/.codex/visualizations/2026/08/02/" + SESSION_ID,
        "session_visualization_files_found": [],
        "note": "The requested session visualization directory contained no files in this workspace; facts are sourced from the corresponding repository benchmark artifacts.",
        "rounds": {
            "current_pmu_pcm": {
                "directory": relpath(current_dir),
                "summary": relpath(current_summary_path),
                "summary_sha256": sha256_file(current_summary_path),
                "manifest": relpath(current_dir / "manifest.json"),
                "manifest_sha256": sha256_file(current_dir / "manifest.json"),
                "methods": ["TBP", "AMD hardware PMU assess_ext", "AMD uProf PCM memory", "process-memory polling"],
            },
            "previous_etw_wpa": {
                "directory": relpath(previous_dir),
                "summary": relpath(previous_summary_path),
                "summary_sha256": sha256_file(previous_summary_path),
                "manifest": relpath(previous_manifest_path),
                "manifest_sha256": sha256_file(previous_manifest_path),
                "methods": ["TBP", "AMD hardware PMU assess_ext", "AMD uProf PCM memory", "ETW", "WPA Reasons export"],
            },
        },
        "throughput": {
            "benchmark_json": relpath(benchmark_path),
            "benchmark_json_sha256": sha256_file(benchmark_path),
            "commands": relpath(previous_dir / "comparison-vspipe-500-r32-3x" / "commands.txt"),
        },
        "wpa": {
            "manifest": relpath(wpa_manifest_path),
            "manifest_sha256": sha256_file(wpa_manifest_path),
            "etl_sha256": str(wpa_manifest.get("etl_sha256", "")).upper(),
            "csv_sha256": str(wpa_manifest.get("csv_sha256", "")).upper(),
            "csv_bytes": wpa_manifest.get("csv_bytes"),
            "raw_files_included": False,
        },
        "ibs": {
            "manifest": relpath(ibs_manifest_path),
            "manifest_sha256": sha256_file(ibs_manifest_path),
            "raw_files_included": False,
        },
    }


def build_conclusions(
    throughput: list[dict[str, Any]],
    current: dict[str, Any],
    previous_etw: dict[str, Any],
    wpa: dict[str, Any],
    ibs: dict[str, Any],
) -> dict[str, list[str]]:
    r32_fixed = [item for item in current["tbp_fixed_kernel"] if item["requests"] == 32]
    r32_pcm = [item for item in current["pcm_memory"] if item["requests"] == 32]
    getfnative_memory = next(item for item in current["process_memory"] if item["case"] == "getfnative")
    speedups = ", ".join(f"{item['case']} {item['speedup']:.3f}x" for item in throughput)
    etw_items = {item["case"]: item for item in previous_etw["etw"]}
    wpa_reason = wpa["distributions"]["wait_reason"][0]
    scheduler = [
        item
        for item in wpa["distributions"]["wait_reason"]
        if item["name"] in ("WrDispatchInt", "WrQuantumEnd")
    ]
    scheduler_ready = sum(float(item["ready_percent"]) for item in scheduler)
    scheduler_rows = sum(int(item["leaf_rows"]) for item in scheduler)
    return {
        "measured_facts": [
            f"Independent 500-frame, 32-request process benchmarks show {speedups} for the new DLL over the fixed old descale.dll baseline.",
            f"Current TBP attributes {min(item['dsmvc_percent'] for item in r32_fixed):.2f}% to {max(item['dsmvc_percent'] for item in r32_fixed):.2f}% of sampled process CPU to dsmvc at r32; columns is the largest bucket in every fixed kernel.",
            f"Current PCM reports {min(item['cumulative_total_gbps'] for item in r32_pcm):.2f}-{max(item['cumulative_total_gbps'] for item in r32_pcm):.2f} GB/s cumulative socket bandwidth across fixed r32 kernels.",
            f"Current getfnative process memory reaches {getfnative_memory['peak_working_set_gib']:.2f} GiB peak WS and {getfnative_memory['steady_working_set_median_gib']:.2f} GiB steady median WS.",
            f"Previous ETW reports {etw_items.get('bicubic_b3', {}).get('dsmvc_percent', 0.0):.2f}% dsmvc CPU share for fixed bicubic and {etw_items.get('getfnative', {}).get('dsmvc_percent', 0.0):.2f}% for the getfnative sweep; these are process-normalized ETW facts from a separate round.",
            f"WPA resolves {wpa['quality']['switch_in_leaf_rows']:,} switch-in leaf rows and {wpa['quality']['wait_event_count']:,} wait events with conservation={wpa['conservation'].get('within_export_precision')}; {wpa_reason['name']} is {wpa_reason['wait_events']:,} wait events and {wpa_reason['wait_percent']:.2f}% of aggregate wait time.",
            f"WrDispatchInt and WrQuantumEnd account for {scheduler_ready:.2f}% of aggregate Ready time across {scheduler_rows:,} switch-in rows.",
            f"IBS diagnostic attempts usable={ibs['usable']}; raw IBS data is intentionally omitted from this pack.",
        ],
        "cross_tool_inferences": [
            "Fixed-kernel r32 work is executor/data-movement dominated; columns, horizontal, and transpose are the review targets.",
            "Socket-wide PCM clustering around 39-40 GB/s supports a shared bandwidth/data-movement ceiling, but does not prove process-local DRAM saturation.",
            "Sweep throughput contains substantial VapourSynth, runtime, kernel, scheduling, and retained-plan work, so inner-loop speedups do not transfer 1:1 to VSPipe FPS.",
            "WPA Ready-state concentration is broad scheduling pressure at 32 requests; it is a lead for review, not proof of a dsmvc synchronization bottleneck.",
        ],
        "not_established": [
            "No same-run old-plugin PMU/TBP/PCM comparison exists; old/new speedup is established only by the independent process benchmark.",
            "The exact limiting cache level (L1/L2/L3/DRAM) is not established; assess_ext is sampled and PCM is socket-wide.",
            "Allocation ownership and planner retention call sites are not established by process-memory polling alone.",
            "WPA has no Ready Thread Stack, so the wake-up call site is unknown.",
            "No API3-versus-API4 A/B profile is included; this pack does not claim API4 would improve throughput.",
        ],
    }


def md_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def build_throughput_markdown(environment: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    rows = [
        (
            item["case"],
            f"{item['old_fps_median']:.3f}",
            f"{item['new_fps_median']:.3f}",
            f"{item['old_fps_mad']:.3f}",
            f"{item['new_fps_mad']:.3f}",
            f"{item['speedup']:.3f}x",
        )
        for item in cases
    ]
    return "\n".join(
        [
            "# Full-process throughput facts",
            "",
            "The old/new comparison uses independent VSPipe processes, 500 frames, 32 requests, and three runs per implementation. These values are external process FPS, not overlapping per-filter timers.",
            "",
            md_table(["Case", "Old median fps", "New median fps", "Old MAD", "New MAD", "Speedup"], rows),
            "",
            "Environment:",
            "",
            f"- `{environment.get('vspipe', '').splitlines()[2] if len(environment.get('vspipe', '').splitlines()) > 2 else environment.get('vspipe', '')}`",
            f"- frames/requests/threads: `{environment.get('frames')}/{environment.get('requests')}/{environment.get('threads')}`",
            f"- input SHA-256: `{environment.get('input_sha256')}`",
            f"- old plugin SHA-256: `{environment.get('old_plugin_sha256')}`",
            f"- new plugin SHA-256: `{environment.get('new_plugin_sha256')}`",
            "",
        ]
    )


def build_profile_markdown(
    current: dict[str, Any],
    title: str = "Current PMU/PCM/TBP facts",
    note: str = "This section is from the current elevated AMD uProf round. ETW was skipped in this round.",
) -> str:
    fixed = [item for item in current["tbp_fixed_kernel"] if item["requests"] == 32]
    sweeps = current["tbp_sweeps"]
    assess = current["assess_ext"]
    pcm = [item for item in current["pcm_memory"] if item["requests"] == 32]
    memory = current["process_memory"]
    return "\n".join(
        [
            f"# {title}",
            "",
            f"{note} Percentages are sampled CPU shares; PCM is socket-wide.",
            "",
            "## Fixed-kernel TBP at r32",
            "",
            md_table(
                ["Case", "dsmvc", "Columns", "Horizontal", "Transpose", "Profiled fps"],
                [
                    (item["case"], f"{item['dsmvc_percent']:.2f}%", f"{item['columns_percent']:.2f}%", f"{item['horizontal_percent']:.2f}%", f"{item['transpose_percent']:.2f}%", f"{item['target']['fps']:.2f}")
                    for item in fixed
                ],
            ),
            "",
            "## Sweep TBP attribution",
            "",
            "Sweep percentages include the rest of the VSPipe process and must not be averaged with fixed-kernel TBP or ETW percentages.",
            "",
            md_table(
                ["Case", "dsmvc", "VapourSynth", "VC runtime", "Kernel", "Top dsmvc functions"],
                [
                    (
                        item["case"],
                        f"{item['dsmvc_percent']:.2f}%",
                        f"{item['vapoursynth_percent']:.2f}%",
                        f"{item['vc_runtime_percent']:.2f}%",
                        f"{item['kernel_percent']:.2f}%",
                        ", ".join(f"{fn['name']} {fn['cpu_percent']:.2f}%" for fn in item["top_dsmvc_functions"][:4]),
                    )
                    for item in sweeps
                ],
            ),
            "",
            "## Assess Extended",
            "",
            md_table(
                ["Case", "Cycle share", "CPI", "L1 miss", "DRAM PTI", "Cache PTI", "L2 PTI", "STLI PTI"],
                [
                    (item["case"], f"{item['dsmvc_cycle_percent']:.2f}%", f"{item['cpi']:.3f}", f"{item['l1_dc_miss_percent']:.2f}%", f"{item['local_dram_refill_pti']:.2f}", f"{item['local_cache_refill_pti']:.2f}", f"{item['local_l2_refill_pti']:.2f}", f"{item['stli_other_pti']:.2f}")
                    for item in assess
                ],
            ),
            "",
            "## PCM memory",
            "",
            md_table(
                ["Case", "Req", "Cumulative GB/s", "Read", "Write", "Interior median", "p95"],
                [
                    (item["case"], item["requests"], f"{item['cumulative_total_gbps']:.2f}", f"{item['cumulative_read_gbps']:.2f}", f"{item['cumulative_write_gbps']:.2f}", f"{item['sample_total_gbps']['interior_median']:.2f}", f"{item['sample_total_gbps']['p95']:.2f}")
                    for item in pcm
                ],
            ),
            "",
            "## Process memory",
            "",
            md_table(
                ["Case", "Req", "Peak WS", "Steady WS median/max", "Private median/max", "Threads"],
                [
                    (item["case"], item["requests"], f"{item['peak_working_set_gib']:.2f} GiB", f"{item['steady_working_set_median_gib']:.2f}/{item['steady_working_set_max_gib']:.2f}", f"{item['steady_private_median_gib']:.2f}/{item['steady_private_max_gib']:.2f}", item["max_threads"])
                    for item in memory
                ],
            ),
            "",
        ]
    )


def build_etw_markdown(previous: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Previous ETW facts",
            "",
            "These module and scheduling facts come from the previous final ETW round. They are not merged into the current PMU/PCM session counts.",
            "",
            md_table(
                ["Case", "VSPipe life", "CPU seconds", "Avg cores", "dsmvc", "VapourSynth", "VC", "Kernel", "Lost"],
                [
                    (item["case"], f"{item['vspipe_lifetime_seconds']:.3f}s", f"{item['vspipe_cpu_seconds']:.3f}", f"{item['vspipe_average_cores']:.2f}", f"{item['dsmvc_percent']:.2f}%", f"{item['vapoursynth_percent']:.2f}%", f"{item['vc_runtime_percent']:.2f}%", f"{item['kernel_percent']:.2f}%", f"{item['lost_events']}/{item['lost_buffers']}")
                    for item in previous["etw"]
                ],
            ),
            "",
            "ETW module shares are normalized within each VSPipe process and are not directly comparable to TBP sample shares.",
            "",
        ]
    )


def build_round_comparison(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    current_fixed = {
        item["case"]: item
        for item in current["tbp_fixed_kernel"]
        if item["requests"] == 32
    }
    previous_fixed = {
        item["case"]: item
        for item in previous["tbp_fixed_kernel"]
        if item["requests"] == 32
    }
    current_pcm = {
        item["case"]: item
        for item in current["pcm_memory"]
        if item["requests"] == 32
    }
    previous_pcm = {
        item["case"]: item
        for item in previous["pcm_memory"]
        if item["requests"] == 32
    }
    cases = []
    for case in sorted(set(current_fixed) & set(previous_fixed)):
        current_item = current_fixed[case]
        previous_item = previous_fixed[case]
        current_bandwidth = current_pcm.get(case, {}).get("cumulative_total_gbps")
        previous_bandwidth = previous_pcm.get(case, {}).get("cumulative_total_gbps")
        cases.append(
            {
                "case": case,
                "current_dsmvc_percent": current_item["dsmvc_percent"],
                "previous_dsmvc_percent": previous_item["dsmvc_percent"],
                "dsmvc_delta_percentage_points": current_item["dsmvc_percent"] - previous_item["dsmvc_percent"],
                "current_pcm_cumulative_gbps": current_bandwidth,
                "previous_pcm_cumulative_gbps": previous_bandwidth,
                "pcm_delta_gbps": (
                    current_bandwidth - previous_bandwidth
                    if current_bandwidth is not None and previous_bandwidth is not None
                    else None
                ),
            }
        )
    return {
        "same_binary": current["manifest"]["plugin_sha256"] == previous["manifest"]["plugin_sha256"],
        "current_round": current["manifest"],
        "previous_round": previous["manifest"],
        "cases": cases,
        "caveat": "The rounds are separate collections, not synchronized A/B runs; deltas describe evidence drift, not a causal optimization effect.",
    }


def build_round_comparison_markdown(comparison: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Profile round comparison",
            "",
            "The current and previous rounds use the same plugin SHA-256, but were collected at different times. This table is a consistency check, not a causal A/B performance result.",
            "",
            md_table(
                ["Case", "Current dsmvc", "Previous dsmvc", "Delta pp", "Current PCM", "Previous PCM", "Delta GB/s"],
                [
                    (
                        item["case"],
                        f"{item['current_dsmvc_percent']:.2f}%",
                        f"{item['previous_dsmvc_percent']:.2f}%",
                        f"{item['dsmvc_delta_percentage_points']:+.2f}",
                        f"{item['current_pcm_cumulative_gbps']:.2f}" if item["current_pcm_cumulative_gbps"] is not None else "n/a",
                        f"{item['previous_pcm_cumulative_gbps']:.2f}" if item["previous_pcm_cumulative_gbps"] is not None else "n/a",
                        f"{item['pcm_delta_gbps']:+.2f}" if item["pcm_delta_gbps"] is not None else "n/a",
                    )
                    for item in comparison["cases"]
                ],
            ),
            "",
            f"Same binary: `{comparison['same_binary']}`.",
            "",
            comparison["caveat"],
            "",
        ]
    )


def build_wpa_markdown(wpa: dict[str, Any]) -> str:
    reasons = wpa["distributions"]["wait_reason"][:12]
    ready = sorted(wpa["distributions"]["wait_reason"], key=lambda item: item["ready_us"], reverse=True)[:8]
    return "\n".join(
        [
            "# WPA wait facts",
            "",
            "This is a compressed analysis of the WPA Reasons export from the requested session's repository artifact. The raw ETL, WPAProfile, and 273 MB CSV are not included.",
            "",
            f"- switch-in leaf rows: `{wpa['quality']['switch_in_leaf_rows']:,}`",
            f"- wait events: `{wpa['quality']['wait_event_count']:,}`",
            f"- conservation within export precision: `{wpa['conservation'].get('within_export_precision')}`",
            f"- ready stack available: `{wpa['stack_coverage']['ready_thread_stack_column_available']}`",
            "",
            "## Latency",
            "",
            md_table(
                ["Metric", "Count", "Median", "MAD", "p95", "p99", "Max", "Sum"],
                [
                    ("Wait", wpa["wait_latency"]["count"], f"{wpa['wait_latency']['median_us']:.3f} us", f"{wpa['wait_latency']['mad_us']:.3f} us", f"{wpa['wait_latency']['p95_us']:.3f} us", f"{wpa['wait_latency']['p99_us']:.3f} us", f"{wpa['wait_latency']['max_us']:.3f} us", f"{wpa['wait_latency']['sum_us'] / 1_000_000:.3f} s"),
                    ("Ready", wpa["ready_latency"]["count"], f"{wpa['ready_latency']['median_us']:.3f} us", f"{wpa['ready_latency']['mad_us']:.3f} us", f"{wpa['ready_latency']['p95_us']:.3f} us", f"{wpa['ready_latency']['p99_us']:.3f} us", f"{wpa['ready_latency']['max_us']:.3f} us", f"{wpa['ready_latency']['sum_us'] / 1_000_000:.3f} s"),
                ],
            ),
            "",
            "## Wait reasons",
            "",
            md_table(
                ["Reason", "Wait events", "Wait share", "Ready share"],
                [(item["name"], item["wait_events"], f"{item['wait_percent']:.2f}%", f"{item['ready_percent']:.2f}%") for item in reasons],
            ),
            "",
            "## Ready-state concentration",
            "",
            md_table(
                ["Previous reason", "Leaf rows", "Ready share"],
                [(item["name"], item["leaf_rows"], f"{item['ready_percent']:.2f}%") for item in ready],
            ),
            "",
            "Interpretation: Ready delay is aggregate thread scheduling delay, not serialized VSPipe wall time. Missing Ready Thread Stack prevents wake-up call-site attribution.",
            "",
        ]
    )


def collect_code_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    top_files = [".gitignore", "CMakeLists.txt", "LICENSE", "README.md", "THIRD_PARTY_NOTICES.md"]
    for name in top_files:
        path = root / name
        if path.is_file():
            candidates.append(path)
    for dirname in ("include", "src", "python", "tests", "benchmarks"):
        base = root / dirname
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix.lower() in ALLOWED_CODE_SUFFIXES:
                candidates.append(path)
    return sorted(set(candidates), key=lambda path: path.relative_to(root).as_posix())


def copy_code_snapshot(root: Path, pack: Path) -> dict[str, Any]:
    code_root = pack / "code"
    files = collect_code_files(root)
    entries = []
    for source in files:
        relative = source.relative_to(root)
        target = code_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entries.append({"path": relative.as_posix(), "bytes": source.stat().st_size, "sha256": sha256_file(source)})

    status = subprocess.run(["git", "status", "--short"], cwd=root, capture_output=True, text=True, check=False)
    diff = subprocess.run(["git", "diff", "--binary"], cwd=root, capture_output=True, text=True, check=False)
    write_text(code_root / "WORKTREE-STATUS.txt", status.stdout)
    write_text(code_root / "git-diff.patch", diff.stdout)
    manifest = {
        "source_root": str(root.resolve()),
        "file_count": len(entries),
        "files": entries,
        "excluded": [".git", "build", "benchmark-results", "__pycache__", "raw profiler outputs"],
    }
    write_json_sorted(code_root / "source-manifest.json", manifest)

    source_zip = code_root / "dsmvc-review-source.zip"
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for entry in entries:
            archive.write(code_root / entry["path"], entry["path"])
        archive.write(code_root / "WORKTREE-STATUS.txt", "WORKTREE-STATUS.txt")
        archive.write(code_root / "git-diff.patch", "git-diff.patch")
    return {"file_count": len(entries), "source_zip": source_zip.name, "source_zip_sha256": sha256_file(source_zip)}


def copy_tested_binary(current_dir: Path, pack: Path) -> dict[str, Any]:
    binary_dir = pack / "binary"
    binary_dir.mkdir(parents=True, exist_ok=True)
    source_dir = current_dir / "binary"
    entries = []
    for name in ("dsmvc.dll", "dsmvc.pdb"):
        source = source_dir / name
        if not source.is_file():
            continue
        target = binary_dir / name
        shutil.copy2(source, target)
        entries.append({"path": name, "bytes": target.stat().st_size, "sha256": sha256_file(target)})
    write_text(binary_dir / "README.txt", "Exact tested current-round dsmvc artifacts. The old baseline DLL is not copied from the user's VS installation.\n")
    return {"source_round": "current-pmu-pcm", "files": entries}


def copy_pmu_event_config(current_dir: Path, pack: Path) -> dict[str, Any] | None:
    source = current_dir / "uprof-assess-ext-config.log"
    if not source.is_file():
        return None
    target = pack / "profiles" / "assess-ext-event-config.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "path": target.relative_to(pack).as_posix(),
        "source": relpath(source),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def write_throughput_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    fields = ["case", "old_fps_median", "new_fps_median", "old_fps_mad", "new_fps_mad", "speedup"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: item[field] for field in fields} for item in cases)


def write_checksums(pack: Path) -> None:
    lines = []
    for path in sorted(pack.rglob("*")):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(pack).as_posix()}")
    write_text(pack / "checksums.sha256", "\n".join(lines) + "\n")


def assert_no_raw_archive_files(archive: Path) -> None:
    forbidden = []
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            lowered = name.lower()
            suffix = Path(lowered).suffix
            if suffix in DISALLOWED_ARCHIVE_SUFFIXES or lowered.endswith("cpu.db") or lowered.endswith("callstack.db"):
                forbidden.append(name)
    if forbidden:
        raise RuntimeError("Raw profiler files found in archive: " + ", ".join(forbidden))


def main() -> int:
    args = parse_args()
    current_dir = args.current_dir.resolve()
    previous_dir = args.previous_dir.resolve()
    pack = args.pack_dir.resolve()
    archive = args.archive.resolve()
    if pack.exists():
        if not args.force:
            raise SystemExit(f"pack directory already exists: {pack}; pass --force to replace it")
        shutil.rmtree(pack)
    if archive.exists():
        if not args.force:
            raise SystemExit(f"archive already exists: {archive}; pass --force to replace it")
        archive.unlink()
    pack.mkdir(parents=True)

    current_summary_path = current_dir / "profile-summary.json"
    previous_summary_path = previous_dir / "profile-summary.json"
    benchmark_path = previous_dir / "comparison-vspipe-500-r32-3x" / "benchmark.json"
    previous_manifest_path = previous_dir / "manifest.json"
    wpa_manifest_path = previous_dir / "wpa-waits-getfnative-reasons" / "getfnative_reasons-manifest.json"
    ibs_manifest_path = previous_dir / "completion-profile" / "ibs-diagnostics-20260803-184700" / "manifest.json"
    completion_path = previous_dir / "completion-summary.json"
    for path in (current_summary_path, previous_summary_path, benchmark_path, previous_manifest_path, wpa_manifest_path, ibs_manifest_path, completion_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    current_profile = load_json(current_summary_path)
    previous_profile = load_json(previous_summary_path)
    benchmark = load_json(benchmark_path)
    completion = load_json(completion_path)
    current = build_current_facts(current_profile)
    previous_pmu_pcm = build_previous_pmu_pcm_facts(previous_profile)
    previous_etw = build_previous_etw_facts(previous_profile)
    round_comparison = build_round_comparison(current, previous_pmu_pcm)
    throughput = compact_throughput(benchmark)
    environment = sanitize_environment(benchmark["environment"])
    ibs_manifest = load_json(ibs_manifest_path)
    ibs = {
        "usable": any(bool(item.get("valid_raw_data")) for item in ibs_manifest.get("results", [])),
        "attempts": len(ibs_manifest.get("results", [])),
        "valid_attempts": sum(bool(item.get("valid_raw_data")) for item in ibs_manifest.get("results", [])),
        "raw_files_included": False,
        "conclusion": "IBS is not a usable evidence source for this pack." if not any(bool(item.get("valid_raw_data")) for item in ibs_manifest.get("results", [])) else "IBS data was usable.",
    }
    provenance = build_provenance(current_dir, previous_dir, current_summary_path, previous_summary_path, benchmark_path, previous_manifest_path, wpa_manifest_path, ibs_manifest_path)
    wpa_manifest = load_json(wpa_manifest_path)
    wpa_manifest["_path"] = str(wpa_manifest_path)
    wpa = compact_wpa(completion, wpa_manifest)
    pmu_event_config = copy_pmu_event_config(current_dir, pack)
    if pmu_event_config is not None:
        current["pmu_event_config"] = pmu_event_config
    conclusions = build_conclusions(throughput, current, previous_etw, wpa, ibs)

    identity = {
        "git_commit": current["manifest"]["git_commit"],
        "new_plugin_sha256": current["manifest"]["plugin_sha256"],
        "new_plugin_pdb_sha256": current["manifest"]["pdb_sha256"],
        "old_plugin_sha256": environment["old_plugin_sha256"],
        "input_sha256": environment["input_sha256"],
        "vspipe_sha256": current["manifest"]["vspipe_sha256"],
        "cpu": current["manifest"]["cpu"],
        "physical_cores": current["manifest"]["physical_cores"],
        "logical_processors": current["manifest"]["logical_processors"],
        "threads": current["manifest"]["threads"],
        "vspipe": environment.get("vspipe"),
        "uprof_cli": current["manifest"]["uprof_cli"],
        "uprof_pcm": current["manifest"]["uprof_pcm"],
        "same_binary_across_profile_rounds": current["manifest"]["plugin_sha256"] == previous_etw["manifest"]["plugin_sha256"],
    }
    facts = {
        "schema_version": 1,
        "pack_id": "dsmvc-review-pack-20260803",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Fact-only consolidation for external expert review; no profiler or benchmark was run while building this pack.",
        "identity": identity,
        "throughput": {"environment": environment, "cases": throughput},
        "profiles": {
            "current_pmu_pcm": current,
            "previous_pmu_pcm": previous_pmu_pcm,
            "previous_etw": previous_etw,
            "round_comparison": round_comparison,
            "wpa": wpa,
            "ibs": ibs,
        },
        "conclusions": conclusions,
        "provenance": provenance,
        "delivery_policy": {
            "included": ["analyzed JSON/CSV/Markdown facts", "environment and SHA-256 metadata", "source snapshot", "exact tested dsmvc.dll and dsmvc.pdb"],
            "excluded": ["ETL", "WPAProfile", "WPA exported raw CSV", "uProf cpu.db/callstack.db", "PCM raw session directories", "raw images and output images", "old baseline DLL from the VS installation"],
        },
    }
    write_json_sorted(pack / "review-facts.json", facts)
    write_json_sorted(pack / "provenance.json", provenance)

    throughput_dir = pack / "throughput"
    write_json_sorted(throughput_dir / "benchmark.json", {"schema_version": 1, "environment": environment, "cases": throughput, "raw_samples_included": False})
    write_throughput_csv(throughput_dir / "benchmark.csv", throughput)
    throughput_md = build_throughput_markdown(environment, throughput)
    write_text(throughput_dir / "benchmark.md", throughput_md)
    commands = previous_dir / "comparison-vspipe-500-r32-3x" / "commands.txt"
    if commands.is_file():
        shutil.copy2(commands, throughput_dir / "commands.txt")

    profile_dir = pack / "profiles"
    write_json_sorted(profile_dir / "current-pmu-pcm-facts.json", current)
    write_text(profile_dir / "current-pmu-pcm-facts.md", build_profile_markdown(current))
    write_json_sorted(profile_dir / "previous-pmu-pcm-facts.json", previous_pmu_pcm)
    write_text(
        profile_dir / "previous-pmu-pcm-facts.md",
        build_profile_markdown(
            previous_pmu_pcm,
            title="Previous PMU/PCM/TBP facts",
            note="This section is from the earlier final AMD uProf round and is kept separate from the current round.",
        ),
    )
    write_json_sorted(profile_dir / "round-comparison.json", round_comparison)
    write_text(profile_dir / "round-comparison.md", build_round_comparison_markdown(round_comparison))
    write_json_sorted(profile_dir / "previous-etw-facts.json", previous_etw)
    write_text(profile_dir / "previous-etw-facts.md", build_etw_markdown(previous_etw))
    wpa_dir = pack / "wpa"
    write_json_sorted(wpa_dir / "wpa-facts.json", wpa)
    write_text(wpa_dir / "wpa-facts.md", build_wpa_markdown(wpa))
    write_json_sorted(pack / "ibs-facts.json", ibs)
    copy_code_snapshot(ROOT, pack)
    copy_tested_binary(current_dir, pack)

    readme = "\n".join(
        [
            "# dsmvc external review pack",
            "",
            "This pack consolidates the latest CPU optimization evidence for the API3 `com.dsmvc.descale` plugin and is intended for review by another team.",
            "",
            "## Start here",
            "",
            "1. `review-facts.md` is the cross-tool fact summary.",
            "2. `throughput/benchmark.md` is the authoritative old/new full-process comparison.",
            "3. `profiles/current-pmu-pcm-facts.md` is the latest elevated TBP/assess_ext/PCM/memory round.",
            "4. `profiles/previous-pmu-pcm-facts.md` preserves the earlier PMU/PCM/TBP round for cross-round review.",
            "5. `profiles/round-comparison.md` checks cross-round consistency without claiming causal A/B deltas.",
            "6. `profiles/previous-etw-facts.md` and `wpa/wpa-facts.md` contain separately sourced ETW/WPA facts.",
            "7. `code/dsmvc-review-source.zip` is the source snapshot; `binary/` contains the exact tested current DLL/PDB.",
            "",
            "## Identity",
            "",
            md_table(
                ["Item", "Value"],
                [
                    ("Git commit", f"`{identity['git_commit']}`"),
                    ("New dsmvc SHA-256", f"`{identity['new_plugin_sha256']}`"),
                    ("Old descale SHA-256", f"`{identity['old_plugin_sha256']}`"),
                    ("Input SHA-256", f"`{identity['input_sha256']}`"),
                    ("CPU", identity["cpu"]),
                    ("VS", "Core R57; API4 host with API3 plugin"),
                    ("Threads / requests", f"{identity['threads']} / {environment.get('requests')}"),
                ],
            ),
            "",
            "## Headline result",
            "",
            *[f"- {item}" for item in conclusions["measured_facts"][:5]],
            "",
            "## Provenance and exclusions",
            "",
            "The requested Codex session visualization directory was empty in this workspace. The pack therefore records the session ID and uses the corresponding checked-in repository artifacts, with current PMU/PCM facts separated from previous ETW/WPA facts.",
            "",
            "Raw profiler artifacts are deliberately excluded. Verify `checksums.sha256` before review.",
            "",
        ]
    )
    write_text(pack / "README.md", readme)
    review_md = "\n".join(
        [
            "# dsmvc review facts",
            "",
            "## Measured facts",
            "",
            *[f"- {item}" for item in conclusions["measured_facts"]],
            "",
            "## Cross-tool interpretation",
            "",
            *[f"- {item}" for item in conclusions["cross_tool_inferences"]],
            "",
            "## Not established",
            "",
            *[f"- {item}" for item in conclusions["not_established"]],
            "",
            "## Source separation",
            "",
            "- Current PMU/PCM/TBP/process memory: `profiles/current-pmu-pcm-facts.*`.",
            "- Previous PMU/PCM/TBP/process memory: `profiles/previous-pmu-pcm-facts.*`.",
            "- Cross-round consistency check: `profiles/round-comparison.*`.",
            "- Previous ETW: `profiles/previous-etw-facts.*`.",
            "- WPA wait analysis: `wpa/wpa-facts.*`.",
            "- IBS: `ibs-facts.json` records only validity; raw IBS files are excluded.",
            "",
        ]
    )
    write_text(pack / "review-facts.md", review_md)

    write_checksums(pack)
    pack.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(pack.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(pack).as_posix())
    assert_no_raw_archive_files(archive)
    archive_sha = sha256_file(archive)
    write_text(Path(str(archive) + ".sha256"), f"{archive_sha}  {archive.name}\n")
    print(json.dumps({"pack": str(pack), "archive": str(archive), "archive_sha256": archive_sha}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
