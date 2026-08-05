#!/usr/bin/env python3
"""Validate and summarize a profile_current_cpu.ps1 result directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROFILE_KINDS = {"tbp", "assess_ext"}
EXPECTED_COUNTS = {
    "tbp": 18,
    "assess_ext": 8,
    "pcm": 7,
    "process_memory": 6,
    "etw_cpu": 2,
}


def csv_row(line: str) -> list[str]:
    return next(csv.reader([line]))


def find_table(lines: list[str], heading: str) -> list[dict[str, str]]:
    for index, line in enumerate(lines):
        if heading in line:
            header_index = index + 1
            while header_index < len(lines) and not lines[header_index].strip():
                header_index += 1
            header = csv_row(lines[header_index])
            rows: list[dict[str, str]] = []
            for data_line in lines[header_index + 1 :]:
                if not data_line.strip():
                    break
                values = csv_row(data_line)
                if len(values) != len(header):
                    break
                rows.append(dict(zip(header, values)))
            return rows
    raise ValueError(f"missing table: {heading}")


def find_field(lines: Iterable[str], name: str) -> str | None:
    for line in lines:
        try:
            row = csv_row(line)
        except (csv.Error, StopIteration):
            continue
        if row and row[0].strip().rstrip(":") == name:
            return row[1].strip() if len(row) > 1 else ""
    return None


def number(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.strip().replace("%", "").replace(",", "")
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", cleaned)
    return float(match.group(0)) if match else None


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def short_function(signature: str) -> str:
    known = (
        "solve_columns_b1_pair",
        "solve_columns_b1_single",
        "solve_columns_b3_pair",
        "solve_columns_b3_single",
        "solve_columns_pair",
        "solve_columns_vector<0>",
        "solve_horizontal_b1",
        "solve_horizontal_b3",
        "solve_horizontal_generic",
        "transpose_source",
        "pack_cpu_plan",
        "form_normal_bands",
        "create_axis_plan",
        "filter_get_frame",
    )
    for name in known:
        if name in signature:
            return name
    match = re.search(r"(?:dsmvc::|namespace'::)([^<(]+)", signature)
    if match:
        return match.group(1).strip()
    return signature[:96]


def module_name(path: str) -> str:
    return Path(path.replace("\\", "/")).name.lower()


def parse_target_output(root: Path, session_name: str) -> dict[str, Any] | None:
    collect_log = root / f"{session_name}-collect.log"
    if not collect_log.is_file():
        return None
    text = collect_log.read_text(encoding="utf-8-sig", errors="replace")
    match = re.search(
        r"Output (\d+) frames in ([0-9.]+) seconds \(([0-9.]+) fps\)", text
    )
    if not match:
        return None
    return {
        "frames": int(match.group(1)),
        "seconds": float(match.group(2)),
        "fps": float(match.group(3)),
        "collect_log": str(collect_log),
    }


def parse_percentage(value: str) -> float:
    parsed = number(value)
    return parsed if parsed is not None else 0.0


def parse_uprof_session(session: dict[str, Any]) -> dict[str, Any]:
    session_path = Path(session["path"])
    report_path = session_path / "report-percentage.csv"
    if not report_path.is_file():
        raise ValueError(f"missing clean percentage report: {report_path}")
    text = report_path.read_text(encoding="utf-8-sig", errors="replace")
    if "cannot be configured together" in text:
        raise ValueError(f"conflicting report options remain in {report_path}")
    lines = text.splitlines()
    function_rows = find_table(lines, "200 HOTTEST FUNCTIONS")
    module_rows = find_table(lines, "200 HOTTEST MODULES")
    plugin_functions: list[dict[str, Any]] = []
    for row in function_rows:
        if module_name(row.get("Module", "")) != "dsmvc.dll":
            continue
        event_key = "CPU_TIME" if "CPU_TIME" in row else "CYCLES_NOT_IN_HALT"
        plugin_functions.append(
            {
                "name": short_function(row["FUNCTION"]),
                "signature": row["FUNCTION"],
                "event_percent": parse_percentage(row[event_key]),
                "metrics": {
                    key: number(value)
                    for key, value in row.items()
                    if key not in {"FUNCTION", "Module"}
                },
            }
        )
    modules: list[dict[str, Any]] = []
    for row in module_rows:
        event_key = "CPU_TIME" if "CPU_TIME" in row else "CYCLES_NOT_IN_HALT"
        modules.append(
            {
                "name": module_name(row["MODULE"]),
                "path": row["MODULE"],
                "event_percent": parse_percentage(row[event_key]),
                "metrics": {
                    key: number(value)
                    for key, value in row.items()
                    if key != "MODULE"
                },
            }
        )
    plugin_module = next((row for row in modules if row["name"] == "dsmvc.dll"), None)
    duration = number(find_field(lines, "Profile Duration"))
    call_stack = find_field(lines, "Call Stack Sampling")
    thread_count = number(find_field(lines, "Thread Count"))
    return {
        "name": session["name"],
        "kind": session["kind"],
        "case": session["case"],
        "requests": int(session["requests"]),
        "path": str(session_path),
        "report": str(report_path),
        "profile_duration_seconds": duration,
        "thread_count": int(thread_count) if thread_count is not None else None,
        "call_stack_sampling": call_stack,
        "plugin_module": plugin_module,
        "top_plugin_functions": plugin_functions[:12],
        "top_modules": modules[:12],
        "symbols_resolved": any("dsmvc::" in row["signature"] for row in plugin_functions),
        "source_paths_resolved": "\\src\\" in text or "/src/" in text,
        "unwanted_cuda_module": any("vsnlm_cuda" in row["path"].lower() for row in modules),
        "target_output": parse_target_output(session_path.parent, session["name"]),
    }


def find_pcm_csv(session_path: Path, name: str) -> Path:
    matches = list(session_path.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"expected one {name} under {session_path}, found {len(matches)}")
    return matches[0]


def parse_pcm(session: dict[str, Any]) -> dict[str, Any]:
    session_path = Path(session["path"])
    cumulative_path = find_pcm_csv(session_path, "report-cumulative.csv")
    timeseries_path = find_pcm_csv(session_path, "report-timeseries.csv")
    cumulative: dict[str, float] = {}
    for line in cumulative_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        row = csv_row(line)
        if row and row[0] in {
            "Total Mem Bw (GB/s)",
            "Total Mem RdBw (GB/s)",
            "Total Mem WrBw (GB/s)",
        }:
            parsed = number(row[1]) if len(row) > 1 else None
            if parsed is not None:
                cumulative[row[0]] = parsed
    lines = timeseries_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    header_index = next(
        index
        for index, line in enumerate(lines)
        if csv_row(line) and csv_row(line)[0] == "Total Mem Bw (GB/s)"
    )
    samples: list[dict[str, float]] = []
    for line in lines[header_index + 1 :]:
        row = csv_row(line)
        if len(row) < 3 or number(row[0]) is None:
            break
        samples.append(
            {
                "total_gbps": float(number(row[0]) or 0.0),
                "read_gbps": float(number(row[1]) or 0.0),
                "write_gbps": float(number(row[2]) or 0.0),
            }
        )
    totals = [sample["total_gbps"] for sample in samples]
    interior = totals[1:-1] if len(totals) > 2 else totals
    total_cumulative = cumulative.get("Total Mem Bw (GB/s)", 0.0)
    if total_cumulative <= 0 or not totals or max(totals) <= 0:
        raise ValueError(f"PCM session has no nonzero data: {session_path}")
    return {
        "name": session["name"],
        "case": session["case"],
        "requests": int(session["requests"]),
        "path": str(session_path),
        "cumulative_total_gbps": total_cumulative,
        "cumulative_read_gbps": cumulative.get("Total Mem RdBw (GB/s)"),
        "cumulative_write_gbps": cumulative.get("Total Mem WrBw (GB/s)"),
        "sample_count": len(samples),
        "sample_total_gbps": {
            "min": min(totals),
            "median": statistics.median(totals),
            "interior_median": statistics.median(interior),
            "p95": percentile(totals, 95.0),
            "max": max(totals),
        },
        "samples": samples,
        "target_output": parse_target_output(session_path.parent, session["name"]),
    }


def parse_process_memory(session: dict[str, Any]) -> dict[str, Any]:
    path = Path(session["path"])
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"empty process memory trace: {path}")

    def maximum(field: str) -> float:
        return max(float(row[field]) for row in rows)

    steady_state_start_ms = 800.0
    steady_rows = [
        row for row in rows if float(row["elapsed_ms"]) >= steady_state_start_ms
    ]
    if not steady_rows:
        steady_rows = rows

    def steady_values(field: str) -> list[float]:
        return [float(row[field]) for row in steady_rows]

    last = rows[-1]

    return {
        "name": session["name"],
        "case": session["case"],
        "requests": int(session["requests"]),
        "path": str(path),
        "samples": len(rows),
        "duration_seconds": maximum("elapsed_ms") / 1000.0,
        "peak_working_set_bytes": maximum("peak_working_set_bytes"),
        "max_working_set_bytes": maximum("working_set_bytes"),
        "max_private_bytes": maximum("private_bytes"),
        "max_virtual_bytes": maximum("virtual_bytes"),
        "max_thread_count": int(maximum("thread_count")),
        "max_handle_count": int(maximum("handle_count")),
        "cpu_seconds": maximum("cpu_seconds"),
        "steady_state_start_ms": steady_state_start_ms,
        "steady_working_set_median_bytes": statistics.median(
            steady_values("working_set_bytes")
        ),
        "steady_working_set_max_bytes": max(steady_values("working_set_bytes")),
        "steady_private_median_bytes": statistics.median(steady_values("private_bytes")),
        "steady_private_max_bytes": max(steady_values("private_bytes")),
        "final_working_set_bytes": float(last["working_set_bytes"]),
        "final_private_bytes": float(last["private_bytes"]),
    }


def duration_seconds(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) != 4:
        raise ValueError(f"unexpected ETW duration: {value}")
    days, hours, minutes = (int(part) for part in parts[:3])
    return days * 86400 + hours * 3600 + minutes * 60 + float(parts[3])


def parse_etw(session: dict[str, Any], hash_etw: bool) -> dict[str, Any]:
    path = Path(session["path"])
    stem = path.with_suffix("")
    trace_path = stem.with_name(stem.name + "-tracestats.txt")
    trace_detail_path = stem.with_name(stem.name + "-tracestats-detail.txt")
    lifetime_path = stem.with_name(stem.name + "-process.txt")
    process_path = stem.with_name(stem.name + "-cswitch-process.txt")
    thread_path = stem.with_name(stem.name + "-cswitch-thread.txt")
    profile_path = stem.with_name(stem.name + "-profile-detail.txt")
    required = (
        trace_path,
        trace_detail_path,
        lifetime_path,
        process_path,
        thread_path,
        profile_path,
    )
    missing = [str(candidate) for candidate in required if not candidate.is_file()]
    if missing:
        raise ValueError(f"missing ETW text exports: {missing}")
    trace_text = trace_path.read_text(encoding="utf-8-sig", errors="replace")
    duration_match = re.search(r"End time .*\(\+\s*([^)]+)\)", trace_text)
    lost_buffers = re.search(r"Total # Lost Buffers\s*:\s*(\d+)", trace_text)
    lost_events = re.search(r"Total # Lost Events\s*:\s*(\d+)", trace_text)
    if not duration_match or not lost_buffers or not lost_events:
        raise ValueError(f"incomplete ETW trace statistics: {trace_path}")
    trace_wall_seconds = duration_seconds(duration_match.group(1))

    trace_detail_text = trace_detail_path.read_text(encoding="utf-8-sig", errors="replace")

    def event_count(label: str) -> int:
        line = next(
            (candidate for candidate in trace_detail_text.splitlines() if label in candidate),
            None,
        )
        if line is None:
            raise ValueError(f"missing {label} event count in {trace_detail_path}")
        columns = line[: line.index(label)].split()
        if len(columns) < 2:
            raise ValueError(f"invalid {label} event row in {trace_detail_path}")
        return int(columns[-2])

    lifetime_text = lifetime_path.read_text(encoding="utf-8-sig", errors="replace")
    lifetime_match = re.search(
        r"^\s*(\d+),\s*(\d+),\s*Process,.*VSPipe\.exe \(\s*\d+\)",
        lifetime_text,
        re.M | re.I,
    )
    if not lifetime_match:
        raise ValueError(f"VSPipe lifetime not found in {lifetime_path}")
    vspipe_lifetime_seconds = (
        int(lifetime_match.group(2)) - int(lifetime_match.group(1))
    ) / 1_000_000.0

    process_text = process_path.read_text(encoding="utf-8-sig", errors="replace")
    process_match = re.search(r"^\s*(\d+),\s+VSPipe\.exe \(\s*\d+\)", process_text, re.M | re.I)
    if not process_match:
        raise ValueError(f"VSPipe process not found in {process_path}")
    vspipe_cpu_seconds = int(process_match.group(1)) / 1_000_000.0

    thread_text = thread_path.read_text(encoding="utf-8-sig", errors="replace")
    thread_times = [
        int(match.group(1)) / 1_000_000.0
        for match in re.finditer(
            r"^\s*(\d+),\s+VSPipe\.exe \(\s*\d+\),\s+\d+", thread_text, re.M | re.I
        )
    ]

    module_weights: dict[str, int] = defaultdict(int)
    profile_text = profile_path.read_text(encoding="utf-8-sig", errors="replace")
    for line in profile_text.splitlines():
        match = re.match(
            r"^\s*VSPipe\.exe \(\s*\d+\),\s*(\d+),\s*[0-9.]+,\s*(.+?)\s*$",
            line,
            re.I,
        )
        if match:
            module_weights[match.group(2).strip().strip('"').lower()] += int(match.group(1))
    total_weight = sum(module_weights.values())
    module_percent = {
        name: value * 100.0 / total_weight for name, value in module_weights.items()
    } if total_weight else {}
    result = {
        "name": session["name"],
        "case": session["case"],
        "requests": int(session["requests"]),
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "trace_wall_seconds": trace_wall_seconds,
        "vspipe_lifetime_seconds": vspipe_lifetime_seconds,
        "lost_buffers": int(lost_buffers.group(1)),
        "lost_events": int(lost_events.group(1)),
        "cswitch_events": event_count("Thread: CSwitch"),
        "ready_thread_events": event_count("Thread: ReadyThread"),
        "sampled_profile_events": event_count("Sampled Profile "),
        "vspipe_cpu_seconds": vspipe_cpu_seconds,
        "vspipe_average_cores": vspipe_cpu_seconds / vspipe_lifetime_seconds,
        "vspipe_thread_count": len(thread_times),
        "top_thread_cpu_seconds": sorted(thread_times, reverse=True)[:12],
        "vspipe_module_percent": dict(
            sorted(module_percent.items(), key=lambda item: item[1], reverse=True)
        ),
        "exports": [str(candidate) for candidate in required],
    }
    if hash_etw:
        result["sha256"] = file_sha256(path)
    return result


def get_metric(session: dict[str, Any], name: str) -> float | None:
    module = session.get("plugin_module")
    return module["metrics"].get(name) if module else None


def function_percent(session: dict[str, Any], needle: str) -> float:
    return sum(
        row["event_percent"]
        for row in session["top_plugin_functions"]
        if needle in row["name"]
    )


def plugin_percent(session: dict[str, Any]) -> float:
    module = session.get("plugin_module")
    return module["event_percent"] if module else 0.0


def fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def gib(value: float) -> float:
    return value / (1024.0**3)


def module_share(session: dict[str, Any], name: str) -> float:
    return sum(
        row["event_percent"] for row in session["top_modules"] if row["name"] == name.lower()
    )


def etw_share(session: dict[str, Any], name: str) -> float:
    return session["vspipe_module_percent"].get(name.lower(), 0.0)


def render_markdown(data: dict[str, Any]) -> str:
    manifest = data["manifest"]
    tbp = data["tbp"]
    assess = data["assess"]
    pcm = data["pcm"]
    memory = data["process_memory"]
    etw = data["etw"]
    expected_counts = data["expected_counts"]
    # Older profile manifests predate the explicit flag; an empty ETW set is
    # unambiguous for those bundles and should not invalidate PCM/PMU review.
    etw_skipped = bool(manifest.get("skip_etw", not etw))
    verdict = (
        "This bundle is suitable for expert review of the exact current CPU DLL. "
        "All expected sessions completed, all TBP reports have user call-stack sampling, "
        "current source symbols resolve, and all PCM sessions contain nonzero "
        "memory-controller data. ETW collection was skipped for this run."
        if etw_skipped
        else
        "This bundle is suitable for expert review of the exact current CPU DLL. All expected sessions completed, all TBP reports have user call-stack sampling, current source symbols resolve, all PCM sessions contain nonzero memory-controller data, and both ETW traces report zero lost events and buffers."
    )
    etw_observation = (
        "ETW was skipped in this bundle; the sweep interpretation uses TBP, PMU, and process-memory polling only."
        if etw_skipped
        else
        "The ETW fixed Bicubic trace is executor-dominated, while the getfnative sweep spends a much smaller fraction inside dsmvc and materially more in the kernel, VapourSynth, and VC runtime. Sweep optimization must therefore include planner/cache allocation and scheduler behavior, not only AVX2 recurrence tuning."
    )
    etw_limit = (
        "ETW was skipped for this run. The process-memory polling traces separate startup high-water marks from post-800 ms values but do not identify allocation call sites."
        if etw_skipped
        else
        "WPR collected CPU/context-switch ETW, not heap allocation stacks. The process-memory polling traces separate startup high-water marks from post-800 ms values but do not identify allocation call sites."
    )
    lines = [
        "# Current CPU profile summary",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "The fixed-kernel profiles are strongly plugin-attributed. The sweep profiles intentionally expose VapourSynth scheduling, runtime memory movement, and planner/cache overhead in addition to the dsmvc executor.",
        "",
        "## Identity",
        "",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- Plugin SHA-256: `{manifest['plugin_sha256']}`",
        f"- PDB SHA-256: `{manifest['pdb_sha256']}`",
        f"- Input SHA-256: `{manifest['image_sha256']}`",
        f"- AMD uProf: `{manifest['uprof_cli_version']}`",
        f"- CPU: `{manifest['cpu']['Name'].strip()}` ({manifest['cpu']['NumberOfCores']}C/{manifest['cpu']['NumberOfLogicalProcessors']}T)",
        f"- VS threads: `{manifest['threads']}`; fixed frames: `{manifest['kernel_frames']}`; sweep frames: `{manifest['sweep_frames']}`; PCM frames: `{manifest['pcm_frames']}`",
        "- Runtime isolation: only `Imwri.dll` is autoloaded; the snapshot `dsmvc.dll` is loaded explicitly.",
        "",
        f"ETW collection: `{'skipped' if etw_skipped else '2 traces collected'}`.",
        "",
        "## Inventory",
        "",
        "| Kind | Complete | Expected |",
        "|---|---:|---:|",
    ]
    for kind, expected in expected_counts.items():
        lines.append(f"| {kind} | {data['session_counts'].get(kind, 0)} | {expected} |")

    lines.extend(
        [
            "",
            "## Fixed-kernel TBP",
            "",
            "Percentages are shares of sampled whole-process CPU time.",
            "",
            "| Case | Req | dsmvc | Columns | Horizontal | Transpose | Other dsmvc |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for session in sorted(
        (row for row in tbp if not row["case"].startswith(("getfnative", "selectkernel"))),
        key=lambda row: (row["case"], row["requests"]),
    ):
        columns = function_percent(session, "solve_columns_")
        horizontal = function_percent(session, "solve_horizontal_")
        transpose = function_percent(session, "transpose_source")
        other = max(0.0, plugin_percent(session) - columns - horizontal - transpose)
        lines.append(
            f"| {session['case']} | {session['requests']} | {fmt(plugin_percent(session))}% | {fmt(columns)}% | {fmt(horizontal)}% | {fmt(transpose)}% | {fmt(other)}% |"
        )

    fixed_tbp = [
        row
        for row in tbp
        if not row["case"].startswith(("getfnative", "selectkernel"))
    ]
    by_case: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for session in fixed_tbp:
        by_case[session["case"]][session["requests"]] = session
    lines.extend(
        [
            "",
            "### Profiled concurrency context",
            "",
            "These are the VSPipe target rates printed inside each uProf collect log, not the authoritative benchmark. They are included only to show scaling shape under the profiler.",
            "",
            "| Case | r1 fps | r8 fps | r32 fps | r8/r1 | r32/r1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case, requests in sorted(by_case.items()):
        rates = {
            value: requests[value]["target_output"]["fps"]
            for value in (1, 8, 32)
        }
        lines.append(
            f"| {case} | {fmt(rates[1], 1)} | {fmt(rates[8], 1)} | {fmt(rates[32], 1)} | {fmt(rates[8] / rates[1], 3)}x | {fmt(rates[32] / rates[1], 3)}x |"
        )

    lines.extend(
        [
            "",
            "## Assess Extended at 32 requests",
            "",
            "Metrics below are for `dsmvc.dll`, not the entire system.",
            "",
            "| Case | Cycle share | CPI | L1 miss | DRAM refill PTI | Local-cache refill PTI | Local-L2 refill PTI | STLI PTI | AVX stalls PTC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for session in sorted(assess, key=lambda row: row["case"]):
        lines.append(
            "| {case} | {share}% | {cpi} | {l1}% | {dram} | {cache} | {l2} | {stli} | {avx} |".format(
                case=session["case"],
                share=fmt(plugin_percent(session)),
                cpi=fmt(get_metric(session, "CPI"), 3),
                l1=fmt(get_metric(session, "%L1_DC_MISSES"), 2),
                dram=fmt(get_metric(session, "L1_DEMAND_DC_REFILLS_LOCAL_DRAM (PTI)"), 2),
                cache=fmt(get_metric(session, "L1_DEMAND_DC_REFILLS_LOCAL_CACHE (PTI)"), 2),
                l2=fmt(get_metric(session, "L1_DEMAND_DC_REFILLS_LOCAL_L2 (PTI)"), 2),
                stli=fmt(get_metric(session, "STLI_OTHER (PTI)"), 2),
                avx=fmt(get_metric(session, "SSE_AVX_STALLS (PTC)"), 4),
            )
        )

    lines.extend(
        [
            "",
            "## PCM memory bandwidth",
            "",
            "PCM is socket-wide and therefore not process-attributed. The interior median drops the first and last one-second samples.",
            "",
            "| Case | Req | Target fps | Cumulative | Read | Write | Median | Interior median | p95 | Range |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for session in sorted(pcm, key=lambda row: (row["case"], row["requests"])):
        samples = session["sample_total_gbps"]
        lines.append(
            f"| {session['case']} | {session['requests']} | {fmt(session['target_output']['fps'])} | {fmt(session['cumulative_total_gbps'])} | {fmt(session['cumulative_read_gbps'])} | {fmt(session['cumulative_write_gbps'])} | {fmt(samples['median'])} | {fmt(samples['interior_median'])} | {fmt(samples['p95'])} | {fmt(samples['min'])}-{fmt(samples['max'])} GB/s |"
        )

    lines.extend(
        [
            "",
            "## VSPipe process memory",
            "",
            "The API peak is a high-water mark and includes startup. Post-warm-up values use samples at or after 800 ms, matching the earlier process-memory report.",
            "",
            "| Case | Req | Duration | API peak WS | Steady WS med / max | Steady private med / max | Final private | Threads | CPU time |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for session in sorted(memory, key=lambda row: (row["case"], row["requests"])):
        lines.append(
            f"| {session['case']} | {session['requests']} | {fmt(session['duration_seconds'])} s | {fmt(gib(session['peak_working_set_bytes']))} GiB | {fmt(gib(session['steady_working_set_median_bytes']))} / {fmt(gib(session['steady_working_set_max_bytes']))} GiB | {fmt(gib(session['steady_private_median_bytes']))} / {fmt(gib(session['steady_private_max_bytes']))} GiB | {fmt(gib(session['final_private_bytes']))} GiB | {session['max_thread_count']} | {fmt(session['cpu_seconds'])} s |"
        )

    lines.extend(
        [
            "",
        "## ETW CPU and scheduling",
            "",
            "Module shares are normalized within the VSPipe process. CPU time is derived from context switches. The section is empty when `-SkipEtw` was used.",
            "",
            "| Case | Trace wall | VSPipe life | VSPipe CPU | Avg cores | dsmvc | VapourSynth | VC runtime | Kernel | Unknown | Threads | CSwitch / Ready / Samples | Lost |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for session in sorted(etw, key=lambda row: row["case"]):
        lines.append(
            f"| {session['case']} | {fmt(session['trace_wall_seconds'])} s | {fmt(session['vspipe_lifetime_seconds'])} s | {fmt(session['vspipe_cpu_seconds'])} s | {fmt(session['vspipe_average_cores'])} | {fmt(etw_share(session, 'dsmvc.dll'))}% | {fmt(etw_share(session, 'vapoursynth.dll'))}% | {fmt(etw_share(session, 'vcruntime140.dll'))}% | {fmt(etw_share(session, 'ntoskrnl.exe'))}% | {fmt(etw_share(session, 'unknown'))}% | {session['vspipe_thread_count']} | {session['cswitch_events']} / {session['ready_thread_events']} / {session['sampled_profile_events']} | {session['lost_events']}/{session['lost_buffers']} |"
        )

    sweep_tbp = [row for row in tbp if row["case"] in {"getfnative", "getfnative_v2", "selectkernel"}]
    lines.extend(
        [
            "",
            "## Sweep attribution",
            "",
            "| Case | dsmvc | VapourSynth | VC runtime | Kernel | Top dsmvc functions |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for session in sorted(sweep_tbp, key=lambda row: row["case"]):
        functions = ", ".join(
            f"`{row['name']}` {fmt(row['event_percent'])}%"
            for row in session["top_plugin_functions"][:4]
        )
        lines.append(
            f"| {session['case']} | {fmt(plugin_percent(session))}% | {fmt(module_share(session, 'vapoursynth.dll'))}% | {fmt(module_share(session, 'vcruntime140.dll'))}% | {fmt(module_share(session, 'ntoskrnl.exe'))}% | {functions} |"
        )

    b3_r32 = next(row for row in tbp if row["case"] == "bicubic_b3" and row["requests"] == 32)
    b5_r32 = next(row for row in tbp if row["case"] == "lanczos3_b5" and row["requests"] == 32)
    b7_r32 = next(row for row in tbp if row["case"] == "spline64_b7" and row["requests"] == 32)
    r1_pcm = next(row for row in pcm if row["case"] == "bicubic_b3" and row["requests"] == 1)
    r32_pcm = next(row for row in pcm if row["case"] == "bicubic_b3" and row["requests"] == 32)
    max_memory = max(memory, key=lambda row: row["steady_working_set_median_bytes"])
    lines.extend(
        [
            "",
            "## Review observations",
            "",
            f"- Bicubic b3 at r32 is almost entirely executor work: columns {fmt(function_percent(b3_r32, 'solve_columns_'))}%, horizontal {fmt(function_percent(b3_r32, 'solve_horizontal_'))}%, transpose {fmt(function_percent(b3_r32, 'transpose_source'))}% of whole-process sampled CPU.",
            f"- Bandwidth 5 and 7 use the paired column path. At r32, Lanczos3 b5 spends {fmt(function_percent(b5_r32, 'solve_columns_pair'))}% in `solve_columns_pair` and {fmt(function_percent(b5_r32, 'solve_horizontal_'))}% in `solve_horizontal_generic`; Spline64 b7 spends {fmt(function_percent(b7_r32, 'solve_columns_pair'))}% and {fmt(function_percent(b7_r32, 'solve_horizontal_'))}% respectively.",
            f"- Bicubic memory bandwidth rises from {fmt(r1_pcm['cumulative_total_gbps'])} GB/s at r1 to {fmt(r32_pcm['cumulative_total_gbps'])} GB/s at r32. Fixed-kernel r32 cases cluster tightly around 38-40.5 GB/s, which supports a shared data-movement ceiling rather than a kernel-specific bandwidth anomaly.",
            f"- The largest steady working-set median is {fmt(gib(max_memory['steady_working_set_median_bytes']))} GiB in `{max_memory['case']}` (steady max {fmt(gib(max_memory['steady_working_set_max_bytes']))} GiB). This is process-level evidence for reviewing plan/cache and per-node frame retention; PCM alone cannot answer that question.",
            f"- {etw_observation}",
            "",
            "## Limitations",
            "",
            "- This bundle profiles only the current dsmvc DLL. It contains no same-run old-plugin control, so it cannot establish an old/new speedup; use the independent three-run, 500-frame process benchmark for that comparison.",
            "- The 500-frame getfnative trace covers only the first Bilinear segment (heights 700.0-749.9). getfnative_v2 covers its Bilinear segment plus the start of the first Bicubic segment. selectkernel cycles through its full 101-candidate list.",
            f"- PCM counters are package-wide; background traffic is included. Use the repeated fixed-case agreement and {'ETW/process attribution' if not etw_skipped else 'process-memory attribution'} together rather than treating PCM as proof of process-local DRAM saturation.",
            "- Only `pcm-memory-*` sessions are valid. The older `pcm-*` directories requested combined memory/cache metrics, contain zero tables with `No data available`, and are deliberately excluded from the manifest and summary.",
            "- Assess Extended uses sampled PMU events. Very small per-function event counts are noisy; module-level metrics and large hot functions are the defensible review surface.",
            f"- {etw_limit}",
            "- TBP percentages are CPU-sample shares, not wall-clock percentages. Cross-check throughput decisions against the separate 500-frame, 32-request, three-run VSPipe benchmark.",
            "",
            "## Review entry points",
            "",
            f"- Manifest: `{data['manifest_path']}`",
            f"- Machine-readable summary: `{data['json_path']}`",
            f"- Exact binary directory: `{Path(manifest['plugin']).parent}`",
            f"- Post-processing command record: `{Path(data['manifest_path']).parent / 'review-postprocess-commands.md'}`",
            "- Each TBP/Assess session contains raw uProf databases plus `report-percentage.csv`.",
            "- Each ETW trace has `cswitch-process`, `cswitch-thread`, `profile-detail`, and `tracestats` text exports beside the raw `.etl`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("--hash-etw", action="store_true")
    args = parser.parse_args()
    root = args.profile_dir.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    sessions = manifest["sessions"]
    incomplete = [row for row in sessions if row["status"] not in {"complete", "existing"}]
    if incomplete:
        raise ValueError(f"incomplete sessions: {[row['name'] for row in incomplete]}")
    counts: dict[str, int] = defaultdict(int)
    for session in sessions:
        counts[session["kind"]] += 1
    skip_etw = bool(manifest.get("skip_etw", counts.get("etw_cpu", 0) == 0))
    expected_counts = dict(EXPECTED_COUNTS)
    if skip_etw:
        expected_counts.pop("etw_cpu", None)
    mismatches = {
        kind: (counts.get(kind, 0), expected)
        for kind, expected in expected_counts.items()
        if counts.get(kind, 0) != expected
    }
    if mismatches:
        raise ValueError(f"unexpected session counts: {mismatches}")

    parsed_uprof = [parse_uprof_session(row) for row in sessions if row["kind"] in PROFILE_KINDS]
    tbp = [row for row in parsed_uprof if row["kind"] == "tbp"]
    assess = [row for row in parsed_uprof if row["kind"] == "assess_ext"]
    if any(row["call_stack_sampling"] != "True" for row in tbp):
        raise ValueError("one or more TBP reports lack call-stack sampling")
    if any(not row["symbols_resolved"] for row in parsed_uprof):
        raise ValueError("one or more uProf reports lack resolved dsmvc symbols")
    if any(row["unwanted_cuda_module"] for row in parsed_uprof):
        raise ValueError("an unwanted vsnlm_cuda module appears in a report")

    output_json = root / "profile-summary.json"
    output_markdown = root / "profile-summary.md"
    result: dict[str, Any] = {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "json_path": str(output_json),
        "markdown_path": str(output_markdown),
        "manifest": {key: value for key, value in manifest.items() if key != "sessions"},
        "session_counts": dict(counts),
        "expected_counts": expected_counts,
        "validation": {
            "all_sessions_complete": True,
            "expected_counts_match": True,
            "tbp_call_stacks": True,
            "symbols_resolved": True,
            "unwanted_cuda_modules": False,
            "pcm_nonzero": True,
            "etw_collected": not skip_etw,
            "etw_zero_loss": None if skip_etw else True,
        },
        "tbp": tbp,
        "assess": assess,
        "pcm": [parse_pcm(row) for row in sessions if row["kind"] == "pcm"],
        "process_memory": [
            parse_process_memory(row) for row in sessions if row["kind"] == "process_memory"
        ],
        "etw": [parse_etw(row, args.hash_etw) for row in sessions if row["kind"] == "etw_cpu"],
    }
    if not skip_etw and any(row["lost_events"] or row["lost_buffers"] for row in result["etw"]):
        raise ValueError("one or more ETW traces lost events or buffers")
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    output_markdown.write_text(render_markdown(result), encoding="utf-8")
    print(output_markdown)
    print(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
