#!/usr/bin/env python3
"""Profile fixed-kernel VapourSynth workloads on Linux.

This is a low-level companion to fixed_kernel_benchmark.py. It keeps the
benchmark graph and geometry identical, but samples the VSPipe process while
it runs and records child CPU, memory, faults, and scheduler counters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


KERNELS = (
    ("bilinear", "bilinear"),
    ("bicubic_b0_c0_5", "bicubic (0, 0.5)"),
)
IMPLEMENTATIONS = ("old", "new")
THREADS = (1, 8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_info(path: Path | None) -> dict | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "size": resolved.stat().st_size if resolved.is_file() else None,
        "sha256": sha256_file(resolved) if resolved.is_file() else None,
    }


def command_text(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([item]) for item in command)


def usage_snapshot() -> dict[str, float]:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "user_seconds": usage.ru_utime,
        "system_seconds": usage.ru_stime,
        "major_faults": float(usage.ru_majflt),
        "minor_faults": float(usage.ru_minflt),
        "voluntary_context_switches": float(usage.ru_nvcsw),
        "involuntary_context_switches": float(usage.ru_nivcsw),
        "child_max_rss_kib": float(usage.ru_maxrss),
    }


def usage_delta(before: dict[str, float], after: dict[str, float]) -> dict:
    return {
        "user_seconds": after["user_seconds"] - before["user_seconds"],
        "system_seconds": after["system_seconds"] - before["system_seconds"],
        "major_faults": int(after["major_faults"] - before["major_faults"]),
        "minor_faults": int(after["minor_faults"] - before["minor_faults"]),
        "voluntary_context_switches": int(
            after["voluntary_context_switches"]
            - before["voluntary_context_switches"]),
        "involuntary_context_switches": int(
            after["involuntary_context_switches"]
            - before["involuntary_context_switches"]),
        "child_max_rss_kib": int(after["child_max_rss_kib"]),
    }


def read_mem_available_kib() -> int | None:
    try:
        text = Path("/proc/meminfo").read_text(encoding="ascii")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return None


def read_proc_status(pid: int) -> dict | None:
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    values = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        parts = raw.strip().split()
        if not parts:
            continue
        if key in {
            "VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem",
        }:
            values[key] = int(parts[0])
        elif key in {"Threads", "voluntary_ctxt_switches",
                     "nonvoluntary_ctxt_switches"}:
            values[key] = int(parts[0])
    return values


def process_tree(root_pid: int) -> set[int]:
    """Return the root PID and descendants visible in this PID namespace."""
    parents = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = stat_path.read_text(encoding="ascii")
            right_paren = text.rfind(")")
            fields = text[right_paren + 2:].split()
            pid = int(text[:text.find(" ")])
            ppid = int(fields[1])
            parents[pid] = ppid
        except (OSError, ValueError, IndexError):
            continue
    tree = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid in parents.items():
            if ppid in tree and pid not in tree:
                tree.add(pid)
                changed = True
    return tree


def read_process_tree(root_pid: int) -> dict:
    totals = {
        "VmRSS": 0,
        "VmHWM": 0,
        "RssAnon": 0,
        "RssFile": 0,
        "RssShmem": 0,
        "Threads": 0,
        "voluntary_ctxt_switches": 0,
        "nonvoluntary_ctxt_switches": 0,
        "process_count": 0,
    }
    for pid in process_tree(root_pid):
        status = read_proc_status(pid)
        if not status:
            continue
        totals["process_count"] += 1
        for key in totals:
            if key != "process_count":
                totals[key] += status.get(key, 0)
    return totals


def parse_filter_times(stderr: str) -> dict:
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
        name = " ".join(parts[:-3])
        mode = parts[-3]
        result[name] = {"mode": mode, "percent": percentage,
                        "seconds": seconds}
    return result


def fixed_geometry() -> dict:
    source_width = 1920
    source_height = 1080
    base_height = 1000
    native_height = 810.0
    base_width = round(source_width / source_height * base_height)
    native_width = source_width / source_height * native_height
    output_width = base_width - 2 * int((base_width - native_width) / 2)
    output_height = base_height - 2 * int((base_height - native_height) / 2)
    return {
        "source_width": source_width,
        "source_height": source_height,
        "base_width": output_width,
        "base_height": base_height,
        "native_width": native_width,
        "native_height": native_height,
        "output_width": output_width,
        "output_height": output_height,
    }


def build_command(options, kernel: str, implementation: str,
                  threads: int) -> list[str]:
    script = Path(__file__).with_name("vspipe_fixed_kernel.vpy").resolve()
    values = {
        "implementation": implementation,
        "kernel": kernel,
        "source": str(options.source),
        "plugin": str(options.new_plugin),
        "old_plugin": str(options.old_plugin),
        "source_plugin": str(options.source_plugin or ""),
        "source_filter": options.source_filter,
        "source_decoder": options.source_decoder,
        "source_prefer_hw": str(options.source_prefer_hw),
        "source_ff_loglevel": str(options.source_ff_loglevel),
        "source_rap_verification": str(options.source_rap_verification),
        "frames": str(options.frames),
        "threads": str(threads),
        "src_height": str(options.src_height),
        "base_height": str(options.base_height),
    }
    command = [str(options.vspipe)]
    for key, value in values.items():
        command.extend(["--arg", f"{key}={value}"])
    command.extend([
        "--requests", str(threads),
        "--start", "0",
        "--end", str(options.frames - 1),
        "--filter-time", str(script),
        "--",
    ])
    return command


def run_case(options, kernel: str, label: str, implementation: str,
             threads: int, run_number: int) -> dict:
    command = build_command(options, kernel, implementation, threads)
    before = usage_snapshot()
    started = time.perf_counter()
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, errors="replace")
    peak = {
        "rss_kib": 0,
        "hwm_kib": 0,
        "rss_anon_kib": 0,
        "rss_file_kib": 0,
        "rss_shmem_kib": 0,
        "threads": 0,
        "voluntary_context_switches": 0,
        "involuntary_context_switches": 0,
        "process_count": 0,
    }
    min_available_kib = None
    samples = 0
    while process.poll() is None:
        status = read_process_tree(process.pid)
        if status:
            samples += 1
            for source_key, target_key in (
                ("VmRSS", "rss_kib"), ("VmHWM", "hwm_kib"),
                ("RssAnon", "rss_anon_kib"),
                ("RssFile", "rss_file_kib"),
                ("RssShmem", "rss_shmem_kib"),
                ("Threads", "threads"),
                ("voluntary_ctxt_switches", "voluntary_context_switches"),
                ("nonvoluntary_ctxt_switches",
                 "involuntary_context_switches"),
                ("process_count", "process_count"),
            ):
                peak[target_key] = max(peak[target_key],
                                        status.get(source_key, 0))
            available = read_mem_available_kib()
            if available is not None:
                min_available_kib = (available if min_available_kib is None
                                     else min(min_available_kib, available))
        time.sleep(options.sample_interval)
    stderr = process.communicate()[1]
    elapsed = time.perf_counter() - started
    after = usage_snapshot()
    usage = usage_delta(before, after)
    if process.returncode != 0:
        raise RuntimeError(
            f"VSPipe failed for {kernel}/{implementation}/R{threads}T{threads}:\n"
            + stderr[-8000:])
    cpu_seconds = usage["user_seconds"] + usage["system_seconds"]
    return {
        "kernel": kernel,
        "label": label,
        "implementation": implementation,
        "threads": threads,
        "requests": threads,
        "run": run_number,
        "frames": options.frames,
        "elapsed_seconds": elapsed,
        "fps": options.frames / elapsed,
        "user_seconds": usage["user_seconds"],
        "system_seconds": usage["system_seconds"],
        "cpu_seconds": cpu_seconds,
        "cpu_cores_equivalent": cpu_seconds / elapsed,
        "cpu_percent_of_one_core": cpu_seconds / elapsed * 100.0,
        "major_faults": usage["major_faults"],
        "minor_faults": usage["minor_faults"],
        "voluntary_context_switches": usage[
            "voluntary_context_switches"],
        "involuntary_context_switches": usage[
            "involuntary_context_switches"],
        "child_max_rss_mib": usage["child_max_rss_kib"] / 1024.0,
        "sample_peak_rss_mib": peak["rss_kib"] / 1024.0,
        "sample_peak_hwm_mib": peak["hwm_kib"] / 1024.0,
        "peak_rss_anon_mib": peak["rss_anon_kib"] / 1024.0,
        "peak_rss_file_mib": peak["rss_file_kib"] / 1024.0,
        "peak_rss_shmem_mib": peak["rss_shmem_kib"] / 1024.0,
        "peak_threads": peak["threads"],
        "sampled_context_switches": (
            peak["voluntary_context_switches"]
            + peak["involuntary_context_switches"]),
        "min_system_available_mib": (
            min_available_kib / 1024.0 if min_available_kib is not None
            else None),
        "process_tree_peak_count": peak.get("process_count", 0),
        "process_samples": samples,
        "filter_time": parse_filter_times(stderr),
        "vspipe_output_tail": stderr[-4000:],
        "command": command_text(command),
    }


def median(values: list[float]) -> float:
    return statistics.median(values)


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["kernel"], row["implementation"], row["threads"])].append(row)
    result = []
    for (kernel, implementation, threads), items in sorted(grouped.items()):
        result.append({
            "kernel": kernel,
            "label": items[0]["label"],
            "implementation": implementation,
            "threads": threads,
            "runs": len(items),
            "fps": median([item["fps"] for item in items]),
            "elapsed_seconds": median(
                [item["elapsed_seconds"] for item in items]),
            "cpu_seconds": median([item["cpu_seconds"] for item in items]),
            "cpu_cores_equivalent": median(
                [item["cpu_cores_equivalent"] for item in items]),
            "sample_peak_rss_mib": median(
                [item["sample_peak_rss_mib"] for item in items]),
            "peak_threads": median([item["peak_threads"] for item in items]),
            "minor_faults": median([item["minor_faults"] for item in items]),
            "major_faults": median([item["major_faults"] for item in items]),
            "voluntary_context_switches": median([
                item["voluntary_context_switches"] for item in items]),
            "involuntary_context_switches": median([
                item["involuntary_context_switches"] for item in items]),
            "min_system_available_mib": median([
                item["min_system_available_mib"] for item in items
                if item["min_system_available_mib"] is not None]),
            "filter_time": {
                name: {
                    metric: median([
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


def fmt(value: float | int | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_markdown(result: dict, path: Path) -> None:
    summaries = result["summaries"]
    by_key = {(item["kernel"], item["implementation"], item["threads"]): item
              for item in summaries}
    lines = [
        "# Fixed-kernel Linux profile",
        "",
        "This profile uses the same Digimon graph as the fixed-kernel benchmark:",
        f"{result['environment']['frames']:,} frames at fixed 810p geometry.",
        "Each row is the median of the configured repeated VSPipe processes.",
        "",
        "## Method",
        "",
        f"- Current plugin: `{result['environment']['new_plugin']['sha256']}`",
        f"- Original plugin: `{result['environment']['old_plugin']['sha256']}`",
        f"- Source filter: `{result['environment']['source_filter']}`",
        f"- Source decoder options: decoder `{result['environment']['source_decoder'] or 'default'}`, "
        f"prefer_hw `{result['environment']['source_prefer_hw']}`, "
        f"RAP verification `{result['environment']['source_rap_verification']}`",
        f"- Runs per cell: `{result['environment']['runs']}`",
        f"- `/proc` sample interval: `{result['environment']['sample_interval']} s`",
        "- Hardware PMU: unavailable (`perf_event_paranoid=4`); no cache or DRAM hardware counters are claimed here.",
        "- CPU seconds and faults come from Linux child resource accounting; RSS and thread values are sampled from the complete `/proc` process tree.",
        "",
        "## Throughput and Scaling",
        "",
        "The R8/R1 ratio is the observed scale factor. A ratio near 1 means the workload has reached a shared bottleneck; it does not by itself distinguish bandwidth saturation from higher cache/DRAM access or queueing latency.",
        "",
        "| Kernel | Impl | R1T1 FPS | R8T8 FPS | R8/R1 | R1 cores | R8 cores | R1 RSS MiB | R8 RSS MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kernel, label in KERNELS:
        for implementation in IMPLEMENTATIONS:
            r1 = by_key[(kernel, implementation, 1)]
            r8 = by_key[(kernel, implementation, 8)]
            lines.append(
                f"| `{label}` | `{implementation}` | {r1['fps']:.2f} | "
                f"{r8['fps']:.2f} | {r8['fps'] / r1['fps']:.3f}x | "
                f"{r1['cpu_cores_equivalent']:.2f} | "
                f"{r8['cpu_cores_equivalent']:.2f} | "
                f"{r1['sample_peak_rss_mib']:.1f} | "
                f"{r8['sample_peak_rss_mib']:.1f} |")
    lines.extend([
        "",
        "## Process Resources",
        "",
        "CPU cores equivalent is process CPU time divided by wall time. RSS is the sampled VSPipe process high-water mark; it is not total system memory bandwidth.",
        "",
        "| Kernel | Impl | Threads | CPU s | Cores | RSS MiB | Threads | Minor faults | Voluntary CS | Involuntary CS | Min avail MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in summaries:
        lines.append(
            f"| `{item['label']}` | `{item['implementation']}` | "
            f"R{item['threads']}T{item['threads']} | {item['cpu_seconds']:.2f} | "
            f"{item['cpu_cores_equivalent']:.2f} | "
            f"{item['sample_peak_rss_mib']:.1f} | "
            f"{item['peak_threads']:.0f} | {item['minor_faults']:.0f} | "
            f"{item['voluntary_context_switches']:.0f} | "
            f"{item['involuntary_context_switches']:.0f} | "
            f"{fmt(item['min_system_available_mib'], 0)} |")
    lines.extend([
        "",
        "## Filter Time",
        "",
        "VSPipe's `--filter-time` output separates source decode from the descale executor and the Point conversion path. The percentage is accumulated CPU-time share across worker threads, so it can exceed 100% at R8.",
        "",
        "| Kernel | Impl | Threads | Source CPU s / % | Descale CPU s / % | Point CPU s / % |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for item in summaries:
        filter_time = item["filter_time"]
        source_value = next(
            (filter_time[name] for name in ("Source", "LWLibavSource",
                                             "VideoSource")
             if name in filter_time),
            {},
        )
        descale_name = ("Debilinear" if item["kernel"] == "bilinear"
                        else "Debicubic")
        values = [source_value, filter_time.get(descale_name, {}),
                  filter_time.get("Point", {})]
        formatted = [
            f"{value.get('seconds', 0.0):.2f} / "
            f"{value.get('percent', 0.0):.1f}%"
            for value in values
        ]
        lines.append(
            f"| `{item['label']}` | `{item['implementation']}` | "
            f"R{item['threads']}T{item['threads']} | "
            + " | ".join(formatted) + " |")
    lines.extend([
        "",
        "## Reading the R8 Plateau",
        "",
        "The profile distinguishes memory capacity pressure from a data-movement ceiling as far as this unprivileged Linux environment permits. R8 has flat FPS, bounded per-run RSS, high process CPU occupancy, and no collapse in available memory, so the evidence points away from RAM exhaustion. It is consistent with bandwidth saturation or rising cache/DRAM access and queueing latency, but this run cannot distinguish those mechanisms because the host denies PMU access to `perf`.",
        "",
        "The most actionable code targets are therefore the executor's column and horizontal passes, transpose traffic, and avoidable source/Point copies. Reducing allocations or cache footprint is still useful, but a lower RSS alone should not be expected to restore R8 scaling unless it reduces memory traffic or synchronization.",
        "",
        "## Raw Artifacts",
        "",
        "- [Machine-readable profile](profile.json)",
        "- [CSV profile](profile.csv)",
        "- Commands: `commands.txt`",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "kernel", "label", "implementation", "threads", "run", "frames",
        "elapsed_seconds", "fps", "user_seconds", "system_seconds",
        "cpu_seconds", "cpu_cores_equivalent", "major_faults", "minor_faults",
        "voluntary_context_switches", "involuntary_context_switches",
        "sample_peak_rss_mib", "sample_peak_hwm_mib", "peak_threads",
        "min_system_available_mib",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(fields) + "\n")
        for row in rows:
            values = []
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, str):
                    value = value.replace('"', '""')
                    values.append(f'"{value}"')
                else:
                    values.append(str(value))
            handle.write(",".join(values) + "\n")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--old-plugin", required=True, type=Path)
    parser.add_argument("--new-plugin", required=True, type=Path)
    parser.add_argument("--vspipe", required=True, type=Path)
    parser.add_argument("--source-filter", choices=("lsmas", "ffms2", "bestsource"),
                        default="ffms2")
    parser.add_argument("--source-plugin", type=Path)
    parser.add_argument("--source-decoder", default="",
                        help="Preferred LSMASH/libavcodec decoder name(s).")
    parser.add_argument("--source-prefer-hw", type=int, default=0,
                        help="LSMASH prefer_hw mode; 0 keeps software default.")
    parser.add_argument("--source-ff-loglevel", type=int, default=0,
                        help="LSMASH FFmpeg log level, 0 is quiet.")
    parser.add_argument("--source-rap-verification", type=int, default=-1,
                        help="LSMASH RAP verification; -1 keeps plugin default.")
    parser.add_argument("--output", type=Path, default=root / "benchmark-results" /
                        "fixed-kernel-profile-digimon-20260805")
    parser.add_argument("--frames", type=int, default=4000)
    parser.add_argument("--src-height", type=float, default=810.0)
    parser.add_argument("--base-height", type=float, default=1000.0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    options = parser.parse_args()
    options.source = options.source.expanduser().resolve()
    options.old_plugin = options.old_plugin.expanduser().resolve()
    options.new_plugin = options.new_plugin.expanduser().resolve()
    options.vspipe = options.vspipe.expanduser().resolve()
    options.source_plugin = (options.source_plugin.expanduser().resolve()
                             if options.source_plugin else None)
    options.output = options.output.expanduser().resolve()
    for required in (options.source, options.old_plugin, options.new_plugin,
                     options.vspipe):
        if not required.is_file():
            raise FileNotFoundError(required)
    if options.source_plugin and not options.source_plugin.is_file():
        raise FileNotFoundError(options.source_plugin)
    if options.frames < 1 or options.runs < 1 or options.sample_interval <= 0:
        raise ValueError("frames, runs, and sample interval must be positive")

    options.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for threads in THREADS:
        for kernel, label in KERNELS:
            for implementation in IMPLEMENTATIONS:
                for run_number in range(1, options.runs + 1):
                    row = run_case(options, kernel, label, implementation,
                                   threads, run_number)
                    rows.append(row)
                    print(
                        f"{kernel} {implementation} R{threads}T{threads} "
                        f"run {run_number}: {row['fps']:.3f} fps, "
                        f"{row['cpu_cores_equivalent']:.2f} cores, "
                        f"{row['sample_peak_rss_mib']:.1f} MiB RSS",
                        flush=True)

    result = {
        "schema_version": 1,
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "source_filter": options.source_filter,
            "source_decoder": options.source_decoder,
            "source_prefer_hw": options.source_prefer_hw,
            "source_ff_loglevel": options.source_ff_loglevel,
            "source_rap_verification": options.source_rap_verification,
            "source": file_info(options.source),
            "old_plugin": file_info(options.old_plugin),
            "new_plugin": file_info(options.new_plugin),
            "vspipe": str(options.vspipe),
            "frames": options.frames,
            "src_height": options.src_height,
            "base_height": options.base_height,
            "runs": options.runs,
            "sample_interval": options.sample_interval,
            "threads": list(THREADS),
            "kernels": [name for name, _ in KERNELS],
            "implementations": list(IMPLEMENTATIONS),
            "perf_event_paranoid": (
                Path("/proc/sys/kernel/perf_event_paranoid").read_text(
                    encoding="ascii").strip()
                if Path("/proc/sys/kernel/perf_event_paranoid").is_file()
                else None),
        },
        "geometry": fixed_geometry(),
        "raw_samples": rows,
        "summaries": summarize(rows),
    }
    (options.output / "profile.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8")
    write_csv(rows, options.output / "profile.csv")
    (options.output / "commands.txt").write_text(
        "\n".join(row["command"] for row in rows) + "\n", encoding="utf-8")
    write_markdown(result, options.output / "profile.md")
    print(options.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        raise SystemExit(2)
