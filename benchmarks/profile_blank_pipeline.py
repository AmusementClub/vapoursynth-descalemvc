#!/usr/bin/env python3
"""Profile the in-memory BlankClip pipeline without multiplexed PMU groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


KERNELS = ("b1", "b3", "b5", "b7")
FORMATS = ("float32", "gray16")
THREADS = (1, 8, 32)
IMPLEMENTATIONS = ("baseline", "candidate")
PERF_GROUPS = {
    "cpi": ("cycles", "instructions"),
    "l2": (
        "instructions",
        "l2_request_g1.all_no_prefetch",
        "l2_cache_misses_from_dc_misses",
    ),
    "fills": (
        "ls_dmnd_fills_from_sys.lcl_l2",
        "ls_dmnd_fills_from_sys.int_cache",
        "ls_dmnd_fills_from_sys.ext_cache_local",
        "ls_dmnd_fills_from_sys.mem_io_local",
        "ls_dmnd_fills_from_sys.mem_io_remote",
    ),
}
OUTPUT_RE = re.compile(
    r"Output\s+(?P<frames>\d+)\s+frames\s+in\s+"
    r"(?P<seconds>[0-9.]+)\s+seconds\s+\((?P<fps>[0-9.]+)\s+fps\)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_info(path: Path) -> dict:
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def perf_paranoid() -> int | None:
    try:
        return int(Path("/proc/sys/kernel/perf_event_paranoid").read_text(
            encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def build_vspipe_command(options, implementation: str, format_name: str,
                         kernel: str, threads: int, frames: int) -> list[str]:
    plugin = (options.baseline_plugin if implementation == "baseline"
              else options.candidate_plugin)
    values = {
        "plugin": plugin,
        "kernel": kernel,
        "format_name": format_name,
        "frames": frames,
        "threads": threads,
        "source_width": options.source_width,
        "source_height": options.source_height,
        "native_height": options.native_height,
        "base_height": options.base_height,
    }
    command = [str(options.vspipe)]
    for key, value in values.items():
        command.extend(["--arg", f"{key}={value}"])
    command.extend([
        "--requests", str(threads),
        "--start", "0",
        "--end", str(frames - 1),
        "--filter-time",
        str(Path(__file__).with_name("vspipe_blank_pipeline.vpy").resolve()),
        "--",
    ])
    return command


def parse_vspipe_output(output: str, context: str) -> dict:
    match = OUTPUT_RE.search(output)
    if not match:
        raise RuntimeError(f"VSPipe did not report FPS for {context}:\n"
                           + output[-5000:])
    frames = int(match.group("frames"))
    seconds = float(match.group("seconds"))
    return {
        "frames": frames,
        "seconds": seconds,
        "fps": float(match.group("fps")),
    }


def normalize_perf_event(name: str) -> str:
    name = name.strip().strip('"')
    if name.startswith("cpu/"):
        name = name[4:]
        slash = name.find("/")
        if slash >= 0:
            name = name[:slash]
    return name.split(":", 1)[0]


def parse_number(value: str) -> float | None:
    value = value.strip()
    if not value or value.startswith("<"):
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def parse_perf_csv(path: Path, expected: tuple[str, ...]) -> dict:
    counts = {}
    running_percent = {}
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle, delimiter=";"):
            if len(row) < 3 or row[0].lstrip().startswith("#"):
                continue
            event = normalize_perf_event(row[2])
            if event not in expected:
                continue
            count = parse_number(row[0])
            if count is None:
                continue
            counts[event] = count
            if len(row) > 4:
                percent = parse_number(row[4])
                if percent is not None:
                    running_percent[event] = percent
    missing = sorted(set(expected) - set(counts))
    if missing:
        raise RuntimeError(f"perf did not count events: {', '.join(missing)}")
    low = {event: percent for event, percent in running_percent.items()
           if percent < 99.0}
    if low:
        detail = ", ".join(f"{event}={percent:.2f}%"
                           for event, percent in sorted(low.items()))
        raise RuntimeError(f"perf multiplexed or descheduled an event group: {detail}")
    return {"counts": counts, "running_percent": running_percent}


def run_checked(command: list[str], context: str) -> str:
    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace", check=False)
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"{context} failed with exit {completed.returncode}:\n"
            + output[-8000:])
    return output


def warm_up(options, implementation: str, format_name: str,
            kernel: str, threads: int) -> None:
    command = build_vspipe_command(
        options, implementation, format_name, kernel,
        threads, options.warmup_frames)
    output = run_checked(command, "profile warmup")
    parse_vspipe_output(output, "profile warmup")


def sample_id(phase: str, format_name: str, kernel: str, threads: int,
              implementation: str, run: int, group: str = "") -> str:
    suffix = f"-{group}" if group else ""
    return (f"{phase}-{format_name}-{kernel}-r{threads}-"
            f"{implementation}-run{run}{suffix}")


def run_perf_sample(options, output: Path, format_name: str, kernel: str,
                    threads: int, implementation: str, run: int,
                    group: str) -> dict:
    events = PERF_GROUPS[group]
    identifier = sample_id(
        "perf", format_name, kernel, threads, implementation, run, group)
    perf_file = output / "perf" / f"{identifier}.csv"
    command = build_vspipe_command(
        options, implementation, format_name, kernel,
        threads, options.perf_frames)
    perf_command = [
        str(options.perf), "stat", "--all-user", "--no-big-num",
        "-x", ";", "-o", str(perf_file),
        "-e", "{" + ",".join(events) + "}",
        "--", *command,
    ]
    warm_up(options, implementation, format_name, kernel, threads)
    started = time.perf_counter()
    process_output = run_checked(perf_command, identifier)
    elapsed = time.perf_counter() - started
    measured = parse_perf_csv(perf_file, events)
    vspipe = parse_vspipe_output(process_output, identifier)
    return {
        "id": identifier,
        "phase": "perf",
        "group": group,
        "format": format_name,
        "kernel": kernel,
        "threads": threads,
        "implementation": implementation,
        "run": run,
        "wall_seconds": elapsed,
        "vspipe": vspipe,
        "counts": measured["counts"],
        "running_percent": measured["running_percent"],
        "raw_file": str(perf_file),
        "command": subprocess.list2cmdline(perf_command),
    }


def parse_pcm_cumulative(path: Path) -> dict:
    wanted = {
        "Total Mem Bw (GB/s)": "total_gbps",
        "Total Mem RdBw (GB/s)": "read_gbps",
        "Total Mem WrBw (GB/s)": "write_gbps",
    }
    result = {}
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0] not in wanted or len(row) < 2:
                continue
            value = parse_number(row[1])
            if value is not None:
                result["cumulative_" + wanted[row[0]]] = value
    expected = {"cumulative_" + name for name in wanted.values()}
    missing = sorted(expected - set(result))
    if missing or result.get("cumulative_total_gbps", 0.0) <= 0.0:
        raise RuntimeError(
            f"PCM report has no usable memory bandwidth ({', '.join(missing)}): "
            f"{path}")
    return result


def parse_pcm_reports(cumulative_path: Path, timeseries_path: Path) -> dict:
    result = parse_pcm_cumulative(cumulative_path)
    with timeseries_path.open(
            encoding="utf-8-sig", errors="replace", newline="") as handle:
        rows = list(csv.reader(handle))
    header_index = next((
        index for index, row in enumerate(rows)
        if row and row[0] == "Total Mem Bw (GB/s)"), None)
    if header_index is None:
        raise RuntimeError(f"PCM timeseries header is missing: {timeseries_path}")
    samples = []
    for row in rows[header_index + 1:]:
        if len(row) < 3:
            break
        values = [parse_number(value) for value in row[:3]]
        if any(value is None for value in values):
            break
        samples.append({
            "total_gbps": values[0],
            "read_gbps": values[1],
            "write_gbps": values[2],
        })
    if not samples or max(sample["total_gbps"] for sample in samples) <= 0.0:
        raise RuntimeError(f"PCM timeseries has no nonzero samples: {timeseries_path}")
    steady = samples[1:-1] if len(samples) > 2 else samples
    for metric in ("total_gbps", "read_gbps", "write_gbps"):
        result[metric] = statistics.median(
            sample[metric] for sample in steady)
    result["timeseries_sample_count"] = len(samples)
    result["timeseries_samples"] = samples
    return result


def run_pcm_sample(options, output: Path, format_name: str, kernel: str,
                   threads: int, implementation: str, run: int) -> dict:
    identifier = sample_id(
        "pcm", format_name, kernel, threads, implementation, run)
    session_root = output / "pcm" / (
        f"{identifier}-{time.time_ns()}")
    command = build_vspipe_command(
        options, implementation, format_name, kernel,
        threads, options.pcm_frames)
    pcm_command = [
        str(options.pcm), "profile", "-m", "memory", "-a",
        "-I", "1000", "-P", "4", "-q", "-O", str(session_root),
        "--", *command,
    ]
    warm_up(options, implementation, format_name, kernel, threads)
    started = time.perf_counter()
    process_output = run_checked(pcm_command, identifier)
    elapsed = time.perf_counter() - started
    vspipe = parse_vspipe_output(process_output, identifier)
    cumulative_reports = list(session_root.rglob("report-cumulative.csv"))
    timeseries_reports = list(session_root.rglob("report-timeseries.csv"))
    if len(cumulative_reports) != 1 or len(timeseries_reports) != 1:
        raise RuntimeError(
            f"expected one PCM cumulative and timeseries report for {identifier}, "
            f"found {len(cumulative_reports)} and {len(timeseries_reports)}")
    bandwidth = parse_pcm_reports(
        cumulative_reports[0], timeseries_reports[0])
    return {
        "id": identifier,
        "phase": "pcm",
        "format": format_name,
        "kernel": kernel,
        "threads": threads,
        "implementation": implementation,
        "run": run,
        "wall_seconds": elapsed,
        "vspipe": vspipe,
        "bandwidth": bandwidth,
        "session": str(session_root),
        "cumulative_report": str(cumulative_reports[0]),
        "timeseries_report": str(timeseries_reports[0]),
        "command": subprocess.list2cmdline(pcm_command),
    }


def write_samples(samples: list[dict], path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(samples, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def sum_optional(values) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def throughput_index(throughput: dict) -> dict:
    return {
        (case["format"], case["kernel"], case["threads"],
         case["implementation"]): case
        for case in throughput["cases"]
    }


def summarize_samples(options, samples: list[dict], throughput: dict) -> list[dict]:
    perf = defaultdict(list)
    pcm = defaultdict(list)
    for sample in samples:
        key = (sample["format"], sample["kernel"], sample["threads"],
               sample["implementation"])
        if sample["phase"] == "perf":
            perf[(key, sample["group"])].append(sample)
        elif sample["phase"] == "pcm":
            pcm[key].append(sample)
    throughput_cases = throughput_index(throughput)
    summaries = []
    for format_name in options.formats:
        for kernel in options.kernels:
            for threads in options.threads:
                for implementation in IMPLEMENTATIONS:
                    key = (format_name, kernel, threads, implementation)
                    item = {
                        "format": format_name,
                        "kernel": kernel,
                        "threads": threads,
                        "implementation": implementation,
                    }
                    throughput_case = throughput_cases[key]
                    item["fps"] = throughput_case["vspipe_fps"]["median"]
                    item["cpu_ms_per_frame"] = throughput_case[
                        "cpu_ms_per_frame"]["median"]

                    groups = {}
                    for group in PERF_GROUPS:
                        group_samples = perf.get((key, group), [])
                        if not group_samples:
                            continue
                        counts = {}
                        for event in PERF_GROUPS[group]:
                            counts[event] = median([
                                sample["counts"][event] for sample in group_samples])
                        groups[group] = {
                            "counts": counts,
                            "frames": median([
                                sample["vspipe"]["frames"]
                                for sample in group_samples]),
                            "fps": median([
                                sample["vspipe"]["fps"]
                                for sample in group_samples]),
                        }
                    item["perf_groups"] = groups
                    cpi_counts = groups.get("cpi", {}).get("counts", {})
                    cpi_frames = groups.get("cpi", {}).get("frames")
                    item["cpi"] = ratio(
                        cpi_counts.get("cycles"), cpi_counts.get("instructions"))
                    instructions_per_frame = ratio(
                        cpi_counts.get("instructions"), cpi_frames)

                    l2_group = groups.get("l2", {})
                    l2_counts = l2_group.get("counts", {})
                    l2_frames = l2_group.get("frames")
                    item["l2_requests_per_frame"] = ratio(
                        l2_counts.get("l2_request_g1.all_no_prefetch"), l2_frames)
                    item["l2_misses_per_frame"] = ratio(
                        l2_counts.get("l2_cache_misses_from_dc_misses"), l2_frames)
                    item["l2_miss_ratio"] = ratio(
                        l2_counts.get("l2_cache_misses_from_dc_misses"),
                        l2_counts.get("l2_request_g1.all_no_prefetch"))

                    fill_group = groups.get("fills", {})
                    fill_counts = fill_group.get("counts", {})
                    fill_frames = fill_group.get("frames")
                    l2_fills = fill_counts.get("ls_dmnd_fills_from_sys.lcl_l2")
                    cache_fills = sum_optional((
                        fill_counts.get("ls_dmnd_fills_from_sys.int_cache"),
                        fill_counts.get("ls_dmnd_fills_from_sys.ext_cache_local"),
                    ))
                    dram_fills = sum_optional((
                        fill_counts.get("ls_dmnd_fills_from_sys.mem_io_local"),
                        fill_counts.get("ls_dmnd_fills_from_sys.mem_io_remote"),
                    ))
                    item["l2_fills_per_frame"] = ratio(l2_fills, fill_frames)
                    item["cache_fills_per_frame"] = ratio(cache_fills, fill_frames)
                    item["dram_fills_per_frame"] = ratio(dram_fills, fill_frames)
                    item["l2_fills_per_kinstruction"] = (
                        item["l2_fills_per_frame"] / instructions_per_frame * 1000.0
                        if item["l2_fills_per_frame"] is not None
                        and instructions_per_frame not in (None, 0.0) else None)
                    item["dram_fills_per_kinstruction"] = (
                        item["dram_fills_per_frame"] / instructions_per_frame * 1000.0
                        if item["dram_fills_per_frame"] is not None
                        and instructions_per_frame not in (None, 0.0) else None)

                    pcm_samples = pcm.get(key, [])
                    for metric in ("total_gbps", "read_gbps", "write_gbps"):
                        item[metric] = median([
                            sample["bandwidth"][metric]
                            for sample in pcm_samples
                            if metric in sample["bandwidth"]])
                    item["cumulative_total_gbps"] = median([
                        sample["bandwidth"]["cumulative_total_gbps"]
                        for sample in pcm_samples
                        if "cumulative_total_gbps" in sample["bandwidth"]])
                    item["pcm_fps"] = median([
                        sample["vspipe"]["fps"] for sample in pcm_samples])
                    item["mb_per_frame"] = (
                        item["total_gbps"] * 1000.0 / item["pcm_fps"]
                        if item["total_gbps"] is not None
                        and item["pcm_fps"] not in (None, 0.0) else None)
                    summaries.append(item)
    return summaries


def fmt(value: float | None, digits: int = 2,
        suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def write_csv(summaries: list[dict], path: Path) -> None:
    fields = (
        "format", "kernel", "threads", "implementation", "fps",
        "cpu_ms_per_frame", "cpi", "l2_requests_per_frame",
        "l2_misses_per_frame", "l2_miss_ratio", "l2_fills_per_frame",
        "cache_fills_per_frame", "dram_fills_per_frame",
        "l2_fills_per_kinstruction", "dram_fills_per_kinstruction",
        "total_gbps", "read_gbps", "write_gbps", "cumulative_total_gbps",
        "pcm_fps",
        "mb_per_frame",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fields})


def write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# Blank pipeline memory profile",
        "",
        "All cases use an in-memory `std.BlankClip`. PMU event groups are "
        "measured in separate VSPipe processes and rejected below 99% running. "
        "GB/s is the median after excluding the first and last PCM sample windows.",
        "",
    ]
    for format_name in result["environment"]["formats"]:
        lines.extend([
            f"## {format_name}",
            "",
            "| Impl | Kernel | Requests | FPS | CPI | L2 req/frame | "
            "L2 miss/frame | L2 miss | L2 fills/frame | DRAM fills/frame | "
            "GB/s | MB/frame |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for item in result["summaries"]:
            if item["format"] != format_name:
                continue
            miss_ratio = item["l2_miss_ratio"]
            miss_percent = miss_ratio * 100.0 if miss_ratio is not None else None
            lines.append(
                f"| `{item['implementation']}` | `{item['kernel']}` | "
                f"R{item['threads']} | {item['fps']:.2f} | "
                f"{fmt(item['cpi'], 3)} | "
                f"{fmt(item['l2_requests_per_frame'], 0)} | "
                f"{fmt(item['l2_misses_per_frame'], 0)} | "
                f"{fmt(miss_percent, 2, '%')} | "
                f"{fmt(item['l2_fills_per_frame'], 0)} | "
                f"{fmt(item['dram_fills_per_frame'], 0)} | "
                f"{fmt(item['total_gbps'], 2)} | "
                f"{fmt(item['mb_per_frame'], 2)} |")
        lines.append("")

    by_key = {
        (item["format"], item["kernel"], item["threads"],
         item["implementation"]): item for item in result["summaries"]}
    lines.extend([
        "## Candidate / baseline",
        "",
        "| Format | Kernel | Requests | FPS | CPI | L2 fills/frame | "
        "DRAM fills/frame | GB/s | MB/frame |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    environment = result["environment"]
    for format_name in environment["formats"]:
        for kernel in environment["kernels"]:
            for threads in environment["threads"]:
                baseline = by_key[(format_name, kernel, threads, "baseline")]
                candidate = by_key[(format_name, kernel, threads, "candidate")]
                fps_ratio = ratio(candidate["fps"], baseline["fps"])
                cpi_ratio = ratio(candidate["cpi"], baseline["cpi"])
                l2_fill_ratio = ratio(
                    candidate["l2_fills_per_frame"],
                    baseline["l2_fills_per_frame"])
                dram_fill_ratio = ratio(
                    candidate["dram_fills_per_frame"],
                    baseline["dram_fills_per_frame"])
                bandwidth_ratio = ratio(
                    candidate["total_gbps"], baseline["total_gbps"])
                traffic_ratio = ratio(
                    candidate["mb_per_frame"], baseline["mb_per_frame"])
                lines.append(
                    f"| `{format_name}` | `{kernel}` | R{threads} | "
                    f"{fmt(fps_ratio, 3, 'x')} | "
                    f"{fmt(cpi_ratio, 3, 'x')} | "
                    f"{fmt(l2_fill_ratio, 3, 'x')} | "
                    f"{fmt(dram_fill_ratio, 3, 'x')} | "
                    f"{fmt(bandwidth_ratio, 3, 'x')} | "
                    f"{fmt(traffic_ratio, 3, 'x')} |")

    lines.extend([
        "",
        "## Pixel consistency",
        "",
        "| Format | Kernel | SHA-256 equal | Different samples | Max error |",
        "|---|---|---:|---:|---:|",
    ])
    selected_formats = set(environment["formats"])
    selected_kernels = set(environment["kernels"])
    for comparison in result["pixel_comparisons"]:
        if comparison["format"] not in selected_formats \
                or comparison["kernel"] not in selected_kernels:
            continue
        lines.append(
            f"| `{comparison['format']}` | `{comparison['kernel']}` | "
            f"{comparison['hashes_equal']} | "
            f"{comparison['differing_samples']} | "
            f"{comparison['maximum_absolute_error']:.9g} |")
    lines.extend([
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(environment, indent=2, sort_keys=True),
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def preflight_perf(options) -> None:
    paranoid = perf_paranoid()
    if paranoid is not None and paranoid > 0:
        raise RuntimeError(
            f"kernel.perf_event_paranoid={paranoid}; set it to 0 before profiling")
    for name, events in PERF_GROUPS.items():
        with tempfile.TemporaryDirectory(prefix="dsmvc-perf-probe-") as directory:
            output = Path(directory) / "perf.csv"
            command = [
                str(options.perf), "stat", "--all-user", "--no-big-num",
                "-x", ";", "-o", str(output),
                "-e", "{" + ",".join(events) + "}",
                "--", "sleep", "0.05",
            ]
            run_checked(command, f"perf group {name}")
            parse_perf_csv(output, events)


def preflight_pcm(options) -> None:
    if not Path("/sys/bus/event_source/devices/amd_df").exists():
        raise RuntimeError(
            "the amd_df event source is unavailable; load the amd_uncore module "
            "before PCM profiling")
    with tempfile.TemporaryDirectory(prefix="dsmvc-pcm-probe-") as directory:
        command = [
            str(options.pcm), "profile", "-m", "memory", "-a",
            "-I", "1000", "-P", "4", "-q", "-O", directory,
            "--", "sleep", "1",
        ]
        run_checked(command, "PCM memory probe")
        cumulative = list(Path(directory).rglob("report-cumulative.csv"))
        timeseries = list(Path(directory).rglob("report-timeseries.csv"))
        if len(cumulative) != 1 or len(timeseries) != 1:
            raise RuntimeError("PCM memory probe did not create both reports")
        parse_pcm_reports(cumulative[0], timeseries[0])


def validate_throughput(options, throughput: dict) -> None:
    environment = throughput["environment"]
    expected = {
        "baseline_plugin": file_info(options.baseline_plugin),
        "candidate_plugin": file_info(options.candidate_plugin),
    }
    for key, actual in expected.items():
        recorded = environment[key]
        if recorded["sha256"] != actual["sha256"]:
            raise RuntimeError(
                f"{key} no longer matches the throughput result: "
                f"{recorded['sha256']} != {actual['sha256']}")
    available = set(throughput_index(throughput))
    required = {
        (format_name, kernel, threads, implementation)
        for format_name in options.formats
        for kernel in options.kernels
        for threads in options.threads
        for implementation in IMPLEMENTATIONS
    }
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"throughput result lacks selected cases: {missing}")


def dry_run(options) -> None:
    cases = (len(options.formats) * len(options.kernels)
             * len(options.threads) * len(IMPLEMENTATIONS) * options.runs)
    perf_runs = cases * len(PERF_GROUPS) if "perf" in options.phases else 0
    pcm_runs = cases if "pcm" in options.phases else 0
    print(f"cases={cases} perf_runs={perf_runs} pcm_runs={pcm_runs}")
    example = build_vspipe_command(
        options, "candidate", options.formats[0], options.kernels[0],
        options.threads[0], options.perf_frames)
    print(subprocess.list2cmdline(example))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-plugin", required=True, type=Path)
    parser.add_argument("--candidate-plugin", required=True, type=Path)
    parser.add_argument("--throughput-result", required=True, type=Path)
    parser.add_argument("--vspipe", required=True, type=Path)
    parser.add_argument("--perf", type=Path, default=Path("/usr/bin/perf"))
    parser.add_argument("--pcm", type=Path, default=Path(
        "/opt/AMDuProf_5.3-521/bin/AMDuProfPcm"))
    parser.add_argument("--output", type=Path, default=root /
                        "benchmark-results/blank-pipeline-memory-profile")
    parser.add_argument("--formats", nargs="+", choices=(
        "float32", "gray8", "gray16", "yuv420p10", "rgb24"),
                        default=list(FORMATS))
    parser.add_argument("--kernels", nargs="+", choices=KERNELS,
                        default=list(KERNELS))
    parser.add_argument("--threads", nargs="+", type=int,
                        default=list(THREADS))
    parser.add_argument("--phases", nargs="+", choices=("perf", "pcm"),
                        default=["perf", "pcm"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--perf-frames", type=int, default=2400)
    parser.add_argument("--pcm-frames", type=int, default=4800)
    parser.add_argument("--warmup-frames", type=int, default=64)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--native-height", type=float, default=810.0)
    parser.add_argument("--base-height", type=float, default=1000.0)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()

    for name in ("baseline_plugin", "candidate_plugin", "throughput_result",
                 "vspipe", "perf", "pcm"):
        value = getattr(options, name).expanduser().absolute()
        if not value.is_file():
            raise FileNotFoundError(value)
        setattr(options, name, value)
    options.output = options.output.expanduser().resolve()
    if options.runs < 1 or options.perf_frames < 1 \
            or options.pcm_frames < 1 or options.warmup_frames < 1:
        raise ValueError("run and frame counts must be positive")
    if any(thread < 1 for thread in options.threads):
        raise ValueError("thread counts must be positive")

    throughput = json.loads(options.throughput_result.read_text(
        encoding="utf-8"))
    validate_throughput(options, throughput)
    if options.dry_run:
        dry_run(options)
        return 0

    if "perf" in options.phases:
        preflight_perf(options)
    if "pcm" in options.phases:
        preflight_pcm(options)

    options.output.mkdir(parents=True, exist_ok=True)
    (options.output / "perf").mkdir(exist_ok=True)
    (options.output / "pcm").mkdir(exist_ok=True)
    samples_path = options.output / "raw-samples.json"
    samples = (json.loads(samples_path.read_text(encoding="utf-8"))
               if samples_path.exists() else [])
    completed_ids = {sample["id"] for sample in samples}

    case_number = 0
    if "perf" in options.phases:
        for format_name in options.formats:
            for kernel in options.kernels:
                for threads in options.threads:
                    for run in range(1, options.runs + 1):
                        for group in PERF_GROUPS:
                            order = list(IMPLEMENTATIONS)
                            if (case_number + run) % 2:
                                order.reverse()
                            for implementation in order:
                                identifier = sample_id(
                                    "perf", format_name, kernel, threads,
                                    implementation, run, group)
                                if identifier in completed_ids:
                                    continue
                                sample = run_perf_sample(
                                    options, options.output, format_name, kernel,
                                    threads, implementation, run, group)
                                samples.append(sample)
                                completed_ids.add(identifier)
                                write_samples(samples, samples_path)
                                print(
                                    f"{identifier}: {sample['vspipe']['fps']:.2f} "
                                    f"FPS, min running "
                                    f"{min(sample['running_percent'].values()):.2f}%",
                                    flush=True)
                            case_number += 1

    if "pcm" in options.phases:
        for format_name in options.formats:
            for kernel in options.kernels:
                for threads in options.threads:
                    for run in range(1, options.runs + 1):
                        order = list(IMPLEMENTATIONS)
                        if (case_number + run) % 2:
                            order.reverse()
                        for implementation in order:
                            identifier = sample_id(
                                "pcm", format_name, kernel, threads,
                                implementation, run)
                            if identifier in completed_ids:
                                continue
                            sample = run_pcm_sample(
                                options, options.output, format_name, kernel,
                                threads, implementation, run)
                            samples.append(sample)
                            completed_ids.add(identifier)
                            write_samples(samples, samples_path)
                            print(
                                f"{identifier}: {sample['vspipe']['fps']:.2f} "
                                f"FPS, {sample['bandwidth']['total_gbps']:.2f} GB/s",
                                flush=True)
                        case_number += 1

    summaries = summarize_samples(options, samples, throughput)
    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "dsmvc_memory_concurrency": os.environ.get(
            "DSMVC_MEMORY_CONCURRENCY", "auto"),
        "perf_event_paranoid": perf_paranoid(),
        "formats": options.formats,
        "kernels": options.kernels,
        "threads": options.threads,
        "implementations": list(IMPLEMENTATIONS),
        "phases": options.phases,
        "runs": options.runs,
        "perf_frames": options.perf_frames,
        "pcm_frames": options.pcm_frames,
        "warmup_frames": options.warmup_frames,
        "perf_groups": PERF_GROUPS,
        "baseline_plugin": file_info(options.baseline_plugin),
        "candidate_plugin": file_info(options.candidate_plugin),
        "throughput_result": str(options.throughput_result),
        "vspipe": file_info(options.vspipe),
        "perf": file_info(options.perf),
        "pcm": file_info(options.pcm),
        "source": {
            "type": "VapourSynth std.BlankClip",
            "width": options.source_width,
            "height": options.source_height,
            "native_height": options.native_height,
            "base_height": options.base_height,
        },
    }
    result = {
        "environment": environment,
        "summaries": summaries,
        "pixel_comparisons": throughput["pixel_comparisons"],
        "raw_samples": samples,
    }
    (options.output / "profile.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    write_csv(summaries, options.output / "profile.csv")
    write_markdown(result, options.output / "report.md")
    print(f"wrote {options.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
