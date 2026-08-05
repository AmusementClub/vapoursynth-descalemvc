#!/usr/bin/env python3
"""Measure cache and memory-hierarchy counters across R1/R8/R16/R32."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


THREADS = (1, 8, 16, 32)
KERNELS = (
    ("bilinear", "bilinear"),
    ("bicubic_b0_c0_5", "bicubic (0, 0.5)"),
)
IMPLEMENTATIONS = ("new",)
PERF_EVENTS = (
    "task-clock",
    "context-switches",
    "page-faults",
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "branch-instructions",
    "branch-misses",
    "l2_cache_accesses_from_dc_misses",
    "l2_cache_hits_from_dc_misses",
    "l2_cache_misses_from_dc_misses",
    "l3_read_miss_latency",
    "ls_any_fills_from_sys.int_cache",
    "ls_any_fills_from_sys.mem_io_local",
    "l2_latency.l2_cycles_waiting_on_fills",
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="ascii").strip()


def cache_topology() -> list[dict]:
    result = []
    seen = set()
    cache_root = Path("/sys/devices/system/cpu")
    for index in sorted(cache_root.glob("cpu[0-9]*/cache/index*")):
        try:
            item = {
                "level": read_text(index / "level"),
                "type": read_text(index / "type"),
                "size": read_text(index / "size"),
                "shared_cpu_list": read_text(index / "shared_cpu_list"),
            }
        except OSError:
            continue
        key = tuple(item.values())
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def cpu_model() -> str:
    model = platform.processor()
    if model:
        return model
    try:
        for line in Path("/proc/cpuinfo").read_text(
                encoding="ascii").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def perf_paranoid() -> str | None:
    path = Path("/proc/sys/kernel/perf_event_paranoid")
    try:
        return read_text(path)
    except OSError:
        return None


def build_vspipe_command(options, kernel: str, implementation: str,
                         threads: int) -> list[str]:
    script = Path(__file__).with_name("vspipe_fixed_kernel.vpy").resolve()
    values = {
        "implementation": implementation,
        "kernel": kernel,
        "source": str(options.source),
        "plugin": str(options.new_plugin),
        "old_plugin": str(options.old_plugin),
        "source_plugin": "",
        "source_filter": options.source_filter,
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


def normalize_event(name: str) -> str:
    name = name.strip().strip('"')
    if name.startswith("cpu/") and name.endswith("/"):
        name = name[4:-1]
    return name.split(":", 1)[0]


def parse_perf_csv(path: Path) -> dict[str, float]:
    values = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            raw_value = row[0].strip()
            event = normalize_event(row[2])
            if event not in PERF_EVENTS or raw_value.startswith("<"):
                continue
            try:
                values[event] = float(raw_value.replace(",", ""))
            except ValueError:
                continue
    return values


def probe_perf_events(perf: str) -> tuple[list[str], dict[str, str]]:
    """Keep selectable events; perf also lists non-selectable metrics."""
    supported = []
    skipped = {}
    for event in PERF_EVENTS:
        probe = subprocess.run(
            [perf, "stat", "--all-user", "--no-big-num", "-x,",
             "-o", os.devnull, "-e", event, "--", "true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, errors="replace", check=False)
        error = probe.stderr.strip()
        if probe.returncode == 0 and "Bad event name" not in error \
                and "Unable to find event" not in error \
                and "not supported" not in error.lower():
            supported.append(event)
        else:
            skipped[event] = error.splitlines()[-1] if error else (
                f"perf exit {probe.returncode}")
    return supported, skipped


def run_case(options, output: Path, kernel: str, implementation: str,
             threads: int, run_number: int) -> dict:
    perf_file = output / "perf" / (
        f"{kernel}-{implementation}-r{threads}t{threads}-run{run_number}.csv")
    stderr_file = output / "stderr" / perf_file.name.replace(".csv", ".log")
    command = build_vspipe_command(options, kernel, implementation, threads)
    perf_command = [
        options.perf,
        "stat",
        "--all-user",
        "--no-big-num",
        "-x,",
        "-o", str(perf_file),
    ]
    for event in options.supported_events:
        perf_command.extend(["-e", event])
    perf_command.extend(["--", *command])

    started = time.perf_counter()
    completed = subprocess.run(
        perf_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, errors="replace", check=False)
    elapsed = time.perf_counter() - started
    stderr_file.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"perf/VSPipe failed for {kernel}/{implementation}/"
            f"R{threads}T{threads}; see {stderr_file}\n"
            + completed.stderr[-4000:])
    return {
        "kernel": kernel,
        "implementation": implementation,
        "threads": threads,
        "run": run_number,
        "frames": options.frames,
        "elapsed_seconds": elapsed,
        "fps": options.frames / elapsed,
        "perf_events": parse_perf_csv(perf_file),
        "perf_csv": str(perf_file),
        "stderr": str(stderr_file),
        "command": " ".join(perf_command),
    }


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["kernel"], row["implementation"], row["threads"])].append(row)
    result = []
    for key, items in sorted(grouped.items()):
        events = {}
        for event in PERF_EVENTS:
            values = [item["perf_events"][event]
                      for item in items if event in item["perf_events"]]
            if values:
                events[event] = median_or_none(values)
        result.append({
            "kernel": key[0],
            "implementation": key[1],
            "threads": key[2],
            "runs": len(items),
            "fps": statistics.median([item["fps"] for item in items]),
            "perf_events": events,
        })
    return result


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def per_instruction(events: dict, event: str) -> float | None:
    return ratio(events.get(event), events.get("instructions"))


def fmt(value: float | None, scale: float = 1.0, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value * scale:.{digits}f}"


def write_csv(summaries: list[dict], path: Path) -> None:
    fields = ["kernel", "implementation", "threads", "runs", "fps"] + list(PERF_EVENTS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            row = {key: item.get(key, "") for key in fields}
            row.update(item["perf_events"])
            writer.writerow(row)


def write_markdown(result: dict, path: Path) -> None:
    summaries = result["summaries"]
    by_key = {(item["kernel"], item["threads"]): item for item in summaries}
    lines = [
        "# Cache hypothesis profile",
        "",
        "This profile tests whether R8 is faster than R16/R32 because the active working set has better cache residency.",
        f"It uses {result['environment']['frames']:,} frames at fixed 810p geometry, fresh VSPipe processes, and {result['environment']['runs']} runs per cell.",
        "",
        "## Method",
        "",
        f"- CPU: `{result['environment']['cpu']}`",
        f"- Cache topology: `{json.dumps(result['environment']['cache_topology'], ensure_ascii=True)}`",
        f"- perf_event_paranoid before run: `{result['environment']['perf_event_paranoid_before']}`",
        f"- Selectable events: `{', '.join(result['environment']['supported_events'])}`",
        f"- Skipped events: `{json.dumps(result['environment']['skipped_events'], ensure_ascii=True)}`",
        "- Counters are process plus child-thread user-mode counts from `perf stat`; unsupported events are shown as `n/a`.",
        "- The wrapper temporarily sets `kernel.perf_event_paranoid=0` and restores the original value on exit.",
        "",
        "## Throughput",
        "",
        "| Kernel | R1T1 FPS | R8T8 FPS | R16T16 FPS | R32T32 FPS | R8/R1 | R16/R8 | R32/R8 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kernel, label in KERNELS:
        values = [by_key.get((kernel, thread)) for thread in THREADS]
        fps = [item["fps"] if item else None for item in values]
        lines.append(
            f"| `{label}` | "
            + " | ".join(fmt(value) for value in fps)
            + f" | {fmt(ratio(fps[1], fps[0]), 1, 3)}x"
            + f" | {fmt(ratio(fps[2], fps[1]), 1, 3)}x"
            + f" | {fmt(ratio(fps[3], fps[1]), 1, 3)}x |")

    lines.extend([
        "",
        "## Cache and Memory Counters",
        "",
        "The strongest support for the hypothesis would be an R8-to-R16/R32 FPS drop accompanied by higher L2 miss ratio, higher DRAM-fill rate, or higher fill-wait latency. A throughput drop without those changes does not establish cache eviction.",
        "",
        "| Kernel | Threads | FPS | Generic cache miss / ref | L2 miss / access | L3 miss latency | Internal-cache fills / 1k instr | DRAM fills / 1k instr | L2 fill-wait cycles / instr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in summaries:
        events = item["perf_events"]
        cache_rate = ratio(events.get("cache-misses"), events.get("cache-references"))
        l2_rate = ratio(events.get("l2_cache_misses_from_dc_misses"),
                         events.get("l2_cache_accesses_from_dc_misses"))
        lines.append(
            f"| `{item['kernel']}` | R{item['threads']}T{item['threads']} | "
            f"{item['fps']:.2f} | {fmt(cache_rate, 100, 2)}% | "
            f"{fmt(l2_rate, 100, 2)}% | "
            f"{fmt(events.get('l3_read_miss_latency'), 1, 2)} | "
            f"{fmt(per_instruction(events, 'ls_any_fills_from_sys.int_cache'), 1000, 3)} | "
            f"{fmt(per_instruction(events, 'ls_any_fills_from_sys.mem_io_local'), 1000, 3)} | "
            f"{fmt(per_instruction(events, 'l2_latency.l2_cycles_waiting_on_fills'), 1, 4)} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "If R8 has the best throughput and R16/R32 show a clear increase in miss/fill/latency counters, the cache-residency hypothesis is supported. If the counters stay flat while throughput falls, the likely causes move toward shared memory bandwidth, scheduler/request overhead, synchronization, or cross-CCD traffic. These counters still do not prove that all working data fits in cache; they show the direction of cache and memory pressure.",
        "",
        "Raw per-run counters are in `profile.json` and the individual `perf` CSV/log files under `perf/` and `stderr/`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "/run/media/owen/1A16B65916B6361B/Users/lsy39/Downloads/cesh/"
        "[LoliHouse] DIGIMON BEATBREAK - 40 [WebRip 1080p HEVC-10bit AAC SRTx2].mkv"))
    parser.add_argument("--new-plugin", type=Path, default=Path(
        "/home/owen/dev/Descale-MVC/out/linux-release-generic-20260805/"
        "build/dsmvc.so"))
    parser.add_argument("--old-plugin", type=Path, default=Path(
        "/home/owen/vapoursynth/lib/python3.14/site-packages/vapoursynth/"
        "plugins/vsrepo/libdescale.so"))
    parser.add_argument("--vspipe", type=Path, default=Path(
        "/home/owen/vapoursynth/bin/vspipe"))
    parser.add_argument("--perf", default="perf")
    parser.add_argument("--source-filter", choices=("lsmas", "ffms2", "bestsource"),
                        default="ffms2")
    parser.add_argument("--frames", type=int, default=4000)
    parser.add_argument("--src-height", type=float, default=810.0)
    parser.add_argument("--base-height", type=float, default=1000.0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=root / "benchmark-results" /
                        "fixed-kernel-cache-hypothesis-digimon-20260805")
    options = parser.parse_args()
    options.source = options.source.expanduser().resolve()
    options.new_plugin = options.new_plugin.expanduser().resolve()
    options.old_plugin = options.old_plugin.expanduser().resolve()
    options.vspipe = options.vspipe.expanduser().resolve()
    options.output = options.output.expanduser().resolve()
    for required in (options.source, options.new_plugin, options.old_plugin,
                     options.vspipe):
        if not required.is_file():
            raise FileNotFoundError(required)
    if options.frames < 1 or options.runs < 1:
        raise ValueError("frames and runs must be positive")
    options.supported_events, skipped_events = probe_perf_events(options.perf)
    if not options.supported_events:
        raise RuntimeError(
            "perf found no selectable events; see current perf permissions and "
            "the event list reported by `perf list`")
    print(
        "perf events: "
        + (", ".join(options.supported_events) or "none"),
        flush=True,
    )
    for event, reason in skipped_events.items():
        print(f"perf event skipped: {event}: {reason}", flush=True)
    options.output.mkdir(parents=True, exist_ok=True)
    (options.output / "perf").mkdir(exist_ok=True)
    (options.output / "stderr").mkdir(exist_ok=True)

    rows = []
    for threads in THREADS:
        for kernel, _label in KERNELS:
            for implementation in IMPLEMENTATIONS:
                for run_number in range(1, options.runs + 1):
                    row = run_case(options, options.output, kernel,
                                   implementation, threads, run_number)
                    rows.append(row)
                    print(
                        f"{kernel} R{threads}T{threads} run {run_number}: "
                        f"{row['fps']:.3f} fps, "
                        f"{len(row['perf_events'])}/{len(PERF_EVENTS)} counters",
                        flush=True)

    summaries = summarize(rows)
    result = {
        "schema_version": 1,
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "cpu": cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "cache_topology": cache_topology(),
            "perf_event_paranoid_before": os.environ.get(
                "CACHE_PROFILE_PARANOID_BEFORE", perf_paranoid()),
            "supported_events": options.supported_events,
            "skipped_events": skipped_events,
            "source_filter": options.source_filter,
            "source": str(options.source),
            "new_plugin": str(options.new_plugin),
            "vspipe": str(options.vspipe),
            "frames": options.frames,
            "src_height": options.src_height,
            "base_height": options.base_height,
            "runs": options.runs,
            "threads": list(THREADS),
            "kernels": [name for name, _label in KERNELS],
            "implementations": list(IMPLEMENTATIONS),
        },
        "raw_samples": rows,
        "summaries": summaries,
    }
    (options.output / "profile.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8")
    write_csv(summaries, options.output / "profile.csv")
    write_markdown(result, options.output / "profile.md")
    print(options.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
