#!/usr/bin/env python3

"""Correlate plugin signpost phases with a separate Metal System Trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import analyze_metal_profile as trace


PLUGIN_SUBSYSTEM = "com.dsmvc.plugin"
CPU_PHASE = "DSMVCPluginCpuFrame"
METAL_BATCH = "DSMVCMetalBatch"
METAL_PHASES = (
    "DSMVCMetalUpload",
    "DSMVCMetalEncode",
    "DSMVCMetalWait",
    "DSMVCMetalDownload",
)
ALL_PHASES = (CPU_PHASE, METAL_BATCH, *METAL_PHASES)


@dataclass(frozen=True)
class Interval:
    start_ns: int
    duration_ns: int
    identifier: int
    name: str
    process: str
    start_thread: str
    end_thread: str
    start_message: str
    end_message: str

    @property
    def end_ns(self) -> int:
        return self.start_ns + self.duration_ns


@dataclass(frozen=True)
class ShaderSample:
    start_ns: int
    duration_ns: int
    name: str
    process: str
    percent_of_kick: float | None

    @property
    def end_ns(self) -> int:
        return self.start_ns + self.duration_ns


def fail(message: str) -> None:
    raise ValueError(message)


def text(row: dict[str, trace.Cell], name: str) -> str:
    cell = row.get(name)
    if cell is None:
        return ""
    return cell.formatted or cell.raw


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plugin_intervals(
    table: trace.TraceTable, target_process: str
) -> tuple[list[Interval], int]:
    if table.schema != "OSSignpostIntervals":
        fail(
            "signpost interval export has schema "
            f"{table.schema!r}, expected 'OSSignpostIntervals'"
        )

    intervals: dict[tuple[object, ...], Interval] = {}
    source_rows = 0
    for row in table.rows:
        if text(row, "subsystem") != PLUGIN_SUBSYSTEM:
            continue
        process = text(row, "process")
        if target_process not in process:
            continue
        name = text(row, "name")
        if name not in ALL_PHASES:
            continue
        source_rows += 1
        interval = Interval(
            start_ns=trace.integer(row.get("start"), f"{name} start"),
            duration_ns=trace.integer(row.get("duration"), f"{name} duration"),
            identifier=trace.integer(row.get("identifier"), f"{name} identifier"),
            name=name,
            process=process,
            start_thread=text(row, "start-thread"),
            end_thread=text(row, "end-thread"),
            start_message=text(row, "start-message"),
            end_message=text(row, "end-message"),
        )
        if interval.duration_ns <= 0:
            fail(f"{name} has a non-positive interval duration")
        key = (
            interval.start_ns,
            interval.duration_ns,
            interval.identifier,
            interval.name,
            interval.process,
            interval.start_thread,
            interval.end_thread,
            interval.start_message,
            interval.end_message,
        )
        intervals[key] = interval
    if not intervals:
        fail("no dsmvc plugin intervals matched the target process")
    return sorted(intervals.values(), key=lambda value: value.start_ns), source_rows


def duration_summary(
    intervals: Sequence[Interval], *, batches: int, output_frames: int,
    routed_frames: int,
) -> dict[str, float | int]:
    result = trace.summary([interval.duration_ns for interval in intervals])
    result["total_per_measured_batch"] = result["total"] / batches
    result["total_per_output_frame"] = result["total"] / output_frames
    result["total_per_routed_frame"] = result["total"] / routed_frames
    return result


def message_integer(interval: Interval, field: str, *, at_end: bool) -> int:
    message = interval.end_message if at_end else interval.start_message
    match = re.search(rf"(?:^|\s){re.escape(field)}=([0-9][0-9,]*)", message)
    if match is None:
        fail(f"{interval.name} is missing {field}=... metadata")
    return int(match.group(1).replace(",", ""))


def decimal(cell: trace.Cell | None, context: str) -> float:
    if cell is None or not cell.raw:
        fail(f"{context}: missing decimal value")
    try:
        return float(cell.raw.replace(",", "").rstrip("%"))
    except ValueError:
        fail(f"{context}: invalid decimal value {cell.raw!r}")


def shader_timeline_report(
    table: trace.TraceTable,
    submissions: trace.TraceTable,
    *,
    target_process: str,
    warmup_batches: int,
    measured_batches: int,
    output_frames: int,
    metal_frames: int,
) -> dict[str, object]:
    if table.schema != "metal-shader-profiler-intervals":
        fail(
            "shader timeline export has schema "
            f"{table.schema!r}, expected 'metal-shader-profiler-intervals'"
        )
    measured_rows, _ = trace.measured_submissions(
        submissions, warmup_batches, measured_batches
    )
    measured_start = min(
        trace.integer(row.get("start"), "shader-window start")
        for row in measured_rows
    )
    measured_end = max(
        trace.integer(row.get("start"), "shader-window start")
        + trace.integer(row.get("duration"), "shader-window duration")
        for row in measured_rows
    )
    shader_prefixes = (
        "inverse_axis_", "convert_f32_to_u8", "convert_f32_to_u16"
    )
    samples: dict[tuple[object, ...], ShaderSample] = {}
    source_rows = 0
    for row in table.rows:
        process = text(row, "process")
        name = text(row, "name")
        if target_process not in process or not name.startswith(shader_prefixes):
            continue
        source_rows += 1
        sample = ShaderSample(
            start_ns=trace.integer(row.get("start"), f"{name} start"),
            duration_ns=trace.integer(row.get("duration"), f"{name} duration"),
            name=name,
            process=process,
            percent_of_kick=(
                decimal(row.get("percent-of-kick"), f"{name} percent")
                if row.get("percent-of-kick") is not None
                and row.get("percent-of-kick").raw
                else None
            ),
        )
        if sample.duration_ns <= 0:
            fail(f"{name} has a non-positive shader sample duration")
        if sample.start_ns >= measured_end or sample.end_ns <= measured_start:
            continue
        key = (sample.start_ns, sample.duration_ns, sample.name, sample.process)
        samples[key] = sample
    if not samples:
        fail("no target-process dsmvc shader timeline samples matched")

    by_name: dict[str, list[ShaderSample]] = {}
    for sample in sorted(samples.values(), key=lambda value: value.start_ns):
        by_name.setdefault(sample.name, []).append(sample)
    total_duration = sum(sample.duration_ns for sample in samples.values())
    pipelines: dict[str, object] = {}
    for name, values in sorted(by_name.items()):
        durations = [value.duration_ns for value in values]
        percentages = [
            value.percent_of_kick
            for value in values
            if value.percent_of_kick is not None
        ]
        pipelines[name] = {
            "sample_count": len(values),
            "duration_ns": trace.summary(durations),
            "duration_share": sum(durations) / total_duration,
            "percent_of_kick": (
                trace.summary(percentages) if percentages else None
            ),
        }
    return {
        "source_rows": source_rows,
        "unique_target_samples": len(samples),
        "measured_command_count": len(measured_rows),
        "measured_window_start_ns": measured_start,
        "measured_window_end_ns": measured_end,
        "sampled_duration_ns": trace.summary(
            [sample.duration_ns for sample in samples.values()]
        ),
        "sampled_duration_per_output_frame_ns": total_duration / output_frames,
        "sampled_duration_per_metal_frame_ns": total_duration / metal_frames,
        "pipelines": pipelines,
        "note": (
            "Shader Timeline durations and percent-of-kick values are sampled "
            "GPU attribution. They are not total GPU residency, retired "
            "instructions, cache misses, or physical memory traffic."
        ),
    }


def unique_threads(intervals: Sequence[Interval]) -> dict[str, int]:
    return {
        "unique_start_threads": len({interval.start_thread for interval in intervals}),
        "unique_end_threads": len({interval.end_thread for interval in intervals}),
        "cross_thread_interval_count": sum(
            interval.start_thread != interval.end_thread for interval in intervals
        ),
    }


def peak_concurrency(intervals: Sequence[Interval]) -> int:
    events = [
        event
        for interval in intervals
        for event in ((interval.start_ns, 1), (interval.end_ns, -1))
    ]
    active = 0
    maximum = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def clipped_peak_concurrency(
    intervals: Sequence[Interval], start_ns: int, end_ns: int
) -> int:
    clipped = [
        Interval(
            max(interval.start_ns, start_ns),
            min(interval.end_ns, end_ns) - max(interval.start_ns, start_ns),
            interval.identifier,
            interval.name,
            interval.process,
            interval.start_thread,
            interval.end_thread,
            interval.start_message,
            interval.end_message,
        )
        for interval in intervals
        if interval.start_ns < end_ns and interval.end_ns > start_ns
    ]
    return peak_concurrency(clipped) if clipped else 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-process", default="vspipe")
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--signpost-intervals", type=Path, required=True)
    parser.add_argument("--submissions", type=Path, required=True)
    parser.add_argument("--encoders", type=Path, required=True)
    parser.add_argument("--gpu-intervals", type=Path, required=True)
    parser.add_argument("--allocated", type=Path, required=True)
    parser.add_argument("--shader-intervals", type=Path)
    parser.add_argument("--shader-submissions", type=Path)
    parser.add_argument("--expected-output-frames", type=int, required=True)
    parser.add_argument("--warmup-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-frames-per-batch", type=int, default=9)
    parser.add_argument("--metal-frames-per-batch", type=int, default=7)
    parser.add_argument("--format", default="p10")
    parser.add_argument("--kernel", default="spline64")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    for name in (
        "expected_output_frames",
        "batch_size",
        "cpu_frames_per_batch",
        "metal_frames_per_batch",
        "requests",
    ):
        if getattr(arguments, name) <= 0:
            fail(f"--{name.replace('_', '-')} must be positive")
    if arguments.warmup_batches < 0:
        fail("--warmup-batches must not be negative")
    if (
        arguments.cpu_frames_per_batch + arguments.metal_frames_per_batch
        != arguments.batch_size
    ):
        fail("CPU and Metal frames must add up to the batch size")
    if arguments.expected_output_frames % arguments.batch_size != 0:
        fail("expected output frames must be divisible by the batch size")
    if not arguments.target_process:
        fail("--target-process must not be empty")
    if (arguments.shader_intervals is None) != (
        arguments.shader_submissions is None
    ):
        fail("--shader-intervals and --shader-submissions must be supplied together")
    if not arguments.plugin.is_file():
        fail(f"plugin does not exist: {arguments.plugin}")

    trace.target_process = arguments.target_process
    interval_table = trace.load_table(arguments.signpost_intervals)
    intervals, source_rows = plugin_intervals(
        interval_table, arguments.target_process
    )
    by_name = {
        name: [interval for interval in intervals if interval.name == name]
        for name in ALL_PHASES
    }
    expected_batches = arguments.expected_output_frames // arguments.batch_size
    expected_cpu_frames = expected_batches * arguments.cpu_frames_per_batch
    if len(by_name[METAL_BATCH]) != expected_batches:
        fail(
            "Metal batch interval count does not match the requested workload: "
            f"expected {expected_batches}, found {len(by_name[METAL_BATCH])}"
        )
    if len(by_name[CPU_PHASE]) != expected_cpu_frames:
        fail(
            "CPU frame interval count does not match the requested route: "
            f"expected {expected_cpu_frames}, found {len(by_name[CPU_PHASE])}"
        )
    for name in METAL_PHASES:
        if len(by_name[name]) != expected_batches:
            fail(
                f"{name} interval count does not match Metal batches: "
                f"expected {expected_batches}, found {len(by_name[name])}"
            )
    if arguments.warmup_batches >= expected_batches:
        fail("warmup batches must be smaller than the captured batch count")

    measured_batches = expected_batches - arguments.warmup_batches
    measured_output_frames = measured_batches * arguments.batch_size
    measured_cpu_frames = measured_batches * arguments.cpu_frames_per_batch
    measured_metal_frames = measured_batches * arguments.metal_frames_per_batch

    batches = by_name[METAL_BATCH][arguments.warmup_batches :]
    batch_ids = {interval.identifier for interval in batches}
    if len(batch_ids) != measured_batches:
        fail("measured Metal batches do not have unique identifiers")
    selected = {
        name: [
            interval for interval in by_name[name]
            if interval.identifier in batch_ids
        ]
        for name in (METAL_BATCH, *METAL_PHASES)
    }
    cpu_offset = arguments.warmup_batches * arguments.cpu_frames_per_batch
    selected[CPU_PHASE] = by_name[CPU_PHASE][
        cpu_offset : cpu_offset + measured_cpu_frames
    ]
    if any(len(selected[name]) != measured_batches for name in METAL_PHASES):
        fail("phase identifiers do not map one-to-one to measured Metal batches")

    batch_metal_frames = [
        message_integer(interval, "frames", at_end=False) for interval in batches
    ]
    if set(batch_metal_frames) != {arguments.metal_frames_per_batch}:
        fail(
            "Metal batch signposts do not match --metal-frames-per-batch: "
            f"found {sorted(set(batch_metal_frames))}"
        )

    upload = selected["DSMVCMetalUpload"]
    download = selected["DSMVCMetalDownload"]
    upload_calls = [
        message_integer(interval, "calls", at_end=True) for interval in upload
    ]
    upload_bytes = [
        message_integer(interval, "bytes", at_end=True) for interval in upload
    ]
    total_calls = [
        message_integer(interval, "calls", at_end=True) for interval in download
    ]
    total_bytes = [
        message_integer(interval, "bytes", at_end=True) for interval in download
    ]
    download_calls = [total - uploaded for total, uploaded in zip(
        total_calls, upload_calls, strict=True)]
    download_bytes = [total - uploaded for total, uploaded in zip(
        total_bytes, upload_bytes, strict=True)]

    phase_reports = {
        name: {
            "duration_ns": duration_summary(
                selected[name],
                batches=measured_batches,
                output_frames=measured_output_frames,
                routed_frames=(
                    measured_cpu_frames
                    if name == CPU_PHASE else measured_metal_frames
                ),
            ),
            "threads": unique_threads(selected[name]),
        }
        for name in ALL_PHASES
    }
    batch_total = phase_reports[METAL_BATCH]["duration_ns"]["total"]
    phase_total = sum(
        phase_reports[name]["duration_ns"]["total"] for name in METAL_PHASES
    )
    for name in METAL_PHASES:
        phase_reports[name]["share_of_metal_batch_time"] = (
            phase_reports[name]["duration_ns"]["total"] / batch_total
        )

    cpu_intervals = selected[CPU_PHASE]
    overlapping_cpu_counts = [
        sum(
            interval.start_ns < batch.end_ns
            and interval.end_ns > batch.start_ns
            for interval in cpu_intervals
        )
        for batch in batches
    ]
    overlapping_cpu_peaks = [
        clipped_peak_concurrency(cpu_intervals, batch.start_ns, batch.end_ns)
        for batch in batches
    ]

    submissions = trace.load_table(
        arguments.submissions, "metal-application-command-buffer-submissions"
    )
    measured_submissions, command_ids = trace.measured_submissions(
        submissions, arguments.warmup_batches, measured_batches
    )
    gpu_report = trace.analyze_gpu_intervals(
        trace.load_table(arguments.gpu_intervals, "metal-gpu-intervals"),
        command_ids,
        measured_batches,
        measured_output_frames,
        measured_metal_frames,
    )
    wait_median = phase_reports["DSMVCMetalWait"]["duration_ns"]["median"]
    gpu_median = gpu_report["per_command_union_duration_ns"]["median"]

    shader_report = None
    if arguments.shader_intervals is not None:
        shader_report = shader_timeline_report(
            trace.load_table(
                arguments.shader_intervals, "metal-shader-profiler-intervals"
            ),
            trace.load_table(
                arguments.shader_submissions,
                "metal-application-command-buffer-submissions",
            ),
            target_process=arguments.target_process,
            warmup_batches=arguments.warmup_batches,
            measured_batches=measured_batches,
            output_frames=measured_output_frames,
            metal_frames=measured_metal_frames,
        )

    report = {
        "schema": "dsmvc-metal-plugin-profile-v1",
        "workload": {
            "target_process": arguments.target_process,
            "plugin": str(arguments.plugin.resolve()),
            "plugin_sha256": sha256_file(arguments.plugin),
            "format": arguments.format,
            "kernel": arguments.kernel,
            "requests": arguments.requests,
            "captured_output_frames": arguments.expected_output_frames,
            "captured_batches": expected_batches,
            "warmup_batches_excluded": arguments.warmup_batches,
            "measured_batches": measured_batches,
            "batch_size": arguments.batch_size,
            "measured_output_frames": measured_output_frames,
            "cpu_frames_per_batch": arguments.cpu_frames_per_batch,
            "metal_frames_per_batch": arguments.metal_frames_per_batch,
            "measured_cpu_frames": measured_cpu_frames,
            "measured_metal_frames": measured_metal_frames,
        },
        "signposts": {
            "source_interval_rows": source_rows,
            "unique_plugin_intervals": len(intervals),
            "duplicate_interval_rows_removed": source_rows - len(intervals),
            "phases": phase_reports,
            "metal_phase_accounting_ratio": phase_total / batch_total,
            "staging": {
                "upload_memcpy_calls_per_batch": trace.summary(upload_calls),
                "upload_bytes_per_batch": trace.summary(upload_bytes),
                "download_memcpy_calls_per_batch": trace.summary(download_calls),
                "download_bytes_per_batch": trace.summary(download_bytes),
                "total_memcpy_calls_per_batch": trace.summary(total_calls),
                "total_copied_bytes_per_batch": trace.summary(total_bytes),
            },
            "cpu_overlap": {
                "observed_cpu_frame_intervals": len(cpu_intervals),
                "routed_cpu_frames_per_batch": arguments.cpu_frames_per_batch,
                "overall_peak_cpu_frame_concurrency": peak_concurrency(cpu_intervals),
                "cpu_intervals_overlapping_metal_batch": trace.summary(
                    overlapping_cpu_counts
                ),
                "peak_cpu_concurrency_during_metal_batch": trace.summary(
                    overlapping_cpu_peaks
                ),
            },
        },
        "metal_trace": {
            "submissions": trace.analyze_submissions(
                measured_submissions,
                measured_batches,
                measured_output_frames,
                measured_metal_frames,
            ),
            "encoders": trace.analyze_encoders(
                trace.load_table(
                    arguments.encoders, "metal-application-encoders-list"
                ),
                command_ids,
                measured_batches,
                measured_output_frames,
                measured_metal_frames,
            ),
            "gpu_intervals": gpu_report,
            "allocations": trace.analyze_allocations(
                trace.load_table(
                    arguments.allocated, "metal-current-allocated-size"
                )
            ),
            "shader_timeline": shader_report,
        },
        "cross_trace_comparison": {
            "wait_median_ns": wait_median,
            "gpu_union_median_ns": gpu_median,
            "wait_minus_gpu_median_ns": wait_median - gpu_median,
            "wait_to_gpu_median_ratio": wait_median / gpu_median,
            "note": (
                "Signpost and Metal data come from separate equivalent runs. "
                "This is a distribution comparison, not per-command timestamp "
                "correlation. DSMVCMetalWait includes command commit and the "
                "CPU-visible waitUntilCompleted interval."
            ),
        },
    }
    arguments.json_out.parent.mkdir(parents=True, exist_ok=True)
    arguments.json_out.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(arguments.json_out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ET.ParseError, OSError, ValueError) as error:
        print(f"plugin profile analysis failed: {error}", file=sys.stderr)
        raise SystemExit(1)
