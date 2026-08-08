#!/usr/bin/env python3

"""Normalize dsmvc CPU/Metal xctrace exports into one JSON report."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


target_process = "dsmvc_metal_route_benchmark"
MEASURED_SIGNPOST = "MeasuredLoop"


@dataclass(frozen=True)
class Cell:
    tag: str
    raw: str
    formatted: str

    def contains(self, text: str) -> bool:
        return text in self.raw or text in self.formatted


@dataclass(frozen=True)
class TraceTable:
    schema: str
    columns: tuple[str, ...]
    rows: tuple[dict[str, Cell], ...]


def fail(message: str) -> None:
    raise ValueError(message)


def load_table(path: Path, expected_schema: str | None = None) -> TraceTable:
    root = ET.parse(path).getroot()
    schema = root.find(".//schema")
    if schema is None:
        fail(f"{path}: missing schema")
    schema_name = schema.get("name", "")
    if expected_schema is not None and schema_name != expected_schema:
        fail(
            f"{path}: expected schema {expected_schema!r}, got {schema_name!r}"
        )
    columns = tuple(
        (column.findtext("mnemonic") or "").strip()
        for column in schema.findall("col")
    )
    if not columns or any(not column for column in columns):
        fail(f"{path}: schema has an unnamed column")

    identifiers = {
        element_id: element
        for element in root.iter()
        if (element_id := element.get("id")) is not None
    }

    def resolve(element: ET.Element) -> ET.Element:
        seen: set[str] = set()
        while (reference := element.get("ref")) is not None:
            if reference in seen:
                fail(f"{path}: cyclic XML reference {reference}")
            seen.add(reference)
            try:
                element = identifiers[reference]
            except KeyError:
                fail(f"{path}: unresolved XML reference {reference}")
        return element

    def make_cell(element: ET.Element) -> Cell:
        target = resolve(element)
        return Cell(
            tag=target.tag,
            raw=(target.text or "").strip(),
            formatted=target.get("fmt", ""),
        )

    rows: list[dict[str, Cell]] = []
    for element in root.findall(".//row"):
        values = list(element)
        if len(values) != len(columns):
            fail(
                f"{path}: row has {len(values)} cells for {len(columns)} columns"
            )
        rows.append(dict(zip(columns, map(make_cell, values), strict=True)))
    return TraceTable(schema_name, columns, tuple(rows))


def integer(cell: Cell | None, context: str) -> int:
    if cell is None or not cell.raw:
        fail(f"{context}: missing integer value")
    try:
        return int(cell.raw)
    except ValueError:
        fail(f"{context}: invalid integer value {cell.raw!r}")


def integer_array(cell: Cell | None, context: str) -> tuple[int, ...]:
    if cell is None or not cell.raw:
        fail(f"{context}: missing integer array")
    try:
        return tuple(int(value) for value in cell.raw.split())
    except ValueError:
        fail(f"{context}: invalid integer array {cell.raw!r}")


def process_matches(row: dict[str, Cell]) -> bool:
    process = row.get("process")
    return process is not None and process.contains(target_process)


def summary(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        fail("cannot summarize an empty value sequence")
    return {
        "count": len(values),
        "total": sum(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
    }


def add_normalized(
    values: Sequence[int], *, batches: int, frames: int, metal_frames: int
) -> dict[str, float | int]:
    result = summary(values)
    result["total_per_measured_batch"] = result["total"] / batches
    result["total_per_output_frame"] = result["total"] / frames
    result["total_per_metal_frame"] = result["total"] / metal_frames
    return result


def command_identifier(row: dict[str, Cell]) -> int | None:
    cell = row.get("cmdbuffer-id")
    if cell is None or not cell.raw:
        return None
    return integer(cell, "command buffer id")


def measured_submissions(
    table: TraceTable, warmups: int, iterations: int
) -> tuple[list[dict[str, Cell]], set[int]]:
    rows = sorted(
        (row for row in table.rows if process_matches(row)),
        key=lambda row: integer(row.get("start"), "submission start"),
    )
    expected = warmups + iterations
    if len(rows) != expected:
        fail(
            "Metal submission count does not match the requested profile: "
            f"expected {expected}, found {len(rows)}"
        )
    measured = rows[warmups:]
    identifiers = {command_identifier(row) for row in measured}
    if None in identifiers or len(identifiers) != iterations:
        fail("measured Metal submissions do not have unique command-buffer ids")
    return measured, {value for value in identifiers if value is not None}


def analyze_submissions(
    rows: Sequence[dict[str, Cell]], batches: int, frames: int, metal_frames: int
) -> dict[str, object]:
    durations = [integer(row.get("duration"), "submission duration") for row in rows]
    encoder_times = [
        integer(row.get("encoder-time"), "submission encoder time") for row in rows
    ]
    encoder_counts = [
        integer(row.get("num-encoders"), "submission encoder count") for row in rows
    ]
    return {
        "count": len(rows),
        "duration_ns": add_normalized(
            durations, batches=batches, frames=frames, metal_frames=metal_frames
        ),
        "reported_encoder_time_ns": add_normalized(
            encoder_times,
            batches=batches,
            frames=frames,
            metal_frames=metal_frames,
        ),
        "reported_encoder_count": summary(encoder_counts),
    }


def analyze_encoders(
    table: TraceTable,
    measured_ids: set[int],
    batches: int,
    frames: int,
    metal_frames: int,
) -> dict[str, object]:
    rows = [
        row
        for row in table.rows
        if process_matches(row) and command_identifier(row) in measured_ids
    ]
    durations = [integer(row.get("duration"), "encoder duration") for row in rows]
    labels: dict[str, int] = {}
    for row in rows:
        label = row.get("encoder-label")
        name = label.formatted if label is not None else ""
        labels[name] = labels.get(name, 0) + 1
    return {
        "count": len(rows),
        "duration_ns": add_normalized(
            durations, batches=batches, frames=frames, metal_frames=metal_frames
        ),
        "labels": labels,
        "note": (
            "Instruments may coalesce adjacent compute encoders; this is the "
            "trace representation, not an API encode-call count."
        ),
    }


def interval_union(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def analyze_gpu_intervals(
    table: TraceTable,
    measured_ids: set[int],
    batches: int,
    frames: int,
    metal_frames: int,
) -> dict[str, object]:
    by_command: dict[int, list[tuple[int, int]]] = {
        identifier: [] for identifier in measured_ids
    }
    row_durations: list[int] = []
    for row in table.rows:
        identifier = command_identifier(row)
        if not process_matches(row) or identifier not in measured_ids:
            continue
        start = integer(row.get("start"), "GPU interval start")
        duration = integer(row.get("duration"), "GPU interval duration")
        by_command[identifier].append((start, start + duration))
        row_durations.append(duration)
    missing = sum(not intervals for intervals in by_command.values())
    if missing:
        fail(f"{missing} measured command buffers have no GPU interval")
    union_durations = [interval_union(intervals) for intervals in by_command.values()]
    return {
        "interval_count": len(row_durations),
        "summed_interval_duration_ns": add_normalized(
            row_durations,
            batches=batches,
            frames=frames,
            metal_frames=metal_frames,
        ),
        "per_command_union_duration_ns": add_normalized(
            union_durations,
            batches=batches,
            frames=frames,
            metal_frames=metal_frames,
        ),
        "note": (
            "Union duration removes overlap between GPU interval rows belonging "
            "to the same command buffer."
        ),
    }


def analyze_allocations(table: TraceTable) -> dict[str, object]:
    sizes = [
        integer(row.get("current-allocated-size"), "Metal allocated size")
        for row in table.rows
        if process_matches(row)
    ]
    if not sizes:
        fail("Metal allocation table has no target-process rows")
    return {
        "sample_count": len(sizes),
        "peak_bytes": max(sizes),
        "final_bytes": sizes[-1],
    }


def signpost_interval(table: TraceTable) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    begin_times: dict[int, int] = {}
    end_times: dict[int, int] = {}
    seen_events: set[tuple[int, int, str]] = set()
    for row in table.rows:
        if not process_matches(row):
            continue
        if not any(cell.contains(MEASURED_SIGNPOST) for cell in row.values()):
            continue
        start_cell = row.get("start") or row.get("timestamp") or row.get("time")
        duration_cell = row.get("duration")
        if start_cell is None:
            continue
        start = integer(start_cell, "signpost start")
        if duration_cell is not None and duration_cell.raw:
            duration = integer(duration_cell, "signpost duration")
            if duration > 0:
                candidates.append((start, start + duration))
            continue
        identifier_cell = row.get("identifier")
        event_cell = row.get("event-type")
        if identifier_cell is None or event_cell is None:
            continue
        identifier = integer(identifier_cell, "signpost identifier")
        event = event_cell.formatted
        event_key = (identifier, start, event)
        if event_key in seen_events:
            continue
        seen_events.add(event_key)
        if event == "Begin":
            begin_times[identifier] = min(begin_times.get(identifier, start), start)
        elif event == "End":
            end_times[identifier] = max(end_times.get(identifier, start), start)
    for identifier in begin_times.keys() & end_times.keys():
        if end_times[identifier] > begin_times[identifier]:
            candidates.append((begin_times[identifier], end_times[identifier]))
    if not candidates:
        return None
    return min(start for start, _ in candidates), max(end for _, end in candidates)


def analyze_cpu_counters(
    counter_table: TraceTable,
    signpost_table: TraceTable,
    *,
    iterations: int,
    output_frames: int,
    cpu_frames: int,
) -> dict[str, object]:
    interval = signpost_interval(signpost_table)
    rows = [row for row in counter_table.rows if process_matches(row)]
    if interval is not None:
        start, end = interval
        rows = [
            row
            for row in rows
            if start <= integer(row.get("timestamp"), "CPU counter timestamp") < end
        ]
    if not rows:
        fail("CPU Counter table has no samples in the measured interval")

    arrays = [integer_array(row.get("value"), "CPU Counter value") for row in rows]
    width = len(arrays[0])
    if width == 0 or any(len(values) != width for values in arrays):
        fail("CPU Counter arrays do not have a stable width")
    totals = [sum(values[index] for values in arrays) for index in range(width)]
    precise_rows = sum(
        integer(row.get("is-precise"), "CPU Counter precision") != 0 for row in rows
    )
    sample_duration = sum(
        integer(row.get("duration"), "CPU Counter duration") for row in rows
    )
    return {
        "measured_signpost_found": interval is not None,
        "sample_count": len(rows),
        "sample_duration_ns": sample_duration,
        "precise_sample_count": precise_rows,
        "bottleneck_sample_weight_totals": totals,
        "bottleneck_sample_weights_per_output_frame": [
            value / output_frames for value in totals
        ],
        "bottleneck_sample_weights_per_cpu_frame": [
            value / cpu_frames for value in totals
        ],
        "iterations": iterations,
        "note": (
            "The default CPU Counters template emits CPU Bottlenecks sample "
            "weight arrays. These values are attribution samples, not exact "
            "retired-instruction or cycle counts."
        ),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-process", default="dsmvc_metal_route_benchmark"
    )
    parser.add_argument("--submissions", type=Path, required=True)
    parser.add_argument("--encoders", type=Path, required=True)
    parser.add_argument("--gpu-intervals", type=Path, required=True)
    parser.add_argument("--allocated", type=Path, required=True)
    parser.add_argument("--cpu-neon", type=Path, required=True)
    parser.add_argument("--cpu-neon-signposts", type=Path, required=True)
    parser.add_argument("--cpu-mixed", type=Path, required=True)
    parser.add_argument("--cpu-mixed-signposts", type=Path, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--cpu-frames", type=int, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    global target_process
    arguments = parse_arguments()
    if not arguments.target_process:
        fail("--target-process must not be empty")
    target_process = arguments.target_process
    for name in ("warmups", "iterations", "batch_size", "cpu_frames"):
        if getattr(arguments, name) <= 0:
            fail(f"--{name.replace('_', '-')} must be positive")
    if arguments.cpu_frames >= arguments.batch_size:
        fail("--cpu-frames must be smaller than --batch-size for a Metal trace")

    submissions = load_table(
        arguments.submissions, "metal-application-command-buffer-submissions"
    )
    measured, command_ids = measured_submissions(
        submissions, arguments.warmups, arguments.iterations
    )
    output_frames = arguments.iterations * arguments.batch_size
    mixed_cpu_frames = arguments.iterations * arguments.cpu_frames
    mixed_metal_frames = output_frames - mixed_cpu_frames

    report = {
        "schema_version": 1,
        "profile": {
            "target_process": target_process,
            "warmup_batches_excluded": arguments.warmups,
            "measured_batches": arguments.iterations,
            "batch_size": arguments.batch_size,
            "measured_output_frames": output_frames,
            "mixed_cpu_frames": mixed_cpu_frames,
            "mixed_metal_frames": mixed_metal_frames,
        },
        "metal": {
            "submissions": analyze_submissions(
                measured, arguments.iterations, output_frames, mixed_metal_frames
            ),
            "encoders": analyze_encoders(
                load_table(arguments.encoders, "metal-application-encoders-list"),
                command_ids,
                arguments.iterations,
                output_frames,
                mixed_metal_frames,
            ),
            "gpu_intervals": analyze_gpu_intervals(
                load_table(arguments.gpu_intervals, "metal-gpu-intervals"),
                command_ids,
                arguments.iterations,
                output_frames,
                mixed_metal_frames,
            ),
            "allocations": analyze_allocations(
                load_table(arguments.allocated, "metal-current-allocated-size")
            ),
        },
        "cpu_counters": {
            "neon": analyze_cpu_counters(
                load_table(arguments.cpu_neon, "CounterMetricAggregatedForProcess"),
                load_table(arguments.cpu_neon_signposts, "os-signpost"),
                iterations=arguments.iterations,
                output_frames=output_frames,
                cpu_frames=output_frames,
            ),
            "mixed": analyze_cpu_counters(
                load_table(arguments.cpu_mixed, "CounterMetricAggregatedForProcess"),
                load_table(arguments.cpu_mixed_signposts, "os-signpost"),
                iterations=arguments.iterations,
                output_frames=output_frames,
                cpu_frames=mixed_cpu_frames,
            ),
        },
    }
    arguments.json_out.parent.mkdir(parents=True, exist_ok=True)
    arguments.json_out.write_text(json.dumps(report, indent=2) + "\n")
    print(arguments.json_out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ET.ParseError, OSError, ValueError) as error:
        print(f"metal profile analysis failed: {error}", file=sys.stderr)
        raise SystemExit(1)
