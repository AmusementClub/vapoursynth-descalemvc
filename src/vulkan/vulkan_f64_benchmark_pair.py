#!/usr/bin/env python3
"""Run paired Vulkan F32 executor measurements against a baseline binary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import TextIO


MINIMUM_SAMPLES = 30


class BenchmarkSkip(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile_summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    median = statistics.median(ordered)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": median,
        "maximum": ordered[-1],
        "mad": statistics.median(abs(value - median) for value in ordered),
    }


class Server:
    def __init__(self, label: str, executable: Path, samples: int,
                 warmups: int) -> None:
        self.label = label
        self.executable = executable
        self.command = [
            str(executable),
            "--server-f32",
            "--samples", str(samples),
            "--warmups", str(warmups),
            "--label", label,
        ]
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        self.fixtures: list[dict[str, str | int]] = []
        self.device_name = ""

    @property
    def stdin(self) -> TextIO:
        if self.process.stdin is None:
            raise RuntimeError("benchmark server stdin is unavailable")
        return self.process.stdin

    @property
    def stdout(self) -> TextIO:
        if self.process.stdout is None:
            raise RuntimeError("benchmark server stdout is unavailable")
        return self.process.stdout

    def read_metadata(self) -> None:
        while True:
            line = self.stdout.readline()
            if not line:
                returncode = self.process.wait()
                stderr = self.process.stderr.read() if self.process.stderr else ""
                message = stderr.strip() or "benchmark server exited before READY"
                if returncode == 77:
                    raise BenchmarkSkip(f"{self.label}: {message}")
                raise RuntimeError(
                    f"{self.label} benchmark server exited {returncode}: {message}")
            fields = next(csv.reader([line]))
            if fields[0] == "FIXTURE" and len(fields) == 5:
                self.fixtures.append({
                    "index": int(fields[1]),
                    "name": fields[2],
                    "operation": fields[3],
                    "half_bandwidth": int(fields[4]),
                })
            elif fields[0] == "READY" and len(fields) == 3:
                expected = int(fields[1])
                self.device_name = fields[2]
                if len(self.fixtures) != expected:
                    raise RuntimeError(
                        f"{self.label} described {len(self.fixtures)} fixtures; "
                        f"expected {expected}")
                return
            else:
                raise RuntimeError(
                    f"{self.label} emitted invalid protocol line: {line.rstrip()}")

    def measure(self, fixture_index: int) -> int:
        self.stdin.write(f"RUN,{fixture_index}\n")
        self.stdin.flush()
        line = self.stdout.readline()
        if not line:
            returncode = self.process.poll()
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(
                f"{self.label} exited during measurement ({returncode}): "
                f"{stderr.strip()}")
        fields = next(csv.reader([line]))
        if (len(fields) != 3 or fields[0] != "RESULT"
                or int(fields[1]) != fixture_index):
            raise RuntimeError(
                f"{self.label} emitted invalid result: {line.rstrip()}")
        duration = int(fields[2])
        if duration <= 0:
            raise RuntimeError(f"{self.label} emitted a nonpositive duration")
        return duration

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.stdin.write("QUIT\n")
                self.stdin.flush()
                self.process.wait(timeout=10)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()


def run(options: argparse.Namespace) -> tuple[dict, int]:
    control = Server(
        "control", options.control, options.samples, options.warmups)
    candidate = Server(
        "candidate", options.candidate, options.samples, options.warmups)
    servers = {"control": control, "candidate": candidate}
    try:
        control.read_metadata()
        candidate.read_metadata()
        if control.fixtures != candidate.fixtures:
            raise RuntimeError("control and candidate fixture inventories differ")
        if control.device_name != candidate.device_name:
            raise RuntimeError("control and candidate selected different devices")

        options.csv_out.parent.mkdir(parents=True, exist_ok=True)
        raw_rows: list[dict[str, str | int]] = []
        control_binary_sha256 = sha256_file(options.control)
        candidate_binary_sha256 = sha256_file(options.candidate)
        totals = {
            "control": [0] * options.samples,
            "candidate": [0] * options.samples,
        }
        with options.csv_out.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "schema", "control_source_sha", "candidate_source_sha",
                "control_binary_sha256", "candidate_binary_sha256",
                "device_name", "fixture", "operation", "half_bandwidth",
                "pair", "pair_order", "variant", "duration_ns",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for fixture in control.fixtures:
                fixture_index = int(fixture["index"])
                for pair in range(options.samples):
                    order = ("control", "candidate") if pair % 2 == 0 else (
                        "candidate", "control")
                    pair_order = "control-candidate" if pair % 2 == 0 else (
                        "candidate-control")
                    for variant in order:
                        duration = servers[variant].measure(fixture_index)
                        totals[variant][pair] += duration
                        row = {
                            "schema": "dsmvc-vulkan-f32-executor-ab-v1",
                            "control_source_sha": options.control_source_sha,
                            "candidate_source_sha": options.candidate_source_sha,
                            "control_binary_sha256": control_binary_sha256,
                            "candidate_binary_sha256": candidate_binary_sha256,
                            "device_name": control.device_name,
                            "fixture": fixture["name"],
                            "operation": fixture["operation"],
                            "half_bandwidth": fixture["half_bandwidth"],
                            "pair": pair,
                            "pair_order": pair_order,
                            "variant": variant,
                            "duration_ns": duration,
                        }
                        writer.writerow(row)
                        handle.flush()
                        raw_rows.append(row)

        fixture_summaries = []
        for fixture in control.fixtures:
            ratios = []
            for pair in range(options.samples):
                sample = {
                    str(row["variant"]): int(row["duration_ns"])
                    for row in raw_rows
                    if row["fixture"] == fixture["name"] and row["pair"] == pair
                }
                if set(sample) != {"control", "candidate"}:
                    raise RuntimeError("raw executor samples are not paired")
                ratios.append(sample["candidate"] / sample["control"])
            fixture_summaries.append({
                "fixture": fixture["name"],
                "candidate_over_control": percentile_summary(ratios),
            })

        suite_ratios = [
            totals["candidate"][pair] / totals["control"][pair]
            for pair in range(options.samples)
        ]
        suite_summary = percentile_summary(suite_ratios)
        gate_limit = 1.0 + options.max_regression
        gate_pass = float(suite_summary["median"]) <= gate_limit
        summary = {
            "schema": "dsmvc-vulkan-f32-executor-ab-summary-v1",
            "status": "passed" if gate_pass else "failed",
            "samples": options.samples,
            "warmups": options.warmups,
            "device_name": control.device_name,
            "control": {
                "path": str(options.control),
                "source_sha": options.control_source_sha,
                "sha256": control_binary_sha256,
                "command": control.command,
            },
            "candidate": {
                "path": str(options.candidate),
                "source_sha": options.candidate_source_sha,
                "sha256": candidate_binary_sha256,
                "command": candidate.command,
            },
            "raw_csv": str(options.csv_out),
            "maximum_regression": options.max_regression,
            "suite_candidate_over_control": suite_summary,
            "fixtures": fixture_summaries,
        }
        return summary, 0 if gate_pass else 2
    finally:
        control.close()
        candidate.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--control", required=True, type=Path)
    result.add_argument("--candidate", required=True, type=Path)
    result.add_argument("--control-source-sha", required=True)
    result.add_argument("--candidate-source-sha", required=True)
    result.add_argument("--samples", type=int, default=MINIMUM_SAMPLES)
    result.add_argument("--warmups", type=int, default=3)
    result.add_argument("--max-regression", type=float, default=0.05)
    result.add_argument("--csv-out", required=True, type=Path)
    result.add_argument("--summary-json", required=True, type=Path)
    return result


def main() -> int:
    options = parser().parse_args()
    options.control = options.control.expanduser().resolve()
    options.candidate = options.candidate.expanduser().resolve()
    options.csv_out = options.csv_out.expanduser().resolve()
    options.summary_json = options.summary_json.expanduser().resolve()
    for executable in (options.control, options.candidate):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(f"benchmark executable is not runnable: {executable}")
    if options.samples < MINIMUM_SAMPLES:
        raise ValueError("--samples must be at least 30")
    if options.warmups < 0:
        raise ValueError("--warmups cannot be negative")
    if not 0.0 <= options.max_regression < 1.0:
        raise ValueError("--max-regression must be in [0, 1)")

    try:
        summary, returncode = run(options)
    except BenchmarkSkip as error:
        summary = {
            "schema": "dsmvc-vulkan-f32-executor-ab-summary-v1",
            "status": "skipped",
            "reason": str(error),
        }
        returncode = 77
    options.summary_json.parent.mkdir(parents=True, exist_ok=True)
    options.summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
