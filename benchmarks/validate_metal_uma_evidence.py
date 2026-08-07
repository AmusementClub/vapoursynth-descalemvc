#!/usr/bin/env python3
"""Validate bounded CPU, fixed-kernel, and GetFnative evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MINIMUM_CPU_RATIO = 0.97
MINIMUM_AUTO_MEDIAN = 1.10
MINIMUM_AUTO_BOOTSTRAP_LOWER = 1.05
EXPECTED_FIXED_FORMATS = {"p8", "p10"}
EXPECTED_FIXED_KERNELS = {"lanczos3", "spline64"}
EXPECTED_FIXED_REQUESTS = {16, 32}


def fail(message: str) -> None:
    raise ValueError(message)


def load(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")
    if not isinstance(document, dict):
        fail(f"{path} is not a JSON object")
    return document


def number(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        fail(f"{label} is not numeric")
    if not math.isfinite(result):
        fail(f"{label} is not finite")
    return result


def plugin_hash(info: object, label: str) -> str:
    if not isinstance(info, dict):
        fail(f"{label} is missing plugin metadata")
    value = info.get("sha256")
    if not isinstance(value, str) or len(value) != 64:
        fail(f"{label} has no SHA-256 identity")
    return value


def validate_cpu(document: dict) -> tuple[dict, str]:
    if document.get("schema_version") != 1:
        fail("CPU regression evidence has an unsupported schema")
    environment = document.get("environment")
    summary = document.get("summary")
    if not isinstance(environment, dict) or not isinstance(summary, list):
        fail("CPU regression evidence is incomplete")
    if len(summary) < 6 or number(environment.get("runs"), "CPU runs") < 5:
        fail("CPU regression evidence is not sufficiently repeated")
    if number(
            environment.get("regression_threshold_percent"),
            "CPU regression threshold") > 3.0:
        fail("CPU regression threshold is looser than 3 percent")

    cells = []
    for item in summary:
        if not isinstance(item, dict):
            fail("CPU regression cell is not an object")
        ratio = number(item.get("vspipe_ratio"), "CPU VSPipe ratio")
        paired = number(
            item.get("paired_ratio_median"), "CPU paired median ratio")
        passed = (
            not bool(item.get("regression"))
            and ratio >= MINIMUM_CPU_RATIO
            and paired >= MINIMUM_CPU_RATIO
        )
        cells.append({
            "kernel": item.get("kernel"),
            "threads": item.get("threads"),
            "vspipe_ratio": ratio,
            "paired_ratio_median": paired,
            "pass": passed,
        })
    failed = [item for item in cells if not item["pass"]]
    if failed:
        labels = [f"{item['kernel']}/R{item['threads']}" for item in failed]
        fail("CPU throughput regression exceeds 3 percent: " + ", ".join(labels))
    return {
        "pass": True,
        "minimum_ratio": min(
            min(item["vspipe_ratio"], item["paired_ratio_median"])
            for item in cells),
        "cells": cells,
    }, plugin_hash(environment.get("api4_plugin"), "CPU candidate")


def performance_cell(item: dict, label: str) -> dict:
    paired = item.get("paired_speedup")
    bootstrap = item.get("paired_bootstrap")
    if not isinstance(paired, dict) or not isinstance(bootstrap, dict):
        fail(f"{label} lacks paired bootstrap evidence")
    paired_median = number(paired.get("median"), f"{label} paired median")
    ratio_of_medians = number(
        bootstrap.get("ratio_of_medians"), f"{label} ratio of medians")
    lower = number(
        bootstrap.get("bootstrap_95_percent_lower"),
        f"{label} bootstrap lower bound")
    resamples = int(number(
        bootstrap.get("bootstrap_resamples"), f"{label} bootstrap resamples"))
    passed = (
        paired_median >= MINIMUM_AUTO_MEDIAN
        and ratio_of_medians >= MINIMUM_AUTO_MEDIAN
        and lower >= MINIMUM_AUTO_BOOTSTRAP_LOWER
        and resamples >= 1000
    )
    return {
        "paired_median_speedup": paired_median,
        "ratio_of_medians": ratio_of_medians,
        "bootstrap_95_percent_lower": lower,
        "bootstrap_resamples": resamples,
        "pass": passed,
    }


def validate_fixed(document: dict) -> tuple[dict, str]:
    if document.get("schema") != "dsmvc-plugin-backend-benchmark-v3":
        fail("fixed-kernel evidence has an unsupported schema")
    environment = document.get("environment")
    comparisons = document.get("comparisons")
    if not isinstance(environment, dict) or not isinstance(comparisons, list):
        fail("fixed-kernel evidence is incomplete")
    if environment.get("candidate_backend") != "auto":
        fail("fixed-kernel candidate backend is not auto")
    if set(environment.get("formats", [])) != EXPECTED_FIXED_FORMATS:
        fail("fixed-kernel formats are not the bounded P8/P10 set")
    if set(environment.get("kernels", [])) != EXPECTED_FIXED_KERNELS:
        fail("fixed-kernel kernels are not the promoted wide-kernel set")
    if set(environment.get("requests", [])) != EXPECTED_FIXED_REQUESTS:
        fail("fixed-kernel requests are not the bounded R16/R32 set")
    if number(environment.get("samples"), "fixed-kernel samples") < 7:
        fail("fixed-kernel evidence needs at least seven paired samples")
    if number(environment.get("frames"), "fixed-kernel frames") > 512:
        fail("fixed-kernel evidence exceeds the bounded frame limit")

    expected_count = (
        len(EXPECTED_FIXED_FORMATS)
        * len(EXPECTED_FIXED_KERNELS)
        * len(EXPECTED_FIXED_REQUESTS)
    )
    if len(comparisons) != expected_count:
        fail(
            f"fixed-kernel evidence has {len(comparisons)} cells; "
            f"expected {expected_count}")
    cells = []
    for item in comparisons:
        if not isinstance(item, dict):
            fail("fixed-kernel comparison is not an object")
        label = f"{item.get('format')}/{item.get('kernel')}/R{item.get('requests')}"
        result = performance_cell(item, label)
        cells.append({
            "format": item.get("format"),
            "kernel": item.get("kernel"),
            "requests": item.get("requests"),
            **result,
        })
    failed = [
        f"{item['format']}/{item['kernel']}/R{item['requests']}"
        for item in cells if not item["pass"]
    ]
    if failed:
        fail("fixed-kernel auto promotion gate failed: " + ", ".join(failed))
    plugin = environment.get("plugin_sha256")
    if not isinstance(plugin, str) or len(plugin) != 64:
        fail("fixed-kernel evidence has no plugin SHA-256")
    return {"pass": True, "cells": cells}, plugin


def validate_getfnative(document: dict) -> tuple[dict, str]:
    if document.get("schema") != "dsmvc-getfnative-plugin-ab-v2":
        fail("GetFnative evidence has an unsupported schema")
    environment = document.get("environment")
    comparisons = document.get("comparisons")
    if not isinstance(environment, dict) or not isinstance(comparisons, list):
        fail("GetFnative evidence is incomplete")
    if environment.get("profile") != "stratified256":
        fail("GetFnative evidence did not use stratified256")
    if environment.get("cases") != ["getfnative"]:
        fail("GetFnative evidence contains an unbounded or unrelated case")
    if environment.get("control_backend") != "cpu":
        fail("GetFnative control backend is not CPU")
    if environment.get("candidate_backend") != "auto":
        fail("GetFnative candidate backend is not auto")
    if number(environment.get("samples"), "GetFnative samples") < 7:
        fail("GetFnative evidence needs at least seven paired samples")
    if len(comparisons) != 1 or comparisons[0].get("candidate_count") != 256:
        fail("GetFnative evidence is not exactly 256 stratified candidates")
    result = performance_cell(comparisons[0], "GetFnative stratified256")
    if not result["pass"]:
        fail("GetFnative stratified256 auto promotion gate failed")

    control_hash = environment.get("control_plugin_sha256")
    candidate_hash = environment.get("candidate_plugin_sha256")
    if control_hash != candidate_hash:
        fail("GetFnative CPU and auto variants did not use the same plugin")
    if not isinstance(candidate_hash, str) or len(candidate_hash) != 64:
        fail("GetFnative evidence has no plugin SHA-256")
    return {"pass": True, "candidate_count": 256, **result}, candidate_hash


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cpu", required=True, type=Path)
    result.add_argument("--fixed", required=True, type=Path)
    result.add_argument("--getfnative", required=True, type=Path)
    result.add_argument("--json-out", required=True, type=Path)
    return result


def main() -> int:
    options = parser().parse_args()
    cpu, cpu_hash = validate_cpu(load(options.cpu))
    fixed, fixed_hash = validate_fixed(load(options.fixed))
    getfnative, getfnative_hash = validate_getfnative(load(options.getfnative))
    hashes = {cpu_hash, fixed_hash, getfnative_hash}
    if len(hashes) != 1:
        fail("performance evidence was not produced by one candidate plugin")
    report = {
        "schema": "dsmvc-metal-uma-performance-evidence-v1",
        "pass": True,
        "candidate_plugin_sha256": hashes.pop(),
        "thresholds": {
            "minimum_cpu_ratio": MINIMUM_CPU_RATIO,
            "minimum_auto_median_speedup": MINIMUM_AUTO_MEDIAN,
            "minimum_auto_bootstrap_95_percent_lower": (
                MINIMUM_AUTO_BOOTSTRAP_LOWER),
        },
        "cpu": cpu,
        "fixed_kernel": fixed,
        "getfnative": getfnative,
    }
    options.json_out.parent.mkdir(parents=True, exist_ok=True)
    options.json_out.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8")
    print(options.json_out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(f"Metal UMA evidence failed: {error}")
        raise SystemExit(1)
