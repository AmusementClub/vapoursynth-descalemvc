#!/usr/bin/env python3
"""Validate source-bound CPU, fixed-kernel, and GetFnative evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from paired_benchmark_support import sha256_file


MINIMUM_AUTO_MEDIAN = 1.03
MAXIMUM_CPU_REGRESSION_PERCENT = 3.0
REQUIRED_ORDER = ["C-A", "A-C", "C-A", "A-C", "C-A", "A-C"]


def fail(message: str) -> None:
    raise ValueError(message)


def load(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        fail(f"{path} does not contain a JSON object")
    return document


def number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        fail(f"{label} is not finite")
    return result


def plugin_hash(file_info, label: str) -> str:
    if not isinstance(file_info, dict):
        fail(f"{label} plugin identity is missing")
    value = file_info.get("sha256")
    if not isinstance(value, str) or len(value) != 64:
        fail(f"{label} plugin SHA-256 is missing")
    return value


def validate_manifest(document: dict, path: Path) -> dict:
    if document.get("schema") != "dsmvc-build-manifest-v1":
        fail("candidate build manifest has an unsupported schema")
    if document.get("kind") != "candidate":
        fail("build manifest does not identify the latest candidate")
    source_fingerprint = document.get("source_fingerprint")
    if not isinstance(source_fingerprint, str) or len(source_fingerprint) != 64:
        fail("build manifest source fingerprint is invalid")
    plugin = document.get("plugin")
    if not isinstance(plugin, dict):
        fail("build manifest has no plugin identity")
    plugin_path = Path(str(plugin.get("path", ""))).resolve()
    plugin_sha256 = plugin.get("sha256")
    if not plugin_path.is_file():
        fail(f"manifest plugin does not exist: {plugin_path}")
    if not isinstance(plugin_sha256, str) or len(plugin_sha256) != 64:
        fail("manifest plugin SHA-256 is invalid")
    if sha256_file(plugin_path) != plugin_sha256:
        fail("manifest plugin SHA-256 does not match the file on disk")
    architecture = str(plugin.get("file", ""))
    if "arm64" not in architecture:
        fail("manifest plugin is not recorded as arm64")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "source_fingerprint": source_fingerprint,
        "plugin_path": str(plugin_path),
        "plugin_sha256": plugin_sha256,
    }


def validate_benchmark_identity(environment: dict, manifest: dict,
                                path_key: str, hash_key: str,
                                label: str) -> None:
    benchmark_path = str(Path(str(environment.get(path_key, ""))).resolve())
    if benchmark_path != manifest["plugin_path"]:
        fail(f"{label} did not execute the manifest plugin path")
    if environment.get(hash_key) != manifest["plugin_sha256"]:
        fail(f"{label} plugin SHA-256 differs from the build manifest")
    if environment.get("source_fingerprint") != manifest["source_fingerprint"]:
        fail(f"{label} source fingerprint differs from the build manifest")
    if environment.get("build_manifest_sha256") != manifest["sha256"]:
        fail(f"{label} build manifest SHA-256 is inconsistent")


def validate_cpu(document: dict) -> tuple[dict, str]:
    if document.get("schema_version") != 1:
        fail("CPU evidence has an unsupported schema")
    environment = document.get("environment")
    summary = document.get("summary")
    samples = document.get("raw_samples")
    if not isinstance(environment, dict) or not isinstance(summary, list) \
            or not isinstance(samples, list) or not summary:
        fail("CPU evidence is incomplete")
    if environment.get("backend") != "cpu":
        fail("CPU regression evidence is not CPU-only")
    if number(
            environment.get("regression_threshold_percent"),
            "CPU regression threshold") > MAXIMUM_CPU_REGRESSION_PERCENT:
        fail("CPU regression threshold is looser than 3 percent")
    regressions = []
    worst_change = float("inf")
    for item in summary:
        if not isinstance(item, dict):
            fail("CPU summary contains a non-object cell")
        change = number(item.get("vspipe_change_percent"), "CPU cell change")
        worst_change = min(worst_change, change)
        if item.get("regression") or change < -MAXIMUM_CPU_REGRESSION_PERCENT:
            regressions.append(
                f"{item.get('kernel')}/T{item.get('threads')}={change:.3f}%")
    if regressions:
        fail("CPU-only regression gate failed: " + ", ".join(regressions))
    return {
        "pass": True,
        "cells": len(summary),
        "samples": len(samples),
        "worst_change_percent": worst_change,
    }, plugin_hash(environment.get("api4_plugin"), "CPU candidate")


def _resource_state(state: dict, label: str, baseline_swapouts: int | None) -> int:
    if not isinstance(state, dict):
        fail(f"{label} system telemetry is missing")
    free_percent = int(number(
        state.get("memory_free_percent"), f"{label} memory free percentage"))
    swapouts = int(number(state.get("swapouts"), f"{label} swapouts"))
    if free_percent < 10:
        fail(f"{label} crossed the memory-pressure stop threshold")
    if baseline_swapouts is not None and swapouts != baseline_swapouts:
        fail(f"{label} recorded swap growth")
    if not isinstance(state.get("load_average"), list):
        fail(f"{label} load average is missing")
    if not isinstance(state.get("pmset_therm"), dict):
        fail(f"{label} pmset thermal evidence is missing")
    return swapouts


def validate_runtime_samples(document: dict, expected_measured: int,
                             expected_warmups: int, label: str,
                             require_resident_reuse: bool) -> dict:
    samples = document.get("samples")
    warmups = document.get("warmup_samples")
    if not isinstance(samples, list) or len(samples) != expected_measured:
        fail(f"{label} does not retain exactly {expected_measured} measured runs")
    if not isinstance(warmups, list) or len(warmups) != expected_warmups:
        fail(f"{label} does not retain exactly {expected_warmups} warm-up runs")
    baseline_swapouts = None
    metal_frames = 0
    resident_producers = 0.0
    resident_hits = 0.0
    eliminated_bytes = 0.0
    gpu_interval_maximum = 0
    control_metal_frames = 0
    maximum_metal_errors = 0
    for index, sample in enumerate([*warmups, *samples]):
        if not isinstance(sample, dict):
            fail(f"{label} run {index} is not an object")
        baseline_swapouts = _resource_state(
            sample.get("system_before"), f"{label} run {index} before",
            baseline_swapouts)
        _resource_state(
            sample.get("system_after"), f"{label} run {index} after",
            baseline_swapouts)
        properties = sample.get("frame_properties")
        if not isinstance(properties, dict):
            fail(f"{label} run {index} has no frame-property evidence")
        properties_path = Path(str(properties.get("path", ""))).resolve()
        if not properties_path.is_file() \
                or sha256_file(properties_path) != properties.get("sha256"):
            fail(f"{label} run {index} frame-property evidence is stale")
        telemetry = sample.get("telemetry")
        if not isinstance(telemetry, dict):
            fail(f"{label} run {index} has no route telemetry")
        run_metal_frames = int(number(
            telemetry.get("metal_frames"), f"{label} run Metal frames"))
        if sample.get("variant") == "control":
            control_metal_frames += run_metal_frames
        else:
            metal_frames += run_metal_frames
            totals = telemetry.get("submission_normalized_totals", {})
            if not isinstance(totals, dict):
                fail(f"{label} candidate run has no submission telemetry")
            resident_producers += number(
                totals.get("resident_producers"), "resident producers")
            resident_hits += number(totals.get("resident_hits"), "resident hits")
            eliminated_bytes += number(
                totals.get("eliminated_staging_bytes"),
                "eliminated staging bytes")
            interval = telemetry.get("gpu_interval_nanoseconds", {})
            if not isinstance(interval, dict):
                fail(f"{label} candidate run has no GPU interval telemetry")
            gpu_interval_maximum = max(
                gpu_interval_maximum,
                int(number(interval.get("maximum"), "GPU interval maximum")))
            diagnostics = telemetry.get("diagnostic_maxima", {})
            if not isinstance(diagnostics, dict):
                fail(f"{label} candidate run has no Metal error telemetry")
            maximum_metal_errors = max(
                maximum_metal_errors,
                int(number(diagnostics.get("metal_errors"), "Metal errors")))
    if control_metal_frames != 0:
        fail(f"{label} CPU control unexpectedly routed frames to Metal")
    if metal_frames <= 0 or resident_producers <= 0:
        fail(f"{label} did not exercise Metal resident producer routing")
    if require_resident_reuse \
            and (resident_hits <= 0 or eliminated_bytes <= 0):
        fail(f"{label} did not exercise cross-consumer resident reuse")
    if gpu_interval_maximum <= 0:
        fail(f"{label} lacks GPU interval evidence")
    if maximum_metal_errors != 0:
        fail(f"{label} recorded Metal execution errors")
    return {
        "measured_runs": len(samples),
        "warmup_runs": len(warmups),
        "candidate_metal_frames": metal_frames,
        "resident_producers": resident_producers,
        "resident_hits": resident_hits,
        "eliminated_staging_bytes": eliminated_bytes,
        "resident_reuse_required": require_resident_reuse,
        "maximum_gpu_interval_nanoseconds": gpu_interval_maximum,
        "swapouts": baseline_swapouts,
    }


def performance_cell(item: dict, label: str) -> dict:
    analysis = item.get("paired_analysis")
    if not isinstance(analysis, dict):
        fail(f"{label} lacks six-pair analysis")
    pairs = analysis.get("pairs")
    wall = analysis.get("wall")
    filter_time = analysis.get("filter")
    if not isinstance(pairs, list) or len(pairs) != 6:
        fail(f"{label} does not contain exactly six timing pairs")
    if [item.get("order") for item in pairs] != REQUIRED_ORDER:
        fail(f"{label} pair order is not balanced C-A/A-C")
    if analysis.get("outlier_policy") != "all successful runs retained":
        fail(f"{label} does not retain all successful runs")
    if not isinstance(wall, dict) or not isinstance(filter_time, dict):
        fail(f"{label} lacks wall/filter analysis")
    paired = wall.get("paired_ratio")
    if not isinstance(paired, dict):
        fail(f"{label} lacks wall paired-ratio summary")
    median = number(paired.get("median"), f"{label} paired median")
    mad = number(paired.get("mad"), f"{label} paired MAD")
    ratio_of_medians = number(
        wall.get("ratio_of_medians"), f"{label} ratio of medians")
    for required in ("observations", "later_observations", "phase",
                     "order_strata", "run_index_ratio_slope",
                     "order_effect_median_difference"):
        if required not in wall:
            fail(f"{label} wall analysis lacks {required}")
    if median < MINIMUM_AUTO_MEDIAN:
        fail(
            f"{label} paired median {median:.6f} is below "
            f"{MINIMUM_AUTO_MEDIAN:.2f}x")
    return {
        "pass": True,
        "paired_median_speedup": median,
        "paired_mad": mad,
        "ratio_of_medians": ratio_of_medians,
        "filter_paired_median_speedup": number(
            filter_time.get("paired_ratio", {}).get("median"),
            f"{label} filter paired median"),
        "run_index_ratio_slope": number(
            wall.get("run_index_ratio_slope"), f"{label} run trend"),
        "order_effect_median_difference": number(
            wall.get("order_effect_median_difference"),
            f"{label} order effect"),
    }


def validate_fixed(document: dict, manifest: dict) -> tuple[dict, str]:
    if document.get("schema") != "dsmvc-plugin-backend-benchmark-v4":
        fail("fixed-kernel evidence has an unsupported schema")
    environment = document.get("environment")
    comparisons = document.get("comparisons")
    if not isinstance(environment, dict) or not isinstance(comparisons, list):
        fail("fixed-kernel evidence is incomplete")
    expected = {
        "formats": ["p10"],
        "kernels": ["spline64"],
        "requests": [16],
        "control_backend": "cpu",
        "candidate_backend": "auto",
        "samples": 6,
        "warmups": 1,
        "frames": 256,
    }
    for key, value in expected.items():
        if environment.get(key) != value:
            fail(f"fixed-kernel environment {key} is not {value!r}")
    if len(comparisons) != 1:
        fail("fixed-kernel evidence is not exactly one representative cell")
    item = comparisons[0]
    if (item.get("format"), item.get("kernel"), item.get("requests")) \
            != ("p10", "spline64", 16):
        fail("fixed-kernel representative is not p10/spline64/R16")
    validate_benchmark_identity(
        environment, manifest, "plugin", "plugin_sha256", "fixed-kernel")
    runtime = validate_runtime_samples(
        document, 12, 2, "fixed-kernel", require_resident_reuse=False)
    result = performance_cell(item, "fixed p10/spline64/R16")
    return {**result, "runtime": runtime}, environment["plugin_sha256"]


def validate_getfnative(document: dict, manifest: dict) -> tuple[dict, str]:
    if document.get("schema") != "dsmvc-getfnative-plugin-ab-v3":
        fail("GetFnative evidence has an unsupported schema")
    environment = document.get("environment")
    comparisons = document.get("comparisons")
    if not isinstance(environment, dict) or not isinstance(comparisons, list):
        fail("GetFnative evidence is incomplete")
    expected = {
        "profile": "stratified32x4",
        "cases": ["getfnative"],
        "control_backend": "cpu",
        "candidate_backend": "auto",
        "samples": 6,
        "warmups": 1,
        "requests": 16,
        "threads": 16,
    }
    for key, value in expected.items():
        if environment.get(key) != value:
            fail(f"GetFnative environment {key} is not {value!r}")
    if len(comparisons) != 1:
        fail("GetFnative evidence is not exactly one bounded case")
    item = comparisons[0]
    if item.get("candidate_count") != 128 \
            or item.get("unique_cell_count") != 32:
        fail("GetFnative evidence is not 32 unique cells / 128 observations")
    validate_benchmark_identity(
        environment, manifest, "candidate_plugin",
        "candidate_plugin_sha256", "GetFnative")
    if environment.get("control_plugin_sha256") != manifest["plugin_sha256"]:
        fail("GetFnative CPU and auto variants did not use one candidate plugin")
    runtime = validate_runtime_samples(
        document, 12, 2, "GetFnative", require_resident_reuse=True)
    result = performance_cell(item, "GetFnative stratified32x4")
    return {
        **result,
        "candidate_count": 128,
        "unique_cell_count": 32,
        "runtime": runtime,
    }, environment["candidate_plugin_sha256"]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--build-manifest", required=True, type=Path)
    result.add_argument("--cpu", type=Path)
    result.add_argument("--fixed", type=Path)
    result.add_argument("--getfnative", type=Path)
    result.add_argument("--json-out", required=True, type=Path)
    return result


def main() -> int:
    options = parser().parse_args()
    for name in ("build_manifest", "cpu", "fixed", "getfnative", "json_out"):
        value = getattr(options, name)
        if value is not None:
            setattr(options, name, value.expanduser().resolve())
    supplied = [options.cpu, options.fixed, options.getfnative]
    if not any(supplied):
        fail("at least one CPU, fixed, or GetFnative artifact is required")
    manifest = validate_manifest(
        load(options.build_manifest), options.build_manifest)
    result = {
        "schema": "dsmvc-metal-uma-evidence-v2",
        "candidate": manifest,
        "thresholds": {
            "maximum_cpu_regression_percent": MAXIMUM_CPU_REGRESSION_PERCENT,
            "minimum_full_e2e_paired_median_speedup": MINIMUM_AUTO_MEDIAN,
            "bootstrap_is_a_gate": False,
        },
    }
    hashes = set()
    if options.cpu:
        result["cpu_regression"], plugin = validate_cpu(load(options.cpu))
        hashes.add(plugin)
    if options.fixed:
        result["fixed_kernel"], plugin = validate_fixed(
            load(options.fixed), manifest)
        hashes.add(plugin)
    if options.getfnative:
        result["getfnative"], plugin = validate_getfnative(
            load(options.getfnative), manifest)
        hashes.add(plugin)
    if hashes != {manifest["plugin_sha256"]}:
        fail("performance evidence was not produced by the manifest plugin")
    result["pass"] = True
    options.json_out.parent.mkdir(parents=True, exist_ok=True)
    options.json_out.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
