#!/usr/bin/env python3
"""Merge a fresh old-only run with an existing new-only release run."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import release_benchmark as report
import fixed_kernel_benchmark as fixed_report


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def plugin_value(environment: dict, key: str):
    value = environment.get(key)
    if value is None:
        return None
    return copy.deepcopy(value)


def merge_environment(old: dict, new: dict) -> dict:
    environment = copy.deepcopy(new.get("environment", {}))
    old_environment = old.get("environment", {})
    new_environment = new.get("environment", {})
    environment.update({
        "implementations": ["old", "new"],
        "old_plugin": plugin_value(old_environment, "old_plugin"),
        "old_plugin_sha256": old_environment.get("old_plugin_sha256"),
        "new_plugin": plugin_value(new_environment, "new_plugin"),
        "new_plugin_sha256": new_environment.get("new_plugin_sha256"),
        "source_filter": new_environment.get(
            "source_filter", old_environment.get("source_filter")),
        "source_decoder": new_environment.get(
            "source_decoder", old_environment.get("source_decoder", "")),
        "source_prefer_hw": new_environment.get(
            "source_prefer_hw", old_environment.get("source_prefer_hw", 0)),
        "source_ff_loglevel": new_environment.get(
            "source_ff_loglevel", old_environment.get("source_ff_loglevel", 0)),
        "source_rap_verification": new_environment.get(
            "source_rap_verification",
            old_environment.get("source_rap_verification", -1)),
        "refresh_mode": "old-and-new",
        "old_performance_refreshed": True,
        "new_performance_reused": True,
    })
    return environment


def merge_e2e(old: dict, new: dict) -> dict:
    old_summaries = {
        item["case"]: item for item in old["performance"]["summaries"]
    }
    new_summaries = {
        item["case"]: item for item in new["performance"]["summaries"]
    }
    summaries = []
    for case in report.CASES:
        old_item = old_summaries[case]
        new_item = new_summaries[case]
        if old_item["candidate_count"] != new_item["candidate_count"]:
            raise ValueError(f"candidate count mismatch for {case}")
        old_values = old_item["old"]
        new_values = new_item["new"]
        old_seconds = old_values["elapsed_seconds"]["median"]
        new_seconds = new_values["elapsed_seconds"]["median"]
        summaries.append({
            "case": case,
            "candidate_count": old_item["candidate_count"],
            "old": copy.deepcopy(old_values),
            "new": copy.deepcopy(new_values),
            "new_speedup": old_seconds / new_seconds,
        })

    old_environment = old.get("environment", {})
    new_environment = new.get("environment", {})
    provenance = copy.deepcopy(new.get("provenance", {}))
    provenance["old_run"] = {
        "source_sha256": old_environment.get("source_sha256"),
        "runner_sha256": old_environment.get("runner_sha256"),
        "vpy_sha256": old_environment.get("vpy_sha256"),
    }
    return {
        "schema_version": 1,
        "environment": merge_environment(old, new),
        "provenance": provenance,
        "performance": {
            "summaries": summaries,
            "raw_samples": (copy.deepcopy(old["performance"]["raw_samples"])
                             + copy.deepcopy(new["performance"]["raw_samples"])),
            "commands": (list(old["performance"]["commands"])
                         + list(new["performance"]["commands"])),
        },
        "errors": {"summaries": [], "commands": [], "files": {}},
    }


def merge_fixed(old: dict, new: dict) -> dict:
    old_cases = {(item["kernel"], item["threads"]): item
                 for item in old["cases"]}
    new_cases = {(item["kernel"], item["threads"]): item
                 for item in new["cases"]}
    if set(old_cases) != set(new_cases):
        raise ValueError("fixed-kernel case sets differ")

    cases = []
    for new_item in new["cases"]:
        key = (new_item["kernel"], new_item["threads"])
        old_item = old_cases[key]
        old_values = old_item["old"]
        new_values = new_item["new"]
        old_seconds = old_values["elapsed_seconds"]["median"]
        new_seconds = new_values["elapsed_seconds"]["median"]
        merged_item = {
            key: copy.deepcopy(value)
            for key, value in new_item.items()
            if key not in ("old", "new", "speedup")
        }
        merged_item["old"] = copy.deepcopy(old_values)
        merged_item["new"] = copy.deepcopy(new_values)
        merged_item["speedup"] = old_seconds / new_seconds
        cases.append(merged_item)

    result = copy.deepcopy(new)
    result["environment"] = merge_environment(
        {"environment": old["environment"]},
        {"environment": new["environment"]},
    )
    result["environment"].update({
        "frames": new["environment"]["frames"],
        "thread_configs": new["environment"]["thread_configs"],
        "threads": new["environment"]["threads"],
        "runs": new["environment"]["runs"],
    })
    result["cases"] = cases
    result["raw_samples"] = (copy.deepcopy(old["raw_samples"])
                              + copy.deepcopy(new["raw_samples"]))
    return result


def artifactize(output: Path, fixed: dict, blank: dict) -> None:
    scaling = output / "fixed-kernel-scaling.svg"
    fixed_report.write_scaling_svg(
        fixed["cases"], fixed["kernels"], ("old", "new"), scaling)
    report.write_blank_scaling(
        blank, output / "blank-fixed-kernel-scaling.svg")

    json_path = output / "release-benchmark.json"
    merged = read_json(json_path)
    merged["artifacts"] = {
        "e2e_scaling": "e2e-scaling.svg",
        "fixed_scaling": "fixed-kernel-scaling.svg",
        "blank_scaling": "blank-fixed-kernel-scaling.svg",
        "old_e2e_root": "../release-old-lsmas-final",
        "new_e2e_root": "../release-new-lsmas-final",
        "error_report": "../release-old-lsmas-final/errors-r32t32/benchmark.json",
        "old_fixed_report": "../release-old-lsmas-final/fixed-kernel/benchmark.json",
        "new_fixed_report": "../release-new-lsmas-final/fixed-kernel/benchmark.json",
        "blank_clip_report": "../blank-fixed-kernel-digimon-810p-release-20260805/benchmark.json",
    }
    merged["environment"].update({
        "refresh_mode": "old-refresh-with-existing-new",
        "old_performance_refreshed": True,
        "new_performance_reused": True,
        "error_data_refreshed": True,
        "blank_performance_refreshed": True,
    })
    json_path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    markdown_path = output / "release-benchmark.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    markdown = markdown.replace(
        "All performance and error results below were freshly measured. The current plugin was built with generic x86-64 Release code and an AVX2/FMA-only executor TU.",
        "The old performance and paired error data were freshly measured in this refresh. Current Release performance is reused from the preceding new-only final-binary refresh and was not rerun here. The current plugin was built with generic x86-64 Release code and an AVX2/FMA-only executor TU.",
    )
    markdown = markdown.replace(
        "../fixed-kernel-digimon-810p-release/scaling.svg",
        "fixed-kernel-scaling.svg",
    )
    markdown = markdown.replace(
        "- E2E per-thread reports: `../e2e-digimon-release-r{1,8,16,32}t{1,8,16,32}/benchmark.json`",
        "- Old E2E per-thread reports: `../release-old-lsmas-final/e2e-r{1,8,16,32}/benchmark.json`\n"
        "- New E2E per-thread reports: `../release-new-lsmas-final/e2e-r{1,8,16,32}/benchmark.json`",
    )
    markdown = markdown.replace(
        "- [Full error report](../e2e-digimon-release-errors-r32t32/benchmark.json)",
        "- [Fresh paired error report](../release-old-lsmas-final/errors-r32t32/benchmark.json)",
    )
    markdown = markdown.replace(
        "- [Full fixed-kernel report](../fixed-kernel-digimon-810p-release/benchmark.json)\n"
        "- [Fixed-kernel CSV](../fixed-kernel-digimon-810p-release/benchmark.csv)",
        "- [Fresh old fixed-kernel report](../release-old-lsmas-final/fixed-kernel/benchmark.json)\n"
        "- [Existing new fixed-kernel report](../release-new-lsmas-final/fixed-kernel/benchmark.json)",
    )
    markdown_path.write_text(markdown, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--old-e2e-root", required=True, type=Path)
    parser.add_argument("--new-e2e-root", required=True, type=Path)
    parser.add_argument("--old-fixed", required=True, type=Path)
    parser.add_argument("--new-fixed", required=True, type=Path)
    parser.add_argument("--errors", required=True, type=Path)
    parser.add_argument("--blank", required=True, type=Path)
    parser.add_argument("--source-plugin", type=Path)
    parser.add_argument("--source-filter", default="lsmas")
    parser.add_argument("--source-decoder", default="")
    parser.add_argument("--source-prefer-hw", type=int, default=0)
    parser.add_argument("--source-ff-loglevel", type=int, default=0)
    parser.add_argument("--source-rap-verification", type=int, default=-1)
    parser.add_argument("--output", type=Path, default=root /
                        "benchmark-results" / "release-benchmark-20260805")
    return parser.parse_args()


def main() -> int:
    options = parse_args()
    options.source = resolve(options.source)
    options.old_e2e_root = resolve(options.old_e2e_root)
    options.new_e2e_root = resolve(options.new_e2e_root)
    options.old_fixed = resolve(options.old_fixed)
    options.new_fixed = resolve(options.new_fixed)
    options.errors = resolve(options.errors)
    options.blank = resolve(options.blank)
    options.output = resolve(options.output)
    if options.source_plugin:
        options.source_plugin = resolve(options.source_plugin)

    perf_results = {}
    for threads in report.THREADS:
        name = f"e2e-r{threads}t{threads}/benchmark.json"
        perf_results[threads] = merge_e2e(
            read_json(options.old_e2e_root / name),
            read_json(options.new_e2e_root / name),
        )
    fixed = merge_fixed(
        read_json(options.old_fixed / "benchmark.json"),
        read_json(options.new_fixed / "benchmark.json"),
    )
    errors = read_json(options.errors)
    blank = read_json(options.blank)
    report.merge_report(options, options.output, perf_results, errors, fixed,
                        blank)
    artifactize(options.output, fixed, blank)
    print(options.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
