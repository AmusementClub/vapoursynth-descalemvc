#!/usr/bin/env python3
"""Evaluate a GRAYS planner candidate against the frozen plugin control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path


CONTROL_SHA256 = (
    "cc27fca30064d90291ebdf1da4b7346e6b39c0383b5b911aee73fd3827900eb0"
)
BLANK_KERNELS = ("spline36", "spline64")
BLANK_REQUESTS = (16, 32)
RSS_PATTERN = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$", re.MULTILINE)
RECLAIM_PATTERN = re.compile(r"^\s*(\d+)\s+page reclaims\s*$", re.MULTILINE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command: list[str]) -> None:
    print("+ " + " ".join(shlex.quote(item) for item in command), flush=True)
    subprocess.run(command, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reusable_artifact(
    path: Path, control_sha256: str, candidate_sha256: str, samples: int,
) -> dict | None:
    if not path.is_file():
        return None
    document = load_json(path)
    environment = document.get("environment", {})
    if (
        environment.get("control_plugin_sha256") != control_sha256
        or environment.get("candidate_plugin_sha256") != candidate_sha256
        or environment.get("samples") != samples
    ):
        return None
    return document


def blank_bootstrap(document: dict, resamples: int, seed: int) -> dict:
    grouped: dict[tuple[str, int], dict[str, list[float]]] = {}
    for sample in document["samples"]:
        key = (sample["kernel"], sample["requests"])
        variants = grouped.setdefault(key, {"control": [], "candidate": []})
        variants[sample["variant"]].append(sample["elapsed_seconds"])

    ordered = []
    for key in sorted(grouped):
        control = grouped[key]["control"]
        candidate = grouped[key]["candidate"]
        if len(control) != len(candidate) or not control:
            raise RuntimeError(f"unpaired BlankClip samples for {key}")
        ordered.append((control, candidate))

    def aggregate(cases: list[tuple[list[float], list[float]]]) -> float:
        ratios = [
            statistics.median(control) / statistics.median(candidate)
            for control, candidate in cases
        ]
        return math.exp(statistics.mean(math.log(value) for value in ratios))

    rng = random.Random(seed)
    distribution = []
    for _ in range(resamples):
        sampled_cases = []
        for control, candidate in ordered:
            indices = [rng.randrange(len(control)) for _ in control]
            sampled_cases.append((
                [control[index] for index in indices],
                [candidate[index] for index in indices],
            ))
        distribution.append(aggregate(sampled_cases))
    distribution.sort()
    lower_index = int(0.025 * (len(distribution) - 1))
    upper_index = int(0.975 * (len(distribution) - 1))
    return {
        "geometric_mean_ratio_of_medians": aggregate(ordered),
        "bootstrap_95_percent_lower": distribution[lower_index],
        "bootstrap_95_percent_upper": distribution[upper_index],
        "bootstrap_resamples": resamples,
    }


def extract_process_memory(document: dict) -> dict:
    values = {variant: {"rss": [], "page_reclaims": []}
              for variant in ("control", "candidate")}
    for sample in document["samples"]:
        tail = sample.get("vspipe_output_tail", "")
        rss = RSS_PATTERN.search(tail)
        reclaims = RECLAIM_PATTERN.search(tail)
        if not rss or not reclaims:
            raise RuntimeError(
                "full getfnative artifact lacks time -l memory accounting")
        variant = sample["variant"]
        values[variant]["rss"].append(int(rss.group(1)))
        values[variant]["page_reclaims"].append(int(reclaims.group(1)))

    result = {}
    for metric in ("rss", "page_reclaims"):
        control = statistics.median(values["control"][metric])
        candidate = statistics.median(values["candidate"][metric])
        result[metric] = {
            "control_median": control,
            "candidate_median": candidate,
            "candidate_over_control": candidate / control,
        }
    return result


def ensure_blank(options: argparse.Namespace, control_hash: str,
                 candidate_hash: str) -> dict:
    output = options.artifact_dir / "blank.json"
    if not options.force:
        reused = reusable_artifact(
            output, control_hash, candidate_hash, options.blank_samples)
        if reused is not None:
            print(f"Reusing {output}", flush=True)
            return reused
    command = [
        sys.executable,
        str(Path(__file__).with_name("blank_plugin_ab_benchmark.py")),
        "--control-plugin", str(options.control_plugin),
        "--candidate-plugin", str(options.candidate_plugin),
        "--vspipe", str(options.vspipe),
        "--json-out", str(output),
        "--kernels", *BLANK_KERNELS,
        "--requests", *(str(value) for value in BLANK_REQUESTS),
        "--frames", "1024",
        "--samples", str(options.blank_samples),
        "--warmups", "1",
        "--bootstrap-resamples", str(options.bootstrap_resamples),
    ]
    run_command(command)
    return load_json(output)


def ensure_getfnative(options: argparse.Namespace, control_hash: str,
                      candidate_hash: str) -> dict:
    output = options.artifact_dir / "getfnative.json"
    if not options.force:
        reused = reusable_artifact(
            output, control_hash, candidate_hash, options.e2e_samples)
        if reused is not None:
            print(f"Reusing {output}", flush=True)
            return reused

    wrapper = options.artifact_dir / "vspipe-time-wrapper.sh"
    wrapper.write_text(
        "#!/bin/sh\nexec /usr/bin/time -l "
        + shlex.quote(str(options.vspipe)) + " \"$@\"\n",
        encoding="ascii",
    )
    wrapper.chmod(0o755)
    command = [
        sys.executable,
        str(Path(__file__).with_name("e2e_plugin_ab_benchmark.py")),
        "--control-plugin", str(options.control_plugin),
        "--candidate-plugin", str(options.candidate_plugin),
        "--vspipe", str(wrapper),
        "--source", str(options.source),
        "--source-plugin", str(options.source_plugin),
        "--source-filter", "ffms2",
        "--profile", "full",
        "--cases", "getfnative",
        "--requests", "16",
        "--threads", "16",
        "--backend", "cpu",
        "--opt", "0",
        "--samples", str(options.e2e_samples),
        "--warmups", "0",
        "--bootstrap-resamples", str(options.bootstrap_resamples),
        "--json-out", str(output),
    ]
    run_command(command)
    return load_json(output)


def evaluate(options: argparse.Namespace) -> dict:
    control_hash = sha256_file(options.control_plugin)
    candidate_hash = sha256_file(options.candidate_plugin)
    identity_pass = (
        control_hash == CONTROL_SHA256 and candidate_hash != control_hash)
    if not identity_pass:
        raise RuntimeError(
            f"unexpected plugin identities: control={control_hash}, "
            f"candidate={candidate_hash}")

    blank = ensure_blank(options, control_hash, candidate_hash)
    blank_aggregate = blank_bootstrap(
        blank, options.bootstrap_resamples, 0x44534D56)
    blank_case_ratios = [item["speedup"] for item in blank["comparisons"]]
    blank_pass = (
        blank_aggregate["geometric_mean_ratio_of_medians"] >= 0.99
        and blank_aggregate["bootstrap_95_percent_lower"] >= 0.98
        and min(blank_case_ratios) >= 0.98
    )
    if not blank_pass:
        return {
            "pass": False,
            "failed_gate": "blank",
            "control_sha256": control_hash,
            "candidate_sha256": candidate_hash,
            "blank": blank_aggregate,
            "blank_case_ratios": blank_case_ratios,
        }

    getfnative = ensure_getfnative(options, control_hash, candidate_hash)
    comparison = getfnative["comparisons"][0]
    memory = extract_process_memory(getfnative)
    e2e_pass = (
        comparison["paired_bootstrap"]["ratio_of_medians"] > 1.01
        and comparison["paired_bootstrap"]["bootstrap_95_percent_lower"] >= 1.0
    )
    memory_pass = (
        memory["rss"]["candidate_over_control"] <= 1.05
        and memory["page_reclaims"]["candidate_over_control"] <= 1.05
    )
    return {
        "pass": e2e_pass and memory_pass,
        "failed_gate": None if e2e_pass and memory_pass
        else ("getfnative" if not e2e_pass else "memory"),
        "control_sha256": control_hash,
        "candidate_sha256": candidate_hash,
        "blank": blank_aggregate,
        "blank_case_ratios": blank_case_ratios,
        "getfnative": comparison,
        "memory": memory,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--control-plugin", required=True, type=Path)
    result.add_argument("--candidate-plugin", required=True, type=Path)
    result.add_argument("--vspipe", required=True, type=Path)
    result.add_argument("--source", required=True, type=Path)
    result.add_argument("--source-plugin", required=True, type=Path)
    result.add_argument("--artifact-dir", required=True, type=Path)
    result.add_argument("--blank-samples", type=int, default=21)
    result.add_argument("--e2e-samples", type=int, default=7)
    result.add_argument("--bootstrap-resamples", type=int, default=50000)
    result.add_argument("--force", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    for name in (
        "control_plugin", "candidate_plugin", "vspipe", "source",
        "source_plugin", "artifact_dir",
    ):
        setattr(arguments, name, getattr(arguments, name).expanduser().resolve())
    for name in (
        "control_plugin", "candidate_plugin", "vspipe", "source",
        "source_plugin",
    ):
        if not getattr(arguments, name).is_file():
            raise FileNotFoundError(getattr(arguments, name))
    if (arguments.blank_samples < 1 or arguments.e2e_samples < 1
            or arguments.bootstrap_resamples < 1000):
        raise ValueError("sample counts must be positive; bootstrap >= 1000")
    arguments.artifact_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate(arguments)
    report_path = arguments.artifact_dir / "evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=True), flush=True)
    raise SystemExit(0 if report["pass"] else 1)
