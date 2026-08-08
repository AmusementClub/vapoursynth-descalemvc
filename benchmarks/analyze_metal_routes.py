#!/usr/bin/env python3

"""Evaluate Metal route benchmark JSON with a deterministic bootstrap gate."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Sequence


def fail(message: str) -> None:
    raise ValueError(message)


def samples(case: dict[str, object], route: str) -> list[float]:
    try:
        raw = case["routes"][route]["timings_ms_per_frame"]["wall"]["raw"]
    except (KeyError, TypeError):
        fail(f"case {case.get('name')!r} is missing wall samples for {route}")
    if not isinstance(raw, list) or len(raw) < 5:
        fail(f"case {case.get('name')!r} route {route} has fewer than five samples")
    result = [float(value) for value in raw]
    if any(not math.isfinite(value) or value <= 0.0 for value in result):
        fail(f"case {case.get('name')!r} route {route} has invalid samples")
    return result


def percentile(values: list[float], probability: float) -> float:
    values.sort()
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def paired_bootstrap_speedup(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    if len(baseline) != len(candidate):
        fail("paired route sample counts differ")
    count = len(baseline)
    paired_speedups = [
        baseline[index] / candidate[index] for index in range(count)
    ]
    generator = random.Random(seed)
    distribution: list[float] = []
    for _ in range(resamples):
        distribution.append(
            statistics.median(
                paired_speedups[generator.randrange(count)] for _ in range(count)
            )
        )
    return {
        "ratio_of_medians": statistics.median(baseline)
        / statistics.median(candidate),
        "paired_median_speedup": statistics.median(paired_speedups),
        "bootstrap_95_percent_lower": percentile(distribution, 0.025),
        "bootstrap_95_percent_upper": percentile(distribution, 0.975),
        "bootstrap_resamples": resamples,
    }


def correctness_pass(
    document: dict[str, object], case: dict[str, object], route: str
) -> bool:
    try:
        correctness = case["routes"][route]["correctness"]
        limits = document["error_limits"]
        return (
            bool(correctness["finite"])
            and float(correctness["maximum_absolute_error"])
            <= float(limits["absolute"])
            and float(correctness["maximum_relative_error"])
            <= float(limits["relative"])
        )
    except (KeyError, TypeError, ValueError):
        fail(f"case {case.get('name')!r} has invalid correctness data for {route}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_json", type=Path)
    parser.add_argument("--baseline-route", default="neon-neon")
    parser.add_argument("--minimum-lower-bound", type=float, default=1.05)
    parser.add_argument("--resamples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0x44534D56)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--assert-route")
    parser.add_argument("--assert-case")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.resamples < 1000:
        fail("--resamples must be at least 1000")
    if not math.isfinite(arguments.minimum_lower_bound):
        fail("--minimum-lower-bound must be finite")

    document = json.loads(arguments.benchmark_json.read_text())
    if document.get("benchmark") not in {
        "dsmvc-metal-routes",
        "dsmvc-metal-yuv-routes",
    }:
        fail("input is not a supported dsmvc Metal route benchmark")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        fail("benchmark has no cases")
    first_routes = cases[0].get("routes")
    if not isinstance(first_routes, dict):
        fail("benchmark has no routes")
    route_names = list(first_routes)
    if arguments.baseline_route not in route_names:
        fail(f"baseline route {arguments.baseline_route!r} is missing")

    route_reports: dict[str, object] = {}
    accepted_routes: list[str] = []
    for route_index, route in enumerate(route_names):
        case_reports: list[dict[str, object]] = []
        route_pass = route != arguments.baseline_route
        for case_index, case in enumerate(cases):
            if not isinstance(case, dict):
                fail("benchmark case is not an object")
            baseline = samples(case, arguments.baseline_route)
            candidate = samples(case, route)
            estimate = paired_bootstrap_speedup(
                baseline,
                candidate,
                resamples=arguments.resamples,
                seed=arguments.seed + route_index * 1009 + case_index,
            )
            correct = correctness_pass(document, case, route)
            clears_gate = (
                route != arguments.baseline_route
                and correct
                and estimate["bootstrap_95_percent_lower"]
                >= arguments.minimum_lower_bound
            )
            route_pass = route_pass and clears_gate
            case_reports.append(
                {
                    "name": case.get("name", ""),
                    "correctness_pass": correct,
                    **estimate,
                    "clears_lower_bound_gate": clears_gate,
                }
            )
        route_reports[route] = {
            "accepted": route_pass,
            "accepted_cases": [
                item["name"]
                for item in case_reports
                if item["clears_lower_bound_gate"]
            ],
            "cases": case_reports,
        }
        if route_pass:
            accepted_routes.append(route)

    report = {
        "schema_version": 1,
        "source": str(arguments.benchmark_json),
        "baseline_route": arguments.baseline_route,
        "minimum_bootstrap_95_percent_lower": arguments.minimum_lower_bound,
        "accepted_routes": accepted_routes,
        "routes": route_reports,
    }
    arguments.json_out.parent.mkdir(parents=True, exist_ok=True)
    arguments.json_out.write_text(json.dumps(report, indent=2) + "\n")

    if arguments.assert_route:
        route = route_reports.get(arguments.assert_route)
        if route is None:
            fail(f"asserted route {arguments.assert_route!r} is missing")
        if arguments.assert_case:
            selected = next(
                (
                    item
                    for item in route["cases"]
                    if item["name"] == arguments.assert_case
                ),
                None,
            )
            if selected is None:
                fail(f"asserted case {arguments.assert_case!r} is missing")
            assertion_pass = selected["clears_lower_bound_gate"]
        else:
            assertion_pass = route["accepted"]
        if not assertion_pass:
            suffix = (
                f" case {arguments.assert_case}"
                if arguments.assert_case
                else " across all cases"
            )
            print(
                f"route gate failed: {arguments.assert_route}{suffix}",
                file=sys.stderr,
            )
            return 1
    print(arguments.json_out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Metal route analysis failed: {error}", file=sys.stderr)
        raise SystemExit(1)
