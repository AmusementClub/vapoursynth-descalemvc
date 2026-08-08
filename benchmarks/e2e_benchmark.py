#!/usr/bin/env python3
"""End-to-end old descale versus dsmvc benchmark on a real video.

The three cases are derived from the supplied GetNative training scripts.  A
fresh VSPipe process measures each implementation, while a separate worker
computes paired old/new error metrics for every candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html.parser
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


CASES = ("getfnative", "getfnative_v2", "selectkernel")
PROFILES = ("smoke", "stratified32x4", "stratified256", "full")
IMPLEMENTATIONS = ("old", "new")
SCRIPT_CASES = {
    "getfnative": "test_getfnative.vpy",
    "getfnative_v2": "test_getfnative_v2.vpy",
    "selectkernel": "test_selectkernel.vpy",
}
RESULT_PREFIX = "DSMVC_E2E_RESULT="
OUTPUT_RE = re.compile(
    r"Output (?P<frames>\d+) frames in (?P<seconds>[0-9.]+) seconds "
    r"\((?P<fps>[0-9.]+) fps\)"
)


def repeated_arange(start: float, stop: float, step: float) -> list[float]:
    values = []
    current = start
    while current < stop:
        values.append(current)
        current += step
    return values


def scaler(kernel: str, b: float = 0.0, c: float = 0.5,
           taps: int = 3) -> dict:
    if kernel == "bicubic":
        name = f"bicubic_b{b:.1f}_c{c:.1f}"
    elif kernel == "lanczos":
        name = f"lanczos{taps}"
    else:
        name = kernel
    return {
        "name": name,
        "kernel": kernel,
        "b": b,
        "c": c,
        "taps": taps,
    }


GETFNATIVE_SCALERS = [
    scaler("bilinear"),
    scaler("bicubic", 1 / 3, 1 / 3),
    scaler("bicubic", 0.0, 0.5),
    scaler("bicubic", 0.0, 1.0),
    scaler("bicubic", 1.0, 0.0),
    scaler("bicubic", 0.0, 0.75),
    scaler("lanczos", taps=2),
    scaler("lanczos", taps=3),
    scaler("lanczos", taps=4),
    scaler("spline16"),
    scaler("spline36"),
]
GETFNATIVE_V2_SCALERS = GETFNATIVE_SCALERS[:6] + [
    scaler("spline16"), scaler("spline36")]
STRATIFIED_GETFNATIVE_SCALERS = [
    GETFNATIVE_SCALERS[index] for index in (0, 2, 7, 10)
]
SELECTKERNEL_PARAMETERS = repeated_arange(0, 1, 0.1)[:10]
SELECTKERNEL_SCALERS = [scaler("bilinear")] + [
    scaler("bicubic", b, c)
    for b in SELECTKERNEL_PARAMETERS
    for c in SELECTKERNEL_PARAMETERS
]
SMOKE_HEIGHTS = {
    "getfnative": [700.0, 719.8, 840.0, 900.0, 951.4, 951.5, 951.6, 979.9],
    "getfnative_v2": [840.0, 859.9, 860.0, 860.1, 879.9],
}
FUNCTIONS = {
    "bilinear": "Debilinear",
    "bicubic": "Debicubic",
    "lanczos": "Delanczos",
    "spline16": "Despline16",
    "spline36": "Despline36",
}
RECIPE_FACTS = {
    "getfnative": {
        "frame": 12493,
        "base_height": 1000,
        "vertical_only": False,
        "ex_thr": 0.015,
        "height_start": 700.0,
        "height_stop": 980.0,
        "height_step": 0.1,
        "fixed_height": None,
        "scaler_count": 11,
    },
    "getfnative_v2": {
        "frame": 358,
        "base_height": 1000,
        "vertical_only": True,
        "ex_thr": 0.015,
        "height_start": 840.0,
        "height_stop": 880.0,
        "height_step": 0.1,
        "fixed_height": None,
        "scaler_count": 8,
    },
    "selectkernel": {
        "frame": 1111,
        "base_height": 1000,
        "vertical_only": False,
        "ex_thr": 0.012,
        "height_start": None,
        "height_stop": None,
        "height_step": None,
        "fixed_height": 719.8,
        "scaler_count": 101,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_info(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "size": resolved.stat().st_size if resolved.is_file() else None,
        "sha256": sha256_file(resolved) if resolved.is_file() else None,
    }


def recipe_candidates(case: str, profile: str) -> list[dict]:
    facts = RECIPE_FACTS[case]
    repetitions = 1
    if case == "getfnative":
        if profile == "full":
            heights = repeated_arange(700.0, 980.0, 0.1)
            scalers = GETFNATIVE_SCALERS
        elif profile == "stratified32x4":
            full_heights = [value / 10.0 for value in range(7000, 9800)]
            heights = [
                full_heights[(2 * index + 1) * len(full_heights) // 16]
                for index in range(8)
            ]
            scalers = STRATIFIED_GETFNATIVE_SCALERS
            repetitions = 4
        elif profile == "stratified256":
            full_heights = [value / 10.0 for value in range(7000, 9800)]
            heights = [
                full_heights[(2 * index + 1) * len(full_heights) // 128]
                for index in range(64)
            ]
            scalers = STRATIFIED_GETFNATIVE_SCALERS
        else:
            heights = SMOKE_HEIGHTS[case]
            scalers = GETFNATIVE_SCALERS
    elif case == "getfnative_v2":
        if profile in ("stratified32x4", "stratified256"):
            raise ValueError(
                "stratified profiles are defined only for getfnative")
        heights = (repeated_arange(840.0, 880.0, 0.1)
                   if profile == "full" else SMOKE_HEIGHTS[case])
        scalers = GETFNATIVE_V2_SCALERS
    else:
        if profile in ("stratified32x4", "stratified256"):
            raise ValueError(
                "stratified profiles are defined only for getfnative")
        heights = [facts["fixed_height"]]
        scalers = SELECTKERNEL_SCALERS
    candidates = []
    for repetition in range(repetitions):
        for scaler_spec in scalers:
            for height in heights:
                candidates.append({
                    "index": len(candidates),
                    "id": f"{scaler_spec['name']}@{height:.1f}",
                    "repetition": repetition,
                    "scaler": scaler_spec,
                    "height": height,
                })
    return candidates


class HtmlMediaParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.media = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag not in ("audio", "video", "source"):
            return
        attributes = dict(attrs)
        value = attributes.get("src") or attributes.get("data-src")
        if value:
            self.media.append({"tag": tag, "src": value})


def html_info(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "exists": False}
    text = resolved.read_text(encoding="utf-8", errors="replace")
    parser = HtmlMediaParser()
    parser.feed(text)
    attachments = sorted(set(
        match.group(1)
        for match in re.finditer(
            r"\[文件:([^\r\n\"]+?\.(?:mkv|vpy|py|zip|pdf|png))\]",
            text, re.IGNORECASE)
    ))
    resource_files = sorted(set(
        match.group(1).split("#", 1)[0]
        for match in re.finditer(
            r"resources/files/([^\\\"'<>\r\n]+)", text)
    ))
    media = list(parser.media)
    media.extend({"tag": "resource", "src": f"resources/files/{name}"}
                 for name in resource_files
                 if name.lower().endswith((".mkv", ".mp4", ".webm", ".ts")))
    return {
        "path": str(resolved),
        "exists": True,
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "media": sorted(media, key=lambda item: item["src"]),
        "attachments": attachments,
        "resource_files": resource_files,
    }


def first_int(pattern: str, text: str):
    match = re.search(pattern, text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def first_float(pattern: str, text: str):
    match = re.search(pattern, text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def script_info(case: str, path: Path) -> dict:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {"case": case, "path": str(resolved), "exists": False}
    text = resolved.read_text(encoding="utf-8", errors="replace")
    facts = RECIPE_FACTS[case]
    frame = first_int(r"src8\s*\[\s*(\d+)\s*\]", text)
    base_height = first_int(r"base_height\s*=\s*(\d+)", text)
    ex_thr = first_float(r"ex_thr\s*=\s*([0-9.]+)", text)
    if ex_thr is None:
        ex_thr = facts["ex_thr"]
    vertical_match = re.search(r"vertical_only\s*=\s*(true|false)", text,
                               re.IGNORECASE)
    vertical_only = (vertical_match.group(1).lower() == "true"
                     if vertical_match else False)
    range_match = re.search(
        r"src_heights\s*=\s*(?:muf|muvsfunc)\.arange\(\s*"
        r"([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", text,
        re.IGNORECASE)
    fixed_height = first_float(
        r"(?m)^(?!\s*#).*?src_heights\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        text)
    parsed = {
        "frame": frame,
        "base_height": base_height,
        "vertical_only": vertical_only,
        "ex_thr": ex_thr,
        "height_start": float(range_match.group(1)) if range_match else None,
        "height_stop": float(range_match.group(2)) if range_match else None,
        "height_step": float(range_match.group(3)) if range_match else None,
        "fixed_height": (None if range_match else fixed_height),
        "source_assignments": re.findall(
            r"(?:^|\n)\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*r?[\"']([^\"']+)",
            text),
    }
    mismatches = []
    for key in ("frame", "base_height", "vertical_only", "ex_thr",
                "height_start", "height_stop", "height_step", "fixed_height"):
        expected = facts[key]
        actual = parsed[key]
        if isinstance(expected, float) and isinstance(actual, (int, float)):
            same = math.isclose(expected, actual, rel_tol=0.0, abs_tol=1e-9)
        else:
            same = expected == actual
        if not same:
            mismatches.append({"field": key, "expected": expected,
                               "actual": actual})
    return {
        "case": case,
        "path": str(resolved),
        "exists": True,
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "parsed": parsed,
        "matches_recipe_facts": not mismatches,
        "mismatches": mismatches,
    }


def parse_case_scripts(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--script expects CASE=PATH")
        case, raw_path = value.split("=", 1)
        if case not in CASES:
            raise ValueError("unknown script case: " + case)
        if case in result:
            raise ValueError("script supplied twice for case: " + case)
        result[case] = Path(raw_path)
    return result


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"median": None, "mad": None, "p95": None,
                "minimum": None, "maximum": None}
    median = statistics.median(ordered)
    p95 = ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]
    return {
        "median": median,
        "mad": statistics.median(abs(value - median) for value in ordered),
        "p95": p95,
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def command_text(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return " ".join(subprocess.list2cmdline([item]) for item in command)


def parse_vspipe_timing(output: str, expected_frames: int) -> dict:
    match = OUTPUT_RE.search(output)
    if match is None:
        raise RuntimeError("VSPipe output did not contain an Output timing line")
    emitted_frames = int(match.group("frames"))
    if emitted_frames != expected_frames:
        raise RuntimeError(
            f"VSPipe emitted {emitted_frames} frames; expected {expected_frames}")
    return {
        "frames": emitted_frames,
        "seconds": float(match.group("seconds")),
        "fps": float(match.group("fps")),
    }


def candidate_ranges(total: int, batch_size: int) -> list[tuple[int, int]]:
    if total < 1:
        return []
    if batch_size <= 0 or batch_size >= total:
        return [(0, total)]
    return [(start, min(total, start + batch_size))
            for start in range(0, total, batch_size)]


def run_vspipe(options, case: str, implementation: str, run: int,
               candidate_start: int = 0, candidate_end: int | None = None,
               warmup: bool = False) -> dict:
    script = Path(__file__).with_name("vspipe_e2e.vpy").resolve()
    all_candidates = recipe_candidates(case, options.profile)
    total_candidates = len(all_candidates)
    if candidate_end is None:
        candidate_end = total_candidates
    if not 0 <= candidate_start < candidate_end <= total_candidates:
        raise ValueError(
            f"candidate range [{candidate_start}, {candidate_end}) is outside "
            f"0..{total_candidates}")
    candidate_count = candidate_end - candidate_start
    command = [options.vspipe]
    args = {
        "implementation": implementation,
        "case": case,
        "profile": options.profile,
        "source": str(Path(options.source).expanduser().resolve()),
        "plugin": str(Path(options.new_plugin).expanduser().resolve()),
        "old_plugin": (str(Path(options.old_plugin).expanduser().resolve())
                       if options.old_plugin else ""),
        "source_plugin": (str(Path(options.source_plugin).expanduser().resolve())
                          if options.source_plugin else ""),
        "source_filter": options.source_filter,
        "source_decoder": options.source_decoder,
        "source_prefer_hw": str(options.source_prefer_hw),
        "source_ff_loglevel": str(options.source_ff_loglevel),
        "source_rap_verification": str(options.source_rap_verification),
        "frame": str(RECIPE_FACTS[case]["frame"]),
        "threads": str(options.threads),
        "backend": options.backend,
        "opt": str(options.opt),
        "cache_mb": str(options.performance_cache_mb),
        "candidate_start": str(candidate_start),
        "candidate_end": str(candidate_end),
    }
    for key, value in args.items():
        command.extend(["--arg", f"{key}={value}"])
    command.extend([
        "--requests", str(options.requests),
        "--start", "0",
        "--end", str(candidate_count - 1),
        "--filter-time",
        str(script),
        "--",
    ])
    start = time.perf_counter_ns()
    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace", check=False,
        env={**os.environ, "DSMVC_CUDA_PLAN_CACHE_MB":
             str(options.cuda_plan_cache_mb)})
    elapsed_ns = time.perf_counter_ns() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"VSPipe failed for {case}/{implementation}/run-{run}:\n"
            + completed.stdout + "\n" + completed.stderr)
    output = completed.stdout + completed.stderr
    vspipe_timing = parse_vspipe_timing(output, candidate_count)
    return {
        "case": case,
        "implementation": implementation,
        "run": run,
        "candidate_count": candidate_count,
        "candidate_total_count": total_candidates,
        "candidate_start": candidate_start,
        "candidate_end": candidate_end,
        "warmup": warmup,
        "requests": options.requests,
        "threads": options.threads,
        "backend": options.backend if implementation == "new" else "baseline",
        "opt": options.opt if implementation == "new" else 0,
        "elapsed_ns": elapsed_ns,
        "candidates_per_second": candidate_count / (elapsed_ns / 1e9),
        "command": command_text(command),
        "vspipe_seconds": vspipe_timing["seconds"],
        "vspipe_candidates_per_second": vspipe_timing["fps"],
        "process_overhead_seconds": max(
            0.0, elapsed_ns / 1e9 - vspipe_timing["seconds"]),
        "vspipe_output_tail": output[-4000:],
    }


def combine_vspipe_batches(case: str, implementation: str, run: int,
                            batches: list[dict], batch_size: int) -> dict:
    if not batches:
        raise ValueError("cannot combine an empty VSPipe batch list")
    expected_start = 0
    for batch in batches:
        if batch["candidate_start"] != expected_start:
            raise RuntimeError(
                f"candidate batches for {case} are not contiguous at "
                f"{expected_start}")
        expected_start = batch["candidate_end"]
    total_candidates = batches[0]["candidate_total_count"]
    if expected_start != total_candidates:
        raise RuntimeError(
            f"candidate batches for {case} cover {expected_start} of "
            f"{total_candidates}")
    elapsed_ns = sum(batch["elapsed_ns"] for batch in batches)
    # VSPipe prints a rounded FPS value. Sum its reported seconds directly so
    # batch aggregation does not add rounding error, especially for short
    # low-candidate smoke runs.
    vspipe_seconds = sum(batch["vspipe_seconds"] for batch in batches)
    return {
        "case": case,
        "implementation": implementation,
        "run": run,
        "candidate_count": total_candidates,
        "candidate_total_count": total_candidates,
        "candidate_start": 0,
        "candidate_end": total_candidates,
        "warmup": False,
        "batch_count": len(batches),
        "batch_size": (total_candidates if batch_size <= 0
                       else min(batch_size, total_candidates)),
        "requests": batches[0]["requests"],
        "threads": batches[0]["threads"],
        "elapsed_ns": elapsed_ns,
        "candidates_per_second": total_candidates / (elapsed_ns / 1e9),
        "vspipe_seconds": vspipe_seconds,
        "vspipe_candidates_per_second": (
            total_candidates / vspipe_seconds if vspipe_seconds > 0 else None),
        "process_overhead_seconds": max(
            0.0, elapsed_ns / 1e9 - vspipe_seconds),
        "command": "\n".join(batch["command"] for batch in batches),
        "batch_commands": [batch["command"] for batch in batches],
        "batches": batches,
    }


def performance_summary(samples: list[dict], case: str,
                        candidate_count: int,
                        implementations: tuple[str, ...]) -> dict:
    result = {"case": case, "candidate_count": candidate_count}
    for implementation in implementations:
        selected = [item for item in samples
                    if item["implementation"] == implementation]
        elapsed = [item["elapsed_ns"] / 1e9 for item in selected]
        throughput = [item["candidates_per_second"] for item in selected]
        vspipe_elapsed = [item["vspipe_seconds"] for item in selected]
        vspipe_throughput = [
            item["vspipe_candidates_per_second"] for item in selected]
        overhead = [item["process_overhead_seconds"] for item in selected]
        result[implementation] = {
            "runs": len(selected),
            "elapsed_seconds": summarize(elapsed),
            "candidates_per_second": summarize(throughput),
            "vspipe_seconds": summarize(vspipe_elapsed),
            "vspipe_candidates_per_second": summarize(vspipe_throughput),
            "process_overhead_seconds": summarize(overhead),
        }
    if set(implementations) == set(IMPLEMENTATIONS):
        old_median = result["old"]["elapsed_seconds"]["median"]
        new_median = result["new"]["elapsed_seconds"]["median"]
        result["new_speedup"] = (old_median / new_median
                                  if old_median and new_median else None)
    else:
        result["new_speedup"] = None
    return result


def load_vs_plugins(core, options, need_old: bool = True,
                    need_new: bool = True) -> None:
    source_namespace = {"lsmas": "lsmas", "ffms2": "ffms2",
                        "bestsource": "bs"}[options.source_filter]
    if options.source_plugin and not hasattr(core, source_namespace):
        core.std.LoadPlugin(path=str(Path(options.source_plugin).resolve()))
    if need_old and not hasattr(core, "descale"):
        core.std.LoadPlugin(path=str(Path(options.old_plugin).resolve()))
    if need_new and not hasattr(core, "dsmvc"):
        core.std.LoadPlugin(path=str(Path(options.new_plugin).resolve()))


def build_geometry(width: int, height: int, src_height: float,
                   vertical_only: bool) -> dict:
    if vertical_only:
        output_height = 1000 - 2 * int((1000 - src_height) / 2)
        return {
            "width": width,
            "height": output_height,
            "src_left": 0.0,
            "src_top": (output_height - src_height) / 2,
            "src_width": float(width),
            "src_height": src_height,
        }
    src_width = width / height * src_height
    base_width = round(width / height * 1000)
    output_width = base_width - 2 * int((base_width - src_width) / 2)
    output_height = 1000 - 2 * int((1000 - src_height) / 2)
    return {
        "width": output_width,
        "height": output_height,
        "src_left": (output_width - src_width) / 2,
        "src_top": (output_height - src_height) / 2,
        "src_width": src_width,
        "src_height": src_height,
    }


def build_descale(core, implementation: str, source, candidate: dict,
                  vertical_only: bool, backend: str):
    scaler_spec = candidate["scaler"]
    arguments = build_geometry(
        source.width, source.height, candidate["height"], vertical_only)
    namespace = core.dsmvc if implementation == "new" else core.descale
    kwargs = dict(arguments)
    if scaler_spec["kernel"] == "bicubic":
        kwargs.update(b=scaler_spec["b"], c=scaler_spec["c"])
    elif scaler_spec["kernel"] == "lanczos":
        kwargs["taps"] = scaler_spec["taps"]
    if implementation == "new":
        kwargs["backend"] = backend
    output = getattr(namespace, FUNCTIONS[scaler_spec["kernel"]])(
        source, **kwargs)
    return output, arguments


def open_source(core, source_filter: str, path: str, options=None):
    if source_filter == "lsmas":
        kwargs = {}
        if options is not None:
            if options.source_decoder:
                kwargs["decoder"] = options.source_decoder
            if options.source_prefer_hw:
                kwargs["prefer_hw"] = options.source_prefer_hw
            if options.source_ff_loglevel:
                kwargs["ff_loglevel"] = options.source_ff_loglevel
            if options.source_rap_verification >= 0:
                kwargs["rap_verification"] = options.source_rap_verification
        return core.lsmas.LWLibavSource(path, **kwargs)
    if source_filter == "ffms2":
        return core.ffms2.Source(path)
    if source_filter == "bestsource":
        return core.bs.VideoSource(path)
    raise ValueError("unknown source filter: " + source_filter)


def resize_with_scaler(core, clip, scaler_spec: dict, arguments: dict,
                       width: int, height: int):
    resize = getattr(core.resize, scaler_spec["kernel"].capitalize())
    kwargs = {
        "src_left": arguments["src_left"],
        "src_top": arguments["src_top"],
        "src_width": arguments["src_width"],
        "src_height": arguments["src_height"],
    }
    if scaler_spec["kernel"] == "bicubic":
        kwargs["filter_param_a"] = scaler_spec["b"]
        kwargs["filter_param_b"] = scaler_spec["c"]
    elif scaler_spec["kernel"] == "lanczos":
        kwargs["filter_param_a"] = scaler_spec["taps"]
    return resize(clip, width, height, **kwargs)


def prop_value(props, name: str):
    try:
        return getattr(props, name)
    except AttributeError:
        return props[name]


def compare_metrics(core, left, right, threshold: float) -> dict:
    difference = core.std.Expr([left, right], ["x y - abs"])
    stats = difference.std.PlaneStats().get_frame(0).props
    squared = core.std.Expr([difference], ["x dup *"])
    squared_stats = squared.std.PlaneStats().get_frame(0).props
    thresholded = core.std.Expr(
        [difference], [f"x {threshold:.17g} > 0 x ?"])
    thresholded_stats = thresholded.std.PlaneStats().get_frame(0).props
    over = core.std.Expr(
        [difference], [f"x {threshold:.17g} > 1 0 ?"])
    over_stats = over.std.PlaneStats().get_frame(0).props
    maximum = float(prop_value(stats, "PlaneStatsMax"))
    mae = float(prop_value(stats, "PlaneStatsAverage"))
    mse = float(prop_value(squared_stats, "PlaneStatsAverage"))
    return {
        "max_abs": maximum,
        "mae": mae,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "psnr": None if mse == 0.0 else -10.0 * math.log10(mse),
        "thresholded_mae": float(
            prop_value(thresholded_stats, "PlaneStatsAverage")),
        "fraction_over_threshold": float(
            prop_value(over_stats, "PlaneStatsAverage")),
    }


def compare_frame_arrays(numpy, left, right, threshold: float,
                         crop: bool) -> dict:
    """Compute the same metrics without building one PlaneStats graph per metric."""
    difference = numpy.abs(left - right)
    if crop:
        difference = difference[5:-5, 5:-5]
    squared = difference * difference
    thresholded = numpy.where(difference > threshold, 0.0, difference)
    over = difference > threshold
    mae = float(numpy.mean(difference, dtype=numpy.float64))
    mse = float(numpy.mean(squared, dtype=numpy.float64))
    return {
        "max_abs": float(numpy.max(difference)),
        "mae": mae,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "psnr": None if mse == 0.0 else -10.0 * math.log10(mse),
        "thresholded_mae": float(
            numpy.mean(thresholded, dtype=numpy.float64)),
        "fraction_over_threshold": float(
            numpy.mean(over, dtype=numpy.float64)),
    }


def crop_for_metric(core, clip):
    return core.std.CropRel(clip, 5, 5, 5, 5)


def clip_description(clip) -> dict:
    return {
        "width": clip.width,
        "height": clip.height,
        "format": clip.format.name,
    }


def frame_hash(frame, clip) -> str:
    digest = hashlib.sha256()
    for plane in range(clip.format.num_planes):
        digest.update(memoryview(frame[plane]).tobytes(order="C"))
    return digest.hexdigest()


def worker_errors(options) -> int:
    import vapoursynth as vs

    try:
        import numpy
    except ImportError:
        numpy = None

    core = vs.core
    # VapourSynth defaults to roughly half of host RAM per process. Error
    # shards run concurrently, so leave an explicit bounded budget per worker.
    core.max_cache_size = options.error_cache_mb
    core.num_threads = options.threads
    load_vs_plugins(core, options)
    src8 = open_source(
        core, options.source_filter,
        str(Path(options.source).expanduser().resolve()), options)
    source_frame = src8[RECIPE_FACTS[options.case]["frame"]]
    source_frame = core.std.ShufflePlanes(source_frame, 0, vs.GRAY)
    source = source_frame.resize.Point(format=vs.GRAYS)
    source_array = None
    if numpy is not None:
        source_array = numpy.asarray(source.get_frame(0)[0])
    all_candidates = recipe_candidates(options.case, options.profile)
    shard_count = options.error_shard_count
    shard_index = options.error_shard_index
    if not 0 <= shard_index < shard_count:
        raise ValueError("error shard index is outside shard count")
    candidates = [candidate for candidate in all_candidates
                  if candidate["index"] % shard_count == shard_index]
    facts = RECIPE_FACTS[options.case]
    rows = []
    for candidate in candidates:
        old, arguments = build_descale(
            core, "old", source, candidate, facts["vertical_only"],
            options.backend)
        new, new_arguments = build_descale(
            core, "new", source, candidate, facts["vertical_only"],
            options.backend)
        old_frame = old.get_frame(0)
        new_frame = new.get_frame(0)
        old_reconstructed = resize_with_scaler(
            core, old, candidate["scaler"], arguments,
            source.width, source.height)
        new_reconstructed = resize_with_scaler(
            core, new, candidate["scaler"], new_arguments,
            source.width, source.height)
        if numpy is not None:
            old_reconstructed_frame = old_reconstructed.get_frame(0)
            new_reconstructed_frame = new_reconstructed.get_frame(0)
            old_array = numpy.asarray(old_frame[0])
            new_array = numpy.asarray(new_frame[0])
            old_reconstructed_array = numpy.asarray(old_reconstructed_frame[0])
            new_reconstructed_array = numpy.asarray(new_reconstructed_frame[0])
            output_old_new = compare_frame_arrays(
                numpy, old_array, new_array, facts["ex_thr"], False)
            reconstruction_old_new = compare_frame_arrays(
                numpy, old_reconstructed_array, new_reconstructed_array,
                facts["ex_thr"], True)
            old_vs_source = compare_frame_arrays(
                numpy, source_array, old_reconstructed_array,
                facts["ex_thr"], True)
            new_vs_source = compare_frame_arrays(
                numpy, source_array, new_reconstructed_array,
                facts["ex_thr"], True)
        else:
            source_crop = crop_for_metric(core, source)
            old_reconstruction_crop = crop_for_metric(core, old_reconstructed)
            new_reconstruction_crop = crop_for_metric(core, new_reconstructed)
            output_old_new = compare_metrics(
                core, old, new, facts["ex_thr"])
            reconstruction_old_new = compare_metrics(
                core, old_reconstruction_crop, new_reconstruction_crop,
                facts["ex_thr"])
            old_vs_source = compare_metrics(
                core, source_crop, old_reconstruction_crop, facts["ex_thr"])
            new_vs_source = compare_metrics(
                core, source_crop, new_reconstruction_crop, facts["ex_thr"])
        rows.append({
            "index": candidate["index"],
            "id": candidate["id"],
            "scaler": candidate["scaler"]["name"],
            "kernel": candidate["scaler"]["kernel"],
            "height": candidate["height"],
            "geometry": arguments,
            "old_output": clip_description(old),
            "new_output": clip_description(new),
            "old_sha256": frame_hash(old_frame, old),
            "new_sha256": frame_hash(new_frame, new),
            "output_shape_equal": (
                old.width, old.height, old.format.id
            ) == (new.width, new.height, new.format.id),
            "output_old_new": output_old_new,
            "reconstruction_old_new": reconstruction_old_new,
            "old_vs_source": old_vs_source,
            "new_vs_source": new_vs_source,
        })
        del old, new, old_reconstructed, new_reconstructed
        if len(rows) % 128 == 0:
            import gc
            gc.collect()
    payload = {
        "schema_version": 1,
        "case": options.case,
        "profile": options.profile,
        "frame": facts["frame"],
        "candidate_count": len(rows),
        "candidate_total_count": len(all_candidates),
        "source": str(Path(options.source).expanduser().resolve()),
        "threads": options.threads,
        "backend": options.backend,
        "cuda_plan_cache_mb": options.cuda_plan_cache_mb,
        "error_cache_mb": options.error_cache_mb,
        "error_shard_count": shard_count,
        "error_shard_index": shard_index,
        "rows": rows,
    }
    output = Path(options.error_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True),
                      encoding="utf-8")
    print(RESULT_PREFIX + json.dumps({
        "case": options.case,
        "candidate_count": len(rows),
        "output": str(output),
    }, separators=(",", ":")))
    return 0


def error_worker_command(options, case: str, output: Path,
                         shard_index: int, shard_count: int,
                         worker_threads: int) -> list[str]:
    command = [
        options.python,
        str(Path(__file__).resolve()),
        "--worker-errors",
        "--case", case,
        "--profile", options.profile,
        "--source", options.source,
        "--old-plugin", options.old_plugin,
        "--new-plugin", options.new_plugin,
        "--source-filter", options.source_filter,
        "--backend", options.backend,
        "--cuda-plan-cache-mb", str(options.cuda_plan_cache_mb),
        "--error-cache-mb", str(options.error_cache_mb),
        "--threads", str(worker_threads),
        "--error-output", str(output),
        "--error-shard-index", str(shard_index),
        "--error-shard-count", str(shard_count),
    ]
    if options.source_plugin:
        command.extend(["--source-plugin", options.source_plugin])
    if options.source_decoder:
        command.extend(["--source-decoder", options.source_decoder])
    command.extend([
        "--source-prefer-hw", str(options.source_prefer_hw),
        "--source-ff-loglevel", str(options.source_ff_loglevel),
        "--source-rap-verification", str(options.source_rap_verification),
    ])
    return command


def run_error_worker(options, case: str, output: Path) -> dict:
    candidate_count = len(recipe_candidates(case, options.profile))
    shard_count = min(options.error_processes, candidate_count)
    worker_threads = (options.error_threads if options.error_threads > 0
                      else max(1, options.threads // shard_count))
    jobs = []
    for shard_index in range(shard_count):
        shard_output = output if shard_count == 1 else output.with_name(
            f"{output.stem}.shard-{shard_index:02d}{output.suffix}")
        command = error_worker_command(
            options, case, shard_output, shard_index, shard_count,
            worker_threads)
        jobs.append((shard_index, shard_output, command, subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="replace",
            env={**os.environ, "DSMVC_CUDA_PLAN_CACHE_MB":
                 str(options.cuda_plan_cache_mb)})))

    payloads = []
    commands = []
    for shard_index, shard_output, command, process in jobs:
        stdout, stderr = process.communicate()
        commands.append(command_text(command))
        if process.returncode != 0:
            raise RuntimeError(
                f"error worker failed for {case} shard {shard_index}:\n"
                + stdout + "\n" + stderr)
        if not shard_output.is_file():
            raise RuntimeError(
                f"error worker did not write {shard_output}")
        payload = json.loads(shard_output.read_text(encoding="utf-8"))
        payload["worker_output_tail"] = (stdout + stderr)[-2000:]
        payloads.append(payload)

    if shard_count == 1:
        result = payloads[0]
    else:
        rows = [row for payload in payloads for row in payload["rows"]]
        rows.sort(key=lambda row: row["index"])
        expected_indices = list(range(candidate_count))
        actual_indices = [row["index"] for row in rows]
        if actual_indices != expected_indices:
            raise RuntimeError(
                f"error worker shards did not cover {case} exactly: "
                f"expected {candidate_count}, got {len(rows)}")
        result = {
            "schema_version": 1,
            "case": case,
            "profile": options.profile,
            "frame": RECIPE_FACTS[case]["frame"],
            "candidate_count": len(rows),
            "candidate_total_count": len(rows),
            "source": str(Path(options.source).expanduser().resolve()),
            "threads": worker_threads,
            "cuda_plan_cache_mb": options.cuda_plan_cache_mb,
            "error_cache_mb": options.error_cache_mb,
            "error_shard_count": shard_count,
            "rows": rows,
        }
    result["command"] = "\n".join(commands)
    result["commands"] = commands
    result["error_processes"] = shard_count
    result["error_worker_threads"] = worker_threads
    result["cuda_plan_cache_mb"] = options.cuda_plan_cache_mb
    result["error_cache_mb"] = options.error_cache_mb
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True),
                      encoding="utf-8")
    return result


def add_ranks(rows: list[dict]) -> None:
    old_order = sorted(
        range(len(rows)),
        key=lambda index: rows[index]["old_vs_source"]["thresholded_mae"])
    new_order = sorted(
        range(len(rows)),
        key=lambda index: rows[index]["new_vs_source"]["thresholded_mae"])
    old_rank = {index: rank for rank, index in enumerate(old_order)}
    new_rank = {index: rank for rank, index in enumerate(new_order)}
    for index, row in enumerate(rows):
        row["old_rank"] = old_rank[index]
        row["new_rank"] = new_rank[index]
        row["rank_change"] = abs(old_rank[index] - new_rank[index])


def error_summary(payload: dict) -> dict:
    rows = payload["rows"]
    add_ranks(rows)
    old_best = min(rows, key=lambda row: row["old_vs_source"]["thresholded_mae"])
    new_best = min(rows, key=lambda row: row["new_vs_source"]["thresholded_mae"])
    scalers = {}
    for scaler_name in sorted({row["scaler"] for row in rows}):
        grouped = [row for row in rows if row["scaler"] == scaler_name]
        old_item = min(
            grouped, key=lambda row: row["old_vs_source"]["thresholded_mae"])
        new_item = min(
            grouped, key=lambda row: row["new_vs_source"]["thresholded_mae"])
        scalers[scaler_name] = {
            "candidate_count": len(grouped),
            "old_best": {"id": old_item["id"],
                         "height": old_item["height"],
                         "mae": old_item["old_vs_source"]["thresholded_mae"]},
            "new_best": {"id": new_item["id"],
                         "height": new_item["height"],
                         "mae": new_item["new_vs_source"]["thresholded_mae"]},
            "best_height_changed": old_item["height"] != new_item["height"],
        }
    return {
        "candidate_count": len(rows),
        "best_old": {"id": old_best["id"],
                     "height": old_best["height"],
                     "mae": old_best["old_vs_source"]["thresholded_mae"]},
        "best_new": {"id": new_best["id"],
                     "height": new_best["height"],
                     "mae": new_best["new_vs_source"]["thresholded_mae"]},
        "best_candidate_changed": old_best["id"] != new_best["id"],
        "max_output_mae": max(
            row["output_old_new"]["mae"] for row in rows),
        "max_output_max_abs": max(
            row["output_old_new"]["max_abs"] for row in rows),
        "max_reconstruction_mae": max(
            row["reconstruction_old_new"]["mae"] for row in rows),
        "max_reconstruction_max_abs": max(
            row["reconstruction_old_new"]["max_abs"] for row in rows),
        "max_rank_change": max(row["rank_change"] for row in rows),
        "algorithm_summary": scalers,
    }


def write_error_csv(payloads: dict[str, dict], output: Path) -> None:
    fields = [
        "case", "profile", "frame", "index", "id", "scaler", "kernel",
        "height", "old_training_mae", "new_training_mae", "old_source_mae",
        "new_source_mae", "old_new_source_mae_delta", "output_old_new_mae",
        "output_old_new_max_abs", "reconstruction_old_new_mae",
        "reconstruction_old_new_max_abs", "old_rank", "new_rank",
        "rank_change",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case, payload in payloads.items():
            for row in payload["rows"]:
                writer.writerow({
                    "case": case,
                    "profile": payload["profile"],
                    "frame": payload["frame"],
                    "index": row["index"],
                    "id": row["id"],
                    "scaler": row["scaler"],
                    "kernel": row["kernel"],
                    "height": row["height"],
                    "old_training_mae": row["old_vs_source"]["thresholded_mae"],
                    "new_training_mae": row["new_vs_source"]["thresholded_mae"],
                    "old_source_mae": row["old_vs_source"]["mae"],
                    "new_source_mae": row["new_vs_source"]["mae"],
                    "old_new_source_mae_delta": (
                        row["new_vs_source"]["mae"]
                        - row["old_vs_source"]["mae"]),
                    "output_old_new_mae": row["output_old_new"]["mae"],
                    "output_old_new_max_abs": row["output_old_new"]["max_abs"],
                    "reconstruction_old_new_mae": (
                        row["reconstruction_old_new"]["mae"]),
                    "reconstruction_old_new_max_abs": (
                        row["reconstruction_old_new"]["max_abs"]),
                    "old_rank": row["old_rank"],
                    "new_rank": row["new_rank"],
                    "rank_change": row["rank_change"],
                })


def version(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, errors="replace",
            check=False)
    except OSError as error:
        return f"unavailable: {error}"
    return (completed.stdout + completed.stderr).strip()


def write_report(result: dict, output: Path) -> None:
    implementations = tuple(result["environment"].get(
        "implementations", IMPLEMENTATIONS))
    (output / "benchmark.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    with (output / "performance.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        fields = ["type", "case", "implementation", "run",
                  "candidate_count", "elapsed_seconds",
                  "candidates_per_second", "vspipe_seconds",
                  "vspipe_candidates_per_second",
                  "process_overhead_seconds", "batch_count", "batch_size",
                  "new_speedup"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in result["performance"]["raw_samples"]:
            writer.writerow({
                "type": "raw",
                "case": sample["case"],
                "implementation": sample["implementation"],
                "run": sample["run"],
                "candidate_count": sample["candidate_count"],
                "elapsed_seconds": sample["elapsed_ns"] / 1e9,
                "candidates_per_second": sample["candidates_per_second"],
                "vspipe_seconds": sample["vspipe_seconds"],
                "vspipe_candidates_per_second": sample[
                    "vspipe_candidates_per_second"],
                "process_overhead_seconds": sample[
                    "process_overhead_seconds"],
                "batch_count": sample["batch_count"],
                "batch_size": sample["batch_size"],
                "new_speedup": "",
            })
        for item in result["performance"]["summaries"]:
            for implementation in implementations:
                writer.writerow({
                    "type": "summary",
                    "case": item["case"],
                    "implementation": implementation,
                    "run": "",
                    "candidate_count": item["candidate_count"],
                    "elapsed_seconds": item[implementation][
                        "elapsed_seconds"]["median"],
                    "candidates_per_second": item[implementation][
                        "candidates_per_second"]["median"],
                    "vspipe_seconds": item[implementation][
                        "vspipe_seconds"]["median"],
                    "vspipe_candidates_per_second": item[implementation][
                        "vspipe_candidates_per_second"]["median"],
                    "process_overhead_seconds": item[implementation][
                        "process_overhead_seconds"]["median"],
                    "batch_count": "",
                    "batch_size": (
                        item["candidate_count"]
                        if result["environment"]["performance_batch_size"] <= 0
                        else min(
                            item["candidate_count"],
                            result["environment"]["performance_batch_size"])),
                    "new_speedup": item["new_speedup"],
                })

    batch_size = result["environment"]["performance_batch_size"]
    if batch_size <= 0:
        batch_description = "one bounded graph per complete candidate scan"
    else:
        batch_description = (
            f"isolated batches of at most {batch_size} candidates")
    lines = [
        "# Descale E2E benchmark",
        "",
        f"Generated: `{result['environment']['timestamp_utc']}`",
        "",
        "Performance samples are fresh VSPipe processes. Wall time includes",
        "source decode, graph construction, descale planning, reconstruction,",
        "PlaneStats, plugin loading, and process shutdown.",
        f"Each case/implementation has {result['environment']['warmup_runs']} "
        f"untimed warm-up run(s) of "
        f"{result['environment']['warmup_candidates']} candidates. Performance "
        f"uses {batch_description} and a "
        f"{result['environment']['performance_cache_mb']} MiB VapourSynth cache. "
        f"The CUDA packed-plan cache is capped at "
        f"{result['environment']['cuda_plan_cache_mb']} MiB. VSPipe's internal "
        "processing time is retained separately. Throwaway warm-up processes "
        "warm driver/module/page caches and GPU clocks, but each measured fresh "
        "process still creates its own CUDA context.",
        "",
        "## Performance",
        "",
    ]
    if set(implementations) == set(IMPLEMENTATIONS):
        lines.extend([
            "| Case | Candidates | Old median (s) | New median (s) | New speedup |",
            "|---|---:|---:|---:|---:|",
        ])
        for item in result["performance"]["summaries"]:
            lines.append(
                f"| `{item['case']}` | {item['candidate_count']} | "
                f"{item['old']['elapsed_seconds']['median']:.6f} | "
                f"{item['new']['elapsed_seconds']['median']:.6f} | "
                f"{item['new_speedup']:.3f}x |")
    else:
        implementation = implementations[0]
        label = "current dsmvc" if implementation == "new" else "old descale"
        lines.extend([
            f"| Case | Candidates | {label} median (s) | {label} candidates/s |",
            "|---|---:|---:|---:|",
        ])
        for item in result["performance"]["summaries"]:
            lines.append(
                f"| `{item['case']}` | {item['candidate_count']} | "
                f"{item[implementation]['elapsed_seconds']['median']:.6f} | "
                f"{item[implementation]['candidates_per_second']['median']:.3f} |")
    if result["errors"]["summaries"]:
        lines.extend([
            "",
            "## Error comparison",
            "",
            "Errors are float-domain metrics on the GRAYS source plane. Best",
            "candidates and ranks use the training script's thresholded error.",
            "Raw `old/new source MAE` is also recorded after the same scaler and a",
            "5-pixel border crop. `old/new output MAE` compares the descaled clips",
            "directly. The CSV contains every candidate and rank movement.",
            "",
            "| Case | Best old | Best new | Candidate changed | Max output abs | Max reconstruction abs |",
            "|---|---|---|---|---:|---:|",
        ])
        for item in result["errors"]["summaries"]:
            lines.append(
                f"| `{item['case']}` | {item['best_old']['id']} "
                f"({item['best_old']['mae']:.9g}) | "
                f"{item['best_new']['id']} ({item['best_new']['mae']:.9g}) | "
                f"{item['best_candidate_changed']} | "
                f"{item['max_output_max_abs']:.9g} | "
                f"{item['max_reconstruction_max_abs']:.9g} |")
        lines.extend([
            "",
            "### Per-algorithm minima",
            "",
            "| Case | Algorithm | Old best height / MAE | New best height / MAE | Height changed |",
            "|---|---|---:|---:|---|",
        ])
        for item in result["errors"]["summaries"]:
            for algorithm, values in item["algorithm_summary"].items():
                lines.append(
                    f"| `{item['case']}` | `{algorithm}` | "
                    f"{values['old_best']['height']:.1f} / "
                    f"{values['old_best']['mae']:.9g} | "
                    f"{values['new_best']['height']:.1f} / "
                    f"{values['new_best']['mae']:.9g} | "
                    f"{values['best_height_changed']} |")
    lines.extend([
        "",
        "## Provenance",
        "",
        "The HTML files are chat-export inputs used to identify the training",
        "media and scripts; they are not decoded as video. The benchmark source",
        "is the explicitly supplied MKV.",
        "",
        "```json",
        json.dumps(result["provenance"], indent=2, ensure_ascii=True),
        "```",
        "",
        "Commands: [`commands.txt`](commands.txt)",
        "",
    ])
    (output / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    commands = result["performance"]["commands"] + result["errors"]["commands"]
    (output / "commands.txt").write_text("\n".join(commands) + "\n",
                                          encoding="utf-8")


def resolve_executable(value: str, name: str) -> str:
    if value:
        return value
    found = shutil.which(name)
    if found:
        return found
    return name


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(
        description="Run old-descale versus dsmvc E2E tests on a real MKV.")
    result.add_argument("--source", required=True,
                        help="Source MKV passed to LSMASHSource.")
    result.add_argument("--old-plugin", default=os.environ.get(
        "DSMVC_OLD_PLUGIN", ""),
        help="Original descale plugin used by each fresh VSPipe process.")
    result.add_argument("--new-plugin", default=os.environ.get(
        "DSMVC_NEW_PLUGIN", ""),
        help="Current dsmvc plugin.")
    result.add_argument("--implementations", nargs="+", choices=IMPLEMENTATIONS,
                        default=list(IMPLEMENTATIONS),
                        help="Implementations to execute; old is not required for new-only runs.")
    result.add_argument("--source-plugin", default=os.environ.get(
        "DSMVC_SOURCE_PLUGIN", ""),
        help="Optional LSMASHSource plugin; auto-loaded when available.")
    result.add_argument("--source-filter", choices=("lsmas", "ffms2", "bestsource"),
                        default=os.environ.get("DSMVC_SOURCE_FILTER", "lsmas"),
                        help="Decoder namespace; default preserves the training scripts.")
    result.add_argument("--source-decoder", default=os.environ.get(
        "DSMVC_SOURCE_DECODER", ""),
        help="Preferred LSMASH/libavcodec decoder name(s).")
    result.add_argument("--source-prefer-hw", type=int, default=int(
        os.environ.get("DSMVC_SOURCE_PREFER_HW", "0")),
        help="LSMASH prefer_hw mode; 0 keeps software default.")
    result.add_argument("--source-ff-loglevel", type=int, default=int(
        os.environ.get("DSMVC_SOURCE_FF_LOGLEVEL", "0")),
        help="LSMASH FFmpeg log level, 0 is quiet.")
    result.add_argument("--source-rap-verification", type=int, default=int(
        os.environ.get("DSMVC_SOURCE_RAP_VERIFICATION", "-1")),
        help="LSMASH RAP verification; -1 keeps plugin default.")
    result.add_argument("--vspipe", default=os.environ.get(
        "DSMVC_VSPIPE", ""))
    result.add_argument("--python", default=os.environ.get(
        "DSMVC_VS_PYTHON", sys.executable),
        help="VapourSynth Python used by the error worker.")
    result.add_argument("--output", default=str(
        root / "benchmark-results" / "e2e-descale-mkv"))
    result.add_argument("--profile", choices=PROFILES,
                        default="full")
    result.add_argument("--cases", nargs="+", choices=CASES,
                        default=list(CASES))
    result.add_argument("--runs", type=int, default=3)
    result.add_argument("--warmup-runs", type=int, default=1,
                        help="Untimed throwaway VSPipe run(s) per case/implementation.")
    result.add_argument("--warmup-candidates", type=int, default=64,
                        help="Candidates in each warm-up run; 0 disables warm-up.")
    result.add_argument("--requests", type=int, default=32)
    result.add_argument("--threads", type=int, default=32)
    result.add_argument(
        "--backend", choices=("auto", "cpu", "metal", "cuda"),
        default="cpu")
    result.add_argument("--opt", choices=(0, 1, 2), type=int, default=0,
                        help="Optional dsmvc CPU path selector; 0 uses default.")
    result.add_argument("--performance-batch-size", type=int, default=0,
                        help="Maximum candidates per isolated performance VSPipe process; 0 uses one graph.")
    result.add_argument("--performance-cache-mb", type=int, default=int(
        os.environ.get("DSMVC_PERFORMANCE_CACHE_MB", "512")),
        help="VapourSynth cache budget for performance processes in MiB.")
    result.add_argument("--cuda-plan-cache-mb", type=int, default=int(
        os.environ.get("DSMVC_CUDA_PLAN_CACHE_MB", "16")),
        help="CUDA packed-plan cache for one-shot candidate scans in MiB.")
    result.add_argument("--error-processes", type=int, default=int(
        os.environ.get("DSMVC_ERROR_PROCESSES", "1")),
        help="Parallel processes used for the per-candidate error sweep.")
    result.add_argument("--error-threads", type=int, default=int(
        os.environ.get("DSMVC_ERROR_THREADS", "0")),
        help="Threads per error worker; 0 divides --threads across workers.")
    result.add_argument("--error-cache-mb", type=int, default=int(
        os.environ.get("DSMVC_ERROR_CACHE_MB", "512")),
        help="VapourSynth cache budget per error worker in MiB.")
    result.add_argument("--skip-performance", action="store_true")
    result.add_argument("--skip-errors", action="store_true")
    result.add_argument("--html", action="append", default=[],
                        help="HTML provenance input; may be repeated.")
    result.add_argument("--script", action="append", default=[],
                        metavar="CASE=PATH",
                        help="Training script provenance input; may be repeated.")
    result.add_argument("--strict-provenance", action="store_true")
    result.add_argument("--worker-errors", action="store_true",
                        help=argparse.SUPPRESS)
    result.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    result.add_argument("--error-output", help=argparse.SUPPRESS)
    result.add_argument("--error-shard-index", type=int, default=0,
                        help=argparse.SUPPRESS)
    result.add_argument("--error-shard-count", type=int, default=1,
                        help=argparse.SUPPRESS)
    return result


def main(options) -> int:
    if options.worker_errors:
        if not options.case or not options.error_output:
            raise ValueError("error worker requires --case and --error-output")
        return worker_errors(options)

    source = Path(options.source).expanduser().resolve()
    old_plugin = (Path(options.old_plugin).expanduser().resolve()
                  if options.old_plugin else None)
    new_plugin = Path(options.new_plugin).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source MKV does not exist: {source}")
    if not options.implementations:
        raise ValueError("at least one implementation is required")
    if len(set(options.implementations)) != len(options.implementations):
        raise ValueError("implementation names must be unique")
    if "old" in options.implementations and (
            old_plugin is None or not old_plugin.is_file()):
        raise FileNotFoundError(
            f"original descale plugin does not exist: {old_plugin}; "
            "pass --old-plugin")
    if not new_plugin.is_file():
        raise FileNotFoundError(f"current dsmvc plugin does not exist: {new_plugin}")
    if options.source_plugin and not Path(options.source_plugin).is_file():
        raise FileNotFoundError(
            f"source plugin does not exist: {options.source_plugin}")
    if options.source_prefer_hw < 0 or options.source_prefer_hw > 7:
        raise ValueError("--source-prefer-hw must be between 0 and 7")
    if options.source_ff_loglevel < 0 or options.source_ff_loglevel > 8:
        raise ValueError("--source-ff-loglevel must be between 0 and 8")
    if options.source_rap_verification not in (-1, 0, 1):
        raise ValueError("--source-rap-verification must be -1, 0, or 1")
    if (options.runs < 1 or options.requests < 1 or options.threads < 1
            or options.warmup_runs < 0 or options.warmup_candidates < 0
            or options.performance_batch_size < 0
            or options.cuda_plan_cache_mb < 16
            or options.cuda_plan_cache_mb > 4096
            or options.error_processes < 1 or options.error_threads < 0
            or options.error_cache_mb < 16
            or options.performance_cache_mb < 16):
        raise ValueError(
            "--runs, --requests, --threads and --error-processes must be "
            "positive; warm-up and batch values cannot be negative; "
            "VapourSynth cache budgets must be at least 16 MiB; the CUDA "
            "plan cache must be between 16 and 4096 MiB")
    if (not options.skip_errors
            and set(options.implementations) != set(IMPLEMENTATIONS)):
        raise ValueError("error comparison requires both implementations; use --skip-errors")

    options.source = str(source)
    options.old_plugin = str(old_plugin) if old_plugin else ""
    options.new_plugin = str(new_plugin)
    options.vspipe = resolve_executable(options.vspipe, "vspipe")
    options.python = str(Path(options.python).expanduser())
    output = Path(options.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    script_paths = parse_case_scripts(options.script)
    script_metadata = {
        case: script_info(case, script_paths[case])
        for case in script_paths
    }
    if options.strict_provenance:
        missing = [case for case in CASES if case not in script_metadata]
        if missing:
            raise ValueError("missing --script provenance for: "
                             + ", ".join(missing))
        mismatched = [case for case, info in script_metadata.items()
                      if not info.get("matches_recipe_facts", False)]
        if mismatched:
            raise ValueError("training script facts do not match: "
                             + ", ".join(mismatched))

    provenance = {
        "source": file_info(source),
        "html": [html_info(Path(item)) for item in options.html],
        "training_scripts": script_metadata,
        "training_script_contract": RECIPE_FACTS,
    }
    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "vspipe": version([options.vspipe, "--version"]),
        "python": options.python,
        "profile": options.profile,
        "runs": options.runs,
        "warmup_runs": options.warmup_runs,
        "warmup_candidates": options.warmup_candidates,
        "performance_batch_size": options.performance_batch_size,
        "performance_cache_mb": options.performance_cache_mb,
        "cuda_plan_cache_mb": options.cuda_plan_cache_mb,
        "requests": options.requests,
        "threads": options.threads,
        "backend": options.backend,
        "opt": options.opt,
        "implementations": list(options.implementations),
        "error_processes": options.error_processes,
        "error_worker_threads": (
            options.error_threads if options.error_threads > 0 else max(
                1, options.threads // options.error_processes)),
        "error_cache_mb": options.error_cache_mb,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "old_plugin": str(old_plugin) if old_plugin else None,
        "old_plugin_sha256": sha256_file(old_plugin) if old_plugin else None,
        "new_plugin": str(new_plugin),
        "new_plugin_sha256": sha256_file(new_plugin),
        "source_plugin": (str(Path(options.source_plugin).resolve())
                          if options.source_plugin else None),
        "source_filter": options.source_filter,
        "source_decoder": options.source_decoder,
        "source_prefer_hw": options.source_prefer_hw,
        "source_ff_loglevel": options.source_ff_loglevel,
        "source_rap_verification": options.source_rap_verification,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "vpy_sha256": sha256_file(
            Path(__file__).with_name("vspipe_e2e.vpy")),
    }

    performance_samples = []
    performance_warmups = []
    performance_summaries = []
    performance_commands = []
    if not options.skip_performance:
        for case in options.cases:
            total_candidates = len(recipe_candidates(case, options.profile))
            ranges = candidate_ranges(
                total_candidates, options.performance_batch_size)
            case_samples = []
            if options.warmup_candidates > 0 and options.warmup_runs > 0:
                warmup_end = min(total_candidates, options.warmup_candidates)
                for implementation in options.implementations:
                    for warmup_run in range(1, options.warmup_runs + 1):
                        warmup = run_vspipe(
                            options, case, implementation, warmup_run,
                            0, warmup_end, warmup=True)
                        performance_warmups.append(warmup)
                        performance_commands.append(warmup["command"])
                        print(
                            f"{case} {implementation} warmup "
                            f"{warmup_run}/{options.warmup_runs}: "
                            f"{warmup['vspipe_candidates_per_second']:.3f} "
                            "VSPipe candidates/s",
                            flush=True)
            for run in range(1, options.runs + 1):
                for implementation in options.implementations:
                    batches = [
                        run_vspipe(
                            options, case, implementation, run,
                            start, end)
                        for start, end in ranges
                    ]
                    sample = combine_vspipe_batches(
                        case, implementation, run, batches,
                        options.performance_batch_size)
                    case_samples.append(sample)
                    performance_samples.append(sample)
                    performance_commands.extend(sample["batch_commands"])
                    print(
                        f"{case} {implementation} run {run}: "
                        f"{sample['candidates_per_second']:.3f} candidates/s",
                        flush=True)
            performance_summaries.append(performance_summary(
                case_samples, case, len(recipe_candidates(case, options.profile)),
                tuple(options.implementations)))

    error_payloads = {}
    error_summaries = []
    error_commands = []
    if not options.skip_errors:
        for case in options.cases:
            error_file = output / f"errors-{case}.json"
            payload = run_error_worker(options, case, error_file)
            error_payloads[case] = payload
            error_summaries.append({"case": case, **error_summary(payload)})
            error_commands.append(payload["command"])
            print(f"{case} error comparison: {payload['candidate_count']} candidates",
                  flush=True)
        write_error_csv(error_payloads, output / "errors.csv")

    result = {
        "schema_version": 1,
        "environment": environment,
        "provenance": provenance,
        "performance": {
            "summaries": performance_summaries,
            "raw_samples": performance_samples,
            "warmup_samples": performance_warmups,
            "commands": performance_commands,
        },
        "errors": {
            "summaries": error_summaries,
            "commands": error_commands,
            "files": {case: f"errors-{case}.json" for case in error_payloads},
        },
    }
    write_report(result, output)
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
