#!/usr/bin/env python3
"""Reproducible old-descale versus dsmvc benchmark for VapourSynth R57."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_IMAGE_SHA256 = "61f9ee1ac858bbadd6a959ba35f5eceb077b8452b91e97a5ce3d39ebc69e20c6"
EXPECTED_OLD_SHA256 = "b02e4a2fbaaf6ba3f7e3cf2ad8a08d8eefab9e5d634e1d829764671d49933000"
RESULT_PREFIX = "DSMVC_BENCH_RESULT="

DIRECT_CASES = {
    "bicubic_0_1": {"function": "Debicubic", "kernel": "bicubic", "b": 0.0, "c": 1.0},
    "bicubic_0_7_0_6": {"function": "Debicubic", "kernel": "bicubic", "b": 0.7, "c": 0.6},
    "bilinear": {"function": "Debilinear", "kernel": "bilinear"},
    "bicubic_default": {"function": "Debicubic", "kernel": "bicubic", "b": 0.0, "c": 0.5},
    "lanczos3": {"function": "Delanczos", "kernel": "lanczos", "taps": 3},
    "spline16": {"function": "Despline16", "kernel": "spline16"},
    "spline36": {"function": "Despline36", "kernel": "spline36"},
}


def scaler(kernel: str, *, b: float = 0.0, c: float = 0.5,
           taps: int = 3) -> dict:
    if kernel == "bicubic":
        name = f"bicubic_b{b:.1f}_c{c:.1f}"
    elif kernel == "lanczos":
        name = f"lanczos{taps}"
    else:
        name = kernel
    return {"name": name, "kernel": kernel, "b": b, "c": c, "taps": taps}


def repeated_arange(start: float, stop: float, step: float) -> list[float]:
    """Match the repeated-addition semantics of muvsfunc.arange."""
    values = []
    current = start
    while current < stop:
        values.append(current)
        current += step
    return values


GETFNATIVE_SCALERS = [
    scaler("bilinear"),
    scaler("bicubic", b=1 / 3, c=1 / 3),
    scaler("bicubic", b=0.0, c=0.5),
    scaler("bicubic", b=0.0, c=1.0),
    scaler("bicubic", b=1.0, c=0.0),
    scaler("bicubic", b=0.0, c=0.75),
    scaler("lanczos", taps=2),
    scaler("lanczos", taps=3),
    scaler("lanczos", taps=4),
    scaler("spline16"),
    scaler("spline36"),
]
GETFNATIVE_V2_SCALERS = GETFNATIVE_SCALERS[:6] + [
    scaler("spline16"), scaler("spline36")]
SELECTKERNEL_PARAMETERS = repeated_arange(0, 1, 0.1)[:10]
SELECTKERNEL_SCALERS = [scaler("bilinear")] + [
    scaler("bicubic", b=b, c=c)
    for b in SELECTKERNEL_PARAMETERS
    for c in SELECTKERNEL_PARAMETERS
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_descale_args(width: int, height: int, src_width: float,
                     src_height: float, base_height: int) -> dict[str, float | int]:
    try:
        from muvsfunc import rescale
        return rescale._get_descale_args(
            width, height, src_width, src_height, base_height)
    except (ImportError, AttributeError):
        base_width = round(width / height * base_height)
        output_width = base_width - 2 * int((base_width - src_width) / 2)
        output_height = base_height - 2 * int((base_height - src_height) / 2)
        return {
            "width": output_width,
            "height": output_height,
            "src_left": (output_width - src_width) / 2,
            "src_top": (output_height - src_height) / 2,
            "src_width": src_width,
            "src_height": src_height,
        }


def get_descale_args_vertical(width: int, src_height: float,
                              base_height: int) -> dict[str, float | int]:
    output_height = base_height - 2 * int((base_height - src_height) / 2)
    return {
        "width": width,
        "height": output_height,
        "src_left": 0.0,
        "src_top": (output_height - src_height) / 2,
        "src_width": float(width),
        "src_height": src_height,
    }


def geometry() -> dict[str, float | int]:
    return get_descale_args(1920, 1080, 951.5 * 16 / 9, 951.5, 1000)


def configure_core(core, threads: int) -> None:
    if threads > 0:
        core.num_threads = threads


def load_new_plugin(core, path: Path) -> int:
    if hasattr(core, "dsmvc"):
        return 0
    start = time.perf_counter_ns()
    core.std.LoadPlugin(path=str(path))
    return time.perf_counter_ns() - start


def ensure_old_plugin(core, path: Path) -> tuple[int, str]:
    if hasattr(core, "descale"):
        return 0, "autoloaded"
    start = time.perf_counter_ns()
    core.std.LoadPlugin(path=str(path))
    return time.perf_counter_ns() - start, "explicit"


def read_sources(core, image: Path, frame_count: int):
    import vapoursynth as vs

    rgb = core.imwri.Read(str(image))
    gray = rgb.resize.Point(format=vs.GRAYS, matrix_s="709")
    rgb = core.std.Loop(rgb, times=frame_count)
    gray = core.std.Loop(gray, times=frame_count)
    return rgb, gray


def public_case_arguments(case: dict) -> dict:
    return {key: value for key, value in case.items()
            if key not in ("function", "kernel")}


def build_direct(core, implementation: str, case: dict, source, args: dict):
    namespace = core.dsmvc if implementation == "new" else core.descale
    function = getattr(namespace, case["function"])
    kwargs = public_case_arguments(case)
    if implementation == "new":
        kwargs["backend"] = "cpu"
    return function(source, **args, **kwargs)


def build_wrapper(core, implementation: str, case: dict, source, args: dict,
                  repo_root: Path, vs_root: Path):
    if implementation == "new":
        sys.path.insert(0, str(repo_root / "python"))
        import dsmvc as wrapper
    else:
        sys.path.insert(0, str(vs_root / "VapourSynthScripts"))
        import descale as wrapper
    function = getattr(wrapper, case["function"])
    kwargs = public_case_arguments(case)
    if implementation == "new":
        kwargs["backend"] = "cpu"
    return function(source, args["width"], args["height"], **kwargs)


def resize_with_scaler(core, clip, scaler_spec: dict, width: int, height: int,
                       args: dict):
    resize = getattr(core.resize, scaler_spec["kernel"].capitalize())
    kwargs = {
        "src_left": args["src_left"],
        "src_top": args["src_top"],
        "src_width": args["src_width"],
        "src_height": args["src_height"],
    }
    if scaler_spec["kernel"] == "bicubic":
        kwargs["filter_param_a"] = scaler_spec["b"]
        kwargs["filter_param_b"] = scaler_spec["c"]
    elif scaler_spec["kernel"] == "lanczos":
        kwargs["filter_param_a"] = scaler_spec["taps"]
    return resize(clip, width, height, **kwargs)


def case_scaler(case: dict) -> dict:
    return scaler(case["kernel"], b=case.get("b", 0.0),
                  c=case.get("c", 0.5), taps=case.get("taps", 3))


def worker_direct(options) -> int:
    core_start = time.perf_counter_ns()
    import vapoursynth as vs
    core = vs.core
    core_initialization_ns = time.perf_counter_ns() - core_start
    configure_core(core, options.threads)
    if options.implementation == "new":
        plugin_load_ns = load_new_plugin(core, Path(options.new_plugin))
        plugin_load_mode = "explicit"
    else:
        plugin_load_ns, plugin_load_mode = ensure_old_plugin(
            core, Path(options.old_plugin))

    case = DIRECT_CASES[options.case]
    frame_count = options.warm_frames + 1
    source_start = time.perf_counter_ns()
    rgb, gray = read_sources(core, Path(options.image), frame_count)
    source = rgb if options.path_kind == "wrapper" else gray
    source.get_frame(0)
    source_decode_ns = time.perf_counter_ns() - source_start
    args = geometry()

    build_start = time.perf_counter_ns()
    if options.path_kind == "wrapper":
        output = build_wrapper(
            core, options.implementation, case, source, args,
            Path(options.repo_root), Path(options.vs_root))
    else:
        output = build_direct(core, options.implementation, case, source, args)
    graph_build_ns = time.perf_counter_ns() - build_start

    cold_start = time.perf_counter_ns()
    output.get_frame(0)
    cold_frame_ns = time.perf_counter_ns() - cold_start
    warm_ns = []
    for frame_number in range(1, frame_count):
        start = time.perf_counter_ns()
        output.get_frame(frame_number)
        warm_ns.append(time.perf_counter_ns() - start)

    result = {
        "implementation": options.implementation,
        "case": options.case,
        "path": options.path_kind,
        "core_initialization_ns": core_initialization_ns,
        "plugin_load_ns": plugin_load_ns,
        "plugin_load_mode": plugin_load_mode,
        "source_decode_ns": source_decode_ns,
        "graph_build_ns": graph_build_ns,
        "cold_frame_ns": cold_frame_ns,
        "first_use_ns": graph_build_ns + cold_frame_ns,
        "warm_frame_ns": warm_ns,
        "threads": core.num_threads,
        "output": {
            "width": output.width,
            "height": output.height,
            "format": output.format.name,
        },
    }
    print(RESULT_PREFIX + json.dumps(result, separators=(",", ":")))
    return 0


def active_frame_hash(frame, clip) -> str:
    digest = hashlib.sha256()
    for plane_index in range(clip.format.num_planes):
        digest.update(memoryview(frame[plane_index]).tobytes(order="C"))
    return digest.hexdigest()


def normalized_plane(core, clip, plane_index: int):
    import vapoursynth as vs

    plane = core.std.ShufflePlanes(clip, plane_index, vs.GRAY)
    if plane.format.id != vs.GRAYS:
        plane = plane.resize.Point(format=vs.GRAYS)
    return plane


def plane_metrics(core, lhs, rhs, plane_index: int) -> dict:
    import vapoursynth as vs

    lhs_plane = normalized_plane(core, lhs, plane_index)
    rhs_plane = normalized_plane(core, rhs, plane_index)
    absolute = core.std.Expr([lhs_plane, rhs_plane], ["x y - abs"])
    squared = core.std.Expr([absolute], ["x dup *"])
    is_float = lhs.format.sample_type == vs.FLOAT
    units = 1.0 if is_float else float((1 << lhs.format.bits_per_sample) - 1)
    native_threshold = 2.0e-5 if is_float else 1.0
    normalized_threshold = native_threshold / units
    threshold = core.std.Expr(
        [absolute], [f"x {normalized_threshold:.17g} > 1 0 ?"])
    absolute_stats = absolute.std.PlaneStats().get_frame(0).props
    squared_stats = squared.std.PlaneStats().get_frame(0).props
    threshold_stats = threshold.std.PlaneStats().get_frame(0).props
    normalized_mse = float(squared_stats.PlaneStatsAverage)
    mse = normalized_mse * units * units
    peak = units
    return {
        "plane": plane_index,
        "units": "float" if is_float else "lsb",
        "max_abs": float(absolute_stats.PlaneStatsMax) * units,
        "mae": float(absolute_stats.PlaneStatsAverage) * units,
        "mse": mse,
        "psnr": "inf" if mse == 0.0 else (
            20.0 * math.log10(peak) - 10.0 * math.log10(mse)),
        "pixels_over_threshold": round(
            float(threshold_stats.PlaneStatsAverage)
            * lhs_plane.width * lhs_plane.height),
        "fraction_over_threshold": float(threshold_stats.PlaneStatsAverage),
    }


def metrics_within_tolerance(clip, metrics: list[dict]) -> bool:
    import vapoursynth as vs

    if clip.format.sample_type == vs.FLOAT:
        return all(item["max_abs"] <= 2.0e-5 and item["mae"] <= 1.0e-6
                   for item in metrics)
    return all(item["max_abs"] <= 1.000001 for item in metrics)


def describe_clip(clip) -> dict:
    import vapoursynth as vs

    color_family = {
        vs.GRAY: "GRAY",
        vs.RGB: "RGB",
        vs.YUV: "YUV",
    }.get(clip.format.color_family, str(clip.format.color_family))
    sample_type = "float" if clip.format.sample_type == vs.FLOAT else "integer"
    return {
        "width": clip.width,
        "height": clip.height,
        "format": clip.format.name,
        "format_id": clip.format.id,
        "color_family": color_family,
        "sample_type": sample_type,
        "bits_per_sample": clip.format.bits_per_sample,
        "planes": clip.format.num_planes,
        "subsampling_w": clip.format.subsampling_w,
        "subsampling_h": clip.format.subsampling_h,
    }


def write_png(core, clip, path: Path) -> None:
    import vapoursynth as vs

    path.parent.mkdir(parents=True, exist_ok=True)
    if clip.format.color_family == vs.RGB:
        render = clip.resize.Point(format=vs.RGB24)
    elif clip.format.color_family == vs.GRAY:
        render = clip.resize.Point(format=vs.GRAY8)
    else:
        render = clip.resize.Point(format=vs.RGB24, matrix_in_s="709")
    pattern = path.with_name(f"{path.stem}-%06d{path.suffix}")
    rendered_path = path.with_name(f"{path.stem}-000000{path.suffix}")
    core.imwri.Write(render, "PNG", str(pattern), overwrite=True).get_frame(0)
    os.replace(rendered_path, path)


def write_difference_png(core, lhs, rhs, path: Path) -> None:
    lhs_plane = normalized_plane(core, lhs, 0)
    rhs_plane = normalized_plane(core, rhs, 0)
    absolute = core.std.Expr([lhs_plane, rhs_plane], ["x y - abs"])
    maximum = float(absolute.std.PlaneStats().get_frame(0).props.PlaneStatsMax)
    scale = 1.0 / maximum if maximum > 0.0 else 1.0
    visual = core.std.Expr([absolute], [f"x {scale:.17g} *"])
    write_png(core, visual, path)


def compare_case(options, case_name: str, path_kind: str,
                 artifact_dir: Path) -> dict:
    import vapoursynth as vs

    core = vs.core
    configure_core(core, options.threads)
    load_new_plugin(core, Path(options.new_plugin))
    ensure_old_plugin(core, Path(options.old_plugin))
    rgb, gray = read_sources(core, Path(options.image), 1)
    source = rgb if path_kind == "wrapper" else gray
    case = DIRECT_CASES[case_name]
    args = geometry()
    if path_kind == "wrapper":
        old = build_wrapper(core, "old", case, source, args,
                            Path(options.repo_root), Path(options.vs_root))
        new = build_wrapper(core, "new", case, source, args,
                            Path(options.repo_root), Path(options.vs_root))
    else:
        old = build_direct(core, "old", case, source, args)
        new = build_direct(core, "new", case, source, args)
    old_frame = old.get_frame(0)
    new_frame = new.get_frame(0)
    metrics = [plane_metrics(core, old, new, plane)
               for plane in range(old.format.num_planes)]
    prefix = f"{case_name}-{path_kind}"
    write_png(core, old, artifact_dir / f"{prefix}-old.png")
    write_png(core, new, artifact_dir / f"{prefix}-new.png")
    write_difference_png(core, old, new, artifact_dir / f"{prefix}-difference.png")

    reconstruction = None
    if path_kind == "direct":
        scaler_spec = case_scaler(case)
        old_reconstructed = resize_with_scaler(
            core, old, scaler_spec, gray.width, gray.height, args)
        new_reconstructed = resize_with_scaler(
            core, new, scaler_spec, gray.width, gray.height, args)
        old_source_metrics = [plane_metrics(core, gray, old_reconstructed, 0)]
        new_source_metrics = [plane_metrics(core, gray, new_reconstructed, 0)]
        old_new_metrics = [plane_metrics(
            core, old_reconstructed, new_reconstructed, 0)]
        write_png(core, old_reconstructed,
                  artifact_dir / f"{prefix}-reconstructed-old.png")
        write_png(core, new_reconstructed,
                  artifact_dir / f"{prefix}-reconstructed-new.png")
        write_difference_png(
            core, old_reconstructed, new_reconstructed,
            artifact_dir / f"{prefix}-reconstructed-difference.png")
        reconstruction = {
            "old_vs_source": old_source_metrics,
            "new_vs_source": new_source_metrics,
            "old_vs_new": old_new_metrics,
        }

    return {
        "case": case_name,
        "path": path_kind,
        "old_output": describe_clip(old),
        "new_output": describe_clip(new),
        "old_sha256": active_frame_hash(old_frame, old),
        "new_sha256": active_frame_hash(new_frame, new),
        "shape_equal": (old.width, old.height, old.format.id)
            == (new.width, new.height, new.format.id),
        "planes": metrics,
        "within_tolerance": metrics_within_tolerance(old, metrics),
        "reconstruction": reconstruction,
    }


def summarize(values: list[int | float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {key: 0.0 for key in
                ("median_ns", "mad_ns", "p95_ns", "min_ns", "max_ns")}
    median = statistics.median(ordered)
    deviations = [abs(value - median) for value in ordered]
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "median_ns": float(median),
        "mad_ns": float(statistics.median(deviations)),
        "p95_ns": float(ordered[p95_index]),
        "min_ns": float(ordered[0]),
        "max_ns": float(ordered[-1]),
    }


def extract_result(completed: subprocess.CompletedProcess) -> dict:
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            result = json.loads(line[len(RESULT_PREFIX):])
            result["stderr"] = completed.stderr.strip()
            return result
    raise RuntimeError(
        f"benchmark worker failed with exit code {completed.returncode}:\n"
        + completed.stdout + "\n" + completed.stderr)


def run_worker(options, implementation: str, case_name: str,
               path_kind: str) -> dict:
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker-direct",
        "--implementation", implementation,
        "--case", case_name,
        "--path-kind", path_kind,
        "--image", options.image,
        "--new-plugin", options.new_plugin,
        "--old-plugin", options.old_plugin,
        "--vs-root", options.vs_root,
        "--repo-root", options.repo_root,
        "--warm-frames", str(options.warm_frames),
        "--threads", str(options.threads),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    result = extract_result(completed)
    result["command"] = subprocess.list2cmdline(command)
    return result


def aggregate_runs(samples: list[dict], implementation: str) -> dict:
    selected = [sample for sample in samples
                if sample["implementation"] == implementation]
    warm = [value for sample in selected for value in sample["warm_frame_ns"]]
    return {
        "implementation": implementation,
        "runs": len(selected),
        "plugin_load_modes": sorted({sample["plugin_load_mode"]
                                      for sample in selected}),
        "core_initialization": summarize(
            [sample["core_initialization_ns"] for sample in selected]),
        "plugin_load": summarize(
            [sample["plugin_load_ns"] for sample in selected]),
        "source_decode": summarize(
            [sample["source_decode_ns"] for sample in selected]),
        "graph_build": summarize(
            [sample["graph_build_ns"] for sample in selected]),
        "cold_frame": summarize(
            [sample["cold_frame_ns"] for sample in selected]),
        "first_use": summarize(
            [sample["first_use_ns"] for sample in selected]),
        "warm_frame": summarize(warm),
    }


def sweep_spec(name: str, profile: str) -> dict:
    if name == "getfnative":
        heights = (repeated_arange(700, 980, 0.1)
                   if profile == "full"
                   else [700.0, 719.8, 840.0, 900.0, 951.4, 951.5, 951.6, 979.9])
        scalers = GETFNATIVE_SCALERS
        vertical_only = False
        ex_thr = 0.015
        source_file = "test_getfnative.vpy"
    elif name == "getfnative_v2":
        heights = (repeated_arange(840, 880, 0.1)
                   if profile == "full"
                   else [840.0, 859.9, 860.0, 860.1, 879.9])
        scalers = GETFNATIVE_V2_SCALERS
        vertical_only = True
        ex_thr = 0.015
        source_file = "test_getfnative_v2.vpy"
    elif name == "selectkernel":
        heights = [719.8]
        scalers = SELECTKERNEL_SCALERS
        vertical_only = False
        ex_thr = 0.012
        source_file = "test_selectkernel.vpy"
    else:
        raise ValueError(f"unknown sweep case: {name}")
    candidates = []
    for scaler_spec in scalers:
        for height in heights:
            candidates.append({
                "index": len(candidates),
                "id": f"{scaler_spec['name']}@{height:.1f}",
                "scaler": scaler_spec,
                "height": height,
            })
    return {
        "name": name,
        "profile": profile,
        "source_file": source_file,
        "base_height": 1000,
        "crop_size": 5,
        "ex_thr": ex_thr,
        "vertical_only": vertical_only,
        "scaler_count": len(scalers),
        "height_count": len(heights),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def build_sweep_graph(core, implementation: str, source, candidate: dict,
                      spec: dict):
    namespace = core.dsmvc if implementation == "new" else core.descale
    scaler_spec = candidate["scaler"]
    if spec["vertical_only"]:
        args = get_descale_args_vertical(
            source.width, candidate["height"], spec["base_height"])
    else:
        src_width = source.width / source.height * candidate["height"]
        args = get_descale_args(
            source.width, source.height, src_width, candidate["height"],
            spec["base_height"])
    kwargs = {"kernel": scaler_spec["kernel"]}
    if scaler_spec["kernel"] == "bicubic":
        kwargs.update(b=scaler_spec["b"], c=scaler_spec["c"])
    elif scaler_spec["kernel"] == "lanczos":
        kwargs["taps"] = scaler_spec["taps"]
    if implementation == "new":
        kwargs["backend"] = "cpu"
    descaled = namespace.Descale(source, **args, **kwargs)
    reconstructed = resize_with_scaler(
        core, descaled, scaler_spec, source.width, source.height, args)
    difference = core.std.Expr(
        [source, reconstructed],
        [f"x y - abs dup {spec['ex_thr']:.17g} > swap 0 ?"])
    if spec["crop_size"] > 0:
        difference = core.std.CropRel(
            difference, *([spec["crop_size"]] * 4))
    return difference.std.PlaneStats()


def evaluate_sweep_graph(graph) -> tuple[float, int]:
    start = time.perf_counter_ns()
    frame = graph.get_frame(0)
    elapsed = time.perf_counter_ns() - start
    return float(frame.props.PlaneStatsAverage), elapsed


def worker_sweep(options) -> int:
    core_start = time.perf_counter_ns()
    import vapoursynth as vs
    core = vs.core
    core_initialization_ns = time.perf_counter_ns() - core_start
    configure_core(core, options.threads)
    if options.implementation == "new":
        plugin_load_ns = load_new_plugin(core, Path(options.new_plugin))
        plugin_load_mode = "explicit"
    else:
        plugin_load_ns, plugin_load_mode = ensure_old_plugin(
            core, Path(options.old_plugin))

    source_start = time.perf_counter_ns()
    _, source = read_sources(core, Path(options.image), 1)
    source_frame = source.get_frame(0)
    del source_frame
    source_decode_ns = time.perf_counter_ns() - source_start
    spec = sweep_spec(options.sweep_case, options.sweep_profile)
    concurrency = options.sweep_concurrency
    if concurrency <= 0:
        concurrency = options.threads if options.threads > 0 else core.num_threads
    concurrency = max(1, min(concurrency, spec["candidate_count"]))

    graph_build_ns = 0
    evaluation_wall_ns = 0
    individual_frame_ns = []
    curve = []
    sweep_start = time.perf_counter_ns()
    executor = ThreadPoolExecutor(max_workers=concurrency) if concurrency > 1 else None
    try:
        candidates = spec["candidates"]
        for batch_start in range(0, len(candidates), concurrency):
            batch = candidates[batch_start:batch_start + concurrency]
            graphs = []
            for candidate in batch:
                start = time.perf_counter_ns()
                graphs.append(build_sweep_graph(
                    core, options.implementation, source, candidate, spec))
                graph_build_ns += time.perf_counter_ns() - start
            evaluation_start = time.perf_counter_ns()
            if executor is None:
                evaluated = [evaluate_sweep_graph(graph) for graph in graphs]
            else:
                evaluated = list(executor.map(evaluate_sweep_graph, graphs))
            evaluation_wall_ns += time.perf_counter_ns() - evaluation_start
            for candidate, (error, elapsed) in zip(batch, evaluated):
                curve.append({
                    "index": candidate["index"],
                    "id": candidate["id"],
                    "scaler": candidate["scaler"]["name"],
                    "height": candidate["height"],
                    "error": error,
                })
                individual_frame_ns.append(elapsed)
            del graphs
            del evaluated
            if batch_start and batch_start % (concurrency * 64) == 0:
                gc.collect()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    sweep_total_ns = time.perf_counter_ns() - sweep_start

    result = {
        "implementation": options.implementation,
        "case": options.sweep_case,
        "profile": options.sweep_profile,
        "core_initialization_ns": core_initialization_ns,
        "plugin_load_ns": plugin_load_ns,
        "plugin_load_mode": plugin_load_mode,
        "source_decode_ns": source_decode_ns,
        "graph_build_ns": graph_build_ns,
        "evaluation_wall_ns": evaluation_wall_ns,
        "individual_frame_ns": individual_frame_ns,
        "sweep_total_ns": sweep_total_ns,
        "threads": core.num_threads,
        "concurrency": concurrency,
        "definition": {key: value for key, value in spec.items()
                       if key != "candidates"},
        "curve": curve,
    }
    print(RESULT_PREFIX + json.dumps(result, separators=(",", ":")))
    return 0


def sweep_checkpoint(options, implementation: str, case_name: str,
                     run_index: int) -> Path:
    identity = {
        "runner": sha256_file(Path(__file__).resolve()),
        "input": sha256_file(Path(options.image)),
        "plugin": sha256_file(Path(
            options.new_plugin if implementation == "new" else options.old_plugin)),
        "threads": options.threads,
        "concurrency": options.sweep_concurrency,
        "profile": options.sweep_profile,
    }
    fingerprint = hashlib.sha256(json.dumps(
        identity, sort_keys=True).encode("ascii")).hexdigest()[:16]
    return (Path(options.output) / ".checkpoints"
            / f"{case_name}-{options.sweep_profile}-{implementation}-"
              f"{run_index}-{fingerprint}.json")


def run_sweep_worker(options, implementation: str, case_name: str,
                     run_index: int) -> dict:
    checkpoint = sweep_checkpoint(options, implementation, case_name, run_index)
    if options.resume and checkpoint.exists():
        return json.loads(checkpoint.read_text(encoding="utf-8"))
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker-sweep",
        "--implementation", implementation,
        "--sweep-case", case_name,
        "--sweep-profile", options.sweep_profile,
        "--sweep-concurrency", str(options.sweep_concurrency),
        "--image", options.image,
        "--new-plugin", options.new_plugin,
        "--old-plugin", options.old_plugin,
        "--vs-root", options.vs_root,
        "--repo-root", options.repo_root,
        "--threads", str(options.threads),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    result = extract_result(completed)
    result["command"] = subprocess.list2cmdline(command)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps(result, ensure_ascii=True), encoding="utf-8")
    return result


def aggregate_sweep(samples: list[dict], implementation: str) -> dict:
    selected = [sample for sample in samples
                if sample["implementation"] == implementation]
    candidate_count = selected[0]["definition"]["candidate_count"]
    totals = [sample["sweep_total_ns"] for sample in selected]
    return {
        "implementation": implementation,
        "runs": len(selected),
        "candidate_count": candidate_count,
        "total": summarize(totals),
        "per_candidate": summarize(
            [value / candidate_count for value in totals]),
        "graph_build": summarize(
            [sample["graph_build_ns"] for sample in selected]),
        "evaluation_wall": summarize(
            [sample["evaluation_wall_ns"] for sample in selected]),
        "individual_frame": summarize(
            [value for sample in selected
             for value in sample["individual_frame_ns"]]),
    }


def compare_sweep_curves(samples: list[dict]) -> dict:
    old_runs = [sample for sample in samples if sample["implementation"] == "old"]
    new_runs = [sample for sample in samples if sample["implementation"] == "new"]
    old_curve = old_runs[0]["curve"]
    new_curve = new_runs[0]["curve"]
    if [item["id"] for item in old_curve] != [item["id"] for item in new_curve]:
        raise RuntimeError("old and new sweep candidate order differs")
    old_order = sorted(range(len(old_curve)), key=lambda index: old_curve[index]["error"])
    new_order = sorted(range(len(new_curve)), key=lambda index: new_curve[index]["error"])
    old_rank = {index: rank for rank, index in enumerate(old_order)}
    new_rank = {index: rank for rank, index in enumerate(new_order)}
    curve = []
    maximum_difference = 0.0
    maximum_rank_change = 0
    for index, (old_item, new_item) in enumerate(zip(old_curve, new_curve)):
        difference = abs(old_item["error"] - new_item["error"])
        rank_change = abs(old_rank[index] - new_rank[index])
        maximum_difference = max(maximum_difference, difference)
        maximum_rank_change = max(maximum_rank_change, rank_change)
        curve.append({
            "index": index,
            "id": old_item["id"],
            "scaler": old_item["scaler"],
            "height": old_item["height"],
            "old_error": old_item["error"],
            "new_error": new_item["error"],
            "absolute_difference": difference,
            "old_rank": old_rank[index],
            "new_rank": new_rank[index],
            "rank_change": rank_change,
        })
    return {
        "candidate_count": len(curve),
        "max_absolute_error_difference": maximum_difference,
        "max_rank_change": maximum_rank_change,
        "best_old": curve[old_order[0]],
        "best_new": curve[new_order[0]],
        "best_changed": old_order[0] != new_order[0],
        "curve": curve,
    }


def parse_build_info(options) -> dict:
    root = Path(options.repo_root)
    cache = root / "build" / "CMakeCache.txt"
    values = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" in line and "=" in line and not line.startswith(("//", "#")):
                key = line.split(":", 1)[0]
                values[key] = line.split("=", 1)[1]
    compiler = values.get("CMAKE_CXX_COMPILER")
    compiler_id = None
    compiler_version = None
    compiler_files = list((root / "build" / "CMakeFiles").glob(
        "*/CMakeCXXCompiler.cmake"))
    if compiler_files:
        compiler_text = compiler_files[-1].read_text(
            encoding="utf-8", errors="replace")
        matches = {
            key: re.search(rf'set\({key} "([^"]+)"\)', compiler_text)
            for key in ("CMAKE_CXX_COMPILER", "CMAKE_CXX_COMPILER_ID",
                        "CMAKE_CXX_COMPILER_VERSION")
        }
        if matches["CMAKE_CXX_COMPILER"]:
            compiler = matches["CMAKE_CXX_COMPILER"].group(1)
        if matches["CMAKE_CXX_COMPILER_ID"]:
            compiler_id = matches["CMAKE_CXX_COMPILER_ID"].group(1)
        if matches["CMAKE_CXX_COMPILER_VERSION"]:
            compiler_version = matches["CMAKE_CXX_COMPILER_VERSION"].group(1)
    return {
        "configuration": Path(options.new_plugin).parent.name,
        "generator": values.get("CMAKE_GENERATOR"),
        "compiler": compiler,
        "compiler_id": compiler_id,
        "compiler_version": compiler_version,
        "getnative_source": values.get("DSMVC_GETNATIVE_SOURCE_DIR"),
        "getnative_commit": "d64af6caa6c18d670eda1f25ee69fb47f5313b69",
        "msvc_runtime": "static (/MT)",
    }


def command_version(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=30)
        return (completed.stdout + completed.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


def environment_info(options) -> dict:
    import vapoursynth as vs

    core = vs.core
    references = {}
    for name in ("6.2-1.png", "test_getfnative.vpy", "test_getfnative_v2.vpy",
                 "test_selectkernel.vpy", "总监培训2026_20260725.html",
                 "总监培训2026_20260726.html"):
        path = Path(options.downloads) / name
        if path.exists():
            references[name] = {"path": str(path.resolve()),
                                "sha256": sha256_file(path)}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER"),
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version,
        "python_executable": sys.executable,
        "vapoursynth": core.version(),
        "vapoursynth_api": str(vs.__api_version__),
        "vspipe": command_version([options.vspipe, "--version"]),
        "threads": core.num_threads if options.threads <= 0 else options.threads,
        "old_plugin": str(Path(options.old_plugin).resolve()),
        "old_plugin_sha256": sha256_file(Path(options.old_plugin)),
        "new_plugin": str(Path(options.new_plugin).resolve()),
        "new_plugin_sha256": sha256_file(Path(options.new_plugin)),
        "input": str(Path(options.image).resolve()),
        "input_sha256": sha256_file(Path(options.image)),
        "input_metadata": {"width": 1920, "height": 1080,
                           "bits": 8, "color_family": "RGB", "alpha": False},
        "geometry": geometry(),
        "build": parse_build_info(options),
        "reference_files": references,
        "command": subprocess.list2cmdline(sys.argv),
    }


def run_vspipe_once(options, implementation: str, case_name: str) -> dict:
    threads = options.threads if options.threads > 0 else 32
    command = [
        options.vspipe,
        "--arg", f"implementation={implementation}",
        "--arg", f"image={Path(options.image).resolve()}",
        "--arg", f"plugin={Path(options.new_plugin).resolve()}",
        "--arg", f"case={case_name}",
        "--arg", f"frames={options.vspipe_frames}",
        "--arg", f"threads={threads}",
        "--requests", str(threads),
        "--end", str(options.vspipe_frames - 1),
        "--filter-time",
        str(Path(options.repo_root) / "benchmarks" / "vspipe_benchmark.vpy"),
        ".",
    ]
    start = time.perf_counter_ns()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter_ns() - start
    if completed.returncode != 0:
        raise RuntimeError("VSPipe benchmark failed:\n" + completed.stdout
                           + "\n" + completed.stderr)
    return {
        "implementation": implementation,
        "case": case_name,
        "frames": options.vspipe_frames,
        "elapsed_ns": elapsed,
        "fps": options.vspipe_frames / (elapsed / 1e9),
        "command": subprocess.list2cmdline(command),
        "filter_time_output": (completed.stdout + completed.stderr).strip(),
    }


def aggregate_vspipe(samples: list[dict], implementation: str) -> dict:
    selected = [sample for sample in samples
                if sample["implementation"] == implementation]
    elapsed = summarize([sample["elapsed_ns"] for sample in selected])
    frames = selected[0]["frames"]
    return {
        "implementation": implementation,
        "runs": len(selected),
        "frames": frames,
        "elapsed": elapsed,
        "median_fps": frames / (elapsed["median_ns"] / 1e9),
    }


def acceptance_result(cases: list[dict], comparisons: list[dict]) -> dict:
    required = {"bicubic_0_1", "bicubic_0_7_0_6"}
    direct = [case for case in cases
              if case["path"] == "direct" and case["name"] in required]
    required_direct_present = {case["name"] for case in direct} == required
    minimum_direct_speedup = min(
        (case["speedup"]["warm_frame"] for case in direct), default=0.0)
    no_required_regression = all(
        case["speedup"]["warm_frame"] >= 1.0 / 1.05
        for case in cases if case["name"] in required)
    numerical = all(item["within_tolerance"] for item in comparisons)
    target = required_direct_present and minimum_direct_speedup >= 1.2
    return {
        "required_direct_present": required_direct_present,
        "minimum_required_direct_speedup": minimum_direct_speedup,
        "direct_speedup_target_1_2x": target,
        "no_required_case_slower_than_5_percent": no_required_regression,
        "all_comparisons_within_tolerance": numerical,
        "passed": target and no_required_regression and numerical,
    }


def write_sweep_plot(sweep: dict, output_dir: Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    comparison = sweep["comparison"]
    x = [item["index"] for item in comparison["curve"]]
    old = [max(item["old_error"], 1e-12) for item in comparison["curve"]]
    new = [max(item["new_error"], 1e-12) for item in comparison["curve"]]
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(x, old, label="old descale", linewidth=1.0)
    axis.plot(x, new, label="dsmvc", linewidth=1.0, alpha=0.8)
    axis.set_yscale("log")
    axis.set_xlabel("Candidate index")
    axis.set_ylabel("Thresholded mean absolute reconstruction error")
    axis.set_title(f"{sweep['name']} ({sweep['profile']})")
    axis.legend()
    figure.tight_layout()
    filename = f"{sweep['name']}-{sweep['profile']}-error-curve.png"
    figure.savefig(output_dir / filename, dpi=140)
    plt.close(figure)
    return filename


def write_results(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sweep in result.get("sweeps", []):
        sweep["error_curve_plot"] = write_sweep_plot(sweep, output_dir)
    (output_dir / "benchmark.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")

    with (output_dir / "benchmark.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        fields = [
            "record_type", "case", "path", "implementation", "phase",
            "plane", "median_ms", "mad_ms", "p95_ms", "min_ms", "max_ms",
            "speedup", "candidate_id", "height", "old_error", "new_error",
            "rank_change", "max_abs", "mae", "mse", "psnr",
            "pixels_over_threshold", "within_tolerance",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in result.get("cases", []):
            for phase in ("core_initialization", "plugin_load", "source_decode",
                          "graph_build", "cold_frame", "first_use", "warm_frame"):
                for implementation in ("old", "new"):
                    stats = case[implementation][phase]
                    writer.writerow({
                        "record_type": "case",
                        "case": case["name"],
                        "path": case["path"],
                        "implementation": implementation,
                        "phase": phase,
                        "median_ms": stats["median_ns"] / 1e6,
                        "mad_ms": stats["mad_ns"] / 1e6,
                        "p95_ms": stats["p95_ns"] / 1e6,
                        "min_ms": stats["min_ns"] / 1e6,
                        "max_ms": stats["max_ns"] / 1e6,
                        "speedup": case["speedup"].get(phase, ""),
                    })
        for comparison in result.get("comparisons", []):
            for plane in comparison["planes"]:
                writer.writerow({
                    "record_type": "comparison",
                    "case": comparison["case"],
                    "path": comparison["path"],
                    "phase": "old_vs_new",
                    "plane": plane["plane"],
                    "max_abs": plane["max_abs"],
                    "mae": plane["mae"],
                    "mse": plane["mse"],
                    "psnr": plane["psnr"],
                    "pixels_over_threshold": plane["pixels_over_threshold"],
                    "within_tolerance": comparison["within_tolerance"],
                })
        for item in result.get("vspipe", []):
            for implementation in ("old", "new"):
                stats = item[implementation]["elapsed"]
                writer.writerow({
                    "record_type": "vspipe",
                    "case": item["case"],
                    "implementation": implementation,
                    "phase": "elapsed",
                    "median_ms": stats["median_ns"] / 1e6,
                    "mad_ms": stats["mad_ns"] / 1e6,
                    "p95_ms": stats["p95_ns"] / 1e6,
                    "min_ms": stats["min_ns"] / 1e6,
                    "max_ms": stats["max_ns"] / 1e6,
                    "speedup": item["speedup"],
                })
        for sweep in result.get("sweeps", []):
            for phase in ("total", "per_candidate", "graph_build",
                          "evaluation_wall", "individual_frame"):
                old_median = sweep["old"][phase]["median_ns"]
                new_median = sweep["new"][phase]["median_ns"]
                speedup = old_median / new_median if new_median else math.inf
                for implementation in ("old", "new"):
                    stats = sweep[implementation][phase]
                    writer.writerow({
                        "record_type": "sweep",
                        "case": sweep["name"],
                        "path": sweep["profile"],
                        "implementation": implementation,
                        "phase": phase,
                        "median_ms": stats["median_ns"] / 1e6,
                        "mad_ms": stats["mad_ns"] / 1e6,
                        "p95_ms": stats["p95_ns"] / 1e6,
                        "min_ms": stats["min_ns"] / 1e6,
                        "max_ms": stats["max_ns"] / 1e6,
                        "speedup": speedup,
                    })
            for item in sweep["comparison"]["curve"]:
                writer.writerow({
                    "record_type": "curve",
                    "case": sweep["name"],
                    "path": sweep["profile"],
                    "phase": "error",
                    "candidate_id": item["id"],
                    "height": item["height"],
                    "old_error": item["old_error"],
                    "new_error": item["new_error"],
                    "rank_change": item["rank_change"],
                })

    comparisons = {(item["case"], item["path"]): item
                   for item in result.get("comparisons", [])}
    lines = [
        "# dsmvc benchmark",
        "",
        f"Generated: `{result['environment']['timestamp_utc']}`",
        "",
        f"Acceptance: **{'PASS' if result['acceptance']['passed'] else 'FAIL'}**",
        "",
        f"Command: `{result['environment']['command']}`",
        "",
        "Direct cases use the recorded fractional geometry. Wrapper cases preserve "
        "the public wrapper API, whose crop offsets default to zero.",
        "",
        "| Case | Path | Format | Old warm ms | New warm ms | Speedup | Max abs | MAE | SHA equal | Numeric |",
        "|---|---|---|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for case in result.get("cases", []):
        comparison = comparisons[(case["name"], case["path"])]
        maximum = max(plane["max_abs"] for plane in comparison["planes"])
        mae = max(plane["mae"] for plane in comparison["planes"])
        lines.append(
            f"| {case['name']} | {case['path']} | "
            f"{comparison['new_output']['format']} | "
            f"{case['old']['warm_frame']['median_ns'] / 1e6:.3f} | "
            f"{case['new']['warm_frame']['median_ns'] / 1e6:.3f} | "
            f"{case['speedup']['warm_frame']:.3f}x | {maximum:.3g} | "
            f"{mae:.3g} | "
            f"{'yes' if comparison['old_sha256'] == comparison['new_sha256'] else 'no'} | "
            f"{'yes' if comparison['within_tolerance'] else 'no'} |")

    lines.extend([
        "", "## Timing statistics", "",
        "Times are process-level samples in milliseconds.", "",
        "The baseline DLL is autoloaded by the R57 environment, so its explicit "
        "plugin-load value is zero and its load cost is included in core "
        "initialization. dsmvc is loaded explicitly from the build directory.", "",
        "| Case | Path | Implementation | Phase | Median | MAD | p95 | Min | Max |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for case in result.get("cases", []):
        for implementation in ("old", "new"):
            for phase in ("core_initialization", "plugin_load", "source_decode",
                          "graph_build", "cold_frame", "first_use", "warm_frame"):
                stats = case[implementation][phase]
                lines.append(
                    f"| {case['name']} | {case['path']} | {implementation} | "
                    f"{phase} | {stats['median_ns'] / 1e6:.3f} | "
                    f"{stats['mad_ns'] / 1e6:.3f} | "
                    f"{stats['p95_ns'] / 1e6:.3f} | "
                    f"{stats['min_ns'] / 1e6:.3f} | "
                    f"{stats['max_ns'] / 1e6:.3f} |")

    reconstructions = [item for item in result.get("comparisons", [])
                       if item.get("reconstruction")]
    if reconstructions:
        lines.extend([
            "", "## Reconstruction error", "",
            "| Case | Old vs source MSE | New vs source MSE | Old PSNR | New PSNR | Old/new max abs |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for item in reconstructions:
            reconstruction = item["reconstruction"]
            old = reconstruction["old_vs_source"][0]
            new = reconstruction["new_vs_source"][0]
            old_new = reconstruction["old_vs_new"][0]
            lines.append(
                f"| {item['case']} | {old['mse']:.6g} | {new['mse']:.6g} | "
                f"{old['psnr']} | {new['psnr']} | {old_new['max_abs']:.3g} |")

    if result.get("vspipe"):
        lines.extend([
            "", "## VSPipe throughput", "",
            "| Case | Frames | Old fps | New fps | Speedup |",
            "|---|---:|---:|---:|---:|",
        ])
        for item in result["vspipe"]:
            lines.append(
                f"| {item['case']} | {item['old']['frames']} | "
                f"{item['old']['median_fps']:.2f} | "
                f"{item['new']['median_fps']:.2f} | {item['speedup']:.3f}x |")

    if result.get("sweeps"):
        lines.extend([
            "", "## Sweep benchmarks", "",
            "| Sweep | Profile | Candidates | Old total s | New total s | Speedup | Max curve delta | Best changed |",
            "|---|---|---:|---:|---:|---:|---:|:---:|",
        ])
        for sweep in result["sweeps"]:
            comparison = sweep["comparison"]
            lines.append(
                f"| {sweep['name']} | {sweep['profile']} | "
                f"{comparison['candidate_count']} | "
                f"{sweep['old']['total']['median_ns'] / 1e9:.3f} | "
                f"{sweep['new']['total']['median_ns'] / 1e9:.3f} | "
                f"{sweep['speedup']:.3f}x | "
                f"{comparison['max_absolute_error_difference']:.3g} | "
                f"{'yes' if comparison['best_changed'] else 'no'} |")
        for sweep in result["sweeps"]:
            comparison = sweep["comparison"]
            lines.extend([
                "",
                f"`{sweep['name']}` best old: `{comparison['best_old']['id']}` "
                f"({comparison['best_old']['old_error']:.6g}); best new: "
                f"`{comparison['best_new']['id']}` "
                f"({comparison['best_new']['new_error']:.6g}); "
                f"maximum rank change: `{comparison['max_rank_change']}`.",
            ])
            if sweep.get("error_curve_plot"):
                lines.extend([
                    "",
                    f"![{sweep['name']} error curve]({sweep['error_curve_plot']})",
                ])

    lines.extend([
        "", "## Acceptance details", "", "```json",
        json.dumps(result["acceptance"], indent=2, ensure_ascii=True),
        "```", "", "## Environment", "", "```json",
        json.dumps(result["environment"], indent=2, ensure_ascii=True),
        "```", "", "Output and difference images: [`images/`](images/)",
        "", "Complete worker commands: [`commands.txt`](commands.txt)", "",
    ])
    (output_dir / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")

    commands = [result["environment"]["command"]]
    for sample in result.get("raw_samples", []):
        if sample.get("command"):
            commands.append(sample["command"])
    for sweep in result.get("sweeps", []):
        for sample in sweep.get("raw_samples", []):
            if sample.get("command"):
                commands.append(sample["command"])
    for item in result.get("vspipe_raw_samples", []):
        commands.append(item["command"])
    (output_dir / "commands.txt").write_text(
        "\n".join(dict.fromkeys(commands)) + "\n", encoding="utf-8")


def orchestrate(options) -> int:
    image_hash = sha256_file(Path(options.image))
    old_hash = sha256_file(Path(options.old_plugin))
    if image_hash != EXPECTED_IMAGE_SHA256:
        raise RuntimeError(f"unexpected input SHA-256: {image_hash}")
    if old_hash != EXPECTED_OLD_SHA256:
        raise RuntimeError(f"unexpected baseline plugin SHA-256: {old_hash}")
    output_dir = Path(options.output)
    artifact_dir = output_dir / "images"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    selected_cases = options.cases or ["bicubic_0_1", "bicubic_0_7_0_6"]
    selected_paths = options.paths or ["direct", "wrapper"]
    cases = []
    comparisons = []
    raw_samples = []
    for path_kind in selected_paths:
        for case_name in selected_cases:
            samples = []
            for implementation in ("old", "new"):
                for _ in range(options.runs):
                    sample = run_worker(options, implementation, case_name, path_kind)
                    samples.append(sample)
                    raw_samples.append(sample)
            old = aggregate_runs(samples, "old")
            new = aggregate_runs(samples, "new")
            speedup = {}
            for phase in ("plugin_load", "graph_build", "cold_frame",
                          "first_use", "warm_frame"):
                denominator = new[phase]["median_ns"]
                numerator = old[phase]["median_ns"]
                speedup[phase] = numerator / denominator if denominator else math.inf
            cases.append({"name": case_name, "path": path_kind,
                          "old": old, "new": new, "speedup": speedup})
            comparisons.append(compare_case(
                options, case_name, path_kind, artifact_dir))

    vspipe_raw = []
    vspipe_results = []
    if not options.skip_vspipe:
        for case_name in [name for name in selected_cases
                          if name in ("bicubic_0_1", "bicubic_0_7_0_6")]:
            samples = []
            for implementation in ("old", "new"):
                for _ in range(options.vspipe_runs):
                    sample = run_vspipe_once(options, implementation, case_name)
                    samples.append(sample)
                    vspipe_raw.append(sample)
            old = aggregate_vspipe(samples, "old")
            new = aggregate_vspipe(samples, "new")
            vspipe_results.append({
                "case": case_name, "old": old, "new": new,
                "speedup": old["elapsed"]["median_ns"]
                    / new["elapsed"]["median_ns"],
            })

    sweeps = []
    if options.sweep_profile != "none":
        sweep_cases = options.sweep_cases or [
            "getfnative", "getfnative_v2", "selectkernel"]
        for case_name in sweep_cases:
            samples = []
            for implementation in ("old", "new"):
                for run_index in range(options.sweep_runs):
                    samples.append(run_sweep_worker(
                        options, implementation, case_name, run_index))
            old = aggregate_sweep(samples, "old")
            new = aggregate_sweep(samples, "new")
            comparison = compare_sweep_curves(samples)
            sweeps.append({
                "name": case_name,
                "profile": options.sweep_profile,
                "definition": samples[0]["definition"],
                "old": old,
                "new": new,
                "speedup": old["total"]["median_ns"]
                    / new["total"]["median_ns"],
                "comparison": comparison,
                "raw_samples": [
                    {key: value for key, value in sample.items()
                     if key not in ("curve", "individual_frame_ns")}
                    for sample in samples
                ],
            })

    acceptance = acceptance_result(cases, comparisons)
    acceptance["sweep_curves_identical_within_1e_7"] = all(
        sweep["comparison"]["max_absolute_error_difference"] <= 1.0e-7
        for sweep in sweeps)
    acceptance["passed"] = (acceptance["passed"]
                            and acceptance["sweep_curves_identical_within_1e_7"])
    result = {
        "schema_version": 2,
        "environment": environment_info(options),
        "acceptance": acceptance,
        "cases": cases,
        "comparisons": comparisons,
        "vspipe": vspipe_results,
        "vspipe_raw_samples": vspipe_raw,
        "sweeps": sweeps,
        "raw_samples": raw_samples,
    }
    write_results(result, output_dir)
    print(output_dir.resolve())
    return 0


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    downloads = Path.home() / "Downloads"
    vs_root = Path(r"D:\okegui\OKEGui\tools\vapoursynth")
    result = argparse.ArgumentParser()
    result.add_argument("--image", default=str(downloads / "6.2-1.png"))
    result.add_argument("--downloads", default=str(downloads))
    result.add_argument("--old-plugin", default=str(
        vs_root / "vapoursynth64" / "plugins" / "descale.dll"))
    result.add_argument("--new-plugin", default=str(
        root / "build" / "Release" / "dsmvc.dll"))
    result.add_argument("--vs-root", default=str(vs_root))
    result.add_argument("--vspipe", default=str(vs_root / "VSPipe.exe"))
    result.add_argument("--repo-root", default=str(root))
    result.add_argument("--output", default=str(root / "benchmark-results"))
    result.add_argument("--runs", type=int, default=7)
    result.add_argument("--warm-frames", type=int, default=32)
    result.add_argument("--threads", type=int, default=32)
    result.add_argument("--cases", nargs="*", choices=sorted(DIRECT_CASES))
    result.add_argument("--paths", nargs="*", choices=("direct", "wrapper"))
    result.add_argument("--vspipe-runs", type=int, default=3)
    result.add_argument("--vspipe-frames", type=int, default=64)
    result.add_argument("--skip-vspipe", action="store_true")
    result.add_argument("--sweep-profile", choices=("none", "smoke", "full"),
                        default="smoke")
    result.add_argument("--sweep-cases", nargs="*",
                        choices=("getfnative", "getfnative_v2", "selectkernel"))
    result.add_argument("--sweep-runs", type=int, default=3)
    result.add_argument("--sweep-concurrency", type=int, default=0)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--worker-direct", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--worker-sweep", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--implementation", choices=("old", "new"), help=argparse.SUPPRESS)
    result.add_argument("--case", choices=sorted(DIRECT_CASES), help=argparse.SUPPRESS)
    result.add_argument("--path-kind", choices=("direct", "wrapper"), help=argparse.SUPPRESS)
    result.add_argument("--sweep-case",
                        choices=("getfnative", "getfnative_v2", "selectkernel"),
                        help=argparse.SUPPRESS)
    return result


def main() -> int:
    options = parser().parse_args()
    if options.worker_direct:
        return worker_direct(options)
    if options.worker_sweep:
        return worker_sweep(options)
    return orchestrate(options)


if __name__ == "__main__":
    raise SystemExit(main())
