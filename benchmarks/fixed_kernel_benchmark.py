#!/usr/bin/env python3
"""Fixed-kernel VapourSynth throughput benchmark on the Digimon MKV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import platform
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


KERNELS = (
    {"name": "bilinear", "label": "bilinear"},
    {"name": "bicubic_b0_c0_5", "label": "bicubic (0, 0.5)"},
    {"name": "lanczos2", "label": "lanczos2"},
    {"name": "lanczos3", "label": "lanczos3"},
    {"name": "lanczos4", "label": "lanczos4"},
    {"name": "lanczos5", "label": "lanczos5"},
    {"name": "lanczos6", "label": "lanczos6"},
    {"name": "spline16", "label": "spline16"},
    {"name": "spline36", "label": "spline36"},
    {"name": "spline64", "label": "spline64"},
)
KERNEL_BY_NAME = {item["name"]: item for item in KERNELS}
IMPLEMENTATIONS = ("old", "new")
DEFAULT_THREADS = (1, 8, 16, 32)
OUTPUT_RE = re.compile(
    r"Output (?P<frames>\d+) frames in (?P<seconds>[0-9.]+) seconds "
    r"\((?P<fps>[0-9.]+) fps\)"
)


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


def command_text(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return " ".join(subprocess.list2cmdline([item]) for item in command)


def version(command: list[str]) -> str:
    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace", check=False)
    return (completed.stdout + completed.stderr).strip()


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    median = statistics.median(ordered)
    deviations = [abs(value - median) for value in ordered]
    p95_index = max(0, (len(ordered) * 95 + 99) // 100 - 1)
    return {
        "median": median,
        "mad": statistics.median(deviations),
        "p95": ordered[p95_index],
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


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


def fixed_geometry(source_width: int, source_height: int,
                   native_height: float, base_height: float) -> dict:
    base_height_int = int(round(base_height))
    base_width = round(source_width / source_height * base_height_int)
    native_width = source_width / source_height * native_height
    output_width = base_width - 2 * int((base_width - native_width) / 2)
    output_height = base_height_int - 2 * int(
        (base_height_int - native_height) / 2)
    return {
        "source_width": source_width,
        "source_height": source_height,
        "base_width": base_width,
        "base_height": base_height,
        "native_width": native_width,
        "native_height": native_height,
        "output_width": output_width,
        "output_height": output_height,
        "src_left": (output_width - native_width) / 2,
        "src_top": (output_height - native_height) / 2,
        "src_width": native_width,
        "src_height": native_height,
    }


def run_sample(options, kernel: dict, implementation: str,
               threads: int, run: int, frames: int,
               warmup: bool = False) -> dict:
    script = Path(__file__).with_name("vspipe_fixed_kernel.vpy").resolve()
    source = Path(options.source).expanduser().resolve()
    new_plugin = Path(options.new_plugin).expanduser().resolve()
    old_plugin = (Path(options.old_plugin).expanduser().resolve()
                  if options.old_plugin else None)
    args = {
        "implementation": implementation,
        "kernel": kernel["name"],
        "source": str(source),
        "plugin": str(new_plugin),
        "old_plugin": str(old_plugin) if old_plugin else "",
        "source_plugin": (str(Path(options.source_plugin).expanduser().resolve())
                          if options.source_plugin else ""),
        "source_filter": options.source_filter,
        "source_decoder": options.source_decoder,
        "source_prefer_hw": str(options.source_prefer_hw),
        "source_ff_loglevel": str(options.source_ff_loglevel),
        "source_rap_verification": str(options.source_rap_verification),
        "frames": str(frames),
        "threads": str(threads),
        "backend": options.backend,
        "src_height": str(options.src_height),
        "base_height": str(options.base_height),
    }
    command = [options.vspipe]
    for key, value in args.items():
        command.extend(["--arg", f"{key}={value}"])
    command.extend([
        "--requests", str(threads),
        "--start", "0",
        "--end", str(frames - 1),
        "--filter-time", str(script),
        "--",
    ])
    start_ns = time.perf_counter_ns()
    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace", check=False)
    elapsed_ns = time.perf_counter_ns() - start_ns
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"VSPipe failed for {kernel['name']}/{implementation}/"
            f"r{threads}t{threads}/run-{run}:\n{output[-8000:]}")
    vspipe_timing = parse_vspipe_timing(output, frames)
    elapsed_seconds = elapsed_ns / 1e9
    return {
        "kernel": kernel["name"],
        "label": kernel["label"],
        "implementation": implementation,
        "run": run,
        "warmup": warmup,
        "frames": frames,
        "requests": threads,
        "threads": threads,
        "elapsed_ns": elapsed_ns,
        "elapsed_seconds": elapsed_seconds,
        "fps": frames / elapsed_seconds,
        "vspipe_seconds": vspipe_timing["seconds"],
        "vspipe_fps": vspipe_timing["fps"],
        "process_overhead_seconds": max(
            0.0, elapsed_seconds - vspipe_timing["seconds"]),
        "command": command_text(command),
        "vspipe_output_tail": output[-4000:],
    }


def case_summaries(samples: list[dict], kernels: list[dict],
                   implementations: tuple[str, ...]) -> list[dict]:
    grouped = {}
    for sample in samples:
        key = (sample["kernel"], sample["threads"])
        grouped.setdefault(key, []).append(sample)
    result = []
    for kernel in kernels:
        for threads in sorted({item["threads"] for item in samples}):
            selected = grouped[(kernel["name"], threads)]
            item = {
                "kernel": kernel["name"],
                "label": kernel["label"],
                "frames": selected[0]["frames"],
                "threads": threads,
                "requests": threads,
            }
            for implementation in implementations:
                implementation_samples = [
                    sample for sample in selected
                    if sample["implementation"] == implementation]
                item[implementation] = {
                    "runs": len(implementation_samples),
                    "elapsed_seconds": summarize([
                        sample["elapsed_seconds"]
                        for sample in implementation_samples]),
                    "fps": summarize([
                        sample["fps"] for sample in implementation_samples]),
                    "vspipe_seconds": summarize([
                        sample["vspipe_seconds"]
                        for sample in implementation_samples]),
                    "vspipe_fps": summarize([
                        sample["vspipe_fps"]
                        for sample in implementation_samples]),
                    "process_overhead_seconds": summarize([
                        sample["process_overhead_seconds"]
                        for sample in implementation_samples]),
                }
            if set(implementations) == set(IMPLEMENTATIONS):
                old = item["old"]["elapsed_seconds"]["median"]
                new = item["new"]["elapsed_seconds"]["median"]
                item["speedup"] = old / new
            else:
                item["speedup"] = None
            result.append(item)
    return result


def svg_polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_scaling_svg(cases: list[dict], kernels: list[dict],
                      implementations: tuple[str, ...], path: Path) -> None:
    by_key = {(item["kernel"], item["threads"]): item for item in cases}
    columns = 5
    panel_width = 270
    panel_height = 245
    margin_x = 20
    margin_y = 88
    rows = (len(kernels) + columns - 1) // columns
    width = margin_x * 2 + columns * panel_width
    height = margin_y + rows * panel_height + 28
    threads = sorted({item["threads"] for item in cases})
    colors = {
        "old": "#64748b",
        "new": "#dc2626",
        "newmetal": "#0f766e",
    }
    labels = {
        "old": "old descale",
        "new": "new CPU",
        "newmetal": "new Metal (1692x952)",
    }
    legend = " / ".join(labels[item] for item in implementations)
    frames = cases[0].get("frames", "") if cases else ""
    frame_label = f"{frames} source frames; " if frames else ""
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="title desc">',
        '<title id="title">Fixed kernel thread scaling</title>',
        f'<desc id="desc">{html.escape(legend)} FPS across the selected thread configurations for each fixed kernel.</desc>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" '
        'font-family="sans-serif" font-size="22" font-weight="700" fill="#111827">'
        + ('Fixed kernel scaling at 810p</text>'
           if "newmetal" not in implementations else
           'Fixed kernel scaling (CPU 810p; Metal 1692x952)</text>'),
        f'<text x="{width / 2:.1f}" y="52" text-anchor="middle" '
        'font-family="sans-serif" font-size="13" fill="#4b5563">'
        f'{frame_label}FPS is external VSPipe wall-clock throughput</text>',
    ]
    legend_width = 170 * len(implementations)
    legend_start = width / 2 - legend_width / 2
    for index, implementation in enumerate(implementations):
        center = legend_start + index * 170 + 85
        color = colors[implementation]
        parts.extend([
            f'<line x1="{center - 45:.1f}" y1="72" x2="{center - 15:.1f}" y2="72" stroke="{color}" stroke-width="3"/>',
            f'<circle cx="{center - 30:.1f}" cy="72" r="4" fill="{color}"/>',
            f'<text x="{center - 5:.1f}" y="77" font-family="sans-serif" font-size="13" fill="#374151">{html.escape(labels[implementation])}</text>',
        ])
    for index, kernel in enumerate(kernels):
        column = index % columns
        row = index // columns
        panel_x = margin_x + column * panel_width
        panel_y = margin_y + row * panel_height
        plot_left = panel_x + 36
        plot_right = panel_x + panel_width - 14
        plot_top = panel_y + 28
        plot_bottom = panel_y + 190
        values = [
            by_key[(kernel["name"], thread)][implementation]["fps"]["median"]
            for thread in threads
            for implementation in implementations
            if by_key[(kernel["name"], thread)][implementation]["fps"]["median"] is not None
        ]
        y_max = max(values) * 1.12 if values else 1.0
        if y_max <= 0:
            y_max = 1.0
        parts.extend([
            f'<g font-family="sans-serif">',
            f'<rect x="{panel_x}" y="{panel_y}" width="{panel_width - 10}" height="220" fill="#f8fafc" stroke="#cbd5e1"/>',
            f'<text x="{panel_x + (panel_width - 10) / 2:.1f}" y="{panel_y + 19}" text-anchor="middle" font-size="14" font-weight="700" fill="#111827">{html.escape(kernel["label"])}</text>',
            f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#334155"/>',
            f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#334155"/>',
        ])
        for tick in (0.0, y_max / 2, y_max):
            y = plot_bottom - (tick / y_max) * (plot_bottom - plot_top)
            parts.append(
                f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" '
                'stroke="#e2e8f0"/>')
            parts.append(
                f'<text x="{plot_left - 5}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-size="9" fill="#475569">{tick:.0f}</text>')
        x_positions = []
        for thread_index, thread in enumerate(threads):
            if len(threads) == 1:
                x = (plot_left + plot_right) / 2
            else:
                x = plot_left + thread_index * (plot_right - plot_left) / (len(threads) - 1)
            x_positions.append(x)
            parts.append(
                f'<text x="{x:.2f}" y="{plot_bottom + 15}" text-anchor="middle" '
                f'font-size="9" fill="#475569">{thread}</text>')
        for implementation in implementations:
            color = colors[implementation]
            points = []
            for x, thread in zip(x_positions, threads):
                value = by_key[(kernel["name"], thread)][implementation]["fps"]["median"]
                if value is None:
                    continue
                y = plot_bottom - value / y_max * (plot_bottom - plot_top)
                points.append((x, y))
            if len(points) >= 2:
                parts.append(
                    f'<polyline points="{svg_polyline(points)}" fill="none" '
                    f'stroke="{color}" stroke-width="2.5"/>')
            for x, y in points:
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{color}"/>')
        parts.append('</g>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_csv(samples: list[dict], cases: list[dict],
              implementations: tuple[str, ...], path: Path) -> None:
    fields = [
        "type", "kernel", "label", "implementation", "threads",
        "requests", "run", "frames", "elapsed_seconds", "fps",
        "median_fps", "vspipe_seconds", "vspipe_fps",
        "process_overhead_seconds", "speedup",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "type": "raw",
                "kernel": sample["kernel"],
                "label": sample["label"],
                "implementation": sample["implementation"],
                "threads": sample["threads"],
                "requests": sample["requests"],
                "run": sample["run"],
                "frames": sample["frames"],
                "elapsed_seconds": f'{sample["elapsed_seconds"]:.9f}',
                "fps": f'{sample["fps"]:.6f}',
                "median_fps": "",
                "vspipe_seconds": f'{sample["vspipe_seconds"]:.9f}',
                "vspipe_fps": f'{sample["vspipe_fps"]:.6f}',
                "process_overhead_seconds": (
                    f'{sample["process_overhead_seconds"]:.9f}'),
                "speedup": "",
            })
        for item in cases:
            for implementation in implementations:
                writer.writerow({
                    "type": "summary",
                    "kernel": item["kernel"],
                    "label": item["label"],
                    "implementation": implementation,
                    "threads": item["threads"],
                    "requests": item["requests"],
                    "run": "",
                    "frames": item["frames"],
                    "elapsed_seconds": f'{item[implementation]["elapsed_seconds"]["median"]:.9f}',
                    "fps": "",
                    "median_fps": f'{item[implementation]["fps"]["median"]:.6f}',
                    "vspipe_seconds": f'{item[implementation]["vspipe_seconds"]["median"]:.9f}',
                    "vspipe_fps": f'{item[implementation]["vspipe_fps"]["median"]:.6f}',
                    "process_overhead_seconds": f'{item[implementation]["process_overhead_seconds"]["median"]:.9f}',
                    "speedup": (f'{item["speedup"]:.6f}'
                                if item["speedup"] is not None else ""),
                })


def compact_speed(item: dict, implementations: tuple[str, ...]) -> str:
    if set(implementations) == set(IMPLEMENTATIONS):
        old = item["old"]["fps"]["median"]
        new = item["new"]["fps"]["median"]
        return f"{old:.3f} -> {new:.3f} ({item['speedup']:.3f}x)"
    implementation = implementations[0]
    return f"{implementation} {item[implementation]['fps']['median']:.3f}"


def write_markdown(result: dict, path: Path) -> None:
    environment = result["environment"]
    geometry = result["geometry"]
    kernels = result["kernels"]
    implementations = tuple(environment["implementations"])
    by_kernel = {item["name"]: [] for item in kernels}
    for item in result["cases"]:
        by_kernel[item["kernel"]].append(item)
    threads = result["environment"]["thread_configs"]
    lines = [
        "# DIGIMON BEATBREAK Fixed Kernel Benchmark",
        "",
        "## Conclusion",
        "",
        f"Each kernel processes the first **{environment['frames']:,}** frames of the supplied MKV at fixed native height **{geometry['native_height']:.1f}p**.",
        f"The selected implementation(s) ({', '.join(implementations)}) are run in fresh VSPipe",
        "processes at R1T1, R8T8, R16T16, and R32T32. FPS is external wall-clock",
        "throughput from process start to exit; it includes source decode, graph",
        "construction, plugin loading, descale, frame delivery, and shutdown.",
        f"Each cell has {environment['warmup_runs']} untimed warm-up run(s) of "
        f"{environment['warmup_frames']} frames. VSPipe's internal processing "
        "time is retained separately in JSON and CSV; use that internal value "
        "for executor A/B decisions and wall time for end-to-end cost. Warm-up "
        "processes warm "
        "driver/module/page caches and GPU clocks, but each measured fresh "
        "process still creates its own CUDA context.",
        "",
        "## Geometry",
        "",
        f"- Source: `{geometry['source_width']}x{geometry['source_height']}`",
        f"- Descale geometry: `base_height={geometry['base_height']:.1f}`, `src_height={geometry['native_height']:.1f}`",
        f"- Descale output: `{geometry['output_width']}x{geometry['output_height']}`",
        f"- Crop offsets: `src_left={geometry['src_left']:.6f}`, `src_top={geometry['src_top']:.6f}`",
        "- Input filter: `" + environment["source_filter"] + "`",
        "- Bicubic: only `(b=0, c=0.5)`",
        "- Selected kernels: " + ", ".join(
            item["label"] for item in kernels),
        "",
        "## Scaling",
        "",
        "![Fixed kernel scaling](scaling.svg)",
        "",
        "Each panel uses its own y-axis range so slow and fast kernels remain readable.",
        "",
        "## Throughput",
        "",
        "Values are FPS (or `old FPS -> current FPS (speedup)` when both implementations are selected); each cell is one independent",
        "thread configuration and each implementation has `runs=1` unless noted.",
        "",
        "| Kernel | " + " | ".join(
            f"R{thread}T{thread}" for thread in environment["threads"]) + " |",
        "|---|" + "|".join(
            ":---:" for _ in environment["threads"]) + "|",
    ]
    for kernel in kernels:
        values = {item["threads"]: compact_speed(item, implementations)
                  for item in by_kernel[kernel["name"]]}
        lines.append("| " + " | ".join(
            [f"`{kernel['label']}`"]
            + [values[thread] for thread in environment["threads"]]) + " |")
    lines.extend([
        "",
        "## Raw Results",
        "",
        "- [Structured JSON](benchmark.json)",
        "- [CSV samples and summaries](benchmark.csv)",
        "- [Commands](commands.txt)",
        "- [Scaling SVG](scaling.svg)",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(environment, indent=2, ensure_ascii=True),
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_options(options) -> None:
    if (options.frames < 1 or options.runs < 1
            or options.warmup_runs < 0 or options.warmup_frames < 0):
        raise ValueError(
            "--frames and --runs must be positive; warm-up values cannot be negative")
    if options.src_height <= 0 or options.base_height <= 0:
        raise ValueError("heights must be positive")
    if not options.threads:
        raise ValueError("at least one thread configuration is required")
    if any(value < 1 for value in options.threads):
        raise ValueError("thread counts must be positive")
    if len(set(options.threads)) != len(options.threads):
        raise ValueError("thread counts must be unique")
    if not options.implementations:
        raise ValueError("at least one implementation is required")
    if len(set(options.implementations)) != len(options.implementations):
        raise ValueError("implementation names must be unique")
    if options.source_prefer_hw < 0 or options.source_prefer_hw > 7:
        raise ValueError("--source-prefer-hw must be between 0 and 7")
    if options.source_ff_loglevel < 0 or options.source_ff_loglevel > 8:
        raise ValueError("--source-ff-loglevel must be between 0 and 8")
    if options.source_rap_verification not in (-1, 0, 1):
        raise ValueError("--source-rap-verification must be -1, 0, or 1")
    unknown = [name for name in options.kernels if name not in KERNEL_BY_NAME]
    if unknown:
        raise ValueError("unknown kernels: " + ", ".join(unknown))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Benchmark fixed descale kernels on the first source frames.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--old-plugin", default="")
    parser.add_argument("--new-plugin", required=True)
    parser.add_argument("--vspipe", default="vspipe")
    parser.add_argument("--source-plugin")
    parser.add_argument("--source-filter", choices=("lsmas", "ffms2", "bestsource"),
                        default="ffms2")
    parser.add_argument("--source-decoder", default="",
                        help="Preferred LSMASH/libavcodec decoder name(s).")
    parser.add_argument("--source-prefer-hw", type=int, default=0,
                        help="LSMASH prefer_hw mode; 0 keeps software default.")
    parser.add_argument("--source-ff-loglevel", type=int, default=0,
                        help="LSMASH FFmpeg log level, 0 is quiet.")
    parser.add_argument("--source-rap-verification", type=int, default=-1,
                        help="LSMASH RAP verification; -1 keeps plugin default.")
    parser.add_argument("--output", default=str(
        root / "benchmark-results" / "fixed-kernel-digimon-810p"))
    parser.add_argument("--frames", type=int, default=4000)
    parser.add_argument("--src-height", type=float, default=810.0)
    parser.add_argument("--base-height", type=float, default=1000.0)
    parser.add_argument("--threads", nargs="*", type=int,
                        default=list(DEFAULT_THREADS))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1,
                        help="Untimed throwaway VSPipe runs per benchmark cell.")
    parser.add_argument("--warmup-frames", type=int, default=256,
                        help="Frames in each warm-up run; 0 disables warm-up.")
    parser.add_argument("--backend",
                        choices=("auto", "cpu", "metal", "vulkan", "cuda"),
                        default="cpu")
    parser.add_argument("--implementations", nargs="+", choices=IMPLEMENTATIONS,
                        default=list(IMPLEMENTATIONS))
    parser.add_argument("--kernels", nargs="*",
                        default=[item["name"] for item in KERNELS])
    options = parser.parse_args()
    validate_options(options)

    source = Path(options.source).expanduser().resolve()
    old_plugin = (Path(options.old_plugin).expanduser().resolve()
                  if options.old_plugin else None)
    new_plugin = Path(options.new_plugin).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source does not exist: {source}")
    if "old" in options.implementations and (
            old_plugin is None or not old_plugin.is_file()):
        raise FileNotFoundError(f"old plugin does not exist: {old_plugin}")
    if not new_plugin.is_file():
        raise FileNotFoundError(f"new plugin does not exist: {new_plugin}")
    if options.source_plugin and not Path(options.source_plugin).is_file():
        raise FileNotFoundError(f"source plugin does not exist: {options.source_plugin}")

    geometry = fixed_geometry(1920, 1080, options.src_height,
                              options.base_height)
    output = Path(options.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_kernels = [KERNEL_BY_NAME[name] for name in options.kernels]
    samples = []
    warmups = []
    for threads in options.threads:
        for kernel in selected_kernels:
            if options.warmup_frames > 0:
                for implementation in options.implementations:
                    for warmup_run in range(1, options.warmup_runs + 1):
                        warmup = run_sample(
                            options, kernel, implementation, threads,
                            warmup_run, options.warmup_frames, warmup=True)
                        warmups.append(warmup)
                        print(
                            f"warmup {kernel['name']} {implementation} "
                            f"R{threads}T{threads} {warmup_run}/"
                            f"{options.warmup_runs}: "
                            f"{warmup['vspipe_fps']:.3f} VSPipe fps",
                            flush=True)
            for run in range(1, options.runs + 1):
                for implementation in options.implementations:
                    sample = run_sample(options, kernel, implementation,
                                        threads, run, options.frames)
                    samples.append(sample)
                    print(
                        f"{kernel['name']} {implementation} R{threads}T{threads} "
                        f"run {run}: {sample['fps']:.3f} fps", flush=True)

    implementations = tuple(options.implementations)
    cases = case_summaries(samples, selected_kernels, implementations)
    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "vspipe": version([options.vspipe, "--version"]),
        "source_filter": options.source_filter,
        "source_decoder": options.source_decoder,
        "source_prefer_hw": options.source_prefer_hw,
        "source_ff_loglevel": options.source_ff_loglevel,
        "source_rap_verification": options.source_rap_verification,
        "source": file_info(source),
        "old_plugin": file_info(old_plugin) if old_plugin else None,
        "new_plugin": file_info(new_plugin),
        "source_plugin": (file_info(Path(options.source_plugin))
                          if options.source_plugin else None),
        "frames": options.frames,
        "thread_configs": [f"R{value}T{value}" for value in options.threads],
        "threads": options.threads,
        "runs": options.runs,
        "warmup_runs": options.warmup_runs,
        "warmup_frames": options.warmup_frames,
        "backend": options.backend,
        "implementations": list(implementations),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "vpy_sha256": sha256_file(
            Path(__file__).with_name("vspipe_fixed_kernel.vpy")),
    }
    result = {
        "schema_version": 1,
        "environment": environment,
        "geometry": geometry,
        "kernels": selected_kernels,
        "cases": cases,
        "raw_samples": samples,
        "warmup_samples": warmups,
    }
    (output / "benchmark.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_csv(samples, cases, implementations, output / "benchmark.csv")
    write_scaling_svg(cases, selected_kernels, implementations,
                      output / "scaling.svg")
    write_markdown(result, output / "benchmark.md")
    write_markdown(result, output / "summary.md")
    (output / "commands.txt").write_text(
        "\n".join(
            sample["command"] for sample in [*warmups, *samples]) + "\n",
        encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        raise SystemExit(2)
