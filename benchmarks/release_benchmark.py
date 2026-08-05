#!/usr/bin/env python3
"""Run and consolidate the release old/new benchmark package."""

from __future__ import annotations

import argparse
import html
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


THREADS = (1, 8, 16, 32)
CASES = ("getfnative", "getfnative_v2", "selectkernel")
KERNELS = (
    ("bilinear", "bilinear"),
    ("bicubic_b0_c0_5", "bicubic (0, 0.5)"),
    ("lanczos2", "lanczos2"),
    ("lanczos3", "lanczos3"),
    ("lanczos4", "lanczos4"),
    ("lanczos5", "lanczos5"),
    ("lanczos6", "lanczos6"),
    ("spline16", "spline16"),
    ("spline36", "spline36"),
    ("spline64", "spline64"),
)


def run(command: list[str], label: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    print(" ".join(subprocess.list2cmdline([item]) for item in command),
          flush=True)
    subprocess.run(command, check=True)


def e2e_args(options, output: Path, threads: int,
             skip_errors: bool = True) -> list[str]:
    command = [
        options.python,
        str(options.e2e_runner),
        "--source", str(options.source),
        "--old-plugin", str(options.old_plugin),
        "--new-plugin", str(options.new_plugin),
        "--vspipe", str(options.vspipe),
        "--python", str(options.vs_python),
        "--source-filter", options.source_filter,
        "--profile", "full",
        "--cases", *CASES,
        "--implementations", "old", "new",
        "--runs", "1",
        "--requests", str(threads),
        "--threads", str(threads),
        "--output", str(output),
    ]
    for path in options.html:
        command.extend(["--html", str(path)])
    for case, path in options.scripts.items():
        command.extend(["--script", f"{case}={path}"])
    if skip_errors:
        command.append("--skip-errors")
    return command


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def e2e_summary(perf_results: dict[int, dict]) -> list[dict]:
    rows = []
    for case in CASES:
        row = {"case": case, "threads": {}}
        for threads, result in perf_results.items():
            summary = next(item for item in result["performance"]["summaries"]
                           if item["case"] == case)
            row["threads"][str(threads)] = summary
        rows.append(row)
    return rows


def svg_polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_e2e_scaling(rows: list[dict], path: Path) -> None:
    width = 1160
    panel_width = 350
    panel_height = 300
    left = 25
    top = 92
    plot_left_offset = 52
    plot_right_offset = 22
    plot_top_offset = 36
    plot_bottom_offset = 242
    height = top + panel_height + 30
    colors = {"old": "#475569", "new": "#dc2626"}
    labels = {"old": "old descale", "new": "current Release"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">E2E candidate thread scaling</title>',
        '<desc id="desc">Old and current Release candidates per second across R1T1, R8T8, R16T16 and R32T32.</desc>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="31" text-anchor="middle" font-family="sans-serif" font-size="24" font-weight="700" fill="#111827">E2E candidate scaling</text>',
        f'<text x="{width / 2:.1f}" y="56" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#475569">Full candidate sweep; higher candidates/s is better</text>',
    ]
    for index, implementation in enumerate(("old", "new")):
        center = width / 2 - 82 + index * 164
        color = colors[implementation]
        parts.extend([
            f'<line x1="{center - 55:.1f}" y1="74" x2="{center - 25:.1f}" y2="74" stroke="{color}" stroke-width="3"/>',
            f'<circle cx="{center - 40:.1f}" cy="74" r="4" fill="{color}"/>',
            f'<text x="{center - 15:.1f}" y="79" font-family="sans-serif" font-size="13" fill="#374151">{labels[implementation]}</text>',
        ])
    for index, row in enumerate(rows):
        panel_x = left + index * panel_width
        plot_left = panel_x + plot_left_offset
        plot_right = panel_x + panel_width - plot_right_offset
        plot_top = top + plot_top_offset
        plot_bottom = top + plot_bottom_offset
        values = [
            row["threads"][str(thread)][implementation]["candidates_per_second"]["median"]
            for thread in THREADS
            for implementation in ("old", "new")
        ]
        y_max = max(values) * 1.12
        parts.extend([
            f'<g font-family="sans-serif">',
            f'<rect x="{panel_x}" y="{top}" width="{panel_width - 18}" height="270" fill="#f8fafc" stroke="#cbd5e1"/>',
            f'<text x="{panel_x + (panel_width - 18) / 2:.1f}" y="{top + 23}" text-anchor="middle" font-size="15" font-weight="700" fill="#111827">{html.escape(row["case"])}</text>',
            f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#334155"/>',
            f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#334155"/>',
        ])
        for tick in (0.0, y_max / 2, y_max):
            y = plot_bottom - tick / y_max * (plot_bottom - plot_top)
            parts.extend([
                f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="#e2e8f0"/>',
                f'<text x="{plot_left - 7}" y="{y + 4:.2f}" text-anchor="end" font-size="9" fill="#475569">{tick:.0f}</text>',
            ])
        x_positions = []
        for thread_index, thread in enumerate(THREADS):
            x = plot_left + thread_index * (plot_right - plot_left) / (len(THREADS) - 1)
            x_positions.append(x)
            parts.append(
                f'<text x="{x:.2f}" y="{plot_bottom + 17}" text-anchor="middle" font-size="10" fill="#475569">{thread}</text>')
        for implementation in ("old", "new"):
            color = colors[implementation]
            points = []
            for x, thread in zip(x_positions, THREADS):
                value = row["threads"][str(thread)][implementation][
                    "candidates_per_second"]["median"]
                y = plot_bottom - value / y_max * (plot_bottom - plot_top)
                points.append((x, y))
            parts.append(
                f'<polyline points="{svg_polyline(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for x, y in points:
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}"/>')
        parts.append('</g>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def fixed_table(fixed: dict) -> dict:
    result = {}
    for sample in fixed["raw_samples"]:
        result.setdefault(sample["kernel"], {}).setdefault(
            sample["threads"], {})[sample["implementation"]] = sample["fps"]
    return result


def merge_report(options, output: Path, perf_results: dict[int, dict],
                 error_result: dict, fixed_result: dict) -> None:
    e2e_rows = e2e_summary(perf_results)
    fixed_values = fixed_table(fixed_result)
    output.mkdir(parents=True, exist_ok=True)
    write_e2e_scaling(e2e_rows, output / "e2e-scaling.svg")
    merged = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "source": str(options.source),
            "source_sha256": fixed_result["environment"]["source"]["sha256"],
            "source_filter": options.source_filter,
            "old_plugin": fixed_result["environment"]["old_plugin"],
            "new_plugin": fixed_result["environment"]["new_plugin"],
            "threads": list(THREADS),
            "frames": fixed_result["environment"]["frames"],
        },
        "e2e_performance": e2e_rows,
        "e2e_errors": error_result["errors"],
        "fixed_kernel": fixed_result,
        "artifacts": {
            "e2e_scaling": "e2e-scaling.svg",
            "fixed_scaling": "../fixed-kernel-digimon-810p-release/scaling.svg",
        },
    }
    (output / "release-benchmark.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8")

    lines = [
        "# Descale MVC Release Benchmark",
        "",
        "## Executive Summary",
        "",
        "This package compares the current Release build against the original descale plugin on the same Digimon source, VapourSynth runtime, decoder, geometry, and thread configurations.",
        "All performance and error results below were freshly measured. The current plugin was built with generic x86-64 Release code and an AVX2/FMA-only executor TU.",
        "",
        "| Workload | Result |",
        "|---|---:|",
        f"| E2E getfnative candidates | {e2e_rows[0]['threads']['32']['new']['candidates_per_second']['median']:.3f} candidates/s at R32T32 |",
        f"| Fixed kernel coverage | {len(fixed_result['cases'])} algorithm/thread/implementation cases, 4,000 frames each |",
        f"| Error coverage | {sum(item['candidate_count'] for item in error_result['errors']['summaries']):,} candidates across three recipes |",
        "",
        "## Build and Provenance",
        "",
        f"- Current plugin: `{fixed_result['environment']['new_plugin']['sha256']}`",
        f"- Original plugin: `{fixed_result['environment']['old_plugin']['sha256']}`",
        f"- Source SHA-256: `{fixed_result['environment']['source']['sha256']}`",
        f"- Source filter: `{options.source_filter}`",
        "- Build: `Release`, `-O3 -DNDEBUG`, generic `x86-64`, AVX2/FMA isolated to `cpu_executor_avx2.cpp`",
        "- Link: version-script export of `VapourSynthPluginInit`, RELRO, NOW, and pthread",
        "- No LTO, PGO, native CPU tuning, or fast-math flags",
        "",
        "## E2E Thread Scaling",
        "",
        "![E2E thread scaling](e2e-scaling.svg)",
        "",
        "The chart reports candidates per second for the complete candidate graph. It includes planner/cache work, FrameEval, reconstruction, Expr, PlaneStats, frame delivery, and VSPipe process overhead.",
        "",
        "| Case | R1T1 old -> new | R8T8 old -> new | R16T16 old -> new | R32T32 old -> new |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in e2e_rows:
        cells = []
        for thread in THREADS:
            item = row["threads"][str(thread)]
            old = item["old"]["candidates_per_second"]["median"]
            new = item["new"]["candidates_per_second"]["median"]
            cells.append(f"{old:.3f} -> {new:.3f} ({old and new / old:.2f}x)")
        lines.append(f"| `{row['case']}` | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## E2E Error Comparison",
        "",
        "The error sweep evaluates every candidate in each recipe on the same training frame. Metrics compare old and current reconstructed output against the source after the benchmark's 5-pixel border crop.",
        "",
        "| Case | Candidates | Best old | Best current | Changed | Max output abs | Max reconstruction abs |",
        "|---|---:|---|---|---|---:|---:|",
    ])
    for item in error_result["errors"]["summaries"]:
        lines.append(
            f"| `{item['case']}` | {item['candidate_count']:,} | "
            f"{item['best_old']['id']} ({item['best_old']['mae']:.6g}) | "
            f"{item['best_new']['id']} ({item['best_new']['mae']:.6g}) | "
            f"{item['best_candidate_changed']} | "
            f"{item['max_output_max_abs']:.6g} | "
            f"{item['max_reconstruction_max_abs']:.6g} |")
    lines.extend([
        "",
        "### Per-algorithm error minima",
        "",
        "| Case | Algorithm | Old best height / MAE | Current best height / MAE | Height changed |",
        "|---|---|---:|---:|---|",
    ])
    for item in error_result["errors"]["summaries"]:
        for algorithm, values in item["algorithm_summary"].items():
            lines.append(
                f"| `{item['case']}` | `{algorithm}` | "
                f"{values['old_best']['height']:.1f} / {values['old_best']['mae']:.6g} | "
                f"{values['new_best']['height']:.1f} / {values['new_best']['mae']:.6g} | "
                f"{values['best_height_changed']} |")
    lines.extend([
        "",
        "## Fixed Kernel Throughput",
        "",
        "Each cell is `old FPS -> current Release FPS (speedup)` for the first 4,000 frames at fixed 810p geometry.",
        "",
        "[Open the full fixed-kernel scaling chart](../fixed-kernel-digimon-810p-release/scaling.svg)",
        "",
        "| Kernel | R1T1 | R8T8 | R16T16 | R32T32 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, label in KERNELS:
        cells = []
        for thread in THREADS:
            values = fixed_values[name][thread]
            old = values["old"]
            new = values["new"]
            cells.append(f"{old:.3f} -> {new:.3f} ({new / old:.2f}x)")
        lines.append(f"| `{label}` | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The Release build removes the previous build-condition confounder: current and original plugins are now compared with the current plugin compiled using the recommended optimized parameters. Fixed-kernel results expose the long-running executor cost, while E2E candidate scanning also includes graph construction and per-candidate statistics. Their scaling curves therefore answer different performance questions.",
        "",
        "The full fixed-kernel table shows where the current executor wins or loses by algorithm and thread count. The error table shows whether those throughput differences change candidate selection or reconstructed output. Together they are the release-facing performance and compatibility record.",
        "",
        "## Raw Artifacts",
        "",
        "- [Merged machine-readable result](release-benchmark.json)",
        "- E2E per-thread reports: `../e2e-digimon-release-r{1,8,16,32}t{1,8,16,32}/benchmark.json`",
        "- [Full error report](../e2e-digimon-release-errors-r32t32/benchmark.json)",
        "- [Full fixed-kernel report](../fixed-kernel-digimon-810p-release/benchmark.json)",
        "- [Fixed-kernel CSV](../fixed-kernel-digimon-810p-release/benchmark.csv)",
        "",
    ])
    (output / "release-benchmark.md").write_text("\n".join(lines),
                                                 encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--old-plugin", required=True, type=Path)
    parser.add_argument("--new-plugin", required=True, type=Path)
    parser.add_argument("--vspipe", required=True, type=Path)
    parser.add_argument("--vs-python", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source-filter", default="ffms2",
                        choices=("lsmas", "ffms2", "bestsource"))
    parser.add_argument("--output-root", type=Path, default=root / "benchmark-results")
    parser.add_argument("--release-output", type=Path,
                        default=root / "benchmark-results" / "release-benchmark-20260805")
    parser.add_argument("--html", action="append", type=Path, default=[])
    parser.add_argument("--script", action="append", default=[],
                        metavar="CASE=PATH")
    parser.add_argument("--skip-run", action="store_true",
                        help="Only consolidate already completed reports")
    return parser.parse_args()


def main() -> int:
    options = parse_args()
    options.source = options.source.expanduser().resolve()
    options.old_plugin = options.old_plugin.expanduser().resolve()
    options.new_plugin = options.new_plugin.expanduser().resolve()
    options.vspipe = options.vspipe.expanduser().resolve()
    # Keep the VapourSynth launcher path spelling. It may be a symlink whose
    # surrounding environment is selected by the launcher directory.
    options.vs_python = options.vs_python.expanduser()
    options.e2e_runner = Path(__file__).with_name("e2e_benchmark.py").resolve()
    options.fixed_runner = Path(__file__).with_name(
        "fixed_kernel_benchmark.py").resolve()
    options.html = [item.expanduser().resolve() for item in options.html]
    options.scripts = {}
    for raw in options.script:
        case, value = raw.split("=", 1)
        options.scripts[case] = str(Path(value).expanduser().resolve())
    for required in (options.source, options.old_plugin, options.new_plugin,
                     options.vspipe, options.vs_python):
        if not required.is_file():
            raise FileNotFoundError(required)

    output_root = options.output_root.expanduser().resolve()
    if not options.skip_run:
        perf_results = {}
        for threads in THREADS:
            output = output_root / f"e2e-digimon-release-r{threads}t{threads}"
            run(e2e_args(options, output, threads),
                f"e2e performance R{threads}T{threads}")
            perf_results[threads] = read_json(output / "benchmark.json")

        error_output = output_root / "e2e-digimon-release-errors-r32t32"
        command = e2e_args(options, error_output, 32, skip_errors=False)
        command.extend([
            "--skip-performance",
            "--error-processes", "4",
            "--error-threads", "1",
        ])
        run(command, "e2e full error sweep")

        fixed_output = output_root / "fixed-kernel-digimon-810p-release"
        fixed_command = [
            options.python,
            str(options.fixed_runner),
            "--source", str(options.source),
            "--old-plugin", str(options.old_plugin),
            "--new-plugin", str(options.new_plugin),
            "--vspipe", str(options.vspipe),
            "--source-filter", options.source_filter,
            "--frames", "4000",
            "--src-height", "810",
            "--base-height", "1000",
            "--threads", *[str(item) for item in THREADS],
            "--runs", "1",
            "--implementations", "old", "new",
            "--kernels", *[name for name, _ in KERNELS],
            "--output", str(fixed_output),
        ]
        run(fixed_command, "fixed kernel full old/new")
    else:
        perf_results = {
            threads: read_json(output_root /
                               f"e2e-digimon-release-r{threads}t{threads}" /
                               "benchmark.json")
            for threads in THREADS
        }

    error_result = read_json(
        output_root / "e2e-digimon-release-errors-r32t32" / "benchmark.json")
    fixed_result = read_json(
        output_root / "fixed-kernel-digimon-810p-release" / "benchmark.json")
    merge_report(options, options.release_output.expanduser().resolve(),
                 perf_results, error_result, fixed_result)
    print(options.release_output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
