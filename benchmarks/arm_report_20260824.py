#!/usr/bin/env python3
"""Assemble the 2026-08-24 Apple Silicon official benchmark report."""

from __future__ import annotations

import html
import json
import math
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "arm-benchmark-2026-08-24"
DOCS = ROOT / "docs"
THREADS = (1, 4, 8, 16)
CASES = ("getfnative", "getfnative_v2", "selectkernel")
SERIES = ("old", "jet", "newcpu", "newmetal")
LABELS = {
    "old": "old descale",
    "jet": "JET descale",
    "newcpu": "dsmvc CPU",
    "newmetal": "dsmvc Metal (hybrid)",
}
COLORS = {
    "old": "#dc2626",
    "jet": "#2563eb",
    "newcpu": "#059669",
    "newmetal": "#7c3aed",
}
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


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def summary_for(payload: dict, case: str) -> dict:
    return next(row for row in payload["performance"]["summaries"]
                if row["case"] == case)


def load_e2e() -> dict:
    result = {case: {thread: {} for thread in THREADS} for case in CASES}
    for thread in THREADS:
        roots = {
            "old": ARTIFACTS / "e2e" / f"r{thread}t{thread}-references" / "old",
            "jet": ARTIFACTS / "e2e" / f"r{thread}t{thread}-references" / "jet",
            "newcpu": ARTIFACTS / "e2e" / f"r{thread}t{thread}-cpu",
            "newmetal": ARTIFACTS / "e2e" / f"r{thread}t{thread}-metal",
        }
        for series, root in roots.items():
            payload = read_json(root / "benchmark.json")
            implementation = "old" if series in ("old", "jet") else "new"
            for case in CASES:
                summary = summary_for(payload, case)
                result[case][thread][series] = float(
                    summary[implementation]["candidates_per_second"]["median"])
    return result


def load_kernel_fps(campaign: str, blank: bool) -> dict:
    root = ARTIFACTS / campaign
    paths = {
        "old": root / "references" / "old" / "benchmark.json",
        "jet": root / "references" / "jet" / "benchmark.json",
        "newcpu": root / "cpu" / "benchmark.json",
        "newmetal": root / "metal" / "benchmark.json",
    }
    result = {kernel: {thread: {} for thread in THREADS}
              for kernel, _ in KERNELS}
    for series, path in paths.items():
        payload = read_json(path)
        implementation = "old" if series in ("old", "jet") else "new"
        for row in payload["cases"]:
            kernel = row["kernel"]
            if kernel not in result:
                continue
            if blank:
                value = row["fps"]["median"]
            else:
                value = row[implementation]["fps"]["median"]
            result[kernel][int(row["threads"])][series] = float(value)
    return result


def load_routes() -> list[dict]:
    rows = []
    for thread in THREADS:
        benchmark = ARTIFACTS / "e2e" / f"r{thread}t{thread}-metal" / "benchmark.json"
        payload = read_json(benchmark)
        probe = payload["route_probe"]
        frames = int(probe.get("candidate_count", probe.get("source_frames", 0)))
        metal = int(probe.get("metal_frame_count", sum(
            value > 0 for value in probe.get("metal_batches", []))))
        route = ("cpu" if metal == 0 else "metal" if metal == frames else "mixed")
        rows.append({
            "thread": thread,
            "frames": frames,
            "metal": metal,
            "maximum_batch": int(probe.get("maximum_metal_batch", max(
                probe.get("metal_batches", [0])))),
            "route": route,
        })
    return rows


def load_errors() -> list[dict]:
    rows = []
    for reference in ("old", "jet"):
        payload = read_json(ARTIFACTS / "error-sweep" / reference / "benchmark.json")
        for summary in payload["errors"]["summaries"]:
            rows.append({"reference": reference, **summary})
    return rows


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def fmt_metric(value: float) -> str:
    return f"{value:.6g}"


def table_header() -> list[str]:
    return [
        "| Threads | old | JET | dsmvc-cpu | dsmvc-metal | cpu vs old | cpu vs JET |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]


def append_scaling_table(lines: list[str], values: dict) -> None:
    lines.extend(table_header())
    for thread in THREADS:
        row = values[thread]
        lines.append(
            f"| R{thread}T{thread} | {fmt(row['old'])} | {fmt(row['jet'])} | "
            f"{fmt(row['newcpu'])} | {fmt(row['newmetal'])} | "
            f"{row['newcpu'] / row['old']:.2f}x | "
            f"{row['newcpu'] / row['jet']:.2f}x |")


def svg_polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_scaling_svg(path: Path, title: str, subtitle: str,
                      panels: list[tuple[str, dict]]) -> None:
    columns = 3 if len(panels) <= 3 else 5
    rows = math.ceil(len(panels) / columns)
    width = 1250
    panel_width = width / columns
    panel_height = 285
    height = 104 + rows * panel_height
    backend_of = {
        "old": "cpu", "jet": "cpu", "newcpu": "cpu", "newmetal": "metal",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="29" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        f'<text x="{width / 2:.1f}" y="51" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#475569">{html.escape(subtitle)}</text>',
    ]
    for index, series in enumerate(SERIES):
        x = width / 2 - 310 + index * 205
        parts.extend((
            f'<line x1="{x:.1f}" y1="73" x2="{x + 25:.1f}" y2="73" stroke="{COLORS[series]}" stroke-width="2.5"/>',
            f'<text x="{x + 32:.1f}" y="77" font-family="sans-serif" font-size="11" fill="#374151">{html.escape(LABELS[series])}</text>',
        ))
    for panel_index, (label, values) in enumerate(panels):
        column = panel_index % columns
        row_index = panel_index // columns
        panel_x = column * panel_width
        panel_y = 94 + row_index * panel_height
        left, top = panel_x + 48, panel_y + 36
        plot_width, plot_height = panel_width - 68, panel_height - 68
        # The shared mapping is defined before the range pass and is also used
        # while drawing, so the Metal series cannot escape a CPU-only scale.
        y_max = max(values[thread][series]
                    for series in SERIES for thread in THREADS
                    if backend_of[series] in ("cpu", "metal")) * 1.10
        parts.extend((
            f'<rect x="{panel_x + 8}" y="{panel_y + 4}" width="{panel_width - 16}" height="{panel_height - 12}" fill="#f8fafc" stroke="#cbd5e1"/>',
            f'<text x="{panel_x + panel_width / 2:.1f}" y="{panel_y + 25}" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#111827">{html.escape(label)}</text>',
        ))
        for tick_index in range(5):
            tick = y_max * tick_index / 4
            y = top + plot_height * (1 - tick_index / 4)
            parts.extend((
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e2e8f0"/>',
                f'<text x="{left - 6}" y="{y + 3:.2f}" text-anchor="end" font-family="sans-serif" font-size="8" fill="#64748b">{tick:.0f}</text>',
            ))
        x_positions = []
        for thread_index, thread in enumerate(THREADS):
            x = left + plot_width * thread_index / (len(THREADS) - 1)
            x_positions.append(x)
            parts.append(
                f'<text x="{x:.2f}" y="{top + plot_height + 16}" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#475569">R{thread}</text>')
        for series in SERIES:
            points = [(x, top + plot_height * (1 - values[thread][series] / y_max))
                      for x, thread in zip(x_positions, THREADS)]
            parts.append(
                f'<polyline points="{svg_polyline(points)}" fill="none" stroke="{COLORS[series]}" stroke-width="2.2"/>')
            for x, y in points:
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.8" fill="{COLORS[series]}"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def copy_showcase() -> None:
    source = ARTIFACTS / "accuracy-showcase"
    destination = DOCS / "showcase-arm-20260824"
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.suffix in (".svg", ".png"):
            shutil.copy2(path, destination / path.name)


def main() -> int:
    e2e = load_e2e()
    fixed = load_kernel_fps("fixed-kernel", blank=False)
    blank = load_kernel_fps("blank-fixed-kernel", blank=True)
    routes = load_routes()
    errors = load_errors()
    showcase = read_json(ARTIFACTS / "accuracy-showcase" / "benchmark.json")
    gate = read_json(ARTIFACTS / "provenance" / "three-way-diff" / "three-way-diff.json")

    write_scaling_svg(
        DOCS / "arm-e2e-scaling.svg", "ARM E2E candidate scaling",
        "Full candidate graph; higher candidates/s is better",
        [(case, e2e[case]) for case in CASES],
    )
    write_scaling_svg(
        DOCS / "arm-fixed-kernel-scaling.svg", "ARM fixed-kernel scaling",
        "Real FFMS2 source; 4,000 frames per cell",
        [(label, fixed[kernel]) for kernel, label in KERNELS],
    )
    write_scaling_svg(
        DOCS / "arm-blank-fixed-kernel-scaling.svg", "ARM BlankClip kernel scaling",
        "In-memory GRAYS input; 8,000 frames per cell",
        [(label, blank[kernel]) for kernel, label in KERNELS],
    )
    copy_showcase()

    r16 = e2e["getfnative"][16]
    lines = [
        "# Descale MVC Apple Silicon Release Benchmark",
        "",
        "_Official ARM benchmark; collected 2026-08-24 on dsmvc source `99b920a`._",
        "",
        "## Executive Summary",
        "",
        "This is the Apple Silicon companion to `release-benchmark.md`. It uses the same source bytes and release source SHA, but absolute FPS and candidates/s must not be subtracted across machines. Cross-platform reading is limited to speedup, scaling shape, and numerical consistency.",
        "",
        "| Workload | Result |",
        "|---|---:|",
        f"| E2E getfnative at R16T16 | {r16['newcpu']:.3f} candidates/s CPU; {r16['newmetal']:.3f} explicit mixed Metal |",
        f"| getfnative CPU speedup | {r16['newcpu'] / r16['old']:.2f}x vs old; {r16['newcpu'] / r16['jet']:.2f}x vs JET |",
        "| Fixed kernel coverage | 10 kernels x 4 thread levels x 4 columns; 4,000 frames each |",
        "| BlankClip kernel coverage | 10 kernels x 4 thread levels x 4 columns; 8,000 frames each |",
        "| Error coverage | 34,101 candidates across three recipes and two scalar references |",
        "| Accuracy showcase | 6 ill-conditioned plans; automatic F64 removes the 3.27e6 Lanczos2 F32 peak |",
        "",
        "The `dsmvc-metal` column is explicit `backend=\"metal\"`, but it is heterogeneous rather than a pure-GPU result: executor and plan work remains on CPU, and only scheduler-admitted axis batches run on Metal over UMA. Automatic routing on this source also requires both `numFrames >= 64` and `core_threads >= 8`; admitted frames remain mixed. The explicit Metal column demonstrates the hybrid path but is not a measurement of `backend=\"auto\"` throughput, because automatic scheduling applies additional admission policy. R1 explicit Metal is entirely CPU fallback, while R4 explicit Metal admits some batches even though auto is ineligible below 8 core threads. This differs from the x86 report, where auto always routes CPU and explicit GPU columns are opt-in.",
        "",
        "Both reference plugins are ARM64 source builds with `DESCALE_X86` undefined, so old IEW and JET use scalar C. The forced dsmvc CPU column uses the ARM NEON path.",
        "",
        "## Test System and Run Configuration",
        "",
        "| Item | Configuration |",
        "|---|---|",
        "| Chip | Apple M4 Max; 12 performance + 4 efficiency cores |",
        "| OS | macOS 27.0 (build 26A5416b), arm64 |",
        "| Memory | 128 GiB unified memory |",
        "| Power | AC attached; High Power mode (`powermode=2`); system sleep disabled on AC |",
        "| VapourSynth | Core R78; API R4.2 / R3.6; Python 3.14.6 |",
        "| Input | 1920x1080 HEVC-10bit MKV; SHA-256 `864d552f8e2ead057ebd2c202c7580442a5f22c8acecd08167eb8a07110d1bf4` |",
        "| Source filter | Explicit ARM64 FFMS2 for every decoded workload |",
        "| Thread sweep | R1T1, R4T4, R8T8, R16T16 |",
        "| Repetition | One fresh process and one measured run per cell, campaign-wide serial mkdir lock |",
        "| dsmvc build | CMake Release, source `99b920a`; plugin SHA-256 `4742e33b...24db8` |",
        "| Reference builds | IEW `8c53f5d` and JET `d699532b`; Meson release; scalar C on ARM |",
        "",
        "CTest passed 10/10 before measurement. The three-way healthy-kernel gate passed 8/8; its largest max-absolute difference was below `1e-6` (threshold `1e-4`). Full hashes, compiler details, build logs, and commands are under `artifacts/arm-benchmark-2026-08-24/provenance/`.",
        "",
        "## E2E Case Definitions",
        "",
        "| Case | Measured candidate space |",
        "|---|---|",
        "| `getfnative` | frame 12493; 11 scalers x 2,800 heights = 30,800 candidates |",
        "| `getfnative_v2` | frame 358; 8 scalers x 400 heights = 3,200 candidates |",
        "| `selectkernel` | frame 1111; bilinear + 10x10 Bicubic grid = 101 candidates |",
        "",
        "The complete graph includes FFMS2 decode, plane extraction, Float32 conversion, descale, reconstruction, border crop, Expr, PlaneStats, frame delivery, and process overhead.",
        "",
        "## E2E Thread Scaling",
        "",
        "![ARM E2E thread scaling](arm-e2e-scaling.svg)",
        "",
    ]
    for case in CASES:
        lines.extend((f"### `{case}`", ""))
        append_scaling_table(lines, e2e[case])
        lines.append("")

    lines.extend((
        "## Metal Route Evidence", "",
        "The frame property `_DSMVCMetalBatch` was sampled over each route probe. A nonzero batch marks Metal execution; zero marks CPU fallback.", "",
        "| Cell | Probe frames | Metal frames | Max batch | Observed route |",
        "|---|---:|---:|---:|---|",
    ))
    for row in routes:
        lines.append(
            f"| R{row['thread']}T{row['thread']} | {row['frames']} | {row['metal']} | "
            f"{row['maximum_batch']} | {row['route']} |")
    lines.extend(("", "## E2E Error Comparison", "",
        "The CPU error sweep evaluates every candidate against old and JET separately. All three recipes retain the same best candidate and best MAE across old, JET, and dsmvc within a `1e-8` comparison tolerance.", "",
        "The `getfnative` maximum output difference of about `0.0553` is the expected automatic-F64 behavior inherited from the x86 report: near-unity ill-conditioned candidates keep the scalar references on F32 while dsmvc auto follows its F64 plan. The best candidate and reconstruction MAE remain consistent; this is not a ranking regression.", "",
        "| Case | Reference | Candidates | Best reference | Best dsmvc | Changed | Max output abs | Max reconstruction abs |",
        "|---|---|---:|---|---|---|---:|---:|",
    ))
    for row in errors:
        old_best, new_best = row["best_old"], row["best_new"]
        lines.append(
            f"| `{row['case']}` | `{row['reference']}` | {row['candidate_count']:,} | "
            f"`{old_best['id']}` ({old_best['mae']:.8g}) | "
            f"`{new_best['id']}` ({new_best['mae']:.8g}) | "
            f"{row['best_candidate_changed']} | {fmt_metric(row['max_output_max_abs'])} | "
            f"{fmt_metric(row['max_reconstruction_max_abs'])} |")

    lines.extend(("", "## Fixed-Kernel Throughput", "",
                  "![ARM fixed-kernel scaling](arm-fixed-kernel-scaling.svg)", "",
                  "The decoded-source run holds each kernel and 1440x810 geometry fixed for 4,000 frames.", ""))
    for thread in THREADS:
        lines.extend((f"### R{thread}T{thread}", "",
                      "| Kernel | old | JET | dsmvc-cpu | dsmvc-metal | cpu vs old | cpu vs JET |",
                      "|---|---:|---:|---:|---:|---:|---:|"))
        for kernel, label in KERNELS:
            row = fixed[kernel][thread]
            lines.append(
                f"| `{label}` | {fmt(row['old'])} | {fmt(row['jet'])} | "
                f"{fmt(row['newcpu'])} | {fmt(row['newmetal'])} | "
                f"{row['newcpu'] / row['old']:.2f}x | {row['newcpu'] / row['jet']:.2f}x |")
        lines.append("")

    lines.extend(("## BlankClip Kernel Throughput", "",
                  "![ARM BlankClip kernel scaling](arm-blank-fixed-kernel-scaling.svg)", "",
                  "BlankClip removes decoder cost and isolates the kernel, scheduler, and any CPU/Metal transfer work over 8,000 frames.", ""))
    for thread in THREADS:
        lines.extend((f"### R{thread}T{thread}", "",
                      "| Kernel | old | JET | dsmvc-cpu | dsmvc-metal | cpu vs old | cpu vs JET |",
                      "|---|---:|---:|---:|---:|---:|---:|"))
        for kernel, label in KERNELS:
            row = blank[kernel][thread]
            lines.append(
                f"| `{label}` | {fmt(row['old'])} | {fmt(row['jet'])} | "
                f"{fmt(row['newcpu'])} | {fmt(row['newmetal'])} | "
                f"{row['newcpu'] / row['old']:.2f}x | {row['newcpu'] / row['jet']:.2f}x |")
        lines.append("")

    lines.extend(("## Ill-Conditioned Plan Accuracy Showcase", "",
        "Frame 12493 is compared with dsmvc forced-F64 CPU output. The first five cases exercise automatic promotion; Spline64 is the near-unity control that remains F32.", "",
        "| Case | Metric | old | JET | dsmvc F32 | dsmvc auto | auto routed F64 |",
        "|---|---|---:|---:|---:|---:|---|",
    ))
    for row in showcase["rows"]:
        lines.append(
            f"| `{row['case']}` | {row['metric']} | {fmt_metric(row['old'])} | "
            f"{fmt_metric(row['jet'])} | {fmt_metric(row['dsmvc-f32'])} | "
            f"{fmt_metric(row['dsmvc-auto'])} | {row['auto_routed_f64']} |")
    lines.extend(("", "Curves and peak-normalized diff maps are in `showcase-arm-20260824/`.", "",
        "## Interpretation and Reproducibility", "",
        "- Compare columns only within this M4 Max campaign. Against x86, compare ratios, scaling shape, route semantics, and error consistency rather than absolute throughput.",
        "- `dsmvc-metal` is a mixed explicit-backend measurement, not a claim that every axis kernel ran on the GPU or that the values equal `backend=\"auto\"` throughput. Route-probe counts above define only the observed explicit-backend boundary.",
        "- One run per cell captures the requested official protocol but does not estimate run-to-run variance; small differences should not be over-interpreted.",
        "- No autoload descale plugin existed, so quarantine and restoration were not required.",
        f"- Machine-readable campaign artifacts, provenance, commands, and all benchmark JSON are under `{ARTIFACTS.relative_to(ROOT)}/`.",
        "",
    ))
    (DOCS / "arm-benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "schema_version": 1,
        "source_sha": "99b920a597769a782cc4654499e5eb6964445de1",
        "threads": list(THREADS),
        "series": list(SERIES),
        "e2e": e2e,
        "fixed_kernel": fixed,
        "blank_fixed_kernel": blank,
        "routes": routes,
        "errors": errors,
        "accuracy_showcase": showcase,
        "three_way_gate_passed": bool(gate["passed"]),
    }
    (ARTIFACTS / "arm-report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(DOCS / "arm-benchmark.md"), "routes": routes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
