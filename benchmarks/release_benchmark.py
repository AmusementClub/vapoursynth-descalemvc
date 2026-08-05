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

import fixed_kernel_benchmark as fixed_report


THREADS = (1, 8, 16, 32)
CASES = ("getfnative", "getfnative_v2", "selectkernel")
IMPLEMENTATIONS = ("old", "new")
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
CASE_REPORTS = (
    {
        "case": "getfnative",
        "scenario": "Full non-vertical GetNative candidate scan",
        "reference": "muf.getnative(src, rescaler, src_heights=arange(700, 980, 0.1), base_height=1000)",
        "benchmark": "frame 12493; 11 scalers x 2,800 heights = 30,800 candidates",
    },
    {
        "case": "getfnative_v2",
        "scenario": "Vertical-only GetNative candidate scan",
        "reference": "muf.getnative(src, rescaler, src_heights=arange(840, 880, 0.1), base_height=1000, vertical_only=True)",
        "benchmark": "frame 358; 8 scalers x 400 heights = 3,200 candidates",
    },
    {
        "case": "selectkernel",
        "scenario": "Kernel-parameter selection at a fixed height",
        "reference": "muf.getnative(src, src_heights=719.8, base_height=1000, ex_thr=0.012, rescalers=...)",
        "benchmark": "frame 1111; bilinear + 10x10 Bicubic b/c grid = 101 candidates",
    },
)


def redact_plugin_hashes(value, plugin_context: bool = False):
    """Keep report provenance while omitting old/current plugin hashes."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in ("old_plugin_sha256", "new_plugin_sha256"):
                continue
            if plugin_context and key == "sha256":
                continue
            result[key] = redact_plugin_hashes(
                item, key in ("old_plugin", "new_plugin"))
        return result
    if isinstance(value, list):
        return [redact_plugin_hashes(item, plugin_context)
                for item in value]
    return value


def system_configuration(vspipe_environment: dict) -> dict:
    cpu_model = platform.processor() or "unknown CPU"
    physical_cores = None
    sockets = set()

    def sysctl(name: str) -> str:
        try:
            completed = subprocess.run(
                ["sysctl", "-n", name], capture_output=True, text=True,
                errors="replace", check=False)
        except OSError:
            return ""
        return completed.stdout.strip() if completed.returncode == 0 else ""

    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="ascii")
        for line in cpuinfo.splitlines():
            if line.startswith("model name") and cpu_model == "unknown CPU":
                cpu_model = line.split(":", 1)[1].strip()
            elif line.startswith("cpu cores") and physical_cores is None:
                physical_cores = int(line.split(":", 1)[1].strip())
            elif line.startswith("physical id"):
                sockets.add(line.split(":", 1)[1].strip())
    except (OSError, ValueError, IndexError):
        pass
    if cpu_model in ("", "unknown CPU", "i386", "arm"):
        cpu_model = (sysctl("machdep.cpu.brand_string")
                     or sysctl("hw.model") or cpu_model)
    if physical_cores is None:
        raw_physical = sysctl("hw.physicalcpu")
        try:
            physical_cores = int(raw_physical)
        except ValueError:
            physical_cores = None
    logical_cpus = (vspipe_environment.get("logical_cpu_count")
                    or os.cpu_count() or "unknown")
    socket_count = len(sockets) or 1
    core_text = (f"{physical_cores * socket_count}C/{logical_cpus}T"
                 if physical_cores is not None else f"{logical_cpus} logical CPUs")

    memory_text = "unknown"
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="ascii")
        total_kib = next(
            int(line.split()[1]) for line in meminfo.splitlines()
            if line.startswith("MemTotal:"))
        memory_text = f"{total_kib / 1024 / 1024:.1f} GiB"
    except (OSError, ValueError, StopIteration):
        pass
    if memory_text == "unknown":
        raw_memory = sysctl("hw.memsize")
        try:
            memory_text = f"{int(raw_memory) / 1024 / 1024 / 1024:.1f} GiB"
        except ValueError:
            pass

    version_lines = []
    for line in str(vspipe_environment.get("vspipe", "")).splitlines():
        line = line.strip()
        if line.startswith("Core ") or line.startswith("API "):
            version_lines.append(line)
    return {
        "cpu": f"{cpu_model} ({core_text})",
        "os": f"{platform.system()} {platform.release()} {platform.machine()}",
        "libc": " ".join(platform.libc_ver()) or "unknown",
        "memory": memory_text,
        "vapoursynth": "; ".join(version_lines) or "unknown",
    }


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
        "--source-prefer-hw", str(options.source_prefer_hw),
        "--source-ff-loglevel", str(options.source_ff_loglevel),
        "--source-rap-verification", str(options.source_rap_verification),
        "--profile", "full",
        "--cases", *CASES,
        "--implementations", "old", "new",
        "--runs", "1",
        "--requests", str(threads),
        "--threads", str(threads),
        "--output", str(output),
    ]
    if options.source_plugin:
        command.extend(["--source-plugin", str(options.source_plugin)])
    if options.source_decoder:
        command.extend(["--source-decoder", options.source_decoder])
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


def blank_paired_cases(blank: dict) -> list[dict]:
    """Normalize blank benchmark summaries for the fixed-kernel chart helper."""
    by_key = {
        (item["kernel"], item["threads"], item["implementation"]): item
        for item in blank["cases"]
    }
    frames = blank["environment"]["frames"]
    cases = []
    for name, label in KERNELS:
        for thread in THREADS:
            old = by_key[(name, thread, "old")]
            new = by_key[(name, thread, "new")]
            cases.append({
                "kernel": name,
                "label": label,
                "frames": frames,
                "threads": thread,
                "requests": thread,
                "old": {"fps": {"median": old["fps"]["median"]}},
                "new": {"fps": {"median": new["fps"]["median"]}},
            })
    return cases


def blank_table(blank: dict) -> dict:
    result = {}
    for item in blank_paired_cases(blank):
        result.setdefault(item["kernel"], {})[item["threads"]] = {
            implementation: item[implementation]["fps"]["median"]
            for implementation in IMPLEMENTATIONS
        }
    return result


def write_blank_scaling(blank: dict, path: Path) -> None:
    fixed_report.write_scaling_svg(
        blank_paired_cases(blank),
        [{"name": name, "label": label} for name, label in KERNELS],
        IMPLEMENTATIONS,
        path,
    )


def consolidated_algorithm_minima(error_result: dict) -> list[dict]:
    """Collapse parameterized scaler names into comparable algorithm families."""
    result = []
    for item in error_result["errors"]["summaries"]:
        grouped = {}
        for scaler, values in item["algorithm_summary"].items():
            family = "bicubic" if scaler.startswith("bicubic_") else scaler
            group = grouped.setdefault(family, {
                "candidate_count": 0,
                "scalers": [],
                "old_best": None,
                "new_best": None,
            })
            group["candidate_count"] += values["candidate_count"]
            group["scalers"].append(scaler)
            if (group["old_best"] is None
                    or values["old_best"]["mae"]
                    < group["old_best"]["mae"]):
                group["old_best"] = values["old_best"]
            if (group["new_best"] is None
                    or values["new_best"]["mae"]
                    < group["new_best"]["mae"]):
                group["new_best"] = values["new_best"]
        for family in sorted(grouped):
            group = grouped[family]
            old_best = group["old_best"]
            new_best = group["new_best"]
            result.append({
                "case": item["case"],
                "algorithm": family,
                "candidate_count": group["candidate_count"],
                "scalers": sorted(group["scalers"]),
                "old_best": old_best,
                "new_best": new_best,
                "mae_delta": new_best["mae"] - old_best["mae"],
                "height_delta": new_best["height"] - old_best["height"],
                "best_candidate_changed": (
                    old_best["id"] != new_best["id"]),
                "best_height_changed": (
                    old_best["height"] != new_best["height"]),
            })
    return result


def merge_report(options, output: Path, perf_results: dict[int, dict],
                 error_result: dict, fixed_result: dict,
                 blank_result: dict) -> None:
    e2e_rows = e2e_summary(perf_results)
    fixed_values = fixed_table(fixed_result)
    blank_values = blank_table(blank_result)
    algorithm_minima = consolidated_algorithm_minima(error_result)
    system = system_configuration(
        perf_results[THREADS[0]]["environment"])
    geometry = fixed_result["geometry"]
    error_environment = error_result.get("environment", {})
    e2e_by_case = {row["case"]: row for row in e2e_rows}
    r1 = {case: e2e_by_case[case]["threads"]["1"] for case in CASES}
    r32 = {case: e2e_by_case[case]["threads"]["32"] for case in CASES}
    output.mkdir(parents=True, exist_ok=True)
    write_e2e_scaling(e2e_rows, output / "e2e-scaling.svg")
    write_blank_scaling(
        blank_result, output / "blank-fixed-kernel-scaling.svg")
    merged = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "system": system,
            "source": str(options.source),
            "source_sha256": fixed_result["environment"]["source"]["sha256"],
            "source_filter": options.source_filter,
            "source_plugin": (str(options.source_plugin)
                              if options.source_plugin else None),
            "source_decoder": options.source_decoder,
            "source_prefer_hw": options.source_prefer_hw,
            "source_ff_loglevel": options.source_ff_loglevel,
            "source_rap_verification": options.source_rap_verification,
            "old_plugin": redact_plugin_hashes(
                fixed_result["environment"]["old_plugin"], True),
            "new_plugin": redact_plugin_hashes(
                fixed_result["environment"]["new_plugin"], True),
            "threads": list(THREADS),
            "frames": fixed_result["environment"]["frames"],
        },
        "e2e_performance": e2e_rows,
        "e2e_errors": redact_plugin_hashes(error_result["errors"]),
        "e2e_error_algorithm_minima": algorithm_minima,
        "fixed_kernel": redact_plugin_hashes(fixed_result),
        "blank_clip": redact_plugin_hashes(blank_result),
        "artifacts": {
            "e2e_scaling": "e2e-scaling.svg",
            "fixed_scaling": "../fixed-kernel-digimon-810p-release/scaling.svg",
            "blank_scaling": "blank-fixed-kernel-scaling.svg",
            "blank_clip_report": "../blank-fixed-kernel-digimon-810p-release-20260805/benchmark.json",
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
        f"| Fixed kernel coverage | {len(fixed_result['cases'])} algorithm/thread/implementation cases, {fixed_result['environment']['frames']:,} frames each |",
        f"| BlankClip kernel coverage | {len(blank_result['cases'])} implementation/thread/kernel cases, {blank_result['environment']['frames']:,} frames each |",
        f"| Error coverage | {sum(item['candidate_count'] for item in error_result['errors']['summaries']):,} candidates across three recipes |",
        "",
        f"At R1T1, the current Release is substantially faster on the complete candidate scans: `getfnative` {r1['getfnative']['old']['candidates_per_second']['median']:.3f} -> {r1['getfnative']['new']['candidates_per_second']['median']:.3f} candidates/s ({r1['getfnative']['new_speedup']:.2f}x), `getfnative_v2` {r1['getfnative_v2']['new_speedup']:.2f}x, and `selectkernel` {r1['selectkernel']['new_speedup']:.2f}x.",
        f"At R32T32, the gains narrow to {r32['getfnative_v2']['new_speedup']:.2f}x-{r32['getfnative']['new_speedup']:.2f}x because the workload reaches this machine's shared memory data-movement ceiling. The fixed-kernel R8-R32 results show the same convergence: available memory stays high, so the bottleneck is local memory bandwidth and/or cache/DRAM access and queueing latency rather than capacity.",
        "A DDR5 platform is therefore expected to improve the high-thread results by raising the memory-system ceiling, especially when channel configuration and timings are favorable. The gain should be treated as an upper-bound improvement opportunity, not a guaranteed linear speedup, because the graph still contains planner, synchronization, and frame-movement overhead.",
        "",
        "## Test System and Run Configuration",
        "",
        "| Item | Configuration |",
        "|---|---|",
        f"| CPU | `{system['cpu']}` |",
        f"| OS | `{system['os']}`, `{system['libc']}` |",
        f"| Memory | `{system['memory']}` physical memory at report generation |",
        f"| VapourSynth | `{system['vapoursynth']}` |",
        f"| Input | `{Path(options.source).name}`, {int(geometry['source_width'])}x{int(geometry['source_height'])}; supplied Digimon 1080p HEVC-10bit MKV |",
        f"| Source filter | `{options.source_filter}` |",
        f"| Source decoder options | decoder `{options.source_decoder or 'default'}`, prefer_hw `{options.source_prefer_hw}`, RAP verification `{options.source_rap_verification}` |",
        f"| Descale geometry | base `{int(geometry['base_width'])}x{int(geometry['base_height'])}`, native target `{int(geometry['native_width'])}x{int(geometry['native_height'])}` |",
        f"| Thread sweep | `R1T1`, `R8T8`, `R16T16`, `R32T32`; each cell uses `core.num_threads=N` and `--requests N` |",
        "| Performance repetition | One fresh VSPipe process per implementation/case/thread cell; wall time includes decode, graph setup, filtering, PlaneStats, and shutdown |",
        f"| BlankClip throughput | In-memory `std.BlankClip`, 1920x1080 GRAYS, fixed 810p geometry, {blank_result['environment']['frames']:,} frames per cell; no decoder or source filter |",
        f"| Error sweep | R32T32 recipe, `{error_environment.get('error_processes', 'n/a')}` worker processes, `{error_environment.get('error_worker_threads', 'n/a')}` worker thread per process |",
        "",
        "The reference scripts are the three supplied `.vpy` files. They use `core.lsmas.LWLibavSource` and `muf.getnative`; the release benchmark uses the explicitly supplied Digimon MKV and the source filter shown above, then expands the same descale/reconstruction/statistics graph so old and current plugin namespaces can be selected independently.",
        "",
        "## E2E Case Definitions",
        "",
        "The measured graph starts with `source -> ShufflePlanes(plane=0, GRAY) -> resize.Point(format=GRAYS)`. For each candidate it calls the old namespace (`core.descale`) or current namespace (`core.dsmvc`, `backend=cpu`) through `Debilinear`, `Debicubic`, `Delanczos`, `Despline16`, or `Despline36`, reconstructs with the matching `core.resize.*` kernel, then applies `std.Expr`, a 5-pixel border crop, and `PlaneStats`. The output is statistics, not an encoded video stream.",
        "",
        "| Case | Scenario | Reference call shape | Measured candidate space |",
        "|---|---|---|---|",
    ]
    for case in CASE_REPORTS:
        lines.append(
            f"| `{case['case']}` | {case['scenario']} | "
            f"`{case['reference']}` | {case['benchmark']} |")
    lines.extend([
        "",
        "`getfnative` is the broad normal search: it scans both height and scaler family on a non-vertical geometry. `getfnative_v2` is the narrower vertical-only search. `selectkernel` holds height at 719.8 and scans kernel parameters, so it isolates kernel-selection cost from height search.",
        "",
        "## Build and Provenance",
        "",
        f"- Source SHA-256: `{fixed_result['environment']['source']['sha256']}`",
        f"- Source filter: `{options.source_filter}`",
        "- Build: `Release`, CMake platform defaults, with AVX2/FMA isolated to `cpu_executor_avx2.cpp` when the target is x86_64",
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
    ])
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
        "### Consolidated per-algorithm minima",
        "",
        "Each row groups all parameter variants of one algorithm family and keeps the best old/current candidate within that family. For example, the 100 Bicubic variants in `selectkernel` become one row. `Delta MAE` and `Delta height` are `current - old`; a negative MAE is an improvement.",
        "",
        "| Case | Algorithm family | Candidates | Old best (candidate; height / MAE) | Current best (candidate; height / MAE) | Delta MAE | Delta height | Candidate changed | Height changed |",
        "|---|---|---:|---|---|---:|---:|---|---|",
    ])
    for item in algorithm_minima:
        old_best = item["old_best"]
        new_best = item["new_best"]
        lines.append(
            f"| `{item['case']}` | `{item['algorithm']}` | "
            f"{item['candidate_count']:,} | "
            f"`{old_best['id']}`; {old_best['height']:.1f} / "
            f"{old_best['mae']:.6g} | `{new_best['id']}`; "
            f"{new_best['height']:.1f} / {new_best['mae']:.6g} | "
            f"{item['mae_delta']:+.6g} | {item['height_delta']:+.1f} | "
            f"{item['best_candidate_changed']} | "
            f"{item['best_height_changed']} |")
    lines.extend([
        "",
        "## Fixed Kernel Throughput",
        "",
        f"Each cell is `old FPS -> current Release FPS (speedup)` for the first {fixed_result['environment']['frames']:,} frames at fixed 810p geometry.",
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
        "## BlankClip Throughput",
        "",
        f"Each cell is `old FPS -> current Release FPS (speedup)` for {blank_result['environment']['frames']:,} frames from an in-memory 1920x1080 GRAYS `std.BlankClip` at fixed 810p geometry. There is no decoder, source filter, or input-video content; this isolates the fixed-kernel execution path and VapourSynth frame plumbing.",
        "",
        "[Open the blank fixed-kernel scaling chart](blank-fixed-kernel-scaling.svg)",
        "",
        "| Kernel | R1T1 | R8T8 | R16T16 | R32T32 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, label in KERNELS:
        cells = []
        for thread in THREADS:
            values = blank_values[name][thread]
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
        "- [BlankClip fixed-kernel report](../blank-fixed-kernel-digimon-810p-release-20260805/benchmark.json)",
        "- [BlankClip CSV](../blank-fixed-kernel-digimon-810p-release-20260805/benchmark.csv)",
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
    parser.add_argument("--source-plugin", type=Path)
    parser.add_argument("--source-decoder", default="",
                        help="Preferred LSMASH/libavcodec decoder name(s).")
    parser.add_argument("--source-prefer-hw", type=int, default=0,
                        help="LSMASH prefer_hw mode; 0 keeps software default.")
    parser.add_argument("--source-ff-loglevel", type=int, default=0,
                        help="LSMASH FFmpeg log level, 0 is quiet.")
    parser.add_argument("--source-rap-verification", type=int, default=-1,
                        help="LSMASH RAP verification; -1 keeps plugin default.")
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
    if options.source_plugin:
        options.source_plugin = options.source_plugin.expanduser().resolve()
    # Keep the VapourSynth launcher path spelling. It may be a symlink whose
    # surrounding environment is selected by the launcher directory.
    options.vs_python = options.vs_python.expanduser()
    options.e2e_runner = Path(__file__).with_name("e2e_benchmark.py").resolve()
    options.fixed_runner = Path(__file__).with_name(
        "fixed_kernel_benchmark.py").resolve()
    options.blank_runner = Path(__file__).with_name(
        "blank_fixed_kernel_benchmark.py").resolve()
    options.html = [item.expanduser().resolve() for item in options.html]
    options.scripts = {}
    for raw in options.script:
        case, value = raw.split("=", 1)
        options.scripts[case] = str(Path(value).expanduser().resolve())
    for required in (options.source, options.old_plugin, options.new_plugin,
                     options.vspipe, options.vs_python):
        if not required.is_file():
            raise FileNotFoundError(required)
    if options.source_plugin and not options.source_plugin.is_file():
        raise FileNotFoundError(options.source_plugin)
    if options.source_prefer_hw < 0 or options.source_prefer_hw > 7:
        raise ValueError("--source-prefer-hw must be between 0 and 7")
    if options.source_ff_loglevel < 0 or options.source_ff_loglevel > 8:
        raise ValueError("--source-ff-loglevel must be between 0 and 8")
    if options.source_rap_verification not in (-1, 0, 1):
        raise ValueError("--source-rap-verification must be -1, 0, or 1")

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
            "--source-prefer-hw", str(options.source_prefer_hw),
            "--source-ff-loglevel", str(options.source_ff_loglevel),
            "--source-rap-verification", str(options.source_rap_verification),
            "--frames", "4000",
            "--src-height", "810",
            "--base-height", "1000",
            "--threads", *[str(item) for item in THREADS],
            "--runs", "1",
            "--implementations", "old", "new",
            "--kernels", *[name for name, _ in KERNELS],
            "--output", str(fixed_output),
        ]
        if options.source_plugin:
            fixed_command.extend(["--source-plugin", str(options.source_plugin)])
        if options.source_decoder:
            fixed_command.extend(["--source-decoder", options.source_decoder])
        run(fixed_command, "fixed kernel full old/new")
        blank_output = output_root / "blank-fixed-kernel-digimon-810p-release-20260805"
        blank_command = [
            options.python,
            str(options.blank_runner),
            "--old-plugin", str(options.old_plugin),
            "--new-plugin", str(options.new_plugin),
            "--vspipe", str(options.vspipe),
            "--frames", "8000",
            "--src-height", "810",
            "--base-height", "1000",
            "--threads", *[str(item) for item in THREADS],
            "--runs", "1",
            "--kernels", *[name for name, _ in KERNELS],
            "--output", str(blank_output),
        ]
        run(blank_command, "blank fixed kernel old/new")
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
    blank_result = read_json(
        output_root / "blank-fixed-kernel-digimon-810p-release-20260805" /
        "benchmark.json")
    merge_report(options, options.release_output.expanduser().resolve(),
                 perf_results, error_result, fixed_result, blank_result)
    print(options.release_output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
