#!/usr/bin/env python3
"""VapourSynth R57 integration and compatibility checks."""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import vapoursynth as vs


FUNCTIONS = {
    "Debilinear": {},
    "Debicubic": {"b": 0.0, "c": 1.0},
    "Delanczos": {"taps": 3},
    "Despline16": {},
    "Despline36": {},
    "Despline64": {},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None,
            f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_clips(old, new, label: str) -> None:
    require((old.width, old.height, old.format.id)
            == (new.width, new.height, new.format.id),
            f"{label}: output shape or format differs")
    old_frame = old.get_frame(0)
    new_frame = new.get_frame(0)
    maximum = 0.0
    total = 0.0
    count = 0
    for plane in range(old.format.num_planes):
        old_view = memoryview(old_frame[plane])
        new_view = memoryview(new_frame[plane])
        require(old_view.shape == new_view.shape,
                f"{label}: plane {plane} shape differs")
        for y in range(old_view.shape[0]):
            for x in range(old_view.shape[1]):
                difference = abs(float(old_view[y, x]) - float(new_view[y, x]))
                maximum = max(maximum, difference)
                total += difference
                count += 1
    mean = total / count if count else 0.0
    if old.format.sample_type == vs.FLOAT:
        require(maximum <= 2.0e-5 and mean <= 1.0e-6,
                f"{label}: float mismatch max={maximum} mean={mean}")
    else:
        require(maximum <= 1.0,
                f"{label}: integer mismatch max={maximum}")


def expect_error(callback, contains: str) -> None:
    try:
        callback()
    except vs.Error as error:
        require(contains.lower() in str(error).lower(),
                f"unexpected error: {error}")
        return
    raise AssertionError(f"expected an error containing {contains!r}")


def direct_call(namespace, name: str, source, **overrides):
    arguments = dict(width=80, height=48)
    arguments.update(FUNCTIONS.get(name, {}))
    arguments.update(overrides)
    return getattr(namespace, name)(source, **arguments)


def run(options) -> None:
    core = vs.core
    core.num_threads = options.threads
    if not hasattr(core, "descale"):
        core.std.LoadPlugin(path=str(Path(options.old_plugin).resolve()))
    core.std.LoadPlugin(path=str(Path(options.plugin).resolve()))

    matches = [plugin for plugin in core.plugins()
               if plugin.identifier == "com.dsmvc.descale"]
    require(len(matches) == 1, "new plugin ID was not registered exactly once")
    require(matches[0].namespace == "dsmvc", "plugin namespace is not dsmvc")
    require({function.name for function in matches[0].functions()} ==
            set(FUNCTIONS) | {"Descale"}, "public function set differs")
    for name in set(FUNCTIONS) | {"Descale"}:
        old_signature = getattr(core.descale, name).signature
        new_signature = getattr(core.dsmvc, name).signature
        require(new_signature == old_signature + "backend:data:opt;",
                f"{name}: signature differs from baseline plus backend")

    float_source = core.std.BlankClip(
        width=96, height=64, format=vs.GRAYS, color=[0.35])
    for name in FUNCTIONS:
        old = direct_call(core.descale, name, float_source)
        new = direct_call(core.dsmvc, name, float_source, backend="cpu")
        compare_clips(old, new, f"function/{name}")

    formats = (
        vs.GRAY8, vs.GRAY16, vs.GRAYS,
        vs.RGB24, vs.RGBS,
        vs.YUV420P10, vs.YUV444PS,
    )
    for format_id in formats:
        source = core.std.BlankClip(width=96, height=64, format=format_id)
        old = direct_call(core.descale, "Debicubic", source)
        new = direct_call(core.dsmvc, "Debicubic", source, backend="auto")
        compare_clips(old, new, f"format/{source.format.name}")

    geometry = {
        "src_left": 0.25,
        "src_top": 0.125,
        "src_width": 79.5,
        "src_height": 47.75,
    }
    for border in (0, 1, 2):
        old = direct_call(core.descale, "Debicubic", float_source,
                          border_handling=border, **geometry)
        new = direct_call(core.dsmvc, "Debicubic", float_source,
                          border_handling=border, backend="cpu", **geometry)
        compare_clips(old, new, f"border/{border}")

    identity_arguments = {"width": 96, "height": 64,
                          "src_width": 96.0, "src_height": 64.0}
    for flags in ({"force": 1}, {"force_h": 1}, {"force_v": 1}):
        old = core.descale.Debicubic(float_source, **identity_arguments, **flags)
        new = core.dsmvc.Debicubic(
            float_source, **identity_arguments, backend="cpu", **flags)
        compare_clips(old, new, "force/" + next(iter(flags)))

    custom_source = core.std.AddBorders(
        core.std.BlankClip(width=64, height=32, format=vs.GRAYS, color=[0.2]),
        left=16, right=16, top=16, bottom=16, color=[0.8])
    custom_kernel = lambda x: max(1.0 - abs(x) / 3.0, 0.0)
    custom_alias = lambda x: max(1.0 - abs(x), 0.0)
    custom_cases = (
        ("custom-kernel", {"custom_kernel": custom_kernel, "taps": 2}),
        ("custom-support", {"custom": custom_kernel, "support": 2}),
        ("custom-precedence", {
            "custom": custom_alias, "custom_kernel": custom_kernel,
            "support": 2, "taps": 1,
        }),
    )
    for label, arguments in custom_cases:
        old_custom = core.descale.Descale(
            custom_source, width=80, height=48, **arguments)
        new_custom = core.dsmvc.Descale(
            custom_source, width=80, height=48, backend="cpu", **arguments)
        compare_clips(old_custom, new_custom, label)

    scalar = core.dsmvc.Debicubic(
        float_source, width=80, height=48, opt=1, backend="cpu")
    avx2 = core.dsmvc.Debicubic(
        float_source, width=80, height=48, opt=2, backend="cpu")
    compare_clips(scalar, avx2, "opt/scalar-vs-avx2")

    for backend in ("metal", "vulkan", "cuda"):
        expect_error(
            lambda backend=backend: core.dsmvc.Debicubic(
                float_source, width=80, height=48, backend=backend),
            f"backend '{backend}' is not compiled")
    expect_error(
        lambda: core.dsmvc.Debicubic(
            float_source, width=80, height=48, backend="invalid"),
        "backend must be")
    expect_error(
        lambda: core.dsmvc.Descale(float_source, width=80, height=48),
        "kernel or custom kernel is required")
    expect_error(
        lambda: core.dsmvc.Debicubic(float_source, width=0, height=48),
        "width must be greater than zero")
    expect_error(
        lambda: core.dsmvc.Debicubic(float_source, width=80, height=7),
        "height must be at least 8")
    subsampled = core.std.BlankClip(width=96, height=64, format=vs.YUV420P10)
    expect_error(
        lambda: core.dsmvc.Debicubic(subsampled, width=79, height=48),
        "incompatible with subsampling")

    old_wrapper = load_module(
        "dsmvc_test_old_wrapper",
        Path(options.vs_root) / "VapourSynthScripts" / "descale.py")
    new_wrapper = load_module(
        "dsmvc_test_new_wrapper", Path(options.repo_root) / "python" / "dsmvc.py")
    rgb = core.std.BlankClip(width=96, height=64, format=vs.RGB24)
    compare_clips(
        old_wrapper.Debicubic(rgb, 80, 48, b=0.0, c=1.0),
        new_wrapper.Debicubic(rgb, 80, 48, b=0.0, c=1.0, backend="cpu"),
        "wrapper/RGB24")
    yuv = core.std.BlankClip(width=96, height=64, format=vs.YUV420P10)
    compare_clips(
        old_wrapper.Debicubic(yuv, 80, 48, gray=True),
        new_wrapper.Debicubic(yuv, 80, 48, gray=True, backend="cpu"),
        "wrapper/gray")
    compare_clips(
        old_wrapper.Debicubic(yuv, 80, 48, yuv444=True),
        new_wrapper.Debicubic(yuv, 80, 48, yuv444=True, backend="cpu"),
        "wrapper/yuv444")

    large_source = core.std.BlankClip(
        width=1920, height=1080, format=vs.RGBS, color=[0.2, 0.4, 0.6])
    for _ in range(5):
        for name, kernel_arguments in FUNCTIONS.items():
            large_output = getattr(core.dsmvc, name)(
                large_source, width=1692, height=952,
                src_left=0.2222222222221717, src_top=0.25,
                src_width=1691.5555555555557, src_height=951.5,
                backend="cpu", **kernel_arguments)
            large_output.get_frame(0)
            large_output.get_frame(0)

    print("dsmvc VapourSynth integration tests passed")


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    vs_root = Path(r"D:\okegui\OKEGui\tools\vapoursynth")
    result = argparse.ArgumentParser()
    result.add_argument("--plugin", default=str(root / "build" / "Release" / "dsmvc.dll"))
    result.add_argument("--old-plugin", default=str(
        vs_root / "vapoursynth64" / "plugins" / "descale.dll"))
    result.add_argument("--vs-root", default=str(vs_root))
    result.add_argument("--repo-root", default=str(root))
    result.add_argument("--threads", type=int, default=32)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
