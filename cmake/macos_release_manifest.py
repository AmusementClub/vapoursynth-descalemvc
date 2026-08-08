#!/usr/bin/env python3
"""Validate and describe an installed macOS arm64 dsmvc package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_PACKAGE_FILES = (
    "python/dsmvc.py",
    "share/dsmvc/LICENSE",
    "share/dsmvc/README.md",
    "share/dsmvc/THIRD_PARTY_NOTICES.md",
    "vapoursynth/dsmvc.so",
)


def run(*arguments: str) -> str:
    return subprocess.run(
        list(arguments), check=True, capture_output=True, text=True,
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, display_path: str) -> dict:
    return {
        "path": display_path,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "file": run("file", str(path)),
    }


def cmake_cache_value(cache: str, name: str) -> str | None:
    prefix = f"{name}:"
    for line in cache.splitlines():
        if line.startswith(prefix) and "=" in line:
            return line.split("=", 1)[1]
    return None


def metal_symbols(output: str) -> list[str]:
    symbols = []
    for line in output.splitlines():
        match = re.fullmatch(
            r"[0-9A-Fa-f]+\s+[Tt]\s+([A-Za-z_][A-Za-z0-9_]*)",
            line.strip(),
        )
        if match:
            symbols.append(match.group(1))
    return sorted(set(symbols))


def write_checksums(package_dir: Path, output: Path) -> None:
    entries = []
    for path in sorted(package_dir.rglob("*")):
        if path.is_file() and path != output:
            relative = path.relative_to(package_dir).as_posix()
            entries.append(f"{sha256_file(path)}  ./{relative}")
    output.write_text("\n".join(entries) + "\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--entrypoints", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--vapoursynth-sdk-ref", required=True)
    parser.add_argument("--vapoursynth-python", required=True, type=Path)
    parser.add_argument("--deployment-target", required=True)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--checksums-out", required=True, type=Path)
    parser.add_argument("--allow-dirty-source", action="store_true")
    options = parser.parse_args()

    repo = options.repo.expanduser().resolve()
    build_dir = options.build_dir.expanduser().resolve()
    package_dir = options.package_dir.expanduser().resolve()
    entrypoints_path = options.entrypoints.expanduser().resolve()
    json_out = options.json_out.expanduser().resolve()
    checksums_out = options.checksums_out.expanduser().resolve()
    # Preserve virtual-environment launcher symlinks: resolving them changes
    # Python's prefix discovery and can hide the environment's site-packages.
    vapoursynth_python = options.vapoursynth_python.expanduser().absolute()
    build_plugin = build_dir / "dsmvc.so"
    plugin = package_dir / "vapoursynth" / "dsmvc.so"
    metallib = (
        build_dir / "generated" / "metal-routes"
        / "dsmvc_metal_routes.metallib"
    )

    for path in (repo, build_dir, package_dir, entrypoints_path,
                 vapoursynth_python,
                 build_plugin, plugin, metallib):
        if not path.exists():
            raise FileNotFoundError(path)
    for relative in REQUIRED_PACKAGE_FILES:
        if not (package_dir / relative).is_file():
            raise FileNotFoundError(package_dir / relative)
    if json_out.parent != package_dir / "share" / "dsmvc":
        raise ValueError("manifest must be written under package/share/dsmvc")
    if checksums_out.parent != package_dir:
        raise ValueError("checksum ledger must be written at the package root")

    head = run("git", "-C", str(repo), "rev-parse", "HEAD")
    if head != options.source_revision:
        raise RuntimeError(
            f"source revision differs: expected {options.source_revision}, got {head}"
        )
    tracked_status = run(
        "git", "-C", str(repo), "status", "--porcelain",
        "--untracked-files=no",
    )
    if tracked_status and not options.allow_dirty_source:
        raise RuntimeError("tracked source changed during the CI build")

    cache_text = (build_dir / "CMakeCache.txt").read_text(encoding="utf-8")
    expected_cache = {
        "BUILD_TESTING": "ON",
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_OSX_ARCHITECTURES": "arm64",
        "CMAKE_OSX_DEPLOYMENT_TARGET": options.deployment_target,
        "DSMVC_ENABLE_CUDA": "OFF",
        "DSMVC_ENABLE_METAL": "ON",
    }
    actual_cache = {
        name: cmake_cache_value(cache_text, name) for name in expected_cache
    }
    if actual_cache != expected_cache:
        raise RuntimeError(
            f"CMake release contract differs: {actual_cache} != {expected_cache}"
        )

    build_sha = sha256_file(build_plugin)
    plugin_sha = sha256_file(plugin)
    if build_sha != plugin_sha:
        raise RuntimeError("installed plugin differs from the build-tree plugin")
    plugin_file = run("file", str(plugin))
    if "Mach-O 64-bit bundle arm64" not in plugin_file:
        raise RuntimeError(f"plugin is not a native arm64 bundle: {plugin_file}")
    architectures = run("lipo", "-archs", str(plugin)).split()
    if architectures != ["arm64"]:
        raise RuntimeError(f"plugin architectures differ: {architectures}")

    exports = sorted(set(run("nm", "-gjU", str(plugin)).splitlines()))
    expected_exports = ["_VapourSynthPluginInit2"]
    if exports != expected_exports:
        raise RuntimeError(
            f"plugin export surface differs: {exports} != {expected_exports}"
        )

    deployment_output = run("xcrun", "vtool", "-show-build", str(plugin))
    minos_match = re.search(r"^\s*minos\s+([0-9.]+)$", deployment_output,
                            re.MULTILINE)
    sdk_match = re.search(r"^\s*sdk\s+([0-9.]+)$", deployment_output,
                          re.MULTILINE)
    if not minos_match or not sdk_match:
        raise RuntimeError("plugin deployment metadata is missing")
    minos = minos_match.group(1)
    if minos != options.deployment_target:
        raise RuntimeError(
            f"plugin minos differs: {minos} != {options.deployment_target}"
        )

    expected_symbols = sorted({
        line.strip() for line in entrypoints_path.read_text(
            encoding="ascii").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    })
    metal_nm = run("xcrun", "--find", "metal-nm")
    actual_symbols = metal_symbols(run(metal_nm, str(metallib)))
    if len(expected_symbols) != 32 or actual_symbols != expected_symbols:
        raise RuntimeError(
            f"Metal inventory differs: {actual_symbols} != {expected_symbols}"
        )

    dependency_lines = run("otool", "-L", str(plugin)).splitlines()[1:]
    dependencies = [line.strip().split(" (", 1)[0]
                    for line in dependency_lines]
    non_system = [dependency for dependency in dependencies
                  if not dependency.startswith(("/System/Library/", "/usr/lib/"))]
    if non_system:
        raise RuntimeError(f"plugin has non-system dependencies: {non_system}")

    runtime = json.loads(run(
        str(vapoursynth_python), "-c",
        "import json,numpy,platform,sys,vapoursynth as vs;"
        "print(json.dumps({'python':sys.version.split()[0],"
        "'machine':platform.machine(),'numpy':numpy.__version__,"
        "'vapoursynth':str(vs.__version__),"
        "'vapoursynth_api':str(vs.__api_version__)}))",
    ))
    if runtime["machine"] != "arm64":
        raise RuntimeError(f"VapourSynth runtime is not arm64: {runtime}")

    json_out.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": "dsmvc-macos-release-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "revision": head,
            "tracked_source_clean": not bool(tracked_status),
        },
        "build": {
            "type": "Release",
            "architecture": "arm64",
            "deployment_target": options.deployment_target,
            "cmake_cache": actual_cache,
            "cmake": run("cmake", "--version").splitlines()[0],
            "xcode": run("xcodebuild", "-version").splitlines(),
            "macos_sdk": run("xcrun", "--sdk", "macosx", "--show-sdk-version"),
            "vapoursynth_sdk_ref": options.vapoursynth_sdk_ref,
            "test_runtime": runtime,
        },
        "plugin": {
            **file_identity(plugin, "vapoursynth/dsmvc.so"),
            "architectures": architectures,
            "exports": ["VapourSynthPluginInit2"],
            "dependencies": dependencies,
            "minos": minos,
            "sdk": sdk_match.group(1),
            "matches_build_tree": True,
        },
        "metallib": {
            **file_identity(
                metallib,
                "build/generated/metal-routes/dsmvc_metal_routes.metallib",
            ),
            "entrypoints": actual_symbols,
            "entrypoint_count": len(actual_symbols),
        },
    }
    json_out.write_text(
        json.dumps(document, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    write_checksums(package_dir, checksums_out)
    print(json.dumps({
        "manifest": str(json_out),
        "checksums": str(checksums_out),
        "plugin_sha256": plugin_sha,
        "metallib_sha256": document["metallib"]["sha256"],
        "metal_entrypoints": len(actual_symbols),
        "minos": minos,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
