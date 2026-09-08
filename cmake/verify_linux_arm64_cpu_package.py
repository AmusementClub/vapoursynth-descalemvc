"""Check a natively built Linux arm64 CPU-only install tree."""

import ctypes
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


build, package = map(Path, sys.argv[1:])
require(platform.system() == "Linux" and platform.machine() == "aarch64",
        "package verification requires a native Linux aarch64 runner")
cache = {}
for line in (build / "CMakeCache.txt").read_text().splitlines():
    if line and not line.startswith(("#", "//")) and "=" in line:
        key, value = line.split("=", 1)
        cache[key.split(":", 1)[0]] = value.upper()
for name in ("DSMVC_ENABLE_CUDA", "DSMVC_ENABLE_VULKAN", "DSMVC_ENABLE_METAL"):
    require(cache.get(name) in {"OFF", "FALSE", "0", "NO"}, f"{name} must be disabled")
require(cache.get("DSMVC_ENABLE_NATIVE_CPU_SIMD") in {"ON", "TRUE", "1", "YES"},
        "native CPU SIMD must be enabled")
require(any(build.rglob("cpu_executor_neon.cpp.o")), "NEON object was not built")

plugin = package / "vapoursynth" / "dsmvc.so"
binary = plugin.read_bytes()
require(binary[:6] == b"\x7fELF\x02\x01", "plugin must be ELF64 little-endian")
require(int.from_bytes(binary[18:20], "little") == 183, "plugin must target AArch64")
library = ctypes.CDLL(str(plugin.resolve()))
require(hasattr(library, "VapourSynthPluginInit2"), "API4 plugin entrypoint is missing")
dynamic = subprocess.check_output(["readelf", "--dynamic", "--wide", str(plugin)], text=True)
dependencies = re.findall(r"\(NEEDED\).*\[(.*?)\]", dynamic)
require(not any(re.search(r"cuda|vulkan|metal", name, re.IGNORECASE) for name in dependencies),
        f"CPU-only package has GPU dependencies: {dependencies}")
linked = subprocess.check_output(["ldd", str(plugin)], text=True)
require("not found" not in linked, f"package has unresolved dependencies: {linked}")
print(json.dumps({"machine": "aarch64", "backends": ["cpu"], "simd": "NEON",
                  "api4_entrypoint": True, "dependencies": dependencies,
                  "plugin_sha256": hashlib.sha256(binary).hexdigest()}, indent=2))
