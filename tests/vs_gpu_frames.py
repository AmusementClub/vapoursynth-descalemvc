"""GPU frame (R81, API 4.3) checks: dsmvc on a GPU resident clip against the
same filter on host frames. Run with an R81 python:  vs_gpu_frames.py <dsmvc plugin>"""
import sys
import numpy as np
import vapoursynth as vs

core = vs.core
core.std.LoadPlugin(sys.argv[1])
FRAMES = 6


def patterned(width, height, fmt):
    blank = core.std.BlankClip(width=width, height=height, format=fmt, length=FRAMES)

    def fill(n, f):
        out = f.copy()
        rng = np.random.default_rng(1000 + n)
        for p in range(out.format.num_planes):
            a = np.asarray(out[p])
            a[:] = (0.25 + 0.5 * rng.random(a.shape, dtype=np.float32)).astype(a.dtype)
        return out

    return core.std.ModifyFrame(blank, blank, fill)


def planes(clip, n):
    f = clip.get_frame(n)
    return [np.array(f[p], copy=True) for p in range(f.format.num_planes)]


def check(name, make, backend="vulkan"):
    src = patterned(1920, 1080, vs.GRAYS if "rgb" not in name and "yuv" not in name
                    else (vs.RGBS if "rgb" in name else vs.YUV444PS))
    host = make(src, backend)                          # frames on the host, own device
    gpu = make(src.std.GPUUpload(), backend).std.GPUDownload()   # frames stay on the GPU
    worst = 0.0
    for n in range(FRAMES):
        for a, b in zip(planes(host, n), planes(gpu, n)):
            assert a.shape == b.shape, (a.shape, b.shape)
            worst = max(worst, float(np.max(np.abs(a - b))))
    print(f"{name:34s} max |host - gpu frames| = {worst:.3g}")
    assert worst == 0.0, f"{name}: GPU frames differ from the host path"


check("2D bicubic gray", lambda c, b: c.dsmvc.Debicubic(c.width if False else 1280, 720, backend=b))
check("rows only (width)", lambda c, b: c.dsmvc.Debicubic(1280, 1080, backend=b))
check("columns only (height)", lambda c, b: c.dsmvc.Debicubic(1920, 720, backend=b))
check("2D lanczos rgb", lambda c, b: c.dsmvc.Delanczos(1280, 720, taps=3, backend=b))
check("2D spline36 yuv444", lambda c, b: c.dsmvc.Despline36(1280, 720, backend=b))


# CUDA reads and writes the same GPU frames through exported memory.
check("cuda 2D bicubic gray", lambda c, b: c.dsmvc.Debicubic(1280, 720, backend=b), "cuda")
check("cuda rows only", lambda c, b: c.dsmvc.Debicubic(1280, 1080, backend=b), "cuda")
check("cuda columns only", lambda c, b: c.dsmvc.Debicubic(1920, 720, backend=b), "cuda")
check("cuda 2D lanczos rgb", lambda c, b: c.dsmvc.Delanczos(1280, 720, taps=3, backend=b), "cuda")
check("cuda 2D spline36 yuv444", lambda c, b: c.dsmvc.Despline36(1280, 720, backend=b), "cuda")
# f64mode=2 forces the Float64 solve (f64mode=1 forces Float32).
for backend in ("vulkan", "cuda"):
    check(f"{backend} Float64 2D (f64mode=2)", lambda c, b: c.dsmvc.Debicubic(1280, 720, f64mode=2, backend=b), backend)
    check(f"{backend} Float64 rows (f64mode=2)", lambda c, b: c.dsmvc.Debicubic(1280, 1080, f64mode=2, backend=b), backend)
    check(f"{backend} Float64 columns (f64mode=2)", lambda c, b: c.dsmvc.Debicubic(1920, 720, f64mode=2, backend=b), backend)

src = patterned(1920, 1080, vs.GRAYS)
gpu = src.std.GPUUpload()
mixed = (gpu.dsmvc.Debicubic(1600, 900, backend="vulkan")
            .dsmvc.Delanczos(1440, 810, taps=3, backend="cuda")
            .dsmvc.Despline36(1280, 720, backend="vulkan")).std.GPUDownload()
mixed_ref = (src.dsmvc.Debicubic(1600, 900, backend="vulkan")
                .dsmvc.Delanczos(1440, 810, taps=3, backend="cuda")
                .dsmvc.Despline36(1280, 720, backend="vulkan"))
for n in range(FRAMES):
    assert np.array_equal(planes(mixed, n)[0], planes(mixed_ref, n)[0]), f"vulkan>cuda>vulkan frame {n}"
print("vulkan -> cuda -> vulkan chain       identical to the host chain (no host copies in between)")
cuda_chain = (gpu.dsmvc.Debicubic(1600, 900, backend="cuda")
                 .dsmvc.Delanczos(1280, 720, taps=3, backend="cuda")).std.GPUDownload()
cuda_ref = (src.dsmvc.Debicubic(1600, 900, backend="cuda")
               .dsmvc.Delanczos(1280, 720, taps=3, backend="cuda"))
for n in range(FRAMES):
    assert np.array_equal(planes(cuda_chain, n)[0], planes(cuda_ref, n)[0]), f"cuda>cuda frame {n}"
print("cuda -> cuda chain                   identical to the host chain")


# Fused 8-16 bit integer descale on GPU frames (both axes change).
def patterned_int(width, height, fmt, limited=True):
    blank = core.std.BlankClip(width=width, height=height, format=fmt, length=FRAMES)

    def fill(n, f):
        out = f.copy()
        rng = np.random.default_rng(2000 + n)
        top = (1 << core.get_video_format(fmt).bits_per_sample) - 1
        lo, hi = (top * 16 // 255, top * 235 // 255) if limited else (0, top)
        for p in range(out.format.num_planes):
            a = np.asarray(out[p])
            a[:] = rng.integers(lo, hi + 1, a.shape, dtype=np.int64).astype(a.dtype)
        return out

    clip = core.std.ModifyFrame(blank, blank, fill)
    return core.std.SetFrameProps(clip, _ColorRange=1 if limited else 0)


def check_int(name, fmt, make, backend, width=1920, height=1080, limited=True):
    src = patterned_int(width, height, fmt, limited)
    host = make(src, backend)
    gpu = make(src.std.GPUUpload(), backend).std.GPUDownload()
    for n in range(FRAMES):
        for a, b in zip(planes(host, n), planes(gpu, n)):
            assert np.array_equal(a, b), f"{name} ({backend}) frame {n}: GPU frames differ"
        assert host.get_frame(n).props["_Range"] == gpu.get_frame(n).props["_Range"], f"{name}: _Range"
    print(f"{name:34s} identical to the host path")


for backend in ("vulkan", "cuda"):
    check_int(f"{backend} yuv420p8 limited", vs.YUV420P8,
              lambda c, b: c.dsmvc.Debicubic(1280, 720, backend=b), backend)
    check_int(f"{backend} yuv420p10 limited", vs.YUV420P10,
              lambda c, b: c.dsmvc.Delanczos(1280, 720, taps=3, backend=b), backend)
    check_int(f"{backend} yuv444p16 full", vs.YUV444P16,
              lambda c, b: c.dsmvc.Despline36(1280, 720, backend=b), backend, limited=False)
    check_int(f"{backend} gray8 (odd byte count)", vs.GRAY8,
              lambda c, b: c.dsmvc.Debicubic(1102, 622, backend=b), backend, width=1670, height=941)
    check_int(f"{backend} yuv420p8 Float64 (f64mode=2)", vs.YUV420P8,
              lambda c, b: c.dsmvc.Debicubic(1280, 720, f64mode=2, backend=b), backend)
    check_int(f"{backend} gray8 odd bytes Float64", vs.GRAY8,
              lambda c, b: c.dsmvc.Debicubic(1102, 622, f64mode=2, backend=b), backend, width=1670, height=941)
    check_int(f"{backend} rgb24 full", vs.RGB24,
              lambda c, b: c.dsmvc.Debicubic(1280, 720, backend=b), backend, limited=False)


# The Float64 nonfinite-status contract holds on GPU frames: an infinite sample fails the frame.
def with_infinity(width, height):
    blank = core.std.BlankClip(width=width, height=height, format=vs.GRAYS, length=2, color=[0.5])

    def poke(n, f):
        out = f.copy()
        np.asarray(out[0])[height // 2, width // 2] = np.inf
        return out

    return core.std.ModifyFrame(blank, blank, poke)


for backend in ("vulkan", "cuda"):
    bad = with_infinity(1920, 1080)
    for label, clip in (("host frames", bad), ("GPU frames", bad.std.GPUUpload())):
        out = clip.dsmvc.Debicubic(1280, 720, f64mode=2, backend=backend)
        out = out.std.GPUDownload() if label == "GPU frames" else out
        try:
            out.get_frame(0)
        except vs.Error as e:
            assert any(k in str(e).lower() for k in ("finite", "nan or infinity")), str(e)
            print(f"{backend} Float64 nonfinite on {label:11s} -> error: {str(e)[-70:]}")
        else:
            raise AssertionError(f"{backend} {label}: an infinite sample must fail the Float64 solve")

# The residency of the input chooses the path, and the result of a chain stays resident.
src = patterned(1920, 1080, vs.GRAYS)
gpu = src.std.GPUUpload()
a = gpu.dsmvc.Debicubic(1600, 900, backend="vulkan")
b = a.dsmvc.Delanczos(1280, 720, taps=3, backend="auto")
chain = b.std.GPUDownload()
ref = src.dsmvc.Debicubic(1600, 900, backend="vulkan").dsmvc.Delanczos(1280, 720, taps=3, backend="vulkan")
for n in range(FRAMES):
    assert np.array_equal(planes(chain, n)[0], planes(ref, n)[0]), f"chain frame {n}"
print("resident chain (bicubic -> lanczos)  identical to the host chain")

# Explicit cpu is a deliberate download boundary.
cpu_on_gpu = gpu.dsmvc.Debicubic(1280, 720, backend="cpu")
cpu_ref = src.dsmvc.Debicubic(1280, 720, backend="cpu")
for n in range(FRAMES):
    assert np.array_equal(planes(cpu_on_gpu, n)[0], planes(cpu_ref, n)[0]), f"cpu boundary frame {n}"
print("backend=cpu on a GPU clip            identical to the cpu result")

# Unsupported requests fail with a remedy, never silently on the CPU.
try:
    core.std.BlankClip(width=1920, height=1080, format=vs.GRAY16, length=2) \
        .std.GPUUpload().dsmvc.Debicubic(1280, 1080, backend="vulkan")
except vs.Error as e:
    print(f"{'one-axis integer GPU clip':36s} -> error: {str(e)[:100]}")
else:
    raise AssertionError("one-axis integer GPU clip: expected an error")
print("OK")

# ---- stress: fan-out, out-of-order requests, concurrency, a small frame cache ----
import hashlib

FRAMES_LONG = 120


def digest(clip, prefetch=1):
    h = hashlib.sha256()
    for frame in clip.frames(prefetch=prefetch, backlog=prefetch * 2):
        for p in range(frame.format.num_planes):
            h.update(np.ascontiguousarray(frame[p]).tobytes())
    return h.hexdigest()


def long_source():
    blank = core.std.BlankClip(width=1920, height=1080, format=vs.GRAYS, length=FRAMES_LONG)

    def fill(n, f):
        out = f.copy()
        a = np.asarray(out[0])
        a[:] = 0.25 + 0.5 * np.random.default_rng(n).random(a.shape, dtype=np.float32)
        return out

    return core.std.ModifyFrame(blank, blank, fill)


core.max_cache_size = 128          # MB: forces the core to recycle GPU frames
src = long_source()
gpu = src.std.GPUUpload()
trunk = gpu.dsmvc.Debicubic(1600, 900, backend="vulkan")
branches = {
    "lanczos": trunk.dsmvc.Delanczos(1280, 720, taps=3, backend="vulkan"),
    "spline36": trunk.dsmvc.Despline36(1280, 720, backend="vulkan"),
    "rows": trunk.dsmvc.Debicubic(1280, 900, backend="vulkan"),
}
host_trunk = src.dsmvc.Debicubic(1600, 900, backend="vulkan")
refs = {
    "lanczos": host_trunk.dsmvc.Delanczos(1280, 720, taps=3, backend="vulkan"),
    "spline36": host_trunk.dsmvc.Despline36(1280, 720, backend="vulkan"),
    "rows": host_trunk.dsmvc.Debicubic(1280, 900, backend="vulkan"),
}
expected = {k: digest(v) for k, v in refs.items()}
for name, branch in branches.items():
    got = digest(branch.std.GPUDownload(), prefetch=16)    # 16 requests in flight
    assert got == expected[name], f"fan-out branch {name} differs under concurrency"
print("fan-out x3 under 16 concurrent requests, 128 MB frame cache: identical")

cuda_trunk = gpu.dsmvc.Debicubic(1600, 900, backend="cuda")
cuda_branches = {
    "lanczos": cuda_trunk.dsmvc.Delanczos(1280, 720, taps=3, backend="cuda"),
    "rows": cuda_trunk.dsmvc.Debicubic(1280, 900, backend="vulkan"),   # CUDA output read by Vulkan
}
cuda_host_trunk = src.dsmvc.Debicubic(1600, 900, backend="cuda")
cuda_refs = {
    "lanczos": cuda_host_trunk.dsmvc.Delanczos(1280, 720, taps=3, backend="cuda"),
    "rows": cuda_host_trunk.dsmvc.Debicubic(1280, 900, backend="vulkan"),
}
for name, branch in cuda_branches.items():
    assert digest(branch.std.GPUDownload(), prefetch=16) == digest(cuda_refs[name]), f"cuda fan-out {name}"
print("cuda fan-out (CUDA and Vulkan consumers) under 16 concurrent requests: identical")

# Delayed consumers: ask for frames out of order, twice, with the cache recycling.
shuffled = branches["lanczos"].std.GPUDownload()
order = list(np.random.default_rng(7).permutation(FRAMES_LONG))[:40]
ref_frames = {n: np.array(refs["lanczos"].get_frame(n)[0], copy=True) for n in order}
for n in order + order[:10]:
    assert np.array_equal(np.asarray(shuffled.get_frame(int(n))[0]), ref_frames[n]), f"frame {n}"
print("out-of-order and repeated requests: identical")

# Many instances on one core share one device runtime and go away with the last one.
many = [gpu.dsmvc.Debicubic(1600 - 16 * i, 900 - 9 * i, backend="vulkan").std.GPUDownload() for i in range(1, 9)]
for clip in many:
    clip.get_frame(3)
del many
print("8 live instances created and released")

# OS handles stay bounded: exported planes are imported once per allocation (or closed).
import os


def open_handles():
    return len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else 0


if open_handles():
    probe = gpu.dsmvc.Debicubic(1600, 900, backend="cuda").dsmvc.Delanczos(1280, 720, taps=3, backend="cuda").std.GPUDownload()
    digest(probe.std.Trim(0, 9), prefetch=8)          # warm up: caches, pools, pipelines
    before = open_handles()
    digest(probe, prefetch=8)                           # 120 more frames through two CUDA filters
    after = open_handles()
    print(f"open file descriptors: {before} -> {after} over {FRAMES_LONG} frames of two CUDA filters")
    assert after - before <= 8, "exported GPU memory handles are leaking"
print("OK (stress)")
