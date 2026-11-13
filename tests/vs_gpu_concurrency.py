"""Full-size GPU checks on host frames: the partitioned solve against the CPU
backend, and many filters sharing one source under concurrent requests (the
shared transposed input) against the same filters run one frame at a time.
A deadlock shows up as the ctest timeout.

    vs_gpu_concurrency.py <dsmvc plugin> [--cuda] [--vulkan]
"""
import hashlib
import sys

import numpy as np
import vapoursynth as vs

core = vs.core
core.std.LoadPlugin(sys.argv[1])
BACKENDS = [name for name in ("cuda", "vulkan") if f"--{name}" in sys.argv[2:]]
FRAMES = 24


def patterned(fmt, width=1920, height=1080, frames=FRAMES):
    blank = core.std.BlankClip(width=width, height=height, format=fmt, length=frames)

    def fill(n, f):
        out = f.copy()
        rng = np.random.default_rng(2000 + n)
        for p in range(out.format.num_planes):
            a = np.asarray(out[p])
            if out.format.sample_type == vs.FLOAT:
                a[:] = (0.25 + 0.5 * rng.random(a.shape, dtype=np.float32)).astype(a.dtype)
            else:
                top = (1 << out.format.bits_per_sample) - 1
                a[:] = rng.integers(top // 4, 3 * top // 4, a.shape, dtype=a.dtype)
        return out

    return core.std.ModifyFrame(blank, blank, fill)


def accuracy(backend):
    # Vectors long enough to be partitioned (src/solve_partition.hpp), bands
    # 1, 3, 5 and 7, an odd size, and a less well-conditioned ratio.
    cases = [
        ("bicubic", "Debicubic", dict(width=1280, height=720)),
        ("bilinear", "Debilinear", dict(width=1280, height=720)),
        ("lanczos3", "Delanczos", dict(width=1280, height=720, taps=3)),
        ("lanczos4", "Delanczos", dict(width=1280, height=720, taps=4)),
        ("odd", "Debicubic", dict(width=1279, height=719)),
        ("1080-to-1040", "Debicubic", dict(width=1848, height=1040)),
    ]
    src = patterned(vs.GRAYS, frames=2)
    for label, name, args in cases:
        cpu = getattr(core.dsmvc, name)(src, backend="cpu", **args)
        gpu = getattr(core.dsmvc, name)(src, backend=backend, **args)
        for n in range(2):
            a = np.asarray(cpu.get_frame(n)[0], dtype=np.float64)
            b = np.asarray(gpu.get_frame(n)[0], dtype=np.float64)
            error = np.abs(a - b)
            assert error.max() <= 2.0e-5 and error.mean() <= 1.0e-6, (
                f"{backend} {label}: max {error.max():.3e} mean {error.mean():.3e}")
    print(f"{backend}: partitioned solve matches the CPU backend ({len(cases)} cases)")


def native_source(fmt, frames=FRAMES):
    # Produced without Python, so requests really overlap (frames made by a
    # Python callback serialise on the GIL and hide slot contention).
    maximum = 1.0 if fmt == vs.GRAYS else float((1 << core.get_video_format(fmt).bits_per_sample) - 1)
    parts = []
    for n in range(frames):
        level = [0.2 + 0.6 * ((n * 7 + k * 3) % 11) / 10 for k in range(4)]
        tiles = [core.std.BlankClip(width=480, height=1080, format=fmt, length=1,
                                    color=[v * maximum if fmt != vs.GRAYS else v]
                                    * core.get_video_format(fmt).num_planes)
                 for v in level]
        parts.append(core.std.StackHorizontal(tiles))
    return core.std.Splice(parts)


def filters(src, backend):
    d = core.dsmvc
    out = [
        d.Debicubic(src, 1280, 720, backend=backend),
        d.Debicubic(src, 1280, 720, b=1 / 3, c=1 / 3, backend=backend),
        d.Delanczos(src, 1280, 720, taps=3, backend=backend),
        d.Debilinear(src, 1600, 900, backend=backend),
        d.Debicubic(src, 1280, 1080, backend=backend),  # rows only
        d.Debicubic(src, 1920, 720, backend=backend),   # columns only
    ]
    if src.format.sample_type == vs.FLOAT:
        out.append(d.Debicubic(src, 1280, 720, f64mode=2, backend=backend))
    return out


def digest(frame):
    h = hashlib.sha1()
    for p in range(frame.format.num_planes):
        h.update(bytes(frame[p]))
    return h.hexdigest()


def fan_out(backend):
    for fmt in (vs.GRAYS, vs.YUV420P16, vs.YUV420P8):
        src = native_source(fmt)
        sequential = [[digest(c.get_frame(n)) for n in range(FRAMES)]
                      for c in filters(src, backend)]
        clips = filters(src, backend)
        seen = {}

        def record(n, f):
            for index, frame in enumerate(f):
                seen[(index, n)] = digest(frame)
            return f[0]

        joined = core.std.ModifyFrame(clips[0], clips, record)
        for _ in joined.frames(prefetch=24, backlog=48):
            pass
        for index in range(len(clips)):
            for n in range(FRAMES):
                assert seen[(index, n)] == sequential[index][n], (
                    f"{backend} {src.format.name}: filter {index} frame {n} differs "
                    "between concurrent and sequential requests")
        print(f"{backend} {src.format.name}: {len(clips)} filters x {FRAMES} frames "
              "identical under concurrent fan-out")


for backend in BACKENDS:
    accuracy(backend)
    fan_out(backend)
print("ok")
