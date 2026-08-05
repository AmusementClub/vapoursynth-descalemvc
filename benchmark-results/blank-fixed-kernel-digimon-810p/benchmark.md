# Blank fixed-kernel benchmark

This compares old descale and current dsmvc without a decoder or source clip. Each run processes 4,000 frames from an in-memory 1920x1080 GRAYS `std.BlankClip` at fixed 810p geometry.

## Throughput

| Kernel | R1 old | R1 new | R1 new/old | R8 old | R8 new | R8 new/old |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bilinear` | 907.97 | 1082.73 | 1.192x | 776.59 | 866.18 | 1.115x |
| `bicubic (0, 0.5)` | 516.68 | 889.10 | 1.721x | 794.30 | 862.74 | 1.086x |

## Filter Time

There is no LSMASH, decoder, or Point conversion in this graph. The remaining filter time is the blank producer and descale node; R8 percentages are accumulated across worker threads.

| Kernel | Impl | Threads | BlankClip s / % | dsmvc s / % |
|---|---|---:|---:|---:|
| `bicubic (0, 0.5)` | `new` | R1T1 | 1.69 / 38.7% | 2.68 / 61.1% |
| `bicubic (0, 0.5)` | `new` | R8T8 | 5.85 / 131.4% | 29.92 / 667.0% |
| `bicubic (0, 0.5)` | `old` | R1T1 | 0.46 / 5.9% | 7.16 / 94.0% |
| `bicubic (0, 0.5)` | `old` | R8T8 | 6.49 / 132.3% | 32.66 / 666.2% |
| `bilinear` | `new` | R1T1 | 1.71 / 48.0% | 1.84 / 51.7% |
| `bilinear` | `new` | R8T8 | 7.18 / 160.9% | 28.50 / 637.3% |
| `bilinear` | `old` | R1T1 | 0.42 / 9.8% | 3.85 / 90.0% |
| `bilinear` | `old` | R8T8 | 8.05 / 160.5% | 31.98 / 637.5% |

## Environment

```json
{
  "timestamp_utc": "2026-08-05T07:02:18.892580+00:00",
  "platform": "Linux-7.0.0-28-generic-x86_64-with-glibc2.43",
  "processor": "",
  "logical_cpu_count": 32,
  "vspipe": "/home/owen/vapoursynth/bin/vspipe",
  "input": {
    "type": "VapourSynth std.BlankClip",
    "width": 1920,
    "height": 1080,
    "format": "GRAYS",
    "color": 0
  },
  "old_plugin": {
    "path": "/home/owen/vapoursynth/lib/python3.14/site-packages/vapoursynth/plugins/vsrepo/libdescale.so",
    "exists": true,
    "size": 71736,
    "sha256": "ca2006ae45a55dc58fa2fb96d77deaee9e7af69065dbbb4a41035e0c1a5de264"
  },
  "new_plugin": {
    "path": "/home/owen/dev/Descale-MVC/out/linux-release-generic-20260805/build/dsmvc.so",
    "exists": true,
    "size": 187272,
    "sha256": "2e96820ca635fba881f889a65777fdcd570292fb4dc25de0d3c932c49afa2052"
  },
  "vpy": {
    "path": "/home/owen/dev/Descale-MVC/benchmarks/vspipe_blank_fixed_kernel.vpy",
    "exists": true,
    "size": 1499,
    "sha256": "c43a476c1599dd701c12bc72251af97b447d41fc8eca37b288f3e03627abde37"
  },
  "frames": 4000,
  "src_height": 810.0,
  "base_height": 1000.0,
  "threads": [
    1,
    8
  ],
  "runs": 3,
  "kernels": [
    "bilinear",
    "bicubic_b0_c0_5"
  ],
  "implementations": [
    "old",
    "new"
  ],
  "runner_sha256": "915278b6297b8e1cf8913b10a627ba5cf37e8ae2eb5d9b8a300ee2889b5b93ec"
}
```
