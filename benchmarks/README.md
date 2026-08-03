# Benchmark runner

Run with the Python interpreter bundled with the target VapourSynth R57
environment. The runner refuses an unexpected input or baseline DLL hash and
never writes into the VapourSynth installation.

```powershell
& 'D:\okegui\OKEGui\tools\vapoursynth\python.exe' benchmarks\benchmark.py
```

The fixed inputs are:

- `C:\Users\lsy39\Downloads\6.2-1.png`
  (`61F9EE1AC858BBADD6A959BA35F5ECEB077B8452B91E97A5CE3D39EBC69E20C6`)
- `D:\okegui\OKEGui\tools\vapoursynth\vapoursynth64\plugins\descale.dll`
  (`B02E4A2FBAAF6BA3F7E3CF2AD8A08D8EEFAB9E5D634E1D829764671D49933000`)

The direct GRAYS path uses `muvsfunc.rescale._get_descale_args` with
`base_height=1000`, `src_height=951.5`, and `src_width=951.5*16/9`, producing
`1692x952`, `src_left=0.2222222222221717`, and `src_top=0.25`. The wrapper path
preserves the public wrapper API and therefore uses its default zero crop
offsets. Reconstruction uses the same resize kernel and the exact descale
geometry, matching `muvsfunc.rescale.Rescaler.upscale`.

The default direct cases are Bicubic `(0, 1)` and Bicubic `(0.7, 0.6)` with
seven independent process repetitions and 32 different warm frame indices.
Use `--cases` to add Bilinear, default Bicubic, Lanczos3, Spline16, and
Spline36.

Three sweep adapters replace the unavailable MKV inputs with `6.2-1.png` but
preserve the source scripts' geometry and error expression:

- `getfnative`: 11 scalers x 2800 heights = 30,800 candidates.
- `getfnative_v2`: 8 scalers x 400 heights = 3,200 candidates, vertical only.
- `selectkernel`: Bilinear plus 100 Bicubic `(b,c)` combinations = 101
  candidates at height 719.8 and `ex_thr=0.012`.

All adapters crop five pixels from each edge before `PlaneStats`. The first
two use `ex_thr=0.015`. Smoke mode samples the long height sweeps while keeping
all kernels. Full mode evaluates every candidate.

```powershell
& 'D:\okegui\OKEGui\tools\vapoursynth\python.exe' benchmarks\benchmark.py `
  --cases bicubic_0_1 bicubic_0_7_0_6 bilinear bicubic_default lanczos3 spline16 spline36 `
  --runs 7 --warm-frames 32 `
  --vspipe-runs 3 --vspipe-frames 64 `
  --sweep-profile full --sweep-runs 3 --resume `
  --output benchmark-results
```

Each checkpoint is keyed by the runner, input, selected plugin, thread count,
concurrency, and profile hashes. `--resume` can therefore reuse completed full
runs without accepting stale results after a rebuild.

Outputs are `benchmark.json`, `benchmark.csv`, `benchmark.md`, `commands.txt`,
old/new PNGs, amplified difference PNGs, and sweep error-curve plots. Reports
include plugin load, graph construction, cold frame, warm frame, VSPipe,
full-sweep, median, MAD, p95, minimum, maximum, hashes, per-plane error,
reconstruction error, best candidate, and rank changes.

The focused whole-process benchmark reproduces the three recorded getnative
graphs through runtime `FrameEval`, launches a fresh VSPipe process for every
sample, and measures wall time through process exit:

```powershell
& 'D:\okegui\OKEGui\tools\vapoursynth\python.exe' `
  benchmarks\vspipe_process_benchmark.py `
  --frames 500 --requests 32 --threads 32 --runs 3
```

The current-binary CPU profile suite builds RelWithDebInfo, snapshots the exact
DLL/PDB and source identity, creates an isolated portable VS runtime containing
only Imwri plus the explicitly loaded dsmvc plugin, and then requests elevation
once for all AMD uProf jobs:

```powershell
& benchmarks\profile_current_cpu.ps1
```

It collects call-graph TBP for bandwidths 1/3/5/7 at 1, 8, and 32 requests;
Assess Extended PMU/cache data and PCM memory-controller data; 500-frame getfnative,
getfnative_v2, and selectkernel profiles; process-memory traces; and two ETW
CPU/context-switch traces. Use `-SkipEtw` when an existing WPR session must be
left untouched. A partially completed output directory can be resumed because
completed sessions are detected by their exported reports.

The hardware-counter portion deliberately uses AMD uProf's `assess_ext` PMU
configuration and `AMDuProfPcm.exe profile -m memory`; it does not use IBS/IBA.
Each result records the exact `assess_ext` event table in
`uprof-assess-ext-config.log`; the PMU report uses uProf's compatible
`--detail --show-percentage` form. PCM is
socket-wide memory-controller data, while `assess_ext` is the CPU PMU sample
source. If a supplied output directory contains a different DLL/PDB, the
runner automatically creates a `-binary-refresh-<timestamp>` sibling instead
of mixing old sessions with the new binary or exiting silently in the elevated
worker. All profiling modes, including `-PcmOnly` and `-TbpOnly`, request one
administrator session so hardware PMU access is consistent.

After collection, generate a validated expert-review summary without elevation:

```powershell
& 'D:\okegui\OKEGui\tools\vapoursynth\python.exe' `
  benchmarks\summarize_cpu_profile.py `
  benchmark-results\profile-current-<commit>-<stamp> --hash-etw
```

The summary command requires the clean `report-percentage.csv` exports and the
Windows Performance Toolkit text exports generated by the current profile
runner. It writes `profile-summary.json` and `profile-summary.md` next to the
raw sessions and fails if session counts, call stacks, symbols, PCM data, or ETW
loss checks do not pass.
