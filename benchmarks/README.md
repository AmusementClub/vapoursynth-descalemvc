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

## Real-video E2E comparison

`e2e_benchmark.py` runs the three supplied training recipes against a real MKV.
It uses `vspipe_e2e.vpy` to load the selected source decoder, extract the training frame,
convert the luma plane to `GRAYS`, descale and reconstruct every candidate. Old
and new performance samples run in separate VSPipe processes. Error workers
pair the two outputs and also compare each reconstruction with the source.

The default `full` profile keeps the training script's kernels, ranges, and
candidate ordering: it evaluates all 30,800 `getfnative` candidates, all 3,200
`getfnative_v2` candidates, and all 101 `selectkernel` candidates. Use the
explicit `--profile smoke` option for a quick check that samples the long
height ranges. The HTML files are provenance inputs: the runner records their
hashes and extracts the referenced training media/scripts; it does not treat
the chat export as a video source.

Example on the supplied Linux VapourSynth environment:

```bash
VS_PY=/home/owen/vapoursynth/bin/python
VSPIPE=/home/owen/vapoursynth/bin/vspipe
OLD_PLUGIN=/home/owen/vapoursynth/lib/python3.14/site-packages/vapoursynth/plugins/vsrepo/libdescale.so
NEW_PLUGIN=/path/to/build/dsmvc.so
SOURCE='/run/media/owen/1A16B65916B6361B/Users/lsy39/Downloads/cesh/[LoliHouse] DIGIMON BEATBREAK - 40 [WebRip 1080p HEVC-10bit AAC SRTx2].mkv'
TRAINING_ROOT='/run/media/owen/1A16B65916B6361B/Users/lsy39/Documents/vf'

"$VS_PY" benchmarks/e2e_benchmark.py \
  --source "$SOURCE" \
  --old-plugin "$OLD_PLUGIN" \
  --new-plugin "$NEW_PLUGIN" \
  --vspipe "$VSPIPE" \
  --python "$VS_PY" \
  --source-filter ffms2 \
  --html "$TRAINING_ROOT/总监培训2026_20260725.html" \
  --html "$TRAINING_ROOT/总监培训2026_20260726.html" \
  --script "getfnative=$TRAINING_ROOT/test_getfnative.vpy" \
  --script "getfnative_v2=$TRAINING_ROOT/test_getfnative_v2.vpy" \
  --script "selectkernel=$TRAINING_ROOT/test_selectkernel.vpy" \
  --strict-provenance \
  --profile full \
  --output benchmark-results/e2e-digimon-full
```

The default `--source-filter lsmas` follows the supplied scripts. On Linux,
`--source-filter ffms2` or `--source-filter bestsource` can be used when an
LSMASHSource binary is unavailable; the selected decoder is recorded in the
report. Add `--source-plugin /path/to/liblsmas.so` when the selected source
plugin is not already autoloaded. Use `--profile full` for the complete ranges
and `--runs 5` (or more) when the machine is otherwise idle. Use
`--profile smoke` when a short validation is preferred. The report writes
`benchmark.json`, `benchmark.md`, `performance.csv`, `errors.csv`, one
`errors-<case>.json` per recipe, and `commands.txt` without modifying the
VapourSynth installation. `new_speedup` is old wall time divided by new wall
time, so values above `1.0` mean the current implementation is faster.

For a full error sweep, `--error-processes 8` partitions candidates by index,
runs the workers independently, and merges them only after checking complete
coverage. If NumPy is available in the VapourSynth Python environment, the
worker computes frame metrics directly from the plane memory; otherwise it
uses the slower PlaneStats fallback.

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
