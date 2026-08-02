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
