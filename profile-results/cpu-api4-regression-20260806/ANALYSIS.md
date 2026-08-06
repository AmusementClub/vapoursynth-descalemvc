# CPU API4 Regression Analysis

Collected on 2026-08-06 to check whether the API4-only migration regressed the
CPU backend. The result is a pass: none of the 12 measured cells crossed the
3% regression threshold.

## Method

The comparison uses two preserved plugin binaries built from the same engine
implementation state:

| Variant | Plugin SHA-256 |
|---|---|
| API3 | `1d3c517813f1fafdd1c649b1771142c4587d7e5c27d3f54c87d59a3f58817653` |
| API4 | `e6a929c7b28f9de97d1f795dc5dc4c972ca7b7bd229c13dce26c910d89d4a411` |

Each cell processes 5,000 1920x1080 GRAYS BlankClip frames at fixed 810p
geometry with `backend=cpu`. The matrix covers bilinear, bicubic `(0, 0.5)`,
and spline64 at R1T1, R8T8, R16T16, and R32T32. There are three measured runs
per cell and each measured process is immediately preceded by an independent
128-frame warm-up process. API3/API4 execution order alternates by cell and
run.

VSPipe's internal FPS is the primary metric and each cell is the ratio of the
two three-run medians. A drop greater than 3% is a regression. Separate
processes prevent one ABI from reusing another ABI's VapourSynth graph state;
alternating order reduces drift bias. The CPU governor reported `powersave`
during collection, so small differences should still be treated as noise.

## Results

| Kernel | R1T1 | R8T8 | R16T16 | R32T32 |
|---|---:|---:|---:|---:|
| bilinear | +91.98% | +15.31% | +12.51% | +2.59% |
| bicubic `(0, 0.5)` | +53.31% | +11.47% | +10.43% | +0.81% |
| spline64 | +27.03% | +3.35% | +3.67% | -0.14% |

The median change across all cells is +10.95%. The only negative median is
spline64 at R32T32: -0.14%, with paired runs spanning -0.48% to +0.27%. That
is well inside the threshold and normal run-to-run variation.

The gain is largest when framework overhead is a large fraction of total
work. At R1T1, VSPipe attributes about 2.11-2.16 seconds to BlankClip in every
API3 run but only 0.51-0.66 seconds in API4. The dsmvc filter's own R1T1 time
also falls modestly. As the kernel becomes more expensive, or R32 fills the
CPU, that lifecycle saving is amortized and the two ABIs converge.

## Isolation And Correctness

The engine objects are byte-identical between the two builds, which rules out
a CPU math or scheduling implementation change as the source of the result:

| Engine object | SHA-256 in both builds |
|---|---|
| `axis_plan.cpp.o` | `4539da97ac376dfe8444b7ad5a692998ea5fed3c085bc5cb7319319b1d589a52` |
| `backend.cpp.o` | `249c55b96c2f70a4377a9098c48f0ed9b26b1f64f29673c68e8ef2dfd801725a` |
| `executor.cpp.o` | `8c070859763d32be8a34240c5754e73057357d5f1096d07038ffd0589d7b9e7d` |
| `cpu_executor.cpp.o` | `920fdd6a24ffcfc38d3c6a793d4905d7985c90d81727568619218658763a28af` |
| `cpu_executor_avx2.cpp.o` | `29aad1880458e8dc721b87709da7c3aa6b5fe47eec37fe88d35568ef3442eba3` |

A separate deterministic, nonuniform GRAYS output check produced identical
API3 and API4 byte-stream hashes for all three kernels:

| Kernel | API3 and API4 SHA-256 |
|---|---|
| bilinear | `3a2fe7832dc8b95a3cf0a7f52a17885b4e7c58d9b64013fe00db9c118c3254fe` |
| bicubic `(0, 0.5)` | `391d63a60f992fce6d57656e827cc9058a951642dea7d51406c490a4eb22618b` |
| spline64 | `591d34abf0defc2f25369d4c82ef05757def5e29eb3825307acf36569e6a1356` |

API4's relevant semantic change is the explicit `rpStrictSpatial` dependency:
output frame `n` requests only input frame `n`. API3's `createFilter` path had
no equivalent declared dependency and VapourSynth had to manage it as a
general request pattern. The identical engine objects, the large reduction in
upstream BlankClip time, and the way the advantage shrinks with heavier or
saturated work strongly indicate that request/cache lifecycle management is
the source of the gain. This benchmark does not claim that the CPU recurrence
or resampling math became faster.

## Conclusion

The API4-only migration causes no measurable CPU regression. It materially
improves fixed-plan throughput at low and medium request concurrency, while
R32 performance remains effectively neutral. This complements the CUDA R32
result: API4 was neutral in that already-saturated GPU workload, but it is a
real lifecycle optimization for these CPU graphs as well as the required base
for future grouped-frame execution.

Raw per-run data are in `benchmark.json` and `benchmark.csv`; the compact table
is in `summary.csv`, and all 144 warm-up and measured VSPipe commands are in
`commands.txt`.
