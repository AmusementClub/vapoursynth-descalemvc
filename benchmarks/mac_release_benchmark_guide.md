# macOS Release Benchmark Guide

这份指南用于让 macOS 上的 agent 复现当前项目的 release benchmark。目标是
比较同一台 Mac 上的 original descale 和当前 Release `dsmvc`，并生成可以与
Linux 结果并排阅读的 Markdown、JSON、CSV 和 SVG 报告。

不要把 Mac 的绝对 FPS 直接和 Ryzen 9 5950X 的 FPS 比较。跨机器最重要的
指标是 old/new speedup、R1T1 到 R8T8/R16T16/R32T32 的 scaling，以及误差和
候选选择是否一致。

## 1. 复现约束

必须使用：

- 同一份 `Descale-MVC` checkout 和当前 Release 源码；
- 同一份 Digimon MKV；
- 同一份两个 HTML provenance 文件和三个 `.vpy` provenance 文件；
- original descale 和 current `dsmvc` 两个与 Mac CPU 架构一致的插件；
- 同一个 source filter 完成全部测试。优先使用 `lsmas`，如果 Mac 环境
  只有 FFMS2，可以全程使用 `ffms2`，但结果必须在报告中注明；
- VapourSynth、VSPipe 和 VapourSynth Python 来自同一个安装/环境。

Linux 当前参考输入的 SHA-256 是：

```text
864d552f8e2ead057ebd2c202c7580442a5f22c8acecd08167eb8a07110d1bf4
```

Mac 上如果 SHA-256 不一致，先不要跑完整 benchmark；那已经不是同一个输入。

## 2. 准备环境

需要 CMake 3.24 或更新版本、C++23 编译器、VapourSynth API3 SDK、VSPipe、
VapourSynth Python，以及所选的 source plugin。Apple Silicon 和 Intel Mac
不能混用插件或 VapourSynth：所有 `.so`/`.dylib`、VSPipe、Python 和宿主进程
都必须是同一架构。

先填写以下变量。路径只需要改这一段：

```bash
set -euo pipefail

REPO="$HOME/dev/Descale-MVC"
VS_PY="/path/to/vapoursynth/bin/python3"
VSPIPE="/path/to/vapoursynth/bin/vspipe"
VS_SDK="/path/to/vapoursynth/sdk"

SOURCE="/path/to/[LoliHouse] DIGIMON BEATBREAK - 40 [WebRip 1080p HEVC-10bit AAC SRTx2].mkv"
OLD_PLUGIN="/path/to/libdescale.so"
TRAINING_ROOT="/path/to/vf"
# Optional. Leave empty when LSMASHSource is already autoloaded.
LSMAS_PLUGIN=""
LSMAS_ARGS=()
if [[ -n "$LSMAS_PLUGIN" ]]; then
    LSMAS_ARGS=(--source-plugin "$LSMAS_PLUGIN")
fi

BUILD="$REPO/build/macos-release"
OUT_ROOT="$REPO/benchmark-results/release-benchmark-macos-$(date +%Y%m%d)"
RELEASE_OUTPUT="$OUT_ROOT/summary"

HTML_1="$TRAINING_ROOT/总监培训2026_20260725.html"
HTML_2="$TRAINING_ROOT/总监培训2026_20260726.html"
SCRIPT_GETNATIVE="$TRAINING_ROOT/test_getfnative.vpy"
SCRIPT_GETNATIVE_V2="$TRAINING_ROOT/test_getfnative_v2.vpy"
SCRIPT_SELECTKERNEL="$TRAINING_ROOT/test_selectkernel.vpy"
```

在 macOS 上用 `shasum` 验证输入，并确认所有文件存在：

```bash
test -f "$SOURCE"
test -f "$OLD_PLUGIN"
test -f "$VS_PY"
test -f "$VSPIPE"
test -f "$VS_SDK/include/VapourSynth.h"
for path in "$HTML_1" "$HTML_2" "$SCRIPT_GETNATIVE" \
    "$SCRIPT_GETNATIVE_V2" "$SCRIPT_SELECTKERNEL"; do
    test -f "$path"
done

SOURCE_SHA256=$(shasum -a 256 "$SOURCE" | awk '{print $1}')
printf 'source sha256: %s\n' "$SOURCE_SHA256"
printf 'macOS: '; sw_vers -productVersion
printf 'machine: '; uname -m
printf 'logical CPUs: '; sysctl -n hw.ncpu
printf 'memory bytes: '; sysctl -n hw.memsize
"$VSPIPE" --version
```

检查 VapourSynth namespace 和二进制架构：

```bash
"$VS_PY" - <<'PY'
import vapoursynth as vs

core = vs.core
print("VapourSynth API:", getattr(vs, "__api_version__", "unknown"))
print("lsmas:", hasattr(core, "lsmas"))
print("ffms2:", hasattr(core, "ffms2"))
print("bestsource:", hasattr(core, "bs"))
PY

file "$VS_PY" "$VSPIPE" "$OLD_PLUGIN"
```

如果选择 `lsmas`，上面的输出必须显示 `lsmas: True`。`release_benchmark.py`
会让 VSPipe 使用已经自动加载的 source plugin；它不会修改 VapourSynth
安装目录。也可以把 `LSMAS_PLUGIN` 设置为插件文件路径，指南中的
`LSMAS_ARGS` 会把 `--source-plugin` 传给所有 benchmark runner。如果 source
plugin 既没有 autoload 也没有显式路径，应改用已经自动加载的 `ffms2` 并在
命令中同步修改 `--source-filter`。

## 2.1 Decoder 对照

当前 benchmark 支持以下 source 参数，默认值保持原有软件解码行为：

```text
--source-decoder ""
--source-prefer-hw 0
--source-ff-loglevel 0
--source-rap-verification -1
```

LSMASH-Works 的 `LWLibavSource` 文档中，`prefer_hw` 的枚举是 CUVID、QSV、
DXVA2、D3D11VA、D3D12VA 和 Vulkan，没有 VideoToolbox。因此 macOS 上不要把
`--source-prefer-hw 3` 当作苹果硬解开关；它很可能直接 fallback 到软件解码。

如果 Mac 使用的 LSMASH-Works 所链接的 FFmpeg 暴露了
`hevc_videotoolbox`，可以用 `--source-decoder` 显式请求它：

```bash
--source-filter lsmas \
--source-decoder 'hevc_videotoolbox,hevc' \
--source-ff-loglevel 5 \
--source-rap-verification 0
```

这里的 `hevc` 是 fallback；只要 `hevc_videotoolbox` 不存在，测试就会退回
软件解码，所以这组结果不能只按目录名称为 hardware decode。`ff_loglevel=5`
用于在 smoke 和首个 full 样本中确认实际 decoder；`rap_verification=0` 是因为
LSMASH 的索引阶段在 RAP verification 开启时会强制使用软件 decoder。完整测试
前必须从 `benchmark.json` 的 `vspipe_output_tail` 或 VSPipe 日志中找到
`videotoolbox`，否则只能标记为 LSMASH software/fallback。

如果 `ffmpeg -decoders` 能看到 `hevc_videotoolbox` 只能作为线索，不能证明
VapourSynth 使用的是同一套 FFmpeg。最终以 LSMASH/VSPipe 的 decoder 日志为准。

第 3 节编译完成并得到 `NEW_PLUGIN` 后，先只跑 current new、bilinear、4000 帧
的四线程配置，分别测三组：

```bash
COMMON_FIXED=(
    --source "$SOURCE"
    --new-plugin "$NEW_PLUGIN"
    --vspipe "$VSPIPE"
    --source-filter lsmas
    --frames 4000
    --src-height 810
    --base-height 1000
    --threads 1 8 16 32
    --runs 3
    --implementations new
    --kernels bilinear
)

# 1. Existing FFMS2 baseline: use --source-filter ffms2 and remove
#    --source-plugin if FFMS2 is autoloaded.
"$VS_PY" "$REPO/benchmarks/fixed_kernel_benchmark.py" \
    "${COMMON_FIXED[@]}" \
    --source-filter ffms2 \
    --output "$OUT_ROOT/fixed-ffms2-new"

# 2. LSMASH software decode.
"$VS_PY" "$REPO/benchmarks/fixed_kernel_benchmark.py" \
    "${COMMON_FIXED[@]}" \
    "${LSMAS_ARGS[@]}" \
    --source-decoder '' \
    --source-prefer-hw 0 \
    --source-rap-verification -1 \
    --output "$OUT_ROOT/fixed-lsmas-sw-new"

# 3. LSMASH explicit VideoToolbox request, only if the smoke log confirms it.
"$VS_PY" "$REPO/benchmarks/fixed_kernel_benchmark.py" \
    "${COMMON_FIXED[@]}" \
    "${LSMAS_ARGS[@]}" \
    --source-decoder 'hevc_videotoolbox,hevc' \
    --source-ff-loglevel 5 \
    --source-rap-verification 0 \
    --output "$OUT_ROOT/fixed-lsmas-videotoolbox-new"
```

`LSMAS_ARGS` 只有在变量非空且插件没有 autoload 时才会包含
`--source-plugin`。三组测试必须使用相同的输入、VapourSynth 版本、线程配置
和 runs。

这个筛选 benchmark 的判断重点是：

- LSMASH software 相对 FFMS2 是否降低 `Source` time；
- VideoToolbox 日志是否真的出现，以及 `Source` time 是否进一步降低；
- `wall time - Source time` 是否变得足够大，让 dsmvc 的 old/new speedup 能
  体现在端到端 FPS；
- 若 LSMASH 和 FFMS2 的 `Source` time 都接近 wall time，则 scale plateau 是
  decoder ceiling，不能归因于 dsmvc 的 cache 或 thread scheduler。

硬件解码即使成功，也不等于整条链路零拷贝。当前 graph 随后要把 decoder 输出
送入 `ShufflePlanes`、`resize.Point(format=GRAYS)` 和 CPU dsmvc；VideoToolbox
帧到 CPU VapourSynth frame 的 transfer/synchronization 仍然可能成为瓶颈。

## 3. 编译 current Release

当前 CMake 会在 x86_64 上单独编译 AVX2/FMA translation unit；Apple Silicon
不会被强制编译 AVX2 路径。macOS 只使用普通 Release，不要加
`-march=native`、全局 `-ffast-math`、LTO 或 PGO：

```bash
cmake -S "$REPO" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DDSMVC_VAPOURSYNTH_SDK="$VS_SDK" \
    -DDSMVC_VS_PYTHON="$VS_PY" \
    -DDSMVC_BASELINE_PLUGIN="$OLD_PLUGIN"

cmake --build "$BUILD" --parallel "$(sysctl -n hw.ncpu)"
ctest --test-dir "$BUILD" --output-on-failure
```

找到新插件并检查架构：

```bash
NEW_PLUGIN=$(find "$BUILD" -type f -name 'dsmvc*' \
    ! -name '*.pdb' ! -name '*.dSYM' -print -quit)
test -n "$NEW_PLUGIN"
printf 'new plugin: %s\n' "$NEW_PLUGIN"
file "$NEW_PLUGIN"
```

`NEW_PLUGIN` 和 `OLD_PLUGIN` 的架构必须相同。Apple Silicon 应看到 `arm64`；
Intel Mac 应看到 `x86_64`。如果 `file` 显示一边是 universal、一边是单架构，
应先固定运行架构再测，不能把 Rosetta 和 native 结果混在一起。

## 4. 先做 smoke test

完整测试耗时较长。先只跑一个 recipe 的少量候选，确认插件、decoder、路径和
API 都正确：

```bash
mkdir -p "$OUT_ROOT/smoke"
"$VS_PY" "$REPO/benchmarks/e2e_benchmark.py" \
    --source "$SOURCE" \
    --old-plugin "$OLD_PLUGIN" \
    --new-plugin "$NEW_PLUGIN" \
    --vspipe "$VSPIPE" \
    --python "$VS_PY" \
    --source-filter lsmas \
    "${LSMAS_ARGS[@]}" \
    --source-decoder '' \
    --source-prefer-hw 0 \
    --source-rap-verification -1 \
    --profile smoke \
    --cases getfnative \
    --implementations old new \
    --runs 1 \
    --requests 1 \
    --threads 1 \
    --html "$HTML_1" \
    --html "$HTML_2" \
    --script "getfnative=$SCRIPT_GETNATIVE" \
    --script "getfnative_v2=$SCRIPT_GETNATIVE_V2" \
    --script "selectkernel=$SCRIPT_SELECTKERNEL" \
    --strict-provenance \
    --output "$OUT_ROOT/smoke"
```

如果 Mac 没有自动加载 `lsmas`，把 smoke 和后续所有命令中的
`--source-filter lsmas` 改成 `--source-filter ffms2`，前提是检查结果中的
`ffms2: True`。不要只在 smoke 阶段切换 decoder。

smoke 必须至少确认：

- old 和 new 都能启动 VSPipe；
- `benchmark.md` 中的 candidate count 正确；
- 没有 `No such filter`、插件架构错误或 VapourSynth API 错误；
- 输出目录已经写出 `benchmark.json`、`performance.csv` 和 `commands.txt`。

## 5. 运行完整 release benchmark

确认 smoke 成功后，运行统一 orchestrator。这里的命令复现当前 release
benchmark 的完整定义：

- E2E performance：`getfnative` 30,800 candidates、`getfnative_v2` 3,200、
  `selectkernel` 101；
- E2E thread sweep：`R1T1`、`R8T8`、`R16T16`、`R32T32`，old/new 分开进程；
- error sweep：三个 recipe 共 34,101 candidates，默认 R32T32；
- fixed kernel：810p、4000 帧、bilinear、Bicubic `(0, 0.5)`、Lanczos2--6、
  Spline16/36/64，old/new 和四种线程配置。

```bash
mkdir -p "$OUT_ROOT"

"$VS_PY" "$REPO/benchmarks/release_benchmark.py" \
    --source "$SOURCE" \
    --old-plugin "$OLD_PLUGIN" \
    --new-plugin "$NEW_PLUGIN" \
    --vspipe "$VSPIPE" \
    --vs-python "$VS_PY" \
    --python "$VS_PY" \
    --source-filter lsmas \
    "${LSMAS_ARGS[@]}" \
    --source-decoder '' \
    --source-prefer-hw 0 \
    --source-rap-verification -1 \
    --html "$HTML_1" \
    --html "$HTML_2" \
    --script "getfnative=$SCRIPT_GETNATIVE" \
    --script "getfnative_v2=$SCRIPT_GETNATIVE_V2" \
    --script "selectkernel=$SCRIPT_SELECTKERNEL" \
    --output-root "$OUT_ROOT" \
    --release-output "$RELEASE_OUTPUT" \
    --fixed-profile "$OUT_ROOT/no-linux-fixed-profile.json"
```

最后一个 `--fixed-profile` 是为了避免误读 Linux 上已有的低层 profile；这个
路径应不存在。macOS 这轮 release bench 不使用 Linux 的 `perf`、`sudo`、
`perf_event_paranoid` 或 `run_cache_hypothesis.sh`。

benchmark 期间不要并行运行编译、播放器、转码器或其他大型任务。每个样本是
新的 VSPipe 进程，wall time 包含 source decode、graph setup、descale、重建、
PlaneStats 和进程退出。

## 6. 内存不足时的处理

性能部分和 fixed-kernel 部分应先完整跑完。完整 error sweep 默认使用四个
worker process；如果 Mac 物理内存较小或出现系统 memory pressure，不要重复
启动整个 release bench。保留已经生成的四个 E2E performance 目录和 fixed
kernel 目录，只单独用一个 error worker 重跑 error 目录：

```bash
ERROR_OUT="$OUT_ROOT/e2e-digimon-release-errors-r32t32"

"$VS_PY" "$REPO/benchmarks/e2e_benchmark.py" \
    --source "$SOURCE" \
    --old-plugin "$OLD_PLUGIN" \
    --new-plugin "$NEW_PLUGIN" \
    --vspipe "$VSPIPE" \
    --python "$VS_PY" \
    --source-filter lsmas \
    "${LSMAS_ARGS[@]}" \
    --source-decoder '' \
    --source-prefer-hw 0 \
    --source-rap-verification -1 \
    --profile full \
    --cases getfnative getfnative_v2 selectkernel \
    --implementations old new \
    --requests 32 \
    --threads 32 \
    --skip-performance \
    --error-processes 1 \
    --error-threads 1 \
    --html "$HTML_1" \
    --html "$HTML_2" \
    --script "getfnative=$SCRIPT_GETNATIVE" \
    --script "getfnative_v2=$SCRIPT_GETNATIVE_V2" \
    --script "selectkernel=$SCRIPT_SELECTKERNEL" \
    --strict-provenance \
    --output "$ERROR_OUT"
```

error 目录生成后，用原来的 release 命令加 `--skip-run` 只重新合并报告。该
命令仍然需要同一组路径参数：

```bash
"$VS_PY" "$REPO/benchmarks/release_benchmark.py" \
    --source "$SOURCE" \
    --old-plugin "$OLD_PLUGIN" \
    --new-plugin "$NEW_PLUGIN" \
    --vspipe "$VSPIPE" \
    --vs-python "$VS_PY" \
    --python "$VS_PY" \
    --source-filter lsmas \
    "${LSMAS_ARGS[@]}" \
    --source-decoder '' \
    --source-prefer-hw 0 \
    --source-rap-verification -1 \
    --html "$HTML_1" \
    --html "$HTML_2" \
    --script "getfnative=$SCRIPT_GETNATIVE" \
    --script "getfnative_v2=$SCRIPT_GETNATIVE_V2" \
    --script "selectkernel=$SCRIPT_SELECTKERNEL" \
    --output-root "$OUT_ROOT" \
    --release-output "$RELEASE_OUTPUT" \
    --fixed-profile "$OUT_ROOT/no-linux-fixed-profile.json" \
    --skip-run
```

上面最后一条命令中的变量名必须和前面保持一致；如果 shell 中使用的是
`SCRIPT_GETNATIVE_V2`，就把对应的 `--script` 写成该变量。

## 7. 验收和回传

完成后检查：

```bash
REPORT="$RELEASE_OUTPUT/release-benchmark.md"
test -f "$REPORT"
test -f "$RELEASE_OUTPUT/release-benchmark.json"
test -f "$RELEASE_OUTPUT/e2e-scaling.svg"
test -f "$OUT_ROOT/e2e-digimon-release-errors-r32t32/benchmark.json"
test -f "$OUT_ROOT/fixed-kernel-digimon-810p-release/benchmark.json"

rg -n "30,800|3,200|34,101|R1T1|R8T8|R16T16|R32T32|Error coverage" \
    "$REPORT"
```

回传整个 `$OUT_ROOT`，至少包括：

- `summary/release-benchmark.md`；
- `summary/release-benchmark.json`；
- `summary/e2e-scaling.svg`；
- 四个 `e2e-digimon-release-r*` 目录；
- `e2e-digimon-release-errors-r32t32`；
- `fixed-kernel-digimon-810p-release`；
- 各目录中的 `commands.txt`、`benchmark.json`、`benchmark.csv` 和 Markdown。

同时回传以下文字信息：

```text
Mac model / chip:
macOS version:
architecture: arm64 or x86_64
physical memory:
VapourSynth version/API:
VSPipe path and version:
source filter: lsmas / ffms2 / bestsource
source SHA-256:
old plugin path and architecture:
new plugin path and architecture:
error processes / error threads:
output directory:
```

报告中的 old/new speedup 大于 `1.0x` 表示 current `dsmvc` 更快。跨 Mac 与
Linux 的分析应优先看 speedup 和 scaling 曲线，不要把不同 CPU、内存系统、
VapourSynth 版本或 source decoder 下的绝对 candidates/s、FPS 直接相减。
