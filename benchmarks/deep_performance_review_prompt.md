# dsmvc static deep performance review prompt

你是负责高性能图像处理、编译器后端、SIMD、VapourSynth 插件 ABI 和数值算法的资深 code reviewer。请对 `dsmvc` 项目做一次面向专家的深度审核。审核的目标不是找格式问题，而是回答：当前实现的算法细节是否正确，planner 前端和 executor 后端的整体方向是否合理，哪些热点还存在可兑现的性能空间，API3/后端抽象是否会限制长期演进，以及 benchmark/profile 结论哪些可靠、哪些不能从现有证据推出。

## 工作范围与输入

运行时请把下面两个占位符替换为实际路径：

- `REPO_ROOT`: 解压后的 dsmvc 源码根目录。
- `PACK_ROOT`: `dsmvc-review-pack-20260803` 解压目录。

优先阅读这些文件，之后再按调用关系扩展搜索：

1. `PACK_ROOT/README.md`
2. `PACK_ROOT/review-facts.md` 和 `PACK_ROOT/review-facts.json`
3. `PACK_ROOT/throughput/benchmark.md`、`benchmark.json`、`benchmark.csv`
4. `PACK_ROOT/profiles/current-pmu-pcm-facts.md`
5. `PACK_ROOT/profiles/previous-pmu-pcm-facts.md`
6. `PACK_ROOT/profiles/round-comparison.md`
7. `PACK_ROOT/profiles/previous-etw-facts.md`
8. `PACK_ROOT/wpa/wpa-facts.md` 和 `PACK_ROOT/ibs-facts.json`
9. `PACK_ROOT/profiles/assess-ext-event-config.txt`
10. `REPO_ROOT/CMakeLists.txt`、`README.md`、`include/dsmvc/engine.hpp`
11. `src/vs_plugin.cpp`、`src/axis_plan.cpp`、`src/backend.cpp`
12. `src/cpu_executor.cpp`、`src/cpu_executor_avx2.cpp`、`src/cpu_packed.hpp`
13. `python/dsmvc.py`、`tests/engine_tests.cpp`、`tests/vs_integration.py`
14. `benchmarks/README.md`、`benchmarks/vspipe_benchmark.vpy`、`benchmarks/benchmark.py`，以及 pack 中的 profile/consolidation 脚本

如果源码只存在于 pack 中，请优先使用 `PACK_ROOT/code/`。不要解压、生成临时源码副本或修改原始仓库/pack；如果某个文件只存在于 source zip 且当前工具不能直接读取，就标注“未直接验证”，不要为了读取它运行解压或构建步骤。旧版 `descale.dll` 可能不在交付包中；若无法取得旧版源码，只能使用公开行为、签名、测试和 benchmark 事实，不能臆测旧实现细节。GetNative-VF 仅作为外部算法参考，不是构建依赖；缺少其源码时明确标注哪些对比未直接验证。

## 证据优先级

把 `PACK_ROOT` 中已经分析好的 profile/benchmark facts 当作本次审核关于性能和环境的唯一 truth source。优先级固定如下：

1. `review-facts.json`、`throughput/benchmark.json`、`profiles/*facts.json`、`wpa/wpa-facts.json` 中的结构化事实；
2. 对应的 Markdown 表格和 provenance/checksum；
3. 源码、测试和 README 对机制、契约和实现意图的描述；
4. 任何外部知识、模型记忆或“按经验应该如此”的推断。

源码可以说明“代码试图做什么”和“某个瓶颈机制是否可能”，不能覆盖 profile 已测出的事实。若源码直觉与 profile 数字冲突，报告冲突并以 profile 数字为测量事实；不要自行重跑来解决冲突。所有性能结论都要标注 `measured`、`inferred` 或 `hypothesis`。

## 已知契约与待核实假设

以下内容是项目声明或当前摘要中的事实候选，不是免审结论。对每一项回到源码、测试或 profile 证据核实：

- 插件 ID 为 `com.dsmvc.descale`，只注册 `dsmvc` namespace，调用形式为 `core.dsmvc.X(...)`；实现 API3 入口，不注册 API4 入口，也不提供 `core.descale` 别名。
- 公开滤镜为 `Debilinear`、`Debicubic`、`Delanczos`、`Despline16`、`Despline36`、`Despline64`、`Descale`；末尾追加可选 `backend:data`。
- `backend=auto/cpu` 解析到 CPU；`metal`、`vulkan`、`cuda` 首版应稳定地返回显式 unsupported 错误，不得静默回退。
- `opt=1` 表示 scalar，`opt=2` 严格要求 AVX2+FMA；默认自动选择，其他数值保持旧行为语义。请确认 API、wrapper、C++ dispatch、错误路径和测试是否完全一致。
- planner 是 inverse-only：使用 Float64 几何/CSR/带状 LDLT 构造，执行端持有不可变 Float32 系数；不保留不需要的 GetNative forward projection 表。
- planner 有精确 key、single-flight、带 entry/bytes 上限的 LRU，geometry cache 与完整 plan cache 分离；AVX2 packed plan 另有共享缓存。
- AVX2 后端包含 b1、b3 的特化求解路径，并为 b5/b7 使用共享的 paired-column path；其他带宽使用 generic vector fallback，较大输入使用列 tile 和最多 4 个 worker。
- `src/vs_plugin.cpp` 延迟到首个 frame 建 plan，并使用 VS frame request 生命周期；Python wrapper 负责 RGB、GRAY、YUV、位深、subsampling、chroma conversion 和 custom alias 行为。

## 静态审核边界与自我校验

使用 `high` 或 `xhigh` effort。可以把互不依赖的静态检查交给子代理并行完成，例如：

- planner/reference：核对 GetNative-VF、原版 descale 公开行为、几何和数值算法；
- SIMD/CPU：核对 AVX2/FMA、memory layout、tile、thread pool、尾部处理和 ISA dispatch；
- VS/API/benchmark：核对 API3 ABI、wrapper、frame scheduling、profile 证据和 benchmark 语义；
- verifier：独立复查高优先级 findings、行号、profile 数字和证据边界。

不要要求或输出内部思考过程。只输出可复核的结论、源码位置、pack 事实和建议。本任务是**静态审核**：禁止编译、运行任何项目代码、运行 Python/VSPipe、加载 DLL、运行 VapourSynth、运行 benchmark、运行 uProf/PCM/TBP/ETW/WPA、执行测试、访问 GPU/PMU，或用新实验替代 pack 中的 profile truth。禁止修改、解压、生成、删除或覆盖任何文件，禁止提交 git、替换 DLL、覆盖 VS 安装目录。只允许静态读取、搜索、解析已经存在的 JSON/CSV/Markdown/文本和查看源码；如果工具会执行代码或改变文件，就不要调用。可以在报告中设计未来的验证实验，但不能在本次审核中执行。

长任务中建立阶段性静态校验：每完成一个子审阅，就用实际读取到的文件、行号或 pack 字段核对结论；没有读取或无法定位的内容必须标为 `unverified`。进度消息只能报告已读取/已核对的证据，不得暗示运行过测试或 profile。除非遇到只能由用户提供的外部输入，否则不要停在“要不要继续”的问题上。

最终审核报告以中文为主，保留英文符号、函数名、路径、性能计数器和错误文本。先给 TL;DR，再给 findings；不要先写漫长的背景介绍。报告应让另一位工程师可以按文件、行号和 pack 字段复核每个重要结论，并把未来实验单独标为验证计划。

## 审核主线

### 1. 先建立架构和调用图

从 `core.dsmvc.*` 的注册和 Python wrapper 开始，画出简洁的调用链：

`VS API3 registration -> filter_create/filter_get_frame -> argument normalization -> AxisRequest -> planner/cache -> packed CPU plan -> horizontal/column inverse solve -> output frame`。

同时标出：

- 哪些状态属于全局 cache，哪些属于 filter instance，哪些属于 frame/request；
- plan、geometry、packed plan、custom kernel callback、worker pool 的所有权和生命周期；
- API3 host 的线程/请求模型与插件内部 worker pool 的关系；
- backend capability interface 在当前 CPU 路径和未来 GPU 路径之间的边界。

对每条边说明可能的锁、分配、同步、复制、格式转换和异常路径。若架构图和源码不一致，以源码为准并列为 finding。

### 2. Planner 前端和数学算法

重点审阅 `src/axis_plan.cpp`、`include/dsmvc/engine.hpp`，并与 GetNative-VF planner 算法对照：

1. 逆向几何的 ratio、pixel center、active length、shift、source/destination 尺寸、half-open/closed 区间是否与 descale 的行为一致。
2. mirror、repeat、zero border 的边界映射是否在负坐标、最后一个像素、单像素/极小尺寸、奇偶尺寸和极端 shift 下正确；是否存在 `size - 0.5`、舍入、负零、NaN、Inf 或整数溢出问题。
3. tap coalescing、权重归一化、重复 index 合并、零权重处理、custom kernel 的回调语义是否改变输出；自定义 kernel 的异常、非有限值和副作用是否有明确契约。
4. Double CSR 到 transpose CSR、带状矩阵构造和 LDLT 分解是否保持稀疏结构、对称性、带宽边界和数值稳定性；Float64 规划后转 Float32 执行的误差是否在所有格式/几何下可控。
5. `PlanKey`、`GeometryKey` 的 bit-level key 是否遗漏影响输出的参数，是否错误地区分或合并 `-0/+0`、NaN、不同但等价的参数；自定义 kernel 为何 bypass cache，是否有更合理的身份/版本策略。
6. `SingleFlightLru` 在并发 build、builder 抛异常、等待者唤醒、eviction、clear、共享 `shared_ptr` 和驻留字节统计下是否无死锁、悬挂引用、重复构造或统计错误。
7. planner 复杂度和分配热点：哪些循环是 `O(source_size * support)`，哪些是隐含的线性搜索/重复排序/重复初始化；geometry cache 能否真正复用 Bicubic 参数族；plan cache 的上限是否与 getnative sweep 的多 GiB 工作集相符。
8. 对比原版 descale 与 GetNative-VF 的职责边界：哪些是算法本体差异，哪些只是数据结构/缓存/执行器差异；不要把“省掉 forward projection”直接当成 end-to-end 加速，要求给出可测量的阶段成本。

请明确指出：planner 的任何优化是否会改变边界像素、kernel normalization、force/force_h/force_v 或 custom kernel 的公开行为。若能提出替代方案，请比较精度、内存、首次 frame 延迟、并发可复用性和实现风险。

### 3. CPU executor、SIMD 和 ISA

重点审阅 `src/cpu_executor.cpp`、`src/cpu_executor_avx2.cpp`、`src/cpu_packed.hpp`：

1. scalar `inverse_axis_f32` 是否是可靠 oracle；AVX2 b1/b3 特化与 paired b5/b7 路径是否数学等价，特别检查 forward recurrence、backward recurrence、带宽顺序、边界条件和 output reuse。
2. 逐一检查 `row_count`、`column_count`、stride、padded source/destination、非 8/16 对齐的尾部；确认 AVX2 load/store 不会越界或读写未定义 padding，确认 fallback 的最后 8 行/最后若干列不会重复或丢失数据。
3. 检查 `inverse_rows_avx2`、`inverse_columns_avx2`、`solve_columns_b1/b3/pair/vector` 的工作划分、列 tile、`thread_local` scratch 和 generic fallback 的内层循环。解释为什么固定 b1/b3 能走专用 path，以及 b5/b7 paired path 是否仍有不必要的分支、标量广播、重复 address calculation 或 cache miss。
4. 复核 `/arch:AVX2`、FMA、`/fp:fast` 与其他目标的 `/fp:strict` 组合；分析 FMA contraction、reassociation、denormal/rounding、编译器版本和 CPU dispatch 对新旧输出一致性的风险。`opt=2` 在编译时没有 AVX2、运行时没有 AVX2/FMA、OSXSAVE/XCR0 不满足时是否都能给出确定错误。
5. 检查 `CpuExecutor` 的 copy/move、`Impl`、packed plan identity、sealed cache、静态 weak cache 和 worker pool 的线程安全。重点寻找：静态析构顺序、barrier 永久等待、异常传播丢失、`in_use_` 导致的串行 fallback、多个 VS frame 同时调用时的 reentrancy、worker oversubscription 和 VS 自己的 thread pool 竞争。
6. 将 profile 热点映射到代码：当前事实中 fixed r32 的 dsmvc 占比约 97-99%，columns 是最大 bucket；sweep 中 `getfnative` 的 dsmvc 比例约 16.40%、`getfnative_v2` 约 26.22%、`selectkernel` 约 40.35%，外部 kernel/VapourSynth/VC runtime 贡献很大。解释哪些优化只能提升核心 kernel，哪些能提升完整 VSPipe 吞吐。
7. 根据 AMD Ryzen 9 5950X/Zen 3 的 256-bit AVX2、L1/L2/L3、load/store、FMA throughput 和内存层级，判断当前实现受算术、load/store、依赖链、cache locality、带宽、线程调度还是 planner/分配限制。不要用 socket-wide PCM 单独证明某个进程已经 DRAM 饱和。
8. 评估具体替代方案，并给出预期收益区间和验证方法：更细/更粗的 tile、fused horizontal+vertical、转置布局、AoS/SoA、按 band/block 的 solve、预取、对齐、非 temporal store、plan/packed-plan 合并、每线程 scratch、固定工作队列、避免 VS oversubscription、AVX-512/AVX10/其他 ISA 的条件性收益。只有能指出瓶颈机制和测量方法的方案才算有效建议。

请给出一个 kernel/ISA 矩阵，至少覆盖：Bilinear、Bicubic、Lanczos3、Spline16、Spline36、Spline64、custom kernel；列出 planner support/half-bandwidth、executor 实现、SIMD 特化、profile 热点、预期收益和风险。明确哪些算法在当前 kernel 调优中受益，哪些只会因 wrapper/planner/cache 改善受益。矩阵中的收益必须标为 `measured`、`inferred` 或 `hypothesis`，不能把静态推测写成 benchmark 结果。

在 SIMD 审核中主动提出新的、尚未实现的探索方向，而不是只评价已有 AVX2 代码。至少从以下角度各提出若干候选，并说明它们适用的 kernel/尺寸/requests：

- **内存带宽和数据量**：估算每输出像素的输入读、系数读、LDLT 中间值读写和最终写回；找出重复 pass、重复转置、过宽 stride、系数复制、无效 padding 和可以消除的临时 buffer；判断 profile 的 38-40 GB/s socket-wide PCM 是否与“带宽受限”假设相容，但不要把它升级为进程级 DRAM 证明。
- **缓存和 locality**：分析 L1D/L2/L3 的工作集、tile/reuse distance、forward/backward recurrence 是否让中间行变冷、packed plan 与 frame data 是否争用 cache、TLB/page locality、对齐、bank/port 压力、预取和 store policy；区分能由源码证明的访问模式、由 assess_ext/PCM 支持的线索和必须新测的 cache-level 结论。
- **SIMD/计算**：分析 FMA 数量、broadcast/load 比例、依赖链、循环展开、寄存器压力、mask/tail、aliasing/alignment、编译器自动向量化和函数边界；比较 b1/b3 特化与 b5/b7 paired path 的机会；讨论 AVX2/FMA、AVX-512/AVX10 或其他 ISA 的条件性收益及频率/功耗/dispatch 风险。
- **算法和布局**：评估融合 horizontal/vertical、避免或重排 transpose、block/banded solve、按 kernel support 生成专用 inner loop、系数压缩/量化、混合精度、plan 与 packed plan 合并，以及 direct convolution 与 inverse solve 的适用边界。
- **并发和调度**：评估 VS core threads、内部最多 4 worker、`262144` work threshold、frame-level parallelism、barrier/atomic、oversubscription、NUMA 和跨 frame 复用；给出不会牺牲 API3 reentrancy 和尾部正确性的替代调度方式。
- **planner/cache/frontend**：评估几何 canonicalization、`-0/+0`/NaN key、Bicubic 参数族复用、single-flight 等待、LRU bytes/entry 上限、首帧 planner cost、packed plan lifetime 和 sweep 的多 GiB working set；提出能减少分配或 retained state 的方向。

要求最终至少列出 8 个不同的性能探索假设。每个假设使用以下字段：`Hypothesis`、`Profile signal`、`Source mechanism`、`Affected cases`、`Expected bound`、`Risk/tradeoff`、`Future experiment`、`Falsifier`。`Expected bound` 只能使用粗粒度区间或排序（例如 `<5%`、`5-15%`、`high leverage/low confidence`），不得伪造精确收益。`Future experiment` 只能是设计方案，不得在本机执行。

### 4. VapourSynth API3、wrapper 和公开行为

检查 `src/vs_plugin.cpp`、`python/dsmvc.py`、`tests/vs_integration.py`：

- API3 C ABI、plugin ID、namespace、function signature、参数顺序、`backend:data:opt` 追加语义是否稳定；是否意外暴露 `core.descale` 或 API4 入口。
- `custom`/`custom_kernel`、`support`/`taps` alias precedence、`kernel` 与 custom 互斥、`force`/`force_h`/`force_v`、border handling、尺寸验证和错误文本是否与 baseline 一致。
- `getFrame` 前后的 node state、plan build、frame ref、custom `VSFuncRef`、core/api 指针和析构是否安全；错误是否能跨 VS API 边界正确释放资源。
- RGB/GRAY/YUV、8/10/16-bit、float、subsampling、`yuv444`、chroma location、alpha/plane selection、Point conversion 和 wrapper 的 Spline36 chroma path 是否改变旧 wrapper 行为或扩大不必要的内存/时间成本。
- API3 host 的缓存/请求语义下，一个 filter instance 是否会重复 build plan，多个 instance 是否共享正确，frame parallelism 与内部 worker pool 是否可能过度订阅。

请单独回答 API4 是否可能改善：注册/参数解析、frame property、异步 request、node cache、线程调度或 GPU interop 中的哪一层。若 API4 不能改善当前主要热点，明确写出“没有证据支持迁移能提高吞吐”，不要把 API4 当作抽象上的自动性能升级。

### 5. Backend 抽象和 GPU 预留接口

审阅 `src/backend.cpp` 和 engine header：

- capability 返回值是否诚实反映“compiled”和“device available”；unsupported 错误是否稳定、可识别、不会静默 CPU fallback。
- `BackendKind` 是否只是字符串分支，还是有可扩展的 executor/plan/data contract；增加 Metal/Vulkan/CUDA 时是否会把 API3 参数解析、planner、CPU layout 和 device synchronization 耦合在一起。
- 评估 backend-neutral plan IR：哪些部分可直接复用（geometry、normalized weights、LDLT factors），哪些必须重排（packed coefficients、memory ownership、command queue、stream/event、host/device copy）。
- 讨论 GPU 后端的实际收益门槛：计划上传、每帧输入输出、同步、多个 VS request、格式转换和小图像 overhead；不能只比较 kernel FLOP。
- 给出最小可行的后端接口演进建议，要求不破坏 API3 ABI，也不把未实现 backend 伪装成可用。

### 6. Benchmark、profile 和证据边界

重新读取 pack 中的 JSON/Markdown，并核对以下事实；本节的数字只允许来自 pack，不得通过运行命令重新测量：

- old/new full-process benchmark 是独立 VSPipe 进程、500 帧、32 requests、每实现 3 次；它是当前 DLL 与旧 `descale.dll` 的吞吐对比，不是同一进程内核热点对比。
- 当前 PMU/PCM/TBP round 与 previous ETW/WPA round 必须分开；相同 plugin/input/CPU identity 不代表采集是同步 A/B。
- TBP 是 sampled CPU share；ETW module share 是另一种归一化；PCM 是 socket-wide memory traffic；process-memory polling 是工作集/私有内存高水位，不是 allocation call-site 证据。
- WPA 的 wait/ready 是聚合 thread delay；没有 Ready Thread Stack，不能指出唤醒调用栈，也不能把 `WrDispatchInt`/`WrQuantumEnd` 直接写成 dsmvc lock。
- IBS 结果已判定不可用；不要用 invalid IBS 支持 cache-level 结论。

请把 profile 热点与源码函数关联起来，并区分：

1. 已由数据直接测量的事实；
2. 多个工具共同支持但仍是推断的机制；
3. 需要新实验才能验证的假设。

针对每个高优先级性能建议，指定最小**未来**验证实验：输入/尺寸、kernel、requests、线程数、warm/cold、重复次数、比较指标、必须记录的 counter，以及怎样判定结果支持或否定该建议。实验设计应覆盖内存带宽、L1/L2/L3/DRAM、cache miss/reuse、SIMD instruction mix、FMA/load-store throughput、线程等待和 planner/cache hit/miss，但不要假设当前 pack 已测到这些维度。优先建议独立 process、固定 frame index、scalar/AVX2、r1/r8/r32、不同尺寸和不同 support 的对照；本次只写设计，不运行任何实验，也不要建议没有可观测指标的“再优化 SIMD”。

如果发现 profile 没有足够证据回答“具体是 L1/L2/L3/DRAM 哪一级限制”，正确输出是：说明已有 counters 能支持到哪一层、哪些结论不能推出，并设计下一轮测量。不能用源码访问模式、单个 PTI、socket-wide PCM 或经验知识冒充已经测得的 cache-level 瓶颈。

### 7. 测试与验证缺口

检查现有 C++/VS integration tests 和 benchmark 是否足以保护：

- planner 数值、极端几何、border、奇偶尺寸、极小尺寸、超大 taps/support、NaN/Inf/negative zero、custom callback 异常与重复 index；
- scalar 与 AVX2 的逐 plane 误差、尾部/stride/padding、随机输入、不同 destination/source 比率；
- b1/b3 特化与 b5/b7 paired path 的覆盖，Spline16/36/64 和 Lanczos taps 的 coverage；
- API3 signature/namespace/alias/force/backend error/AVX2 unavailable；
- RGB/YUV/GRAY、整数/float、subsampling/chroma location、wrapper output format 和 frame properties；
- 多线程 cache single-flight、clear/eviction、多个 filter instance、内部 pool 与 VS threads 的组合；
- old/new output hash、max error、MAE/PSNR、performance regression threshold。

对于缺失测试，按“最小成本、最大风险覆盖”排序，并给出未来测试的伪代码或命令，但不要直接编辑或执行文件。

## Findings 格式（必须遵守）

报告 findings 时按严重性从高到低排序。每条 finding 使用下面字段：

```text
[P0/P1/P2/P3] [CORRECTNESS/PERF/ABI/NUMERIC/CONCURRENCY/MEMORY/ARCH/EVIDENCE] Short title
Location: relative/path:line (or exact symbol/region if line is unavailable)
Claim: one precise sentence
Evidence: source/test/profile fact, with command or file reference
Mechanism: why this can happen; separate observed fact from inference
Impact: correctness, public behavior, memory, cold start, warm throughput, or future backend cost
Recommendation: smallest concrete fix or design change, without implementing it
Validation plan: future test/profile/experiment that would prove or falsify the claim; do not run it in this review
Confidence: high/medium/low
```

不要把纯风格偏好写成 finding。没有可定位证据的内容放到 `Open questions / unverified`，不能伪装成 bug。若一个问题同时影响 correctness 和 performance，按最严重类别列一次，并在 impact 中说明两者。

## 最终报告结构

按以下顺序输出：

1. **TL;DR**：不超过 10 行，先说是否发现阻塞性 correctness/ABI 问题，当前最大性能机会在哪个阶段，以及 profile 证据的主要限制。
2. **Findings**：按 P0 到 P3；没有 findings 时明确写 `No actionable findings at this severity`。
3. **Architecture map**：planner/cache/executor/VS/backend 的调用和所有权摘要。
4. **Algorithm audit**：planner 数学、LDLT、Float64/Float32、边界和 cache key 的表格。
5. **Kernel/ISA matrix**：七类公开 kernel 的 executor、SIMD、热点、收益和风险。
6. **Performance opportunity ranking**：每个机会的瓶颈机制、预计收益区间（例如 `<5%`、`5-15%`、`15%+`，不要伪造精确数字）、适用 kernel、实现风险和验证实验。
7. **API3/API4 and backend assessment**：保持 API3 的必要性、API4 可能/不可能改善的层次、GPU 接口演进建议。
8. **Test/profile gaps**：按优先级给出可执行的补测清单。
9. **Review verdict**：区分“可以合并/需要修复后再审/证据不足”，并列出下一轮最小行动清单。

每个数字都给出来源和采集轮次。不要把 current 与 previous round 的百分比平均，不要把 socket-wide PCM 当成单进程带宽，不要把 aggregate Ready time 当成 wall-time loss。输出应是专家可以审阅和复现的事实报告，而不是泛泛的优化建议。
