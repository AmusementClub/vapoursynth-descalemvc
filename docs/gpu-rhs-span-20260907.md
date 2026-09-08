# Vulkan / Metal 有序 RHS 稀疏长度展开实验

> Status update (2026-09-08): the admitted routes are now enabled in normal builds and the experiment switches have been removed. CUDA wide/generic expansion was rejected during broader testing; Vulkan and Metal use bounded dedicated routes. See [default-promotion validation](gpu-default-promotion-20260908.md). Results below describe the original experiment.


已将 CUDA 的融合 F32 RHS 思路移植到 Vulkan 和 Metal，两个开关均默认关闭。Lanczos3 的直接融合执行在本机 Metal 和 RTX 5080 Vulkan 上均有收益；完整流程的结果取决于调度和计划复用，不能据此推广到所有核函数或所有 GPU。

## 最终结果

| 后端 / 场景 | 基线 → 实验（同一代表配对） | 配对收益中位数 | 配对范围 |
|---|---:|---:|---:|
| metal / 直接融合执行 | 5.326 → 3.882 ms/frame | +27.11% | +25.18% … +27.62% |
| metal / 固定尺寸 BlankClip | 1565.240 → 1702.840 FPS | +8.79% | +5.58% … +12.64% |
| metal / 32 个候选高度扫描 | 248.010 → 255.770 FPS | +3.13% | -9.66% … +8.46% |
| vulkan / 直接融合执行 | 3.653 → 3.083 ms/frame | +15.60% | +15.55% … +15.64% |
| vulkan / 固定尺寸 BlankClip | 580.180 → 580.650 FPS | +0.08% | -0.23% … +0.10% |
| vulkan / 32 个候选高度扫描 | 97.790 → 101.080 FPS | +3.36% | +2.89% … +3.48% |

**直接执行列是墙钟耗时下降，VSPipe 两类是 FPS 增加。** 每项先计算同主机每对 A/B 的变化率，再取中位数；表中的绝对数取该中位数对应的一对，以保持算术一致。直接执行各 3 对；Vulkan VSPipe 各 3 对；Metal VSPipe 各 5 对。执行顺序 AB/BA 交替。

Metal 固定尺寸流程有一致的正向结果，候选扫描仍波动较大，未建立稳定收益。Vulkan 默认自适应的固定计划热路径接近持平，候选扫描约 +3.36% FPS。两台设备的绝对时间不用于横向比较。

## 实现范围

- 新开关：`DSMVC_VULKAN_RHS_SPAN_EXPERIMENT`、`DSMVC_METAL_RHS_SPAN_EXPERIMENT`，均为 `OFF`。
- 半带宽 H5 选择稀疏元素数 6/7，H7 选择 9/10，H11 选择 13/14；其他数量、短边界和其他半带宽沿用通用循环。这里计算 CSR 稀疏元素数量，不是首尾索引之间的距离。
- 保持单一累加器、原顺序和显式 `fma`，没有分组归约或新增精度放宽。
- Vulkan 只给 F32 inverse 着色器传入实验宏；独立 RHS、solve、transpose、convert 及 F64 代码保持原有实现。
- Metal 修改共享的 F32 inverse 实现，涵盖直接和 batch 包装入口；整数输入专用内核保持原有实现。CPU 调度与回退逻辑没有修改。

## 测量边界

直接执行：同一预解码 F32 帧，1920×1080 → 1692×952，active height 951.5，Lanczos3/taps=3，F32，Mirror 边界。先准备计划、分配内存、读入源帧，至少预热 0.25 秒，再计时 64 次二维执行。包含每次调用实际发生的暂存、提交、GPU 工作、同步和回读；文件读取、计划构建、初次缓存填充、最终输出文件写入与哈希不在计时内。Metal 使用 `MetalFloatExecutor`、batch=1、threads=32；Vulkan 使用 `VulkanExecutor`，并设置 `DSMVC_VULKAN_SPLIT_RHS=0` 以测量融合路线，源帧生命周期固定以允许输入缓存。**这不是只计着色器的 GPU event 时间。**

VSPipe：原有 GetNative 风格图，`backend=metal/vulkan`、`f64mode=1`、默认 Symmetric 边界，R78。Metal 为 R8T8，Vulkan 为 R1T1。BlankClip 先预取 256 帧，再测后续 256 帧；候选扫描使用已填充一次的固定实际输入，32 个高度覆盖 700–980（含 951.5），包含重建、Expr 阈值 0.015、裁边 5 像素和 PlaneStats，并在独立几何上预热。Vulkan 此处恢复默认自适应 RHS。所有最终计时进程均小于 30 秒，最长 1.423 秒。

Metal 插件本身是 CPU/GPU 混合调度。另行执行且先收集帧、后计算哈希的并发校验中，BlankClip 256 帧分别有 40/37 帧进入 GPU，候选扫描 32 帧分别有 11/14 帧进入 GPU；GPU 分配不同，像素与统计值依然一致。此独立观察说明路线性质，不代表每次 FPS 计时具有完全相同的 GPU 分配。

## 正确性和默认关闭控制

- Metal：基线 CTest 12/12；最终开关 ON 13/13，OFF 13/13。
- Vulkan：基线 CTest 34/34；最终 ON 35/35，含 SPIR-V 校验、强制融合、强制拆分、非 coherent 内存、缓存淘汰及 F64 测试。
- 新增共享 GPU 回归测试：H0/1/3/5/7/9/11 × 横/纵两个方向，稀疏长度 0–17，非连续有序索引，33 向量尾部、stride 填充和输出哨兵；与有序 `std::fma` 逐位比较。Metal 还覆盖 batch=2。测试同时链接基线与最终实现通过。
- 现有 Metal 数值测试新增真实 Lanczos6/H11 横向、纵向和二维求解，均与有序 CPU 参考 0 ULP。
- 两端直接执行各 6 次计时输出的 SHA-256 全部一致；每端另有 33 帧 VSPipe 像素与 PlaneStats 精确 A/B 校验。Metal 另有上述 288 帧并发校验，全部精确一致，Metal 错误数为 0。
- Vulkan OFF 的全部 **22 个 SPIR-V 文件与基线逐字节相同**；ON 的 14 个非 inverse 文件也逐字节相同。最终 OFF 修复前后 ON 的 SPIR-V 逐字节相同。
- Metal OFF 的最终 **metallib 与基线逐字节相同**；OFF 修复前后的 ON metallib 也逐字节相同。预处理后的程序 token 同样通过比对。中间 AIR 文件的哈希不同，未把 AIR 宣称为相同。

## 环境与保留记录

Metal：Apple M4 Max，128 GiB，Apple Metal compiler 32023.921，MetalToolchain v27.1.5252.6；AC 供电，测试后的 `pmset` 未记录热或性能警告。Vulkan：NVIDIA GeForce RTX 5080，驱动 595.84，Linux x86_64；显式选择 NVIDIA ICD，未使用 llvmpipe；shaderc 2026.1-1 / glslang 16.1.0-1。缺少的 Vulkan 编译依赖仅通过 apt 下载并解压到本次隔离目录，没有系统安装。

源帧 SHA-256：`d50727fbfe6a6cb47eee915d5e92372292c41f7280d4d80d8db37312817e36bd`。

第一次版本的默认 OFF 声明顺序改变了 Vulkan 编译输出，已修复并重新构建、测试；`stage1/` 保留最初结果。随后另一项 GetNative/VideoToolbox 工作与 Metal 复测同时运行，这批数据保存在 `stage2-interference/` 并标记无效，未用于最终表格。等待外部作业结束后重新完成最终 Metal 配对。macOS VSPipe 初次启动缺少运行时配置，使用本次目录内的 `vs-config/` 解决，未修改全局配置。

## 复现与证据

原始脚本、日志、源代码快照、所有测量二进制和输出在 `artifacts/gpu-rhs-span-20260907/`。主要入口：`build_metal_final.sh`、`build_vulkan_final.sh`、`build_direct.py`、`direct_pairs.py`、`vs_pairs.py`、`parallel_proof.py`。`results.csv` 与 `decision.json` 保存完整配对摘要，`delivery-checks.json` 保存交付检查。

Vulkan 的独立目录已完整归档并下载，`vulkan-evidence.tar.gz` SHA-256 为 `671541603de3daf36699263f68172f8ceea318edbc9fda2e0cf391dbd4e3f206`；解压后逐个核验 2194 个文件。基线 189 个源文件中，本任务只修改 CMake、两个着色器和 Metal 数值测试，其余 185 个保持原哈希；另新增 `tests/gpu_rhs_span_tests.cpp`。最终测量源码共 190 个文件。本文档在测量后生成，不属于编译输入清单。

没有提交或推送。两个实验保留默认关闭，只支持对所测硬件、配置和 Lanczos3 案例的性能结论；H7/H11 等分支已做正确性覆盖，尚未在这次任务中测量性能。

启用方式：在对应后端已开启的 CMake 构建中追加 `-DDSMVC_METAL_RHS_SPAN_EXPERIMENT=ON` 或 `-DDSMVC_VULKAN_RHS_SPAN_EXPERIMENT=ON`。
