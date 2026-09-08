# GPU RHS 默认启用与编译参数清理（2026-09-08）

已将通过补测的有序稀疏 RHS 实现纳入三个 GPU 后端的正常构建，删除三个实验开关。默认启用的范围经过收窄：CUDA 只展开专用 H5/H7；Vulkan 只展开 32 线程的固定 H5/H7/H11 着色器；Metal 使用专用 H5/H7/H11 入口。未命中的稀疏长度继续执行原顺序循环。CPU、Float64、独立 RHS、整数输入着色器以及后端选择和自适应调度策略保持原有契约。

这里的默认启用指后端内部实现。CUDA/Vulkan 的构建开关仍默认关闭；Apple ARM64 的 Metal 构建默认开启，显式 Metal 运行仍是 CPU/Metal 混合调度。没有把 GPU 后端强行改为全局默认。

## 准入范围与补测修复

- CUDA：撤回通用路径的 H11 展开。原实验版 Lanczos6 大倍率缩小慢 8.11%，Lanczos5 慢 4.07%，blur 慢 3.42%；最终分别为 +0.07%、+0.19%、+0.02% 的直接执行耗时降幅，恢复为基本持平。H5/H7 常见稀疏计数的收益保留。
- Vulkan：从所有融合入口缩到三个专用 32 线程入口，避免通用着色器膨胀；通用 32/64 线程路径和其他固定带宽保留原实现。22 个最终 SPIR-V 与经过矩阵补测的收窄版本逐字节一致，其中 19 个也与原版一致。
- Metal：新增四个 H11 普通/转置/批处理入口，并将通用函数的固定带宽改为模板参数，使未使用的展开代码在编译时消除。整数入口仍使用原先五项索引；H11 整数输入回到通用入口。入口清单由 32 项变为 36 项，两个产物检查器验证完整集合及唯一性。
- 未将所有负值隐藏为噪声：最终 CUDA Lanczos3 大倍率缩小为 −0.99%；Metal 三个大倍率缩小案例为 −1.74%、−0.56%、−1.24%。因此这次默认实现不代表每种几何都加速；完整结果和配对范围如下。

## 编译参数审查

- 项目显式布尔开关由 9 个减为 5 个（不含 CTest 自带 BUILD_TESTING）：删除三个 RHS 实验开关；旧 DSMVC_BUILD_METAL_EXPERIMENTS 缓存只迁移一次到 DSMVC_ENABLE_METAL，然后删除旧项。四个基准/验证脚本改用规范名称。
- 将十处重复的测试/基准编译配置合并到 dsmvc_configure_tool。保留语言标准、警告、异常、严格浮点和数值参考测试所需选项。
- 移除 Apple Release 重复的 -O3、MSVC ISA 源文件重复的 /fp:strict，以及个人 D:/okegui SDK/Python/旧插件路径。保留 Full LTO、ISA 参数、静态 CRT 和 CUDA 的四个 native/两个 PTX 架构配置。
- CUDA Release 使用 CMake 的 -O3；-lineinfo 只在 RelWithDebInfo 开启，该配置仍明确用 -O3 覆盖默认 -O2。Debug 不再被强制 -O3。
- 修复 CMAKE_OSX_DEPLOYMENT_TARGET 没有传给自定义 Metal 编译命令的问题。正式比较的原版和候选均以 macOS 13.3 为目标；严格 Metal 浮点参数保持不变。
- 发现 Vulkan F64 策略测试使用 assert，Release 的 NDEBUG 会移除检查。改为始终执行的 require；用 Release 参数编译通过，并用故意失败的条件验证其非零退出。没有为这个问题增加一个编译开关。

## 正确性与构建证据

| 检查 | 结果 |
|---|---|
| 最终 Metal Release / macOS 13.3 目标 | CTest 13/13；36 个入口；最终 metallib 与测量版逐字节一致 |
| 最终 CUDA Release | CTest 9/9，含 42 个 RHS 路径/带宽组合；native SM75/86/89/120 与 PTX75/120 产物清单通过 |
| 最终 Vulkan Release | CTest 35/35；修复后的 Release F64 策略检查另行通过正例与反例验证 |
| Apple Metal 关闭的 CPU 构建 | CTest 5/5 |
| 配置审查 | Metal/CUDA 的 Debug、Release、RelWithDebInfo 实际命令；54 条 Metal/Vulkan 主机编译命令的有效选项保持一致；旧 Metal 缓存迁移后规范开关仍可正常切换 |
| 直接 GPU 图像矩阵 | 三个后端各 17 案例；每案例三组交替 A/B；所有成对最终图像 SHA-256 相同 |
| 最终 Vulkan 默认二进制复核 | 17 案例真实图像输出均与原版 SHA-256 相同，计时版与最终版 22 个 SPIR-V 相同 |
| 串行完整图验证 | 每后端 200 帧，共 600 帧；图像 SHA-256 与 PlaneStats 相同 |
| SM120 指令范围 | 28 个内核中只改变 3 个融合 inverse；其余 25 个（含 F64、独立 RHS、转换）SASS 指令不变 |
| 本机安装包审查 | arm64、minos 13.3、唯一 API4 导出、仅系统依赖、36 项 Metal 入口、安装文件与构建文件一致 |

包审查使用 --allow-dirty-source，并明确记录本地修改状态及本机 R78 头文件来源；这不是一次干净源码的发布 CI。SM75/86/89 只完成构建与产物验证，硬件执行证据限于 RTX 5080 / SM120；Metal 硬件证据限于 Apple M4 Max。没有宣称 Windows、其他 GPU 或 macOS 13.3 实机已经运行通过。

### 混合调度的哈希边界

额外的 R8T8 并发完整图检查曾在 Lanczos3 的第 6 个候选发现 SHA 不同，但 PlaneStats 完全相同。原版自身三次重复也出现这两种结果。原始 descale 输出的差异只位于宽度 1346 的最后两列：CPU 工作线程可用时走 inverse_axis_f32，忙碌回退时 NEON 尾列走 inverse_axis_f32_ordered。使用原版库分别重放这两条尾列算法，精确复现两个原始图像哈希：

```text
6718efb11fb1baa2554bf75d093a4548ee2aa918db8c290c0c28dd0ea281ae82
50a63c5ca9c9dc7fc112e96eb24aba1a402e2cb71a037b10cccb838a2c5d2677
```

原始第 6 帧最大差异 2.98e-7；经过重建/阈值后的差异只有 2 个像素、最大 1.49e-8，正负差异抵消使评分统计相同。原版 A/A 变化及首次失败记录完整保留，没有把并发 SHA 检查改写成通过。固定 GPU 路径仍执行比特一致检查；混合 CPU/Metal 调度使用项目既有的 3e-6 Float32 容差。

独立并发原始图像验证先排空异步请求再读像素，保证实际存在 CPU/Metal 混合执行。三种核各 32 个候选，新旧版本最大绝对差异如下：

| 核函数 | 最大绝对差异 | 容差 | 精确相同帧数 |
|---|---:|---:|---:|
| lanczos3 | 4.17232513e-07 | 3e-6 | 27/32 |
| spline64 | 3.57627869e-07 | 3e-6 | 30/32 |
| lanczos6 | 1.01327896e-06 | 3e-6 | 27/32 |

两版在这些并发验证中均实际执行了 GPU 帧，Metal 错误计数均为零。CPU 运行源码保持任务开始时的字节内容；此次未更改既有尾列/并行调度契约。

## 性能方法

输入为同一预解码 1920×1080 Float32 帧，SHA-256：`d50727fbfe6a6cb47eee915d5e92372292c41f7280d4d80d8db37312817e36bd`。每后端独占本次 GPU 工作，按 AB / BA / AB 交替，计时前验证实际二进制及进程状态。CUDA/Vulkan 在 RTX 5080、NVIDIA 595.84 上测试；Vulkan 显式指定 NVIDIA ICD。Metal 在 Apple M4 Max 128 GiB 上测试。

直接执行用真实 MetalFloatExecutor / CudaExecutor / VulkanExecutor，强制 CUDA/Vulkan 融合 RHS，仅有一个在途帧。建计划、读文件在计时外，至少预热 0.35 秒，再以原版校准约 3.5 秒的相同次数执行。计时是含必要传输、提交、同步和回读的主机墙钟，不是 GPU event 的纯内核时间。源缓存保持有效。4K 图像通过平铺原帧生成，tail 图像用裁切生成，不包含视频解码。

四种几何：1080 = 1920×1080→1692×952，active height 951.5；tail = 257×145→226×128，active 127.5；4k = 3840×2160→3384×1904，active 1903.5；down = 1920×1080→854×480，active 479.5。均为 Float32、Symmetric 边界、居中小数偏移。

完整流程用 VSPipe：CUDA/Vulkan R1T1、Metal R8T8，保留默认自适应 RHS / Metal 混合调度。BlankClip 测固定几何的热路径；candidate 是 700–980 高度区间的多候选重建、Expr 阈值、裁边、PlaneStats 完整图，计数按原版校准到约 6 秒，每对使用同一候选集合。输入只填充一次。图像验证在计时外进行，串行每核验证 8 个 BlankClip 帧与 32 个分散候选，并未声称所有计时帧都做了哈希。每个子进程限制 30 秒；正式性能采样实际最长 8.08 秒。

各百分比均为每对比值的中位数；正数表示改善。不能用两列各自中位数相除来替代配对比值。

### 直接 GPU 执行：耗时降幅

| 后端 | 核函数 | 几何 | 配对耗时降幅中位数 | 三对范围 |
|---|---|---|---:|---:|
| cuda | bilinear | 1080 | +0.07% | -0.56% … +0.16% |
| cuda | bicubic | 1080 | -0.63% | -0.67% … -0.07% |
| cuda | lanczos3 | 1080 | +10.57% | +10.55% … +10.58% |
| cuda | spline64 | 1080 | +7.08% | +7.03% … +7.43% |
| cuda | lanczos5 | 1080 | +0.19% | -0.12% … +0.23% |
| cuda | lanczos6 | 1080 | +0.07% | -0.04% … +0.09% |
| cuda | lanczos9 | 1080 | -0.02% | -0.31% … +0.13% |
| cuda | blur | 1080 | +0.02% | +0.01% … +0.04% |
| cuda | lanczos3 | tail | +13.24% | +13.19% … +13.27% |
| cuda | spline64 | tail | +7.49% | +7.38% … +7.78% |
| cuda | lanczos6 | tail | -0.10% | -0.17% … -0.02% |
| cuda | lanczos3 | 4k | +7.61% | +7.34% … +8.58% |
| cuda | spline64 | 4k | +6.57% | +6.35% … +7.24% |
| cuda | lanczos6 | 4k | -0.27% | -0.28% … -0.06% |
| cuda | lanczos3 | down | -0.99% | -1.03% … -0.91% |
| cuda | spline64 | down | +1.61% | +1.57% … +1.67% |
| cuda | lanczos6 | down | +0.07% | -0.28% … +0.21% |
| vulkan | bilinear | 1080 | +0.06% | -0.05% … +0.44% |
| vulkan | bicubic | 1080 | +0.00% | +0.00% … +0.06% |
| vulkan | lanczos3 | 1080 | +15.58% | +15.35% … +15.59% |
| vulkan | spline64 | 1080 | +19.84% | +19.83% … +19.86% |
| vulkan | lanczos5 | 1080 | -0.10% | -0.13% … -0.02% |
| vulkan | lanczos6 | 1080 | +19.39% | +16.15% … +19.68% |
| vulkan | lanczos9 | 1080 | +0.14% | +0.02% … +0.14% |
| vulkan | blur | 1080 | +0.01% | -0.13% … +0.12% |
| vulkan | lanczos3 | tail | +16.23% | +16.19% … +16.23% |
| vulkan | spline64 | tail | +17.16% | +17.03% … +17.20% |
| vulkan | lanczos6 | tail | +20.26% | +20.20% … +20.33% |
| vulkan | lanczos3 | 4k | +7.80% | +7.68% … +7.99% |
| vulkan | spline64 | 4k | +10.21% | +10.09% … +10.36% |
| vulkan | lanczos6 | 4k | +12.33% | +12.04% … +12.72% |
| vulkan | lanczos3 | down | +4.38% | +4.38% … +4.42% |
| vulkan | spline64 | down | +5.46% | +5.43% … +5.46% |
| vulkan | lanczos6 | down | +5.63% | +5.61% … +5.64% |
| metal | bilinear | 1080 | -0.25% | -0.27% … +0.74% |
| metal | bicubic | 1080 | -0.40% | -0.60% … +0.03% |
| metal | lanczos3 | 1080 | +27.15% | +26.02% … +27.61% |
| metal | spline64 | 1080 | +30.52% | +29.89% … +31.33% |
| metal | lanczos5 | 1080 | +0.26% | -0.45% … +0.42% |
| metal | lanczos6 | 1080 | +29.43% | +29.36% … +30.97% |
| metal | lanczos9 | 1080 | -0.05% | -0.26% … -0.02% |
| metal | blur | 1080 | -0.24% | -1.31% … +0.79% |
| metal | lanczos3 | tail | +18.28% | +17.13% … +19.54% |
| metal | spline64 | tail | +20.34% | +19.86% … +20.48% |
| metal | lanczos6 | tail | +24.58% | +23.04% … +25.22% |
| metal | lanczos3 | 4k | +26.27% | +26.12% … +26.71% |
| metal | spline64 | 4k | +28.33% | +28.19% … +28.42% |
| metal | lanczos6 | 4k | +28.41% | +28.32% … +28.52% |
| metal | lanczos3 | down | -1.74% | -2.03% … -1.29% |
| metal | spline64 | down | -0.56% | -1.08% … -0.19% |
| metal | lanczos6 | down | -1.24% | -1.38% … +0.04% |

### 完整 VSPipe 图：FPS 提升

| 后端 | 核函数 | 场景 | 配对 FPS 提升中位数 | 三对范围 |
|---|---|---|---:|---:|
| cuda | bilinear | blank | -0.01% | -0.01% … +0.27% |
| cuda | bilinear | candidate | -0.08% | -1.00% … +0.18% |
| cuda | lanczos3 | blank | +0.15% | -0.26% … +0.33% |
| cuda | lanczos3 | candidate | +1.68% | +1.59% … +1.85% |
| cuda | spline64 | blank | -0.04% | -0.19% … +0.02% |
| cuda | spline64 | candidate | +1.09% | +0.27% … +1.30% |
| cuda | lanczos6 | blank | -0.03% | -0.10% … +0.15% |
| cuda | lanczos6 | candidate | +0.07% | +0.01% … +0.35% |
| cuda | lanczos9 | blank | +0.11% | +0.05% … +0.13% |
| cuda | lanczos9 | candidate | -0.03% | -0.26% … +0.10% |
| vulkan | bilinear | blank | +0.03% | -0.07% … +0.27% |
| vulkan | bilinear | candidate | +0.26% | -0.21% … +13.74% |
| vulkan | lanczos3 | blank | +0.12% | -0.60% … +0.59% |
| vulkan | lanczos3 | candidate | +3.31% | +3.12% … +3.81% |
| vulkan | spline64 | blank | +0.16% | -0.29% … +0.57% |
| vulkan | spline64 | candidate | +6.08% | +6.07% … +6.18% |
| vulkan | lanczos6 | blank | +0.03% | -0.10% … +0.09% |
| vulkan | lanczos6 | candidate | +4.58% | +4.36% … +4.90% |
| vulkan | lanczos9 | blank | +0.00% | -0.04% … +0.13% |
| vulkan | lanczos9 | candidate | -0.02% | -0.24% … +0.07% |
| metal | bilinear | blank | -0.13% | -1.43% … +1.40% |
| metal | bilinear | candidate | +1.63% | -1.22% … +5.39% |
| metal | lanczos3 | blank | +13.37% | +11.85% … +14.23% |
| metal | lanczos3 | candidate | +4.09% | -2.52% … +4.96% |
| metal | spline64 | blank | +10.13% | -3.46% … +11.32% |
| metal | spline64 | candidate | -2.78% | -4.10% … -0.91% |
| metal | lanczos6 | blank | +5.80% | -5.21% … +8.13% |
| metal | lanczos6 | candidate | +2.13% | -9.24% … +3.73% |
| metal | lanczos9 | blank | +0.10% | -6.22% … +0.76% |
| metal | lanczos9 | candidate | -2.32% | -7.85% … -1.86% |

CUDA/Vulkan 的固定热路径通常切换到独立 RHS，因此融合优化对 BlankClip 接近持平。候选扫描在已测配置上，CUDA Lanczos3/Spline64 为 +1.68%/+1.09%；Vulkan Lanczos3/Spline64/Lanczos6 为 +3.31%/+6.08%/+4.58%。这些不能推广成所有线程数和 GPU 的结果。

Metal 直接 GPU 的常见配置收益明确，但完整候选扫描未得到稳定收益结论。补充的同一原版二进制 A/A 也有明显波动：

| A/A 核函数 | 三对 FPS 波动 |
|---|---:|
| spline64 | +22.59% / +1.10% / -2.97% |
| lanczos9 | -1.23% / +0.83% / -0.91% |

上述 A/A 数据原样保留，包括明显偏离的配对。不能把 Metal 混合图的中位数当作稳定的整个应用加速比例。

## 证据与版本

工作目录：`artifacts/gpu-default-promotion-20260908/`。`initial-source` / `initial-manifest.json` 是本任务前快照；`trial-source`、`template-source`、`final-source` 是逐次构建快照。Metal 正式性能数据来自 macOS 13.3 目标的 template 构建；最终移除宏后的 metallib 与其逐字节相同。CUDA 正式性能数据来自最终全部发布架构构建；Vulkan 正式性能数据来自收窄后的 trial 构建，最终 22 个着色器相同且 17 案例实际图像再次逐字节核验。

`final-source` 是冻结的运行时源码快照。之后只补充 README/历史报告状态、当前报告及 Vulkan 策略测试修复；该测试的最终源码单独以最终 Release 命令编译并验证。完整交付源码清单及任务差异另存，不覆盖冻结测量快照。

本地 `stage1`/`stage2` 与 Linux 归档的 `stage1`/`cuda-stage1` 保留被收窄前的结果。`metal-parallel-*`、`parallel-diag-*`、`parallel-raw-*`、`cpu-route-proof.json` 保留并发哈希边界的失败、重放和原始图像。`metal-aa` 保留同二进制调度波动。

`linux-evidence.tar.gz` 保留远端完整构建、源快照、输入、输出、命令、逐对记录、SASS、配置审查与日志；`linux-integrity.json` 记录归档和全部 3376 个文件的本地 SHA-256 核验，零差异。未创建云实例、未改其他项目、未提交或推送。
