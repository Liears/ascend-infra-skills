# P12 — 立项：target 路径 decode 融合算子（2026-10-28）

> P10 §14 关闭了 torch 物化组合（6.8× 性能劣化，死因=15+ 算子 × 36 层的
> launch/标量开销），但**需求本身没有变小**：P7 归因显示 SparseFlashMla
> 占步长 8.3%（18.6 次/步 × 170µs/次），其 decode 需求仅 0.65MB/63MFLOP
> （MAC 利用率 12.7%，99% 是通用 tiling 机器的固定开销）。
> 本项目用一个**单派发融合核**吃掉这份开销。

## 目标与验收

| 项 | 门槛 |
|---|---|
| 形状 | decode CSA：T∈{6,12,18,24}（R=T/6），W=128+K=512 列，H=16/rank，D=512，bf16 |
| 数值 | 与官方 `npu_sparse_flash_mla` bf16 级一致（离线拟合台，复用 P10 语义规格） |
| 性能（离线） | 图内设备时间比官方 **≥30% 快**（µbench，NPUGraph 捕获计时） |
| 性能（引擎） | 交替 A/B ≥2% tok/s，A=5.89，GSM8K 100/100 |

## 技术路线（按投入递增，逐级设止损）

1. **Triton-Ascend 单内核**（首选）：本仓 vllm_ascend/ops/triton 系已在
   npugraph_ex 生产环境验证；P9 的 `mqa_triton.py` 骨架（online-softmax +
   tl.dot + 索引 gather）直接扩展——加 ori 滑窗/per-token 位置/block table
   两级寻址/sinks/双源拼接。图内单次派发，launch 开销被 replay 消化。
   P9 图外测得的 ~150µs Triton launch 税在图内不存在（vllm 自家 Triton
   核全在图内跑）。
2. **AscendC 定制核**（若 Triton 差）：csrc/attention/sparse_flash_mla/
   有官方 arch22 全源码 + 构建系统，可复制剥离 decode 专路。
3. **止损**：若路线 1 离线性能 < 官方 90%（即更慢），直接转路线 2 评估，
   不做无谓打磨。

## 语义规格（P10 §12-13 逆向成果，施工图）

decode（cseq=原始单位）：
- 行号 = cmp 索引值直用；条目全部消费；`cbt_v>=0` 才有效
- 非有限 score 列跳过（NaN-V 垃圾块防护）
- ori 滑窗 [pos-127, pos] 因果；pos = seq_lens - q_len + (t - cu_start)
- sinks 加分（fp32）→ softmax → 双源 V 加权
- 两级寻址：col → (bt[req, col//128] × 128 + col%128) → 池行

prefill（eager）不动，继续走官方算子。

## 里程碑

| # | 内容 | 产出 |
|---|---|---|
| M1 | 离线 µbench：官方算子图内基线（精确 decode 形状） | ✅ **42-61µs/call**（eager 237-244） |
| M2 | Triton 核 + 数值对拍 | ✅ **med=0.00000 big=0%（四个形状全对）** |
| M3 | 图内性能对决 + 参数扫描 | ❌ **止损触发：最佳 69µs vs 42µs（164%），路线关闭** |
| M4 | 引擎接线 | 不再需要 |
| M5 | 上游化 | — |

## M1-M3 实录（2026-10-28/29）

### 基线（NPUGraph 捕获计时，n=30×20 中位）

| T | R | 官方 graph µs | 官方 eager µs |
|---|---|---|---|
| 6 | 1 | **42.4** | 237.7 |
| 12 | 2 | 44.6 | 236.1 |
| 18 | 3 | 46.5 | 241.1 |
| 24 | 4 | 61.3 | 243.5 |

（P7 引擎归因的 170µs 是混合形状平均；精确 decode 形状的图内真实成本 42-61µs。
eager 237µs 印证图外测量的 launch 税。）

### Triton 路线战报（v1→v4 五轮）

| 版本 | 结构 | 结果 |
|---|---|---|
| v1 | 每 (t,h) program, 元素乘+reduce | 编译过（缩块后）；**274µs**，NaN 防护比官方还干净 |
| v2 | 每 t program, tl.dot 双循环 | bishengir 编译失败 |
| v3 | prep+attn 两核拆分, 双池 where-load | where-of-two-loads 编译失败 |
| v5 | unmasked gather + vm-where | **数值 med=0.00000 big=0%（四形状全对）**；89µs（attn 69 + prep 20） |
| v4 | 列 5 分裂 + merge 核 | attn 部分结果完美、merge 单测完美，**但跨核乱序（eager 10/10、图内同样）** |

**数值目标达成**（P10 语义规格 → 可运行 Triton 核，与官方逐位一致）；
**性能目标未达**：单核上限 69µs（T=6）= 官方 1.65×；赢面需要的列分裂并行被跨核
乱序封死。

### Triton-Ascend 3.2.0 工具链四硬伤（本轮实证）

1. BK≥128 的 dot 块 → MLIR "tiling basic block too large" 编译失败
2. masked + gathered 二维 load 组合 → ConvertLinalgRToBinary/PlanMemory 失败
3. **相邻 Triton 核背靠背发射跨核乱序**（bishengIR 生成核内部侧流异步存储；
   `stream=` 参数已废弃且"当前流"承诺不兑现；eager/图内均复现，重跑时对时错）
4. P9 时代能编译的核（mqa_triton.py）现在编不过——工具链向后不兼容

（"本仓 Triton 已在生产验证"的前提也证伪：rms_norm.py 的 Triton 核无任何
生产引用，生产 norm 走 _C_ascend 融合算子。）

### 判决与路线 2 交接

- **路线 1（Triton）按章程止损关闭**：最佳 69µs < 90%×42µs=37.8µs 门槛
- **路线 2（AscendC 定制核）**：`csrc/attention/sparse_flash_mla/` 有官方
  arch22 CSA 全源码 + 构建系统（ninja/CANN op build 已在用）；复制剥离
  decode 专路（T≤24 固定、W+K=640 列单趟、去掉通用 tiling 机器）。
  预算量级：多天 C++/AscendC 工程。语义规格（P10 §12-13）+ µbench 台
  （tools/ubench_decode.py）+ 数值参考（tools/smla_decode_v4.py 的 torch
  合并版可当独立验证器）全部就绪。

## 2026-10-29 补充：重新归因推翻 M1 的悲观锚——奖金额回到 10%+

用 vLLM profiler（`--profiler-config '{"profiler":"torch"}'` +
`/start_profile`）对**当前生产栈**跑 384-token decode 窗口，rank0 的
`op_statistic`（msprof 导出，CSV 归档 `op_statistic_20261028_current_stack.csv`）：

| 排名 | 算子 | 占设备时间 | 次数 | avg |
|---|---|---|---|---|
| 1 | **SparseFlashMla** | **14.6%** | 2560 | **144.3µs** |
| 2 | HcPre | 11.4% | 5498 | 52.5µs |
| 3 | GroupedMatmulSwigluQuantV2 | 8.7% | 2749 | 80.2µs |
| 4-9 | QuantBMMV3/MatMulV2/aiv_all_reduce/TBMM/HcPost/GMM | 各 4-6% | | |
| — | QuantLightningIndexerV2 | 1.9% | 512 | 92.9µs |
| — | SparseFlashMlaMetadata | 1.1% | 192 | 144.4µs |

**关键修正：引擎内 SparseFlashMla 实测 144.3µs/次**——M1 离线 µbench 的
42µs 低估了 3.4×（MIX_ 型算子的 profiler 计口径含子核流水占用，和/或引擎
调度条件比 µbench 重）。P7 的 170µs 归因反而是准的。

**奖金额重估**：若 AscendC 专核把 144µs 压到 25-30µs（需求仅 0.65MB/63MFLOP，
带宽下限 ~5µs），可省设备时间 369→~75ms / 2529ms 总量 ≈ **10-12%**。
路线 2 的多天投入完全正当。

其余榜单情报：
- 通信机器（HcPre/HcPost/aiv_all_reduce）合计 22.5%——P2 理论地板之上的
  每次调用开销（52µs×5498）可能是另一个杠杆
- MoE grouped matmul 链 19.2%
- TransposeBatchMatMul 5.8%（2749 次 53.5µs）——上游布局问题？

### 路线 2 开工清单（下一会话）

1. csrc 构建链打通：复制 SparseFlashMla 为 decode 专用 op（改名
   `SparseFlashMlaDecode`），CMake 注册，ninja 编译，装进
   ASCEND_CUSTOM_OPP_PATH vendor 目录，离线 µbench 复测（先不改代码，
   确认副本与官方逐位一致 + 计时同口径）
2. 特化剥离：固定 T≤32/W+K=640/单趟调度，去掉 metadata 驱动的多核分派
3. 数值验收：拟合台（bf16 级）→ µbench（对官方 ≥3× 快）→ 引擎三件套

## 路线 2 M1 达成（2026-10-29）：改名副本编译+安装成功

`SparseFlashMlaDecode` 已编过全链并装进 vendor 包（源码补丁归档
`smla-decode-op-m1.patch`，61 文件 +21962 行）：

- 构建链：`build_aclnn.sh`（名单驱动）→ CMake/ninja（增量，csrc/build
  保留）→ CPack .run → 安装到 `_cann_ops_custom/vendors/custom_transformer`
  （会先清空重建，安装前备份了原包 `csrc/_cann_ops_custom.bak`）
- 三个坑（全有解）：
  1. ODR：副本的 `GetOptionalStorageShape`/`SMLATilingCheck::` 等无
     前缀助手与原版撞符号 → infershape 助手加 `static`、全目录
     `SMLA→SMLAD` 类名前缀重命名
  2. 构建中途失败留下陈旧 `aic-ascend910b-ops-info.json`（ninja restat
     时序）→ 删 json 强制从 ini 重生成
  3. 安装会清空 vendor 目录且引擎正在用 → 备份 + 停机安装
- 验收：44 算子注册（`SparseFlashMlaDecode` ✓）、`aclnnSparseFlashMlaDecode`
  符号导出（×2）、内核 `.o` 落 `tbe/kernel/ascend910b/`、
  **重建包驱动生产引擎健康（A=5.89、148.3 tok/s 中位）**

### 下一步（M2：离线驱动）

写 pybind JIT 包装（复刻 cann_ops_transformer 的
`ops/csrc/sparse_flash_mla.cpp` 模式调 `aclnnSparseFlashMlaDecode`），
拟合台对拍副本 vs 原版（应逐位一致——同代码），µbench 同口径计时，
然后开始真正的特化剥离。

## 路线 2 M2 达成（2026-10-29）：副本逐位一致 + 离线驱动就绪

- **pybind JIT 包装**（`tools/smla_decode_wrap.cpp`）：sed 克隆原版包装，
  换 aclnn 调用名；JIT 需 5 个 include/链接修正（graph 头、libascendcl、
  libtorch_npu、vendor lib 的 LD_LIBRARY_PATH）
- **冒烟**（`tools/decode_op_smoke.py`）：
  `T=6: bit-diff max=0.000000 nonzero=0, 41.5µs`；
  `T=24: 0.000000, 60.2µs`——**改名副本经全链与原版逐位一致、同速**
  （JIT 包装→aclnnGetWorkspaceSize/aclnn→自编译内核二进制）
- 开发循环定型：停引擎→改 tiling/kernel→`build_aclnn.sh` 增量→smoke
  对拍+计时→恢复生产

### 下一步（M3：特化剥离——真正开始动刀）

按 144µs 引擎实况 → 25-30µs 目标，剥离方向（对照
`sparse_flash_mla_decode_tiling.cpp` 2539 行的通用机器）：
1. 固定 decode 假设：T≤32、W=128+K=512、H=16、D=512、bf16、PA_BBND
   —— 让 tiling 走单分支（去掉 mode/topk/layout 的全排列组合）
2. 调度简化：`SplitBalanced`/多核分派按 decode 形状特化（S2=512+128
   单趟、G=1 固定）
3. 验收：smoke 逐位（同一语义）→ 拟合台 bf16 级 → µbench ≥3× 快
   → 引擎接线（dsa_v41 调 `_C_ascend` 处换调用）→ 三件套

## M3 前置判定（2026-10-29 夜）：时间对 SEQ 完全平坦——固定开销坐实

K 缩放实验被 checker 锁死（cmp_topk 只允许 512/1024），改 SEQ 缩放
（T=6/R=1/K=512，NPUGraph 图内）：

| SEQ | orig µs |
|---|---|
| 2048 | 41.2 |
| 8676 | 41.0 |
| 32768 | 41.2 |
| 65536 | 41.4 |

**完全平坦**——列数固定 640 时，实际数据搬运/计算可忽略，41µs 全部是
固定机器开销（kernel 启动 + metadata 驱动的分派 + 逐块 mask 机器 +
跨核归约 + FIXPIPE）。P7 的"99% 固定开销"预测被直接测量坐实。

**设计推论**：特化核不需要任何聪明调度——目标是**砍机器**。最小核
（读 q/640 slots → FMA → online softmax → 写出）理论上界 ~5-15µs；
引擎口径 144µs → 15µs = 每步省 ~2.4ms ≈ **8-10% tok/s**。

手术面（kernel 入口已读）：910B3 → `SMLAD_OP_IMPL(SparseFlashMlaDecodeCsa, ...)`
（arch22，979 行 csa_kernel.h + block_cube/vector 头）；tiling 侧
`SplitBalanced`/`DoOpTiling`（2539 行）。多核分派在 **metadata AICPU**
侧（引擎每步构建，共享受原版算子）——特化核可选择忽略 metadata 分派、
内核自算静态列划分（T≤32 时 core i 处理列 [i·64,(i+1)·64) + 定槽归约）。

## M3 手术单（行级锚点，2026-10-29 深夜测绘完成）

`ProcessBalance`（csa_kernel.h:778-873）的通用机器全貌：

```
AIC: 4× CrossCoreSetFlag(MTE2) ──┐ 立方/向量管线握手
for bN2 (批×N2):                 │
  GetActualSeqLenQ/KV/CmpKV  ← GM 逐项标量读
  for gS1 (mBaseSize 块):        │
    掩码四件套(oriMaskL/R, cmpMaskRight)
    GetSparseActualSeqLen → CountValidCmpSparseLen  ← 逐条目走索引表(标量)
    UpdateInnerLoopCond / split 数计算
    for s2 (per-core 列区间, metadata 定):
      PreloadPipeline(gloop, cmpLoop, ...)  ← 预取缓存 + MM1/MM2 流水
AIV: 4× CrossCoreWaitFlag ───────┘
```

**方案：不改这台机器，另起 `SparseFlashMlaDecodeCsaFast`**（同文件新类，
入口按形状运行时分派）：

1. **Init**：只取 q/oriKV/cmpKV/indices/两 bt/sinks/cu_q/seq_lens——
   **完全跳过 metadata 分派解析**；req/pos 由 cu_q/seq_lens 直算
   （Triton v5 已验证的广播比较法）
2. **Process**：静态划分——T≤24 × H=16 = ≤96 个 (t,h) 工作对，~40 核
   2.4 波；每核 128 ori 列 + 512 cmp 条目单趟 online softmax（复用
   `block_cube.h` 的 CopyInKv/MM 块），无 bN2/gS1/s2 通用循环、无预取
   缓存、无 CountValidCmpSparseLen（有效性 = idx≥0 & bt≥0 & 有限，
   即 P10 规格的 decode 语义）
3. **入口分派**：`sparse_flash_mla_decode.cpp` 运行时判
   `s1≤32 && CSA && bf16 && TND && PA_BBND → Fast`，其余形状回落原路径
4. **验收循环**（已定型）：增量构建 → smoke 对拍（decode 形状对官方
   bf16 级，拟合台口径）→ µbench ≥3× → 引擎接线（dsa_v41 的
   `_C_ascend.npu_sparse_flash_mla` 换 `npu_sparse_flash_mla_decode`）
   → 三件套

风险与对策：AscendC 模板契约（SMLAD_OP_IMPL 的 tiling data 消费）从简
——Fast 类只读 baseParams 的少量字段；数值验收以 P10 规格为准
（decode 形状下与官方 bf16 级一致，拟合台三重验证）。

## 风险

- ~~Triton tl.dot 小块效率~~（已实证：69µs 上限）
- ~~npugraph 捕获约束~~（已实证：捕获可行，乱序是另一回事）
- 路线 2：AscendC 开发周期与 csrc 构建链的学习成本；官方核的 tiling 机器
  剥离后需重验数值（拟合台现成）
