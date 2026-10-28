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

## 风险

- ~~Triton tl.dot 小块效率~~（已实证：69µs 上限）
- ~~npugraph 捕获约束~~（已实证：捕获可行，乱序是另一回事）
- 路线 2：AscendC 开发周期与 csrc 构建链的学习成本；官方核的 tiling 机器
  剥离后需重验数值（拟合台现成）
