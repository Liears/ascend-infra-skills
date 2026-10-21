# P7 — 用可信仪器重做算子级归因：AIC 只有 20% 时间在算乘加（2026-10-20）

> 承接 P6 的撤回：P6 里所有"带宽利用率 X%"的说法都基于 `npu-smi` 的失效列，已作废。
> 本轮换了三个**可验证**的仪器，重新做归因，并实测了一个优化方向（负结果）。
> 设备：8×910B3（A2）。注意 **A2 每卡 20 cube + 40 vector**（A3 是 24/48，不要混用）。

## 0. 结论

**decode 是"小 batch 下的固定开销主导"，不是带宽受限，也不是算力受限。**

| 事实 | 数值 | 仪器 |
|---|---:|---|
| 单流 decode AICore 实占 | **53.4%**（空闲 0%，负载 53%，8 卡一致） | `npu-smi info watch -s a` |
| 步长 | 31.59 ms（profile 口径） | `kernel_details.csv`，HcPre 锚点 |
| **AIC 忙** | **11.94 ms（38%）** | 同上，并集去重叠 |
| AIV 忙 | 5.88 ms（19%） | 同上 |
| **AIC∩AIV 重叠** | **0.45 ms（仅 4%）** | 区间求交 |
| AIC∩通信 / AIV∩通信 | **0.00 / 0.01 ms** | 同上 |
| 完美流水理论上界 | **11.94 ms = 2.65×** | — |

**AIC 的 9.10 ms core-time 里：**

| 用途 | ms/步 | 占比 |
|---|---:|---:|
| **MAC（真正的乘加）** | **1.81** | **19.9%** |
| MTE2（HBM→L1 搬权重/激活） | 4.64 | 51.0% |
| MTE1（L1→L0） | 3.41 | 37.5% |
| **FIXPIPE** | **2.02** | 22.2% |
| **SCALAR（标量开销）** | **3.95** | 43.4% |

⇒ **矩阵核 80% 的时间不在算乘加。** 其中 SCALAR+FIXPIPE 合计 5.98 ms = **步长的 18.9%**。

## 1. 三个可信仪器（P6 用错的那个已弃用）

| 仪器 | 可信度 | 用法 |
|---|---|---|
| `npu-smi info watch -i <npu> -c 0 -d 1 -s a` | ✅ **可信**（空闲 0%、打满 100%） | 每卡一个 AI Core 芯片，`-c` 恒为 0；`-i` 是 NPU ID 0–7 |
| `kernel_details.csv`（逐算子 + 资源分解） | ✅ **可信** | 由 profiler 在**应用运行时**产出；字段见下 |
| `npu-smi info` 快照的 `AICore(%)` / `HBM Bandwidth Usage Rate` | ❌ **不可信** | 空闲读 93%、压满读 0%；NPU 行第 4 字段是 `Power(W)` 不是利用率 |

`kernel_details.csv` 里最有价值的几列（此前一直没被用起来）：

| 列 | 含义 |
|---|---|
| `aicore_time(us)` | 该算子占用 cube 的时间 |
| `aic_mac_time(us)` | 其中真正做乘加的时间 |
| `aic_mte2_time(us)` | 其中从 HBM 搬数据的时间 |
| `aic_fixpipe_time(us)` | 其中 fixpipe（量化/回写）时间 |
| `aic_scalar_time(us)` | 其中标量指令时间 |
| `aiv_time(us)` / `aiv_vec_time` | 向量核占用 |

**MAC 占比 = 算力是否真的在用。** 19.9% 意味着不是。

## 2. 最大的单项：SparseFlashMla 占步长 10%，MAC 只占它自己 12.7%

| 量 | 值 |
|---|---:|
| 次数 | 18.6 /步（每层约 0.46 次） |
| 时长 | 3.17 ms/步（**10.0% 步长**） |
| 其中 MAC | 148.3 ms 累计（**仅 12.7%** of aicore） |
| 其中 MTE2 | 621.6 ms（53.3%） |
| 其中 FIXPIPE | 658.6 ms（56.4%） |
| 其中 SCALAR | 506.1 ms（43.4%） |
| 单次 | 170 µs，**只有 6 个 token** |
| Block Num | 20（= A2 全部 cube 核，已开满） |
| 实测算力 | **0.148 TFLOPS** |

`170 µs / 6 token = 28 µs/token`。这是典型的**小 batch 固定开销主导**：
启动 20 个核、走完 FIXPIPE/SCALAR 流水所需的时间，远大于 6 个 token 真正需要的乘加。

## 3. 实测过的优化方向：MoE 权重预取（**中性，无收益**）

依据 cannbot `model-infer-prefetch`：GMM 是 memory-bound（MTE2 51%），且有安全依赖窗口
（gate 算完 → experts 前）。`torch_npu.npu_prefetch` 官方文档明确"该接口支持图模式"，
且 vllm-ascend 全仓**零处使用**。

实现：在 `DeepseekV4MoE.forward` 里，`router_logits = F.linear(...)` 之后、
`self.experts(...)` 之前插入
`torch_npu.npu_prefetch(w13_weight, router_logits, max_size, 0)`，
开关用**运行时文件**（`/tmp/v41_moe_prefetch`，0.25 s 轮询）——这样能在
**同一次引擎会话内切换**做 A/B，彻底消除预热差异（见 §3.1）。

**交替 A/B 协议**（off→on→off→on，各 8 次探针取中位，引擎已充分预热）：

| 状态 | 第 1 轮 | 第 2 轮 | 平均 |
|---|---:|---:|---:|
| off | 144.9 | 139.6 | 142.3 |
| on（566 MB = w13 全量） | 146.0 | 141.7 | **143.9** |

差异 **+1.1%**，远小于单轮测量噪声（134–157）。**判中性：既不加速也不减速。**

> ⚠️ **第一次做这个实验时得到的"−19%"是错的**——那时每个配置重启引擎后只测 8 次，
> 全部落在预热爬坡期（见 §3.1），把预热差异误当成了预取代价。用交替协议重做后消失。

**中性结果的信息量**：如果 AIC 的空闲是在等权重数据，预取应该能加速。它没有
⇒ **空闲不是在等数据**。这与算子归因（§2 的 SCALAR/FIXPIPE 主导）互相印证。

补丁存档：`moe-prefetch-NEUTRAL.patch`（运行时开关版），默认关闭。

### 3.1 ★ 方法论教训：这台引擎要预热 15+ 轮才进稳态

重启后逐次探针（同一配置，未做任何改动）：

| 次数 | tok/s |
|---|---|
| 1–3 | 161–163 |
| 4–5 | 145–150 |
| 6–8 | 130–148 |
| 9–11 | 152–164 |
| 12–14 | 160–168 |
| 15+（稳态） | **163.4–166.6**，中位 164.8 |

**含义**：
1. 任何"重启引擎 → 测 8 次"的 A/B 协议**全部不可靠**——预热期的波动（130–168）
   比大多数优化的真实效果还大。
2. 正确协议是二者之一：
   - **预热 ≥15 轮再测**（每次配置切换后都要重新预热）；或
   - **运行时开关 + 交替 A/B/A/B**（推荐，见 §3 的做法）——慢漂移被平均掉。
3. 本文件 §3 之外，**本会话更早的几组重启式 A/B 也受此影响**：
   AllToAll 实验（163.3 vs 165.6 的"+1.4% 无差异"）方向上可能仍对（差异本就小），
   但严格说需要用交替协议重做才能定论。路径 A vs 路径 B（163 vs 119）差异 27%，
   远超预热效应（~15%），结论仍然成立。

## 4. 下一步该往哪（按证据排序）

| # | 方向 | 依据 | 预期 | 状态 |
|---:|---|---|---|---|
| 1 | **减 SCALAR+FIXPIPE（5.98 ms/步 = 19%）** | 这两项不是计算也不是必需搬运 | 若砍半 ≈ 1.5× | 未做 |
| 2 | **SparseFlashMla 的小 batch 开销** | 10% 步长，MAC 仅 12.7% | 若对齐大 batch 效率可达数倍 | 未做 |
| 3 | 让 AIC 与 AIV 真并行（当前重叠 4%） | 完美流水上界 2.65× | 1.4–2.6× | 需拆批，实测代价 ×1.7（P6）⇒ 已否决 |
| 4 | 通信与计算重叠（当前 0.00 ms） | 通信 5.12 ms=16% | 最多 1.19× | 未做 |
| 5 | MoE 权重预取 | — | — | **本轮实测否决** |

**#1 和 #2 是同一件事的两面**：都是"小 batch + 固定开销"。真正该问的是
**为什么 SCALAR 要花 3.95 ms**——43% 的 cube 时间在跑标量指令，这在正常 GEMM 里不应该出现。

## 5. 复现方法

```bash
# AICore 真实占用（唯一可信读法）
python3 aic.py 5           # 空闲应 0%，负载应 ~53%

# 算子级归因（需要 kernel_details.csv；该文件由 profiler 在应用运行时产出）
python3 per_step.py  <...>/ASCEND_PROFILER_OUTPUT/kernel_details.csv   # 逐算子 ms/步
python3 aic_split.py <...>/kernel_details.csv                          # MAC/MTE2/FIXPIPE/SCALAR
python3 overlap.py   <...>/kernel_details.csv                          # AIC∩AIV 重叠、空闲
python3 scalar_ops.py <...>/kernel_details.csv                         # 谁在烧 SCALAR
python3 sparse_detail.py <...>/kernel_details.csv                      # SparseFlashMla 明细
```

`msprof --export` 在本机不可用（需要应用在同一进程里运行），但**`kernel_details.csv`
可以在 profiler 正常 stop 后直接读到**——前提是那次 profile 是由 profiler 自己
在运行时完成的（产物目录里会有 `ASCEND_PROFILER_OUTPUT` 和 `FRAMEWORK`）。
