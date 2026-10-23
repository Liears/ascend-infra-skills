# P8 — "攻"的战报：框架层已穷尽，边界在 CANN 算子内核（2026-10-23）

> 承接 P7 的归因（SCALAR+FIXPIPE 占步长 18.9%）。本文记录沿这条线攻击到内核边界的
> 全过程：哪些路走通了、哪些被堵死、堵在哪里、正确的下一步入口在哪。
> 设备：8×910B3（A2，20 cube + 40 vector）。

## 0. 结论

**单流 decode 的优化在 vllm/vllm-ascend 框架层已穷尽。剩余的 19% 靶点
（SCALAR+FIXPIPE）全部位于 CANN 预编译算子内核内部，需要内核源码或算子团队。**

| 靶点 | 占步长 | 归属 | 框架层可动？ |
|---|---:|---|---|
| SparseFlashMla | **8.3%** | CANN `cann_ops_transformer` 内核 | ❌ |
| GMM1 (SwigluQuantV2) | 3.1% | CANN `aclnnGroupedMatmul*` 内核 | ❌ |
| QuantBatchMatmulV3 | 2.3% | 框架层（可合并投影）| ⚠️ 合并后 **<1%**，放弃 |
| GMM2 | 1.6% | CANN 内核 | ❌ |
| HcPre / MatMulV2 | 各 1.3% | mHC 结构 / 投影 | ❌ 结构固有 |

## 1. 本轮走通的路

### 1.1 SCALAR+FIXPIPE 完整归因（按算子排全序）

| 算子 | 次/步 | SCALAR | FIXPIPE | 占步长 |
|---|---:|---:|---:|---:|
| SparseFlashMla | 18.6 | 1.15 | 1.49 | **8.3%** |
| GroupedMatmulSwigluQuantV2 | 20.0 | 0.82 | 0.15 | 3.1% |
| QuantBatchMatmulV3 | 105.3 | 0.64 | 0.08 | 2.3% |
| GroupedMatmul (GMM2) | 20.0 | 0.43 | 0.07 | 1.6% |
| HcPre | 40.1 | 0.38 | 0.03 | 1.3% |
| MatMulV2 | 73.8 | 0.37 | 0.04 | 1.3% |

**SparseFlashMla 的 FIXPIPE（1.49 ms）占全步 FIXPIPE 的 74%**——
18.6 次 × 80 µs/次，是**每次调用的固定流水排空成本**，不是数据量问题
（单次输出仅 49 KB）。调用参数（`ori_mask_mode=4` 滑窗、`layout_q=TND`、
`topk_value_mode=1`）逐一核对过，均已是合理值，外部无可调项。

### 1.2 QuantBatchMatmulV3 的形状拆解（105 次/步的真身）

| shape（x; weight Nz; out） | 对应投影 | 次数 |
|---|---|---:|
| `6,5120 → 1280` | **wq_a**（q_lora 降维） | 40/步 |
| `6,5120 → 512` | **wkv**（单 KV 头） | ~40/步 |
| `6,5120 → 576` | kv_a (512+64 rope) | 少量 |
| `6,288 → 5120` | indexer 投影 | 少量 |

单次 14 µs，其中 SCALAR 约半。wq_a 与 wkv 输入相同（可合并成 5120→1792），
但合并只省 45 次调用 × ~7 µs 标量 ≈ **0.3 ms < 1%**。改动涉及量化权重 concat
与加载路径，风险中、收益 <1%，**判定不值得，放弃**。

## 2. 本轮被堵死的路（不要重复）

### 2.1 vllm 主仓的 fusion pass 全是 CUDA-only

引擎日志 `pass_config` 显示一批诱人的开关全关着：
`fuse_attn_quant` / `fuse_allreduce_rms` / `fuse_gemm_comms` /
`enable_qk_norm_rope_fusion`。**查实**：这些 pass 的 import 在
`vllm/compilation/passes/pass_manager.py:32-58` 被
`if current_platform.is_cuda_alike() / is_cuda()` 门控，
**Ascend 后端根本没有实现**——打开开关不会生效，不是配置遗漏。

### 2.2 vllm_ascend 自己的三个 fusion pass 默认已全开

`fuse_norm_quant` / `fuse_qknorm_rope` / `fuse_muls_add`（`ascend_config.py:70-72`
默认 True）。日志 "Enabled custom fusions: norm_quant, act_quant" 是 vllm 主仓
视角；ascend 侧 pass 已挂载，未命中的部分是 pattern 不匹配（结构问题，非开关）。

### 2.3 sparse_flash_mla.cpp 不是内核

`cann_ops_transformer/ops/csrc/sparse_flash_mla.cpp`（196 行）只是 host 侧
参数校验与 aclnn 调用封装。设备内核在预编译的 CANN 包内，**无源码**，
无法按 cannbot 案例做地址对齐 / AIC:AIV 比例调整。

## 3. 正确的下一步（给内核工程师 / 下个会话的入口）

cannbot `ascendc-evolution-knowledge` 的调优速查表给出了与我们的 profile
完全吻合的处方：

| 我们的观测 | cannbot 处方 | 出处 |
|---|---|---|
| SparseFlashMla fixpipe_ratio 56.4% | **FlashAttention 案例：GM 地址 512B 对齐，fixpipe 80%→55%** | quickref 案例 3 |
| GMM SCALAR 高（AIC:AIV 失配） | **GroupedMatmul 案例：AIC:AIV 比例调整 29% + workspace 份数 14% + DoubleBuffer 3% = 41%** | quickref 案例 1 |
| grouped matmul 深度优化 | **X10 策略：设备侧分组调度（empty group 跳过）+ NZ plane 权重静态缓存** | evolution-strategies X10 |

**三条可行路径**（按现实程度排序）：

1. **升级 CANN 并复测**：rc2 容器（CANN 9.2.0）已就位。在 9.2 上重采
   `kernel_details.csv` 对比 9.1 的 SCALAR/FIXPIPE——若官方算子已改善，
   路径 B 的性价比会变。注意 vllm 版本不同，需用算子级对比而非端到端。
2. **向算子团队提需求**：附上本文的 per-op 数据
   （SparseFlashMla: 18.6 次/步、M=6、fixpipe 56%、scalar 43%；
   GMM1: SCALAR 0.82ms），请求小 batch 路径的 fixpipe/标量开销优化。
3. **自定义内核替换**（重）：照端口仓 `gmm1 armF` 的 opp_override 先例 +
   cannbot X10 策略自写 grouped matmul。工作量大，仅当 1/2 不可行时考虑。

## 4. 一句话战报

**归因做到了内核门口，框架层的弹药全部打完（融合已开、通信已试、prefetch 中性、
投影合并 <1%），剩下的 19% 是 CANN 算子内部的小 batch 固定开销——
这是"要更快就得动内核"的实测证明，不是猜测。**
## 5. 补充发现：A2 走的是 SparseFlashMla 的 "HCA" 兼容路径

尝试构造该算子的独立微基准时，aclnn 逐条抛出的参数约束揭示了 A2/A3 的专属限制
（每条都是 kernel 参数校验层的原文）：

| 约束（EZ1001 原文） | 含义 |
|---|---|
| `ori_topk is reserved and should only be 0 on A2/A3` | **A2/A3 不支持 ori 侧 topk 稀疏选择** |
| `cmp_mask_mode should only be 3 on A2/A3` | 压缩侧只支持 causal 掩码 |
| `cmp_ratio should be 128 on A2/A3 when cmp_topk is 0 (HCA without cmp_sparse_indices)` | **A2/A3 有专属 "HCA" 模式** |
| `cmp_residual_kv must be provided` | HCA 模式的额外必需输入 |

**含义**：`SparseFlashMla` 的公开接口是按 A5 的通用稀疏路径设计的；
A2/A3 被限制进一条参数面更窄的 HCA 兼容路径。这与它在我们的 profile 里
SCALAR 43% / FIXPIPE 56% 的形态互相印证——**很可能是"为 A5 设计的内核
在 A2 上以兼容模式跑"，而不是 A2 原生优化实现**。

这给 §3 的"向算子团队提需求"添了最具体的一条：
> SparseFlashMla 在 910B3(A2) 的 HCA 路径上，M=6 decode 时
> SCALAR 占 43%、FIXPIPE 占 56%（每次调用固定 ~80µs）。
> 请评估为 A2 提供原生 tiling / 路径的可行性。

微基准脚本 `tools/smla_bench.py` 存档（已通过 metadata 参数校验，
主算子 tiling 阶段失败；要跑通需从引擎真实调用栈抓完整 kwargs）。
