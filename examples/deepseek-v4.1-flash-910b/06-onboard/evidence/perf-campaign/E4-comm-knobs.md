# E4 — 通信域残留旋钮（cannbot 启发的收尾）：中性

在 P2 trace（=当前最优栈）分解上，通信占 39%。对照 cannbot-skills 的通信/算子域
知识，逐一评估剩余杠杆后只剩两个从未测过的便宜旋钮，今晚补测：

| 旋钮 | 来源 | 结果 |
|---|---|---|
| HCCL_BUFFSIZE=1024 + HCCL_OP_EXPANSION_MODE=AIV（上游 A3 配方） | cannbot communication 域 | 中位 136.5 vs 同会话对照 129.5 = **+5%（噪声带内，判中性）** |
| enable_mc2_hierarchy_comm | ascend_config | 不测：A3 专用层级通信路径，A2 无益（与 fused_mc2 同族，此前已定价） |
| SuperKernel | cannbot model-infer-superkernel | **不适用**：需 A3 + ge_graph；我们是 A2 + aclgraph |
| HcPre+RmsNorm 融合（Track B） | 自家仓 TRACKB-FUSION-VERDICT | 上限 ~1ms/step（≈3%），kernel 已验证但接线不划算（判决文档在案） |

E4a 协议：512-token 长生成探针，预热 5 丢 3 测 + 对照臂 11 样本（剔除 1 个坏桶）。
理论预期一致：batch-1 的集合通信是延迟受限（10KB 级消息），缓冲区/展开模式类
旋钮作用在带宽——中性是预期结果，不是意外。

## 终局杠杆表（单流延续口径，174.3 基准）

| 层 | 状态 | 依据 |
|---|---|---|
| 投机（A=5.82/6，97% 上限） | **到顶** | 数学上限；K>5 两个实现均输（E3×2） |
| 框架旋钮（图/mc2/overlap/通信 env） | **到顶** | E1/E2 已收；E4 中性 |
| 微融合（norm-quant 0.3-1.2%、HcPre ~3%） | **低于接线成本** | 诚实算术在案 |
| 大 kernel（SparseFlashMla 11% / 量化 GEMM ~14%） | **vendor/上游域** | 混合量化 SparseFlashMla 在上游 rc2+，即 PR #18967 路线 |

⇒ 当前软件栈内基本到顶；下一步增益在上游 rebase（CANN 9.2 + 171 commits 的
kernel 工作），而它被 PR #18967（draft 图）是否合入门控。
