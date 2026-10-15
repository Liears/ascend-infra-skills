# 实战案例 — DeepSeek-V4.1-Flash 在昇腾 910B（A2）上

[English](README.md) | 中文

用同一个真实问题把每个技能端到端跑一遍：

> "把 DeepSeek-V4.1-Flash 在我们 8×910B3 节点上跑起来，到底需要做什么？"

本案例**不是假设性的**。编号目录里的输出都是技能的真实输出，2026-10-14
在本机采集。凡是某一步依赖了我们没用到的硬件（运行中的 V4.1 服务）的地方，
都如实标注，没有伪造。

---

## 初始状况

- 硬件：8 × 昇腾 910B3（A2），每卡 64 GB HBM，驱动 25.5.2 / toolkit 8.1.RC1
- 现有栈：vLLM-Ascend `0.23.1.dev14`，torch_npu `2.10.0.post4`
- 模型：DeepSeek-V4.1-Flash —— 384 专家 MoE、MLA、稀疏索引器、Engram
  表、MTP、视觉塔
- 开始时未知的问题：vLLM-Ascend 到底支不支持这个模型？

---

## 01 — 检索：有没有人已经做过？

```bash
python3 skills/ascend-open-source-finder/scripts/find_open_source.py \
    --model "DeepSeek-V4.1-Flash" --hardware 910B
```

输出：[`01-discovery/finder-output.md`](01-discovery/finder-output.md)

**结果：** 一个仓评为 `verified` ——
`chiro2001/deepseek-v4.1-flash-ascend910B`（Apache-2.0，24 star）。另有
两个较弱的 `claimed` 线索。

**为什么先做这步：** 它改变了整件事的形态。问题从"怎么写一个 V4.1
移植"变成"审计这个移植并采用它"。如果写完移植才发现它，就白干了。

---

## 02 — 审计：里面到底有什么？

```bash
git clone --depth 1 <url> /tmp/dsv41-ascend
python3 skills/ascend-open-source-finder/scripts/audit_repo.py /tmp/dsv41-ascend
```

输出：[`02-audit/audit-output.md`](02-audit/audit-output.md)

**结果（摘要）：**

- **33 个补丁** —— 12 个打在 `vllm_ascend`，4 个打在 `vllm`，2 个
  ModelSlim 量化配方，另有实验性的。这是真移植，不是宣称。
- REPRO.md、EXPECTED_PERF.md、CORRECTNESS_STATUS.md 齐全。
- 入口脚本：`a2/scripts/serve_a2_offload.sh`（987 行），构建与自检脚本。
- **红旗：工具链出现多个版本** —— CANN 9.1.0 与 9.2.0，驱动 25.5.5 与
  26.1.1。引用任何性能数字之前必须先弄清它属于哪个基线。

这条红旗是整个案例最有用的输出：它把"有一个移植"变成了"这里是具体的
待解问题"。

---

## 03 — 容量：HBM 的算术长什么样？

```bash
python3 skills/llm-serving-capacity-planner/scripts/capacity_analyzer.py \
    03-capacity/sample-startup.log --requests 8192 32768 131072
```

输出：[`03-capacity/capacity-output.md`](03-capacity/capacity-output.md)

输入 `03-capacity/sample-startup.log` 用的是从
`vllm_ascend/worker/worker.py` 和 `vllm/v1/core/kv_cache_utils.py` 里
读出来的**逐字日志格式**，容量数字来自真实 V4.1 部署文档。它标注为
sample：格式是真的、数字是示例性的，不是我们跑过的服务采集。

**结果：** 工具解释了 HBM 的分账（权重 / 激活 / non-torch /
**NPU graph** 显存 / KV），报出 KV 池容量，并按三个请求长度估算并发
——basis 列标明每一行是池上限还是换算值。

---

## 04 — Profiling：NPU trace 长什么样？

两次采集，都在**真实硬件**上（见 `04-profiling/`）：

**(a) 小型冒烟负载** —— [`04-profiling/profile-output.md`](04-profiling/profile-output.md)。
一个小 NPU 任务用 `torch_npu.profiler` 以 Level1 + PipeUtilization 采
出 46 列 `kernel_details.csv`。确立的是*格式*：昇腾算子名
（`aclnnAddmm_MatMulCommon_MatMulV2`）、两类事件、pipe 列依赖。
CUDA 时代的分析器一样都读不了。

**(b) 运行中的 DeepSeek-V4.1-Flash 服务** ——
[`04-profiling/real-service-trace.md`](04-profiling/real-service-trace.md)。
在运行中的 8×910B3 服务器上 `/start_profile` → 发请求 → `/stop_profile`
采集，再用 `torch_npu.profiler.analyse()` 离线解析：
**229,642 行 kernel 记录，486 MB trace**。结果是整个案例要产出的发现：

| 核类型 | 设备时间占比 |
|---|---:|
| **COMMUNICATION** | **83.06** |
| MIX_AIC | 7.95 |
| AI_VECTOR_CORE | 5.05 |
| AI_CORE | 2.64 |

占头的算子全是集合通信（`AivKernel` 41.5%、`allReduce` 39.4%、
`alltoallv`、`allGather`）。**这是大 MoE 在昇腾上 EP/TP 并行的正常
形态**，它意味着调计算 kernel 最多能动 ~17% 的时间——一个按 GPU 思路
写的分析器会把这结论看反。

---

## 05 — Day-0：计划应该长什么样？

```bash
python3 skills/model-day0-support/scripts/plan_template.py \
    --model "DeepSeek-V4.1-Flash" --chip 910B3
```

输出：[`05-day0/day0-plan.md`](05-day0/day0-plan.md)

加上
[`skills/model-day0-support/references/deepseek-v41-case-study.md`](../../skills/model-day0-support/references/deepseek-v41-case-study.md)
里的分析。

**结果：** 一张任务依赖图，每个任务写明它改哪个 vLLM-Ascend 文件、需要
什么证据 —— 以及**Engram 是进度风险**这个发现，因为它的常驻 host 表在
运行时里没有对应物。

---

## 06 — 上板：真的跑起来了

见 [`06-onboard/RUNBOOK.zh.md`](06-onboard/RUNBOOK.zh.md)（流程）和
[`06-onboard/evidence/`](06-onboard/evidence/)（真实产物）。

**结果：** 权重下载（304 GB，72 分片）→ Engram 重组（sha256 通过）→
补丁镜像构建（15 项补丁逐字节校验）→ TP8 起服（health 200，推理返回
正确）→ KV 2,942,071 tokens → 采到真 trace。详细数字和红旗都在
evidence 里。

---

## 这个案例证明了什么

| 技能 | 在这里贡献了什么 |
|---|---|
| `ascend-open-source-finder` | 把一个开放问题变成一个具体的仓 |
| `repo audit` | 把一个仓变成补丁清单**外加一个尖锐的待解问题** |
| `llm-serving-capacity-planner` | 把 HBM 算术摊开，含昇腾特有的 graph 显存分账 |
| `npu-profiler-analysis` | 确立昇腾 trace 里有什么、与 CUDA 差在哪 |
| `model-day0-support` | 产出写明文件与门禁的计划，并指出 Engram 是风险 |

## 这个案例不主张什么

- **不**主张任何精度结论。`run_test.sh` 跑的是 `MODE=quick`，GSM8K
  没有评测。
- **不**主张视觉通路可用 —— 视觉用例被跳过（没有官方图片目录）。
- **不**背书被审计仓里的任何性能数字。审计明确标出工具链版本有歧义，
  而且我们实测 27.2 tok/s @8K 远低于该仓的 A3 数字（A2 本就是更慢的
  设备，且这次没有 PGO 产物）。
- **不**主张 DRAM 卸载生效。验证器的总结行说生效了，但它下面每条判据
  都写着 `未验` —— fail-open bug 详见
  [`06-onboard/evidence/README.zh.md`](06-onboard/evidence/README.zh.md)。
- **不**把被审计仓的宣称当作已验证 —— 只认它的产物清单（以及现在，
  它在本机上的实际行为）。

以上每一步都遵守 [`docs/EVIDENCE-RULES.md`](../../docs/EVIDENCE-RULES.md)：
做了什么就写什么，附上产物，没做的标明没做。

> **关于编号目录里的工具输出**：`01-discovery/`、`02-audit/`、
> `03-capacity/`、`04-profiling/`、`05-day0/` 下的 `.md` 文件是技能的
> **原始输出，保持英文不翻译** —— 它们是证据产物，翻译它们等于改证据。
> 中文叙述都在本文件和 RUNBOOK/evidence 的中文版里。
