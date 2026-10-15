---
name: ascend-infra-skills
version: 0.1.0
description: 面向昇腾 NPU 的 AI-Infra 工作技能集——开源移植检索、跨框架推理基准测试、NPU profiler 分析、KV/HBM 容量规划、模型 Day-0 支持程序，全部执行统一的证据纪律。
author:
  name: Liears
license: Apache-2.0
keywords:
  - ascend
  - npu
  - cann
  - vllm-ascend
  - mindie
  - sglang
  - profiler
  - benchmark
  - day0
  - llm-serving
---

# ascend-infra-skills

[English](README.md) | 中文

给编码 agent 补上在**昇腾 NPU** 上做 AI-Infra 工作所需的操作记忆。方法学移植自
[BBuf/AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS)
（NVIDIA/CUDA 生态），适配到昇腾栈（CANN / torch_npu / vLLM-Ascend / MindIE）。

## 为什么是独立项目

原版技能假设了 CUDA：它的硬件表里是 H200/B200，日志解析器匹配
`cuda_graph` 字符串，profiler 读取器按 CUPTI 内核名解析。在昇腾上这些**每一项都不一样**——
不是改个名字的区别，是结构性的不同（cube/vector 核分离、卡与 device
两种计数单位、图模式要消耗真实的 HBM）。

这套技能编码的是实际差异，并且在证据不足时**拒绝下结论**。

## 技能列表

| 技能 | 什么时候用 |
|---|---|
| [`ascend-open-source-finder`](skills/ascend-open-source-finder/) | "这个模型有没有人已经移植到昇腾？"——动手之前先拿到按证据评级排序的候选清单 |
| [`llm-serving-auto-benchmark`](skills/llm-serving-auto-benchmark/) | 在同一 workload 和 SLA 下对比 vLLM-Ascend / MindIE / SGLang-Ascend，并拒绝无效对比 |
| [`llm-serving-capacity-planner`](skills/llm-serving-capacity-planner/) | 解释 NPU HBM 去了哪、KV 预算是多少、能装多少并发请求 |
| [`npu-profiler-analysis`](skills/npu-profiler-analysis/) | 读 torch_npu trace：哪些昇腾算子占大头、核类型分布、host/device 平衡 |
| [`model-day0-support`](skills/model-day0-support/) | 把一个新架构变成 NPU 上的 Day-0 计划：PR 依赖图、验证矩阵、发布门禁 |
| [`model-pr-history-knowledge`](skills/model-pr-history-knowledge/) | 改模型代码之前，先看上游已经对它做过什么 |
| [`ascend-compute-simulation`](skills/ascend-compute-simulation/) | 从 config 算逐算子 FLOPs 与形状；MFU 只在有可信来源的峰值下计算 |

## 安装

### Claude Code 插件

```
/plugin marketplace add Liears/ascend-infra-skills
/plugin install ascend-infra-skills@ascend-infra-skills
```

### 任意技能运行时（Claude Code、Codex、Kimi 等）

```bash
git clone https://github.com/Liears/ascend-infra-skills.git
cd ascend-infra-skills
SKILL_DIR=~/.claude/skills          # 或 ${CODEX_HOME:-~/.codex}/skills
mkdir -p "$SKILL_DIR"
for d in skills/*/; do
  [ -f "$d/SKILL.md" ] && ln -sfn "$PWD/${d%/}" "$SKILL_DIR/$(basename "$d")"
done
```

技能只需要 Python 3.9+ 标准库。**分析**不需要 NPU——采集 trace 和跑
benchmark 需要，读它们不需要。

## 证据纪律

每个技能都遵守 [`docs/EVIDENCE-RULES.md`](docs/EVIDENCE-RULES.md)。简版：

- 不带 CANN/驱动版本的 benchmark 行，不算一行数据；
- 卡（card）和 device 是两个不同的单位，必须写清是哪个；
- 投机解码模型上的 tok/s，不注明口径就是无意义的数字；
- 一份 trace 不能跨 CANN 版本复用；
- "能跑"不是结论。

## 与其他项目的关系

- **CANNBot**（`gitcode.com/cann/cannbot`）已覆盖算子/CANN 层和 NPU
  模型优化执行：profiling 采集、KV cache、融合算子、量化、图模式。
  本项目**不重复**这些——凡 CANNBot 已有的执行能力，技能直接指向它而不
  重写。本项目提供的是它上面那一层：跨框架 benchmark 纪律、上游 PR 考古、
  Day-0 程序设计、开源移植检索。
- **BBuf/AI-Infra-Auto-Driven-SKILLS** 是方法学源头；证据规则和技能形态
  移植自它。

## 已验证的端到端案例

[`examples/deepseek-v4.1-flash-910b/`](examples/deepseek-v4.1-flash-910b/)
记录了 DeepSeek-V4.1-Flash（W4A8）在 8×910B3（A2）上的完整流程：检索开源
实现 → 审计 → 下载重组权重 → 构建补丁镜像 → TP8 起服 → 采集 229,642 行
真实 trace → 分析。其中 83% 的设备时间是通信算子——这是大 MoE 在昇腾上
EP/TP 并行的正常形态，也是 GPU 思路的分析器会看反的结论。

这个案例同时抓出了两个技能自身的 bug（合成数据抓不到的那种），已修复并
加了回归测试。

## 许可证

Apache-2.0，见 [LICENSE](LICENSE)。
