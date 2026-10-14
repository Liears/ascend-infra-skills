---
name: ascend-infra-skills
version: 0.1.0
description: Agent-ready skills for AI-infrastructure work on Ascend NPUs — open-source port discovery, cross-framework serving benchmarks, NPU profiler analysis, KV/HBM capacity planning, and model Day-0 support programs, all under a strict evidence discipline.
author:
  name: ascend-infra-skills contributors
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

Skills that give a coding agent the operational memory for AI-infrastructure
work on **Ascend NPUs**, adapted from the methodology of
[BBuf/AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS)
(NVIDIA/CUDA) to the Ascend stack (CANN / torch_npu / vLLM-Ascend / MindIE).

## Why a separate project

The original skills assume CUDA: their hardware tables list H200/B200, their
log parsers match `cuda_graph` strings, their profiler readers expect CUPTI
kernel names. On Ascend every one of those is different — not merely renamed,
but structurally different (a cube/vector core split, a cards-vs-devices
distinction, a graph mode that costs real HBM).

These skills encode what actually differs, and refuse to answer when the
evidence is not there.

## Skills

| Skill | Use it when |
|---|---|
| [`ascend-open-source-finder`](skills/ascend-open-source-finder/) | "Has anyone already ported this model to Ascend?" — ranked, evidence-rated candidates before you write anything |
| [`llm-serving-auto-benchmark`](skills/llm-serving-auto-benchmark/) | Compare vLLM-Ascend / MindIE / SGLang-Ascend under one workload and SLA, and refuse comparisons that are not valid |
| [`llm-serving-capacity-planner`](skills/llm-serving-capacity-planner/) | Explain where NPU HBM went, what the KV budget is, how many concurrent requests fit |
| [`npu-profiler-analysis`](skills/npu-profiler-analysis/) | Read a torch_npu trace: which Ascend ops dominate, core split, host/device balance |
| [`model-day0-support`](skills/model-day0-support/) | Turn a new architecture into a Day-0 plan on NPU: PR DAG, validation matrix, release gates |
| [`model-pr-history-knowledge`](skills/model-pr-history-knowledge/) | Before changing a model path, see what upstream already did to it |

## Install

### Claude Code plugin

```
/plugin marketplace add <this repo>
/plugin install ascend-infra-skills@ascend-infra-skills
```

### Any skill runtime (Claude Code, Codex, Kimi, …)

```bash
git clone <this repo> ascend-infra-skills
cd ascend-infra-skills
SKILL_DIR=~/.claude/skills          # or ${CODEX_HOME:-~/.codex}/skills
mkdir -p "$SKILL_DIR"
for d in skills/*/; do
  [ -f "$d/SKILL.md" ] && ln -sfn "$PWD/${d%/}" "$SKILL_DIR/$(basename "$d")"
done
```

Skills need only Python 3.9+ with the standard library. Nothing here requires an
NPU to *analyze* — capturing traces and running benchmarks does, but reading
them does not.

## The evidence discipline

Every skill obeys [`docs/EVIDENCE-RULES.md`](docs/EVIDENCE-RULES.md). The short
version:

- a benchmark row without the CANN/driver version is not a row;
- cards and devices are different units; say which;
- tok/s on a speculative-decoding model is meaningless without the basis stated;
- a trace is not reused across CANN versions;
- "it works" is not a claim.

## Relationship to other projects

- **CANNBot** (`gitcode.com/cann/cannbot`) covers the operator/CANN layer and
  NPU model optimization execution: profiling capture, KV cache, fusion,
  quantization, graph mode. These skills **do not duplicate** that; where an
  execution capability exists there, the skill points to it rather than
  reimplementing it. This project supplies the layer above: cross-framework
  benchmark discipline, upstream-PR archaeology, Day-0 program design, and
  open-source-port discovery.
- **BBuf/AI-Infra-Auto-Driven-SKILLS** is the methodological ancestor; the
  evidence rules and skill shapes are ported from it.

## License

Apache-2.0. See [LICENSE](LICENSE).
