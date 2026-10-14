# Worked example — DeepSeek-V4.1-Flash on Ascend 910B (A2)

An end-to-end run of every skill against one real question:

> "Bring DeepSeek-V4.1-Flash up on our 8×910B3 node. What do we actually need
> to do?"

This example is **not hypothetical**. The outputs in the numbered folders are
the real output of the skills, captured on 2026-10-14 on this machine. Where a
step depends on hardware we did not use (a live V4.1 service), that is stated
rather than faked.

---

## The situation

- Hardware: 8 × Ascend 910B3 (A2), 64 GB HBM each, CANN 25.5.2-driver / 8.1.RC1 toolkit
- Stack present: vLLM-Ascend `0.23.1.dev14`, torch_npu `2.10.0.post4`
- Model: DeepSeek-V4.1-Flash — 384-expert MoE, MLA, sparse indexer, Engram
  tables, MTP, vision tower
- Unknown at the start: does vLLM-Ascend even support this model?

---

## 01 — Discovery: has someone already done this?

```bash
python3 skills/ascend-open-source-finder/scripts/find_open_source.py \
    --model "DeepSeek-V4.1-Flash" --hardware 910B
```

Output: [`01-discovery/finder-output.md`](01-discovery/finder-output.md)

**Result:** one repo rated `verified` —
`chiro2001/deepseek-v4.1-flash-ascend910B` (Apache-2.0, 24 stars). Two weaker
`claimed` leads also surfaced.

**Why this step first:** it changed the entire shape of the work. The question
stopped being "how do we write a V4.1 port" and became "audit this port and
adopt it". Discovering this after writing a port would have wasted the effort.

---

## 02 — Audit: what is actually in it?

```bash
git clone --depth 1 <url> /tmp/dsv41-ascend
python3 skills/ascend-open-source-finder/scripts/audit_repo.py /tmp/dsv41-ascend
```

Output: [`02-audit/audit-output.md`](02-audit/audit-output.md)

**Result, abbreviated:**

- **33 patches** — 12 against `vllm_ascend`, 4 against `vllm`, 2 ModelSlim
  quant recipes, plus experimental ones. This is a real port, not a claim.
- REPRO.md, EXPECTED_PERF.md, CORRECTNESS_STATUS.md present.
- Entry points: `a2/scripts/serve_a2_offload.sh` (987 lines), build and
  self-check scripts.
- **Red flag: the toolchain appears in multiple versions** — CANN 9.1.0 and
  9.2.0, driver 25.5.5 and 26.1.1. Which baseline the performance numbers
  belong to must be resolved before citing any of them.

That red flag is the single most useful output of the whole example: it
converted "there is a port" into "here is the specific open question".

---

## 03 — Capacity: what does the HBM arithmetic look like?

```bash
python3 skills/llm-serving-capacity-planner/scripts/capacity_analyzer.py \
    03-capacity/sample-startup.log --requests 8192 32768 131072
```

Output: [`03-capacity/capacity-output.md`](03-capacity/capacity-output.md)

Input `03-capacity/sample-startup.log` uses the **exact log strings** read from
`vllm_ascend/worker/worker.py` and `vllm/v1/core/kv_cache_utils.py`, with the
capacity figures from the real V4.1 deployment docs. It is labelled a sample:
it is a real format with illustrative numbers, not a capture from a service we
ran.

**Result:** the tool explains the HBM split (weights / activation / non-torch /
**NPU graph** memory / KV), reports the KV pool capacity, and estimates
concurrency at three request lengths — flagging in the basis column whether a
row is pool-bound or scaled.

---

## 04 — Profiling: what does an NPU trace look like?

```bash
python3 skills/npu-profiler-analysis/scripts/analyze_profile.py /tmp/prof-out
```

Output: [`04-profiling/profile-output.md`](04-profiling/profile-output.md)

This ran on a **real trace captured on this machine**: a small NPU workload
profiled with `torch_npu.profiler` at Level1 + PipeUtilization, producing the
46-column `kernel_details.csv` and a `trace_view.json`.

**Result:** kernel attribution by Ascend op type, the accelerator-core split
(AI_CORE / AI_VECTOR_CORE / MIX_AIV), and the host/device event balance. The
point is the *format*: Ascend op names (`aclnnAddmm_MatMulCommon_MatMulV2`),
the two event families, and the pipe-column dependency. A CUDA-era analyzer
reads none of this.

---

## 05 — Day-0: what would the plan be?

```bash
python3 skills/model-day0-support/scripts/plan_template.py \
    --model "DeepSeek-V4.1-Flash" --chip 910B3
```

Output: [`05-day0/day0-plan.md`](05-day0/day0-plan.md)

Plus the analysis in
[`skills/model-day0-support/references/deepseek-v41-case-study.md`](../../skills/model-day0-support/references/deepseek-v41-case-study.md).

**Result:** a task DAG that names, per task, the vLLM-Ascend file it touches
and the evidence it needs — and the finding that **Engram is the schedule
risk**, because its host-resident tables have no analogue in the runtime.

---

## What this example demonstrates

| Skill | What it contributed here |
|---|---|
| `ascend-open-source-finder` | turned an open question into one named repo |
| `repo audit` | turned a repo into a patch inventory **and one sharp open question** |
| `llm-serving-capacity-planner` | made the HBM arithmetic explicit, including the Ascend-specific graph-memory bucket |
| `npu-profiler-analysis` | established what an Ascend trace contains and how it differs from CUDA |
| `model-day0-support` | produced a plan that names files and gates, and identified Engram as the risk |

## What this example does NOT claim

- It does **not** claim to have served DeepSeek-V4.1-Flash. That is the next
  phase and needs the 273 GB W4A8 weights plus a running vLLM-Ascend service.
- It does **not** endorse any performance number in the audited repo. The audit
  explicitly flags that the toolchain versions are ambiguous.
- It does **not** treat the audited repo's claims as verified — only its
  artefact inventory.

Every step above follows [`docs/EVIDENCE-RULES.md`](../../docs/EVIDENCE-RULES.md):
state what was done, attach the artifact, and mark what was not done.
