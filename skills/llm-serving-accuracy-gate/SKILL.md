---
name: llm-serving-accuracy-gate
description: Verify a deployed model's accuracy on Ascend with a three-layer gate — determinism + chat-mode smoke (L1), a sensitive reasoning detector like AIME with official sampling params (L2), and coverage gates for vision/long-context/code (L3). Use after a model is serving, when the user asks whether the deployment is accurate, wants to run GSM8K/AIME/evalscope/aisbench, needs to compare against the official tech report, or must decide whether a quantized port is release-ready. Refuses comparisons across mismatched harnesses or sampling protocols.
---

# LLM Serving Accuracy Gate

Answer "is the deployed model actually accurate" with gates that can fail —
not with a number floating free of its protocol.

## What this skill gates against

Two different questions are routinely conflated, and the gate separates them:

1. **Fidelity** — did the port/quantization preserve the model? Compared
   against the reference implementation or the port repo's own acceptance
   gates. A drop here is a bug.
2. **Capability** — does the model meet a quality bar? Compared against the
   official tech report. A gap here may be expected (e.g. W4A8 vs the
   official MXFP8/MXFP4 weights) and is a measurement, not a bug.

## Workflow

### Step 1 — Find the official baseline before running anything

Like `ascend-open-source-finder`, do not run evals blind. Collect:

| What | Where | Record |
|---|---|---|
| Official benchmark numbers | the model's tech report (often shipped inside the weight dir) | benchmark, metric, page |
| Official sampling protocol | the report's evaluation section | temperature, top-p, effort, k |
| Port repo's acceptance gates | its CORRECTNESS/REPRO docs | dataset, mode, threshold |

For DeepSeek-V4.1-Flash the worked extraction lives in
`references/official-baselines-v41.md` (AIME 2026 Pass@1 94–100% across
effort 25–100; temperature 1.0 / top-p 0.95; GSM8K 8-shot EM 90.8–93.0 —
all with page citations). **Never quote a baseline without its protocol.**

### Step 2 — Align the protocol or downgrade the claim

If you cannot match the official protocol (harness, sampling, effort,
thinking mode), the run is **self-comparison only**: valid for A/B within
your deployment, invalid as a claim against the official number. Say so in
the output; do not print the two numbers side by side without the caveat.

The GSM8K trap, from real data: the official 8-shot EM is **90.8–93.0**,
while a chat-mode run with the official encoder and thinking reached
**198/200 = 99%** — the same benchmark, 8 points apart, purely from
protocol. A gate that ignores this will "pass" anything.

### Step 3 — Run the gates

| Gate | Content | When | Catches | Pass rule |
|---|---|---|---|---|
| **L1 smoke** | determinism check (same prompt ×N, identical logprobs) + GSM8K-200 chat mode, conc 4 | every build | port/quant/chat-template bugs | drop >2 vs baseline = fail (baseline ±1 is noise) |
| **L2 sensitive** | AIME 2026, **avg@4 minimum**, official sampling (temp 1.0, top-p 0.95, effort pinned and recorded), thinking on | pre-release | quantization damage to reasoning — the first thing W4A8 hurts | report mean + per-run spread; compare only against same-protocol numbers |
| **L3 coverage** | vision set (this family is multimodal) + long-context + LiveCodeBench | pre-delivery | axes AIME cannot see | per-axis thresholds set from the port repo's gates |

Why AIME needs avg@4: **30 problems means 1 problem = 3.3 points.** A
single pass cannot distinguish a real 6-point regression from two lucky
guesses. At the official ceiling (94–100%) variance is partially masked; a
quantized deployment sits lower, where the variance bites.

### Step 4 — Verdict with the evidence block

```bash
python3 scripts/check_accuracy_gate.py results.json --gate L1 --baseline 198/200
```

The script emits PASS/FAIL plus the extended evidence block. Exit code 1 on
FAIL — gateable in CI.

## When a gate fails: the escalation ladder

A FAIL is a diagnosis, not the end. Each rung answers a different question,
and skipping rungs wastes the most expensive tool first:

| Rung | Question | Tool |
|---|---|---|
| 1. Determinism | is the output *stable*? | same prompt ×N, identical logprobs (L1a) |
| 2. Task accuracy | is the output *right*? | GSM8K / AIME gates (L1b / L2) |
| 3. Tensor localization | **where** does it diverge? | layer-wise tensor dump & compare |

Rung 3 exists because rung 2 says *that* the deployment is wrong, never
*where*. The technique (absorbed from the ecosystem — see
`../../references/ascend-ecosystem.md` §7): capture intermediate tensors per
layer during forward and compare across runs or across GPU/NPU
implementations, to find the first layer whose output diverges. Tooling
lives outside this suite (`awesome-ascend-skills`'s
`inference-precision-tensor-dump-compare`, and CANNBot's precision-debug
skills); this skill's job is to know the ladder and hand off at rung 3.

**Operational gotcha worth recording before you need it**: tensor dumps on
vLLM require `TORCHDYNAMO_DISABLE=1` — with torch.compile active the dump
logs silently never appear. A "dump produced nothing" session is usually
this, not a broken tool.

## Harness rules

| Harness | Use it when | Caveat |
|---|---|---|
| port repo's own runner (e.g. `acc_eval.py`) | comparing against that repo's published gates | its `--conc 4` serializes prefill — passing it does NOT prove concurrent-batch correctness |
| **evalscope** | community-comparable numbers, OpenAI-compatible endpoint, AIME support | numbers comparable to official reports only if sampling is matched |
| **aisbench** | comparing against Huawei-published gates (vllm-ascend CI uses it) | pulls its own datasets from ModelScope |

**One harness per comparison.** Mixing harnesses inside one comparison
invalidates it. The harness identity, dataset version, sampling params,
effort level, and k all go into the evidence block.

## Deployment-specific checks before trusting a number

- **Effort actually applied.** Reasoning-effort moves AIME by up to 6 points;
  a mis-mapped effort silently produces a wrong-but-plausible score. Verify
  the mapping engages before reading any AIME result.
- **Speculative decode changes determinism.** For accuracy runs, pin the
  draft setting (on or off) and record it; do not compare across draft
  settings.
- **Chat template present.** L1 exists partly because harness bugs around
  the chat template are the most common failure on a new port.
- **HCCL_DETERMINISTIC=true costs accuracy** (observed: GSM8K 91/100 vs
  198/200). It is a diagnostic tool, never a delivery setting.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/check_accuracy_gate.py` | Verdict a result JSON against a gate; emit the extended evidence block; exit 1 on FAIL |

## References

- `references/gate-definitions.md` — the three gates in full, thresholds and their sources
- `references/official-baselines-v41.md` — the extracted DeepSeek-V4.1-Flash baselines with page citations
- `references/harness-notes.md` — protocol details of each harness, including the prefill-serialization caveat
- `../../docs/EVIDENCE-RULES.md` — the base evidence rules this extends
