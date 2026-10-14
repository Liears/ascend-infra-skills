---
name: model-day0-support
description: Turn a new model architecture into a reviewable Ascend Day-0 support plan — a PR DAG, a validation matrix, and release gates. Use when a model must be brought up on vLLM-Ascend / SGLang-Ascend / MindIE for the first time, when asked what work a new model needs on NPU, whether an existing port is release-ready, or to lay out the order of porting tasks. Maps the model's architecture onto the Ascend runtime's actual extension points.
---

# Ascend Model Day-0 Support

Turn "we need model X on NPU" into an ordered, verifiable work plan.

## What Day-0 means here

Day-0 support is not "it runs". It is the set of work that makes a new
architecture **servable and defensible** on Ascend:

- the model loads from the published weights, in the published quant format
- it produces correct output (checked, not asserted)
- it runs the framework's fast paths (fused attention, fused MoE, graph mode)
  rather than silently falling back
- its behavior is captured in artifacts a reviewer can re-run

## Workflow

### 0. Establish the gap

Run `ascend-open-source-finder` first. A large share of Day-0 work already
exists in a personal repo. If it does, the plan is "audit and adopt", not
"write from scratch". Only proceed to build when nothing usable exists, or when
the existing work does not target your chip/stack.

### 1. Map the architecture to Ascend extension points

For each architectural feature, name the exact file in the runtime that must
change. Do not write "add attention support" — write the path.

| Architecture feature | Ascend runtime extension point |
|---|---|
| New attention variant (MLA / sparse / compressed) | `vllm_ascend/attention/`, `vllm_ascend/ops/` |
| MoE with a new routing/scoring function | `vllm_ascend/ops/fused_moe/token_dispatcher.py` |
| Model-specific layers | `vllm_ascend/models/<ModelName>.py` |
| RoPE / position variants | `vllm_ascend/ops/rope_*.py` |
| Vision tower | `vllm_ascend/models/`, plus a multimodal processor |
| MTP / speculative head | `vllm_ascend/models/<Model>_mtp.py`, plus a scheduler/vocab wiring |
| Graph-mode compatibility | graph-capture path, `torch.compile` / aclgraph integration |
| Quant format (W8A8 / W4A8 / MXFP8) | `msmodelslim` recipe + runtime quant config |

**The extension points above are read from a real vLLM-Ascend tree** — see
`references/ascend-extension-points.md` for the paths and what lives in each.

### 2. Classify each task by evidence requirement

| Task class | Example | Evidence it needs |
|---|---|---|
| Loading | weights load, no shape error | load log, weight count |
| Numerical | attention produces right numbers | a check vs a reference, on the real path |
| Performance | a fused path actually engaged | a profiler trace showing the fused op ran |
| Compatibility | works with graph mode / PD / prefix cache | a run in that mode, not an inference from eager |

The last column is the point. **"It works" without evidence for the specific
claim is not a gate pass.**

### 3. Build the PR DAG

Separate work that can land independently from work that cannot. A typical
Ascend Day-0 DAG runs:

```
[1] model skeleton loads                      (blocking everything)
[2] attention path (eager, correctness-first) (blocking [3],[4])
[3] MoE / FFN path                            (blocking runnable)
[4] quant config + weight mapping             (blocking memory-fit)
[5] fast paths: fused attn, fused MoE         (independent, perf)
[6] graph mode                                (after [2]-[5] stable)
[7] speculative / MTP head                    (independent, perf)
[8] multimodal tower                          (independent, if applicable)
```

Each node carries: the file it touches, how it is validated, and which other
nodes it blocks.

### 4. Define release gates

Minimum gates for a Day-0 claim:

1. **Load gate** — weights load; the parameter count matches the config.
2. **Accuracy gate** — a task-based check (e.g. GSM8K-style, N samples) passes
   at the expected rate, on the real path, with the fast paths enabled.
3. **Fast-path gate** — the profiler shows the fused kernel ran; no silent
   fallback to a reference implementation.
4. **Memory gate** — the model fits the target chip at the intended context
   length, with the KV budget stated.
5. **Mode gate** — runs under the deployment modes you claim (eager, and graph
   mode if claimed; PD if claimed).

A gate is passed only with its artifact. State the CANN/driver on the artifact.

### 5. Sanitize before publishing

If the port was developed against a private fork or internal data, the public
artifact must not carry it. See `references/day0-contract.md`.

## The failure mode this skill prevents

The most expensive Day-0 mistake on Ascend is a **silent fallback**: the model
runs, the output is correct, and the fused path never engaged, so the
performance is far below what the chip can do — with no error anywhere. Gate 3
exists solely to catch it. Never accept an end-to-end pass as proof a kernel
ran; assert the kernel itself.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/plan_template.py` | Emit a Day-0 plan skeleton (DAG + gates) to fill in |
| `scripts/check_day0_bundle.py` | Validate that a candidate Day-0 bundle has every required artifact |

## References

- `references/ascend-extension-points.md` — where each kind of model work lands in the vLLM-Ascend tree
- `references/day0-contract.md` — what a public Day-0 bundle must and must not contain
- `references/deepseek-v41-case-study.md` — a worked Day-0 case: DeepSeek-V4.1-Flash on 910B/A2
- `../../docs/EVIDENCE-RULES.md` — the evidence rules every gate obeys
