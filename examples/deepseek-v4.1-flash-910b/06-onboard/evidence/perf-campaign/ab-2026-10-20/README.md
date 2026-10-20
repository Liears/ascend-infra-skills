# A/B: upstream rc2 + PR #18967 vs port-repo stack (2026-10-20, 1017 weights)

Same machine, same session, same weights, same probe protocol
(`decode_probe.py longgen-prompt.json 512`, 8K context, 512-token forced decode,
runs 1-2 discarded as warmup, median of 6 measured).

| path | stack | tok/s (median) | A | TTFT |
|---|---|---|---:|---:|---:|
| A | port-repo `dsv41-a2:v11` (dev5097 + 15 patches, `DRAFT_GRAPH=1`, 45 `V41_*` knobs) | **163.3** | 5.89 | 1.30 s |
| B | `quay.io/ascend/vllm-ascend:v0.30.0rc2` + cherry-pick PR #18967 | **119.1** | 5.89 | 1.17 s |

Raw path A (2026-10-20 05:2x): 152.3 / 161.9 / 160.8 / 162.3 / 163.0 / 163.3 / 163.0 / 164.1
Raw path B (2026-10-20 04:5x): 112.3 / 117.9 / 118.1 / 119.1 / 119.2 / 118.7 / 119.2 / 119.1

Accuracy: path B GSM8K-100 = **99/100** (official encoder, `/v1/completions`,
serialized prefill, conc 4, 88.6 s). Path A baseline 100/100.

## Reading

- The PR alone gives **1.24×** (96.3 upstream-eager → 119.1 with the PR), not the
  1.81× the internal stack shows — because the internal stack carries more than
  the PR: 45 `V41_*` knobs + `VLLM_ADMISSION_GATE` + engram/o_proj/rope/moe
  code paths that are **not** in PR #18967.
- Path A's speedup is **not** attributable to PR #18967 alone. Any doc that
  presents 174/163 as "what the upstream path gives you" is wrong.
- Per-step: A = 512/163.3*6/5.89... ⇒ ~36 ms/step; B = ~49 ms/step at equal A
  (5.89 = acceptance at ceiling), so the delta is pure step time, not sampling.
