# Official Baselines — DeepSeek-V4.1-Flash

Extracted 2026-10-15 from `DeepSeek_V41_Tech_Report.pdf` (51 pages, shipped
inside the weight package at `/mnt/public/models/DeepSeek-V4.1-Flash/`).
Every number carries its page so it can be re-checked.

**These are the official-weights numbers (MXFP8 attention / MXFP4 experts per
the config). A W4A8 ModelSlim deployment should expect a gap — measuring that
gap is the point of L2, not a reason to hide it.**

---

## Sampling protocol (p32, p35)

- Reasoning-intensive benchmarks: **temperature 1.0, top-p 0.95** (p32 states
  "temperature and top-p of 1.0" for the Arena-style set; the agentic runs on
  p35 use "temperature 1.0, top-p 0.95")
- Agent runs: 1M-token context, max_steps 500 (p35)
- Reasoning-effort is a deployment knob, 25–100

## AIME 2026 (p35, p48, p49, p50)

- Metric: **Pass@1 (%)**
- Effort 25 → 100: **94% → 100%** (chart axis p50); "AIME 2026 reaches a full
  100%" at max effort (p49)
- Output length scales with effort: 4.6k → 11.4k tokens per response (p48)
- Part of the 8-benchmark reasoning average (AIME 2026, Apex 2025 Shortlist,
  GPQA Diamond, HLE, IMO-AnswerBench, LiveCodeBench, MathArena-Apex,
  SimpleQA-Verified — p35)

**Implication for gating:** effort moves the score by ~6 points. An AIME run
whose effort is not pinned and recorded is uninterpretable.

## GSM8K (p23, p24)

- Metric: **EM, 8-shot**
- Reported: **90.8 / 92.6 / 93.0** (three comparison columns, p24 table)

**The protocol trap, with real numbers:** the public Ascend port's own
acceptance gate ran GSM8K-200 in **chat mode with the official encoder and
thinking**, conc 4, and scored **198/200 = 99%** (three runs: 198, 199, 197).
Same benchmark, 8 points above the official 8-shot EM — the difference is
protocol, not model quality. Both numbers are correct; they are not
comparable to each other.

## Other reported benchmarks (first-seen pages)

GPQA Diamond (Pass@1, p33), HLE, Codeforces (rating), LiveCodeBench, HellaSwag
(EM, 0-shot, p24), HumanEval (Pass@1, 0-shot, p24), MATH (EM, 4-shot, p24),
MMLU-Pro, BBH, DROP, MMMU (vision), AGIEval, SuperGPQA, SimpleQA-Verified,
LongBench, DeepSWE v1.1 / Terminal-Bench v2.1 / SWE-Bench Pro (agentic).

## What the port repo gates on (its CORRECTNESS_STATUS.md)

| Gate | Value | Rule |
|---|---|---|
| Vision | 23/23 | — |
| GSM8K-200, chat mode, conc 4 | 198 / 199 / 197 out of 200 | drop >2 vs baseline = unusable; baseline ±1 is noise |
| Determinism (`clean`) | same prompt ×N, identical top-1 logprob | separate axis from accuracy |
| HCCL_DETERMINISTIC=true | GSM8K drops to 91/100 | diagnostic only, never delivery |

## Updating this file

When a new model is gated, add its own section with the same discipline:
benchmark, metric, page/section, sampling protocol. A baseline without its
protocol is not admissible (see `../../docs/EVIDENCE-RULES.md`).
