# Config search — live run report (2026-10-15)

Question: for a 128K-context × 16-concurrent SLA on DeepSeek-V4.1-Flash W4A8
(8×910B3, TP8), which GPU_UTIL is optimal?

## Phase 0 — calibration (live service @ 0.92)
- KV pool: 13.60 GiB = 2,942,180 tokens; non-KV 43.49 GiB; HBM 60.96 GiB
- baseline TTFT (8K fixed prompt, warm, n=3): median 0.943 s

## Phase 1 — plan (offline, planner output)
- 0.94 pruned: forbidden region (measured 6× prefill collapse)
- survivors: 0.90 (pred. 12.38 GiB / 20 conc), 0.88 (pred. 11.16 GiB / 18 conc)
- budget 2 restarts ≈ 30 min

## Phase 2 — Trial 1 live (GPU_UTIL=0.90)
| Quantity | Predicted | Measured | Error |
|---|---:|---:|---:|
| KV pool | 12.38 GiB | 12.38 GiB | — |
| KV tokens | 2,678,323 | 2,678,374 | **0.002%** |
| concurrency @128K | 20 | 20 | — |
| TTFT (warm, n=5) | — | 0.928 s median | — |

**Methodology note (found the hard way):** the first probe of a freshly
started service returned TTFT [15.1, 6.7, 0.94] s — the first requests absorb
graph capture / JIT warmup. After 5 discarded warmup requests the probe was
tight: [0.922…0.935] s. Baseline and trial are only comparable when both are
warm. (Absorbed into the skill's references.)

## Early stop — Trial 2 (0.88) skipped, per the strategy's rule
- The SLA needs 2,097,152 tokens. Baseline 0.92 provides 2.94M (22 conc),
  trial 0.90 provides 2.68M (20 conc) — both pass; 0.92 passes by more.
- TTFT is statistically flat in this region (0.943 vs 0.928 s, difference
  inside the baseline's own spread 0.943–1.030 s).
- Remaining trial 0.88 extrapolates to strictly LESS KV (2.41M) — it cannot
  beat the baseline on the binding dimension and TTFT is flat.
  ⇒ "every remaining trial extrapolates at most marginally better" → stop.

## Verdict
**Winner: GPU_UTIL = 0.92 (the baseline).** It maximizes KV headroom under
the SLA (22 vs 20 concurrent @128K) at indistinguishable TTFT. The search
cost one restart instead of three: 0.94 was pruned offline with the measured
reason, 0.88 was skipped by the early-stop rule.

## What this run validated
1. The planner's offset-calibrated extrapolation: 0.002% error on KV tokens.
2. The forbidden-region constraint: the documented 0.94 cliff was never
   re-paid for (15 min saved).
3. The early-stop rule: one more restart saved with the arithmetic shown.
4. A methodology gap found and fixed: warm-up before probing a fresh service.
