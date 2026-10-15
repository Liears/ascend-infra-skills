---
name: extreme-perf-campaign
description: Orchestrate an extreme performance-optimization campaign on a served model on Ascend — from theory (communication model, achievable bound) through framework-layer lever experiments to operator-level fusion, with accuracy gates on every change and honest headroom arithmetic that stops work before it wastes restarts. Use when the user wants extreme/max performance tuning, a systematic optimization campaign, "跑一遍极致性能优化", or asks what to optimize next on a serving deployment. Model-agnostic; the worked example is DeepSeek-V4.1-Flash on 8x910B3.
---

# Extreme Performance Campaign

The orchestration layer over the other skills: it decides **what order to try
things, when to gate, and when to stop**. The individual measurements live in
the specialized skills; this skill is the campaign discipline that made them
pay off — ported from a real campaign (27.2 → 172 tok/s single-stream, 1.33×
on real workload, zero accuracy loss, every number evidenced).

## The core principle

**Theory ranks the levers. Gates guard every change. Honest arithmetic stops
the work before it wastes restarts.** On Ascend one config change costs a
10–20 min restart — the campaign's economics are restart-budget economics.

## Campaign phases

```
P0 theory      → lever ranking (predicted payoff per restart)
P1 framework   → one knob at a time, ranked; gate every change
P2 re-trace    → fresh trace on the WINNING config (the structure shifts!)
P3 operator    → library check → kernel fallback → headroom arithmetic → wire or stop
P4 stabilize   → soak + full gates on the final config
P5 upstream    → drift quantification; price the rebase before doing it
```

### P0 — Theory first (offline, no restarts)

- `ascend-compute-simulation`: FLOPs/token, comm volume by layout
  (`comm_volume.py`), achievable bound (`theoretical_optimum.py`)
- Output: a **lever table** ranked by predicted payoff per restart, each with
  its basis (e.g. "comm is 83% of step and latency-bound → layout/collective
  count is the lever; kernel tuning ceiling is 17%")
- The ranking is a hypothesis — P1 exists to test it — but never skip it:
  an unranked campaign burns restarts on the visible lever instead of the
  binding one.

### P1 — Framework-layer experiments

Rules (each one earned its place by saving a restart):

1. **One knob at a time**, nearest-neighbor from baseline (the
   `serving-config-search` discipline).
2. **Gate every change**: a fast accuracy gate (GSM8K-class, 2 min) after
   each config before believing any speed number. A config that answers
   wrongly is not a winner at any speed.
3. **Warm the service before probing** — first requests absorb graph
   capture/JIT (measured: 15.1 s cold vs 0.94 s warm on the same probe).
4. **Early-stop when the ordering makes the answer deterministic** — the
   remaining trials extrapolate strictly worse → stop, take the simpler config.
5. **A dead lever is a documented deliverable**, not a failure. Record the
   walls (example: dynamic-K hit three distinct walls — not baked / graph-mode
   incompatibility / patch-layering inconsistency — each with its escape
   hatch noted for later).

### P2 — Re-trace the winning config

**The optimization changes the profile.** In the worked example the
communication share fell 83% → 39% and compute grew to 60% — the operator
ceiling moved from ≤17% to much more. Operator-layer decisions made on the
baseline trace would have targeted the wrong things. Re-trace, then
re-rank.

### P3 — Operator layer, in this exact order

1. **Check the CANN library first.** `dir(torch_npu)` lists fused ops —
   but the registry lies: names can fail lazy import. Probe each with
   `getattr` before designing around it (worked example: the add+norm+quant
   fusion existed and hit 4.97× micro; the no-add variant did not exist).
2. **Check the model's actual layer flow** before assuming the fused op
   drops in. Structures like mHC (hyper-connections) break the standard
   `Add → norm` adjacency the fused kernels assume.
3. **Triton-Ascend fallback** for missing fusions: chunked loops for row ops
   (UB is ~192 KB — a naive full-row BLOCK overflows), validate against the
   aclnn reference (±1 quant step / ≥95% exact match is the parity bar).
4. **The honest headroom arithmetic — mandatory before wiring**:
   run `scripts/fusion_headroom.py`. Eager micro-benchmarks overstate
   in-graph gains (launches are amortized inside graphs; at small batch the
   tensors are tens of KB). If the computed end-to-end gain is below the
   wiring cost (restart + gates), **stop and record** — the kernel ships as
   a validated artifact for the higher-batch case.

### P4 — Stabilize

- Soak: alternating load and idle rounds with health checks (the worked
  example: 5/6 clean rounds; round 6 ended by an externally-removed
  container — see lessons-learned for telling those apart).
- Full gates on the final config, evidence blocks complete.

### P5 — Upstream track

Your pinned base ages fast (worked example: 5 weeks behind = 171
model-relevant commits, including upstream's proper fix of a function we had
hand-patched). Before rebasing:

1. Quantify the drift: `scripts/upstream_drift.py` (commits in window,
   perf-relevant subset, which local patch keywords already exist upstream).
2. **Price the rebase first**: run upstream HEAD natively with your weights
   and probe the same 口径 — the gap to your optimized stack IS the value of
   your local patches. Rebase only what the gap proves necessary.

## What this skill refuses to do

- Chase a lever the theory didn't rank (the visible ≠ the binding).
- Trust a speed number from an ungated config.
- Wire a validated kernel whose computed end-to-end payoff is below its
  wiring cost.
- Treat "engine died" as a bug without reading the shutdown logs (graceful
  SIGTERM = external kill on a shared box).

## Scripts

| Script | Purpose |
|---|---|
| `scripts/fusion_headroom.py` | The should-I-wire calculator: eager gain → in-graph end-to-end % with the arithmetic shown |
| `scripts/upstream_drift.py` | Base-vs-upstream drift: commit counts, perf subset, local-patch overlap check |

## References

- `references/campaign-playbook.md` — the full playbook, phase by phase, with the worked example's numbers at each decision point
- `references/lessons-learned.md` — the trap catalog (10 entries, each with its symptom and its tell)
- `../../docs/EVIDENCE-RULES.md` — every campaign number obeys them

## Relationship to the other skills

This skill **orchestrates**: theory from `ascend-compute-simulation`,
experiments from `serving-config-search`, gates from
`llm-serving-accuracy-gate`, traces from `npu-profiler-analysis`. It adds the
sequence, the discipline, and the stop rules.
