# Search Strategy

Why phases instead of a grid, and the rules that keep the search honest.

---

## The cost model that shapes everything

| Operation | Cost (this box, measured) |
|---|---|
| Service restart, new config | ~10–17 min (static-kernel compile dominates) |
| Service restart, same config | minutes (kernel cache hits) |
| Offline extrapolation | milliseconds |

A 4×3×3 grid = 36 restarts ≈ 9 hours. The same answer is usually reachable
in 4–8 restarts, because most knobs are monotone and their interactions are
rare. The strategy below spends restarts only where extrapolation is
uncertain.

## The four phases

### Phase 0 — calibrate

One restart at the known-good baseline. The capacity analyzer produces
`non_kv_gib`, `kv_tokens`, `hbm_gib`. This anchors every extrapolation.

**Extrapolation model** (deliberately simple, and stated so it can be
checked): KV GiB at util u ≈ `hbm_gib × u − non_kv_gib`; KV tokens via the
calibration's tokens-per-GiB. It ignores second-order effects (graph-memory
changes with MAX_SEQS, activation changes with workload) — which is why
Phase 2 exists: extrapolation prunes, measurement decides.

### Phase 1 — prune offline

A candidate is pruned when its **predicted concurrency** (pool tokens ÷
request length) is below the SLA's minimum. The arithmetic is emitted with
every prune so a human can check any single decision.

Also pruned: configs violating `pair` constraints, configs inside
`forbidden` regions, and configs identical to already-planned trials.

### Phase 2 — measure the survivors, in order

Ordering rule: **nearest-neighbor from the baseline** — one-knob moves
first, then two-knob moves. Rationale: interactions are the exception; if
one-knob moves explain the SLA, the search stops before paying for
interactions.

**Warm up a fresh service before probing it.** Found the hard way on the
first live run: a freshly started service's first requests absorb graph
capture and JIT warmup — the same probe that measured 0.94 s on a warm
service returned 15.1 s / 6.7 s / 0.94 s on a cold one. Discard ~5 warmup
requests, then probe; a baseline measured warm is not comparable to a trial
measured cold. This is the config-layer echo of the runtime layer's
"warmup as a sanity gate" phase below.

Early-stop rules:

- a trial misses SLA by a wide margin **and** every remaining trial
  extrapolates strictly worse → stop
- a trial meets SLA **and** every remaining trial extrapolates at most
  marginally better (< a stated epsilon) → stop, take the simpler config

Every trial records a full benchmark row (same workload as every other
trial; the comparator's match keys enforce this).

### Phase 3 — verdict, with the accuracy hook

The comparator picks the SLA-met winner. **If the winner moved an
accuracy-coupled knob, the verdict is provisional** until the accuracy
gate's L1 re-passes on that exact config. The plan marks such trials
upfront, so this cannot be forgotten after a long search.

## Interacting knobs

Two knobs interact when the effect of one depends on the other's value.
Known cases in this ecosystem:

- KV32 slot repack × BAT_TOKENS (hard pairing — boot failure or silent
  misread)
- MAX_SEQS × graph memory (more buckets, more HBM — enters the capacity
  budget)
- GPU_UTIL × workload activation (the 0.94 cliff is an activation-headroom
  effect; a lighter workload may tolerate it)

When an interaction is suspected but unknown, the planner offers a
**crossing trial** (both knobs moved together) rather than a full
sub-grid — one restart answers "do they interact" well enough to decide
whether more is worth it.

## The runtime layer (after the launch winner is deployed)

Launch configs are restart-costly, so this skill prunes and orders them
sparingly. Runtime load — concurrency and request rate — is the opposite:
each probe costs seconds to minutes with no restart, so a full adaptive
search is affordable. The algorithm below is absorbed from
`awesome-ascend-skills`'s `vllm-bench-serve` auto-optimization strategy
(source: `../../../references/ascend-ecosystem.md` §7).

**Problem**: given SLO constraints (e.g. `p99_ttft ≤ 500 ms`,
`mean_tpot ≤ 50 ms`, `success_rate ≥ 95%`, `goodput_ratio ≥ 0.9`), find the
maximum concurrency (or request rate) where all constraints hold.

**Four phases:**

1. **Warmup as a sanity gate.** Probe at minimum load (concurrency 1, ~50
   prompts). **If the SLO already fails here, abort** — no search can help;
   the service or the SLO is wrong. This single check prevents wasted hours.
2. **Exponential probe** (×2 per step, ≤15 iterations): find the upper bound
   where the SLO first breaks. Coarse sampling (≈3× the probe value, floor
   50 prompts) — the goal is the bracket, not precision.
3. **Binary search** (≤8 iterations) inside the bracket. Finer sampling
   (≈6×, floor 100).
4. **Validation.** Re-run at the winner with high sampling (≈10×, floor 200)
   — a winner that only held at 3× sampling is not a winner.

**Sample scaling is the point**: P99 is meaningless at 30 samples. The
multiplier grows with the phase because confidence requirements grow with
the phase.

**Search modes** (pick one dimension at a time; joint 2-D search is a
last resort and must be budgeted aloud to the user first):
concurrency-only; rate-only; fixed-concurrency search rate;
fixed-rate search concurrency.

## What the search is NOT allowed to do

- Select a config on throughput alone while an SLA row shows missed.
- Select a config whose win rests on an accuracy-coupled change without an
  L1 re-pass.
- Mix 口径 (MAX_SEQS 4 rows with 32 rows in one comparison).
- Leave a trial's row without the evidence block — the comparator rejects
  it, and the search result inherits that rejection.
