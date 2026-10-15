---
name: serving-config-search
description: Find the best deployment configuration for a served model on Ascend under an SLA — GPU_UTIL, MAX_SEQS, BAT_TOKENS, prefix caching, TP/EP degrees — by planning a pruned, ordered trial sequence instead of a grid sweep. Use when the user asks for the optimal serving config, wants to tune launch parameters, needs to know which GPU_UTIL/MAX_SEQS to pick, or wants to trade KV capacity against prefill speed. Prunes infeasible configs offline before any service restart, flags accuracy-coupled knobs, and emits trials in the benchmark comparator's row format.
---

# Serving Config Search

Answer "which launch configuration should I deploy" with a searched, evidenced
answer — not a default, and not a blind sweep.

## Why this is not a grid search

On this hardware, **one config change = one service restart ≈ 10–17 minutes**
(static-kernel compilation dominates; the kernel cache only helps when the
config is unchanged). A 4×3×3 knob grid is 36 restarts ≈ 9 hours of machine
time for information that is mostly predictable offline.

The search therefore runs in phases, and only the survivors touch hardware:

```
Phase 0  标定      one restart at a known-good config
                    └─ capacity analyzer reads the memory split + KV pool
Phase 1  剪枝      offline: extrapolate every candidate, prune SLA-infeasible
Phase 2  实测      only survivors, one knob at a time, ordered by expected value
Phase 3  裁决      existing benchmark comparator picks the SLA-met winner
                    └─ if the winner moved an accuracy-coupled knob → re-run L1
```

## The two search layers (absorbed from the ecosystem)

This skill searches the **launch-config layer** — knobs that require a
service restart. There is a second, cheaper layer above it:

| Layer | Knobs | Probe cost | Strategy |
|---|---|---|---|
| **Launch config** (this skill) | GPU_UTIL, BAT_TOKENS, MAX_SEQS, TP/EP, quant | **~10–17 min per probe** (restart + kernel compile) | prune offline, order nearest-neighbor, few trials |
| **Runtime load** (after launch is fixed) | concurrency, request rate | **seconds–minutes per probe**, no restart | warmup-abort → exponential ×2 → binary → validation |

The runtime layer is worth searching immediately after the launch winner is
deployed: the best `MAX_SEQS` is a guess until the actual SLO-breaking
concurrency is measured. See `references/search-strategy.md` §"The runtime
layer" for the phase algorithm (absorbed from
`awesome-ascend-skills`'s `vllm-bench-serve` auto-optimization — source
recorded in `../../references/ascend-ecosystem.md` §7).

## Workflow

### Phase 0 — calibrate (one restart)

Serve once at the known-good baseline (from the RUNBOOK / port repo
defaults). Feed the startup log to `llm-serving-capacity-planner` and record:

- `non_kv_gib` — weights + activation + non-torch + graph memory
- `kv_tokens` — the KV pool at this GPU_UTIL
- `hbm_gib` — per-device total

These three numbers make every other GPU_UTIL value predictable offline.

### Phase 1 — plan and prune (offline, no restarts)

```bash
python3 scripts/plan_config_search.py search.json
```

The input (`search.json`) names the baseline, the knob axes to explore, the
constraints, the calibration from Phase 0, and the SLA. The script:

1. **Enumerates** candidate configs, honouring constraints:
   - `pair` — knobs that must move together (e.g. the KV32 slot repack and
     BAT_TOKENS: changing one without the other fails to boot **or silently
     misreads**)
   - `forbidden` — known-bad regions, with the measured reason (e.g.
     GPU_UTIL 0.94: activation headroom collapses, measured 6× prefill
     slowdown)
2. **Extrapolates** each candidate: KV tokens ≈ `hbm × util − non_kv`,
   converted to tokens via the calibration's KB/token; concurrency ≈
   pool / request_len
3. **Prunes** anything that cannot meet the SLA's minimum concurrency —
   with the arithmetic shown, so the pruning is checkable
4. **Orders** survivors: baseline-adjacent first (one knob moved), then
   two-knob moves — nearest-neighbor exploration, because knob interactions
   are the exception, not the rule
5. **Budgets** the wall-clock: survivors × restart cost
6. **Flags** every trial that touches an accuracy-coupled knob (BAT_TOKENS)
   — such a trial cannot be declared the winner until the L1 gate re-passes

Output: `trials.json` — an ordered list, each entry with its config,
predicted capacity, prune status and reason.

### Phase 2 — run the survivors (live)

For each surviving trial, in order:

1. restart with the trial's env (the port repo's serve script)
2. run a fixed probe workload (same input/output/concurrency as every other
   trial — this is the benchmark skill's match-key discipline)
3. record the row into `rows.json` in the comparator's schema, with the
   full evidence block

**Stop early when the answer is already determined**: if a trial misses its
SLA by a wide margin and every remaining trial is a strictly worse
extrapolation, the search is over. The plan script's ordering exists to make
early stops possible.

### Phase 3 — verdict

```bash
python3 ../llm-serving-auto-benchmark/scripts/compare_results.py rows.json
```

The existing comparator picks the winner (and refuses unrecorded rows). If
the winning config moved an **accuracy-coupled** knob, the win is provisional
until `llm-serving-accuracy-gate` L1 re-passes on that config — a fast config
that answers wrongly is not a winner.

## The knob catalog (what each knob really does)

See `references/ascend-config-knobs.md` for the full catalog with sources.
The three that most often decide the outcome:

| Knob | Real effect (measured) | Coupling |
|---|---|---|
| `GPU_UTIL` | 0.92 → 8K prefill 1.14 s; 0.94 → 8.0–8.6 s (6×) for +1 GB KV | none |
| `BAT_TOKENS` | 2048 → long-context accuracy collapse (~0% at 260K, 127 chunks); 8192 → 32 chunks, clean | **accuracy-coupled**; pairs with KV32 repack |
| `MAX_SEQS` | sets graph-capture buckets (32 → max bucket 192, ~5 extra buckets) and the production concurrency shape | changes the perf 口径 vs quick mode |

## Scripts

| Script | Purpose |
|---|---|
| `scripts/plan_config_search.py` | Enumerate + constrain + extrapolate + prune + order + budget, from `search.json` → `trials.json`. Fully offline. |

## Relationship to the other skills

This skill is the **search loop over three existing ones**:

- feasibility from `llm-serving-capacity-planner` (Phase 0/1)
- trial records and verdict from `llm-serving-auto-benchmark` (Phase 2/3)
- accuracy safety from `llm-serving-accuracy-gate` (Phase 3, for
  accuracy-coupled winners)

It adds only what they lack: the constrained enumeration, the offline
extrapolation, the ordering, and the budget.

## References

- `references/ascend-config-knobs.md` — the knob catalog, each with its measured effect and source
- `references/search-strategy.md` — why phases instead of a grid; early-stop rules; interacting knobs
- `../../docs/EVIDENCE-RULES.md` — every trial row obeys the benchmark evidence rules
