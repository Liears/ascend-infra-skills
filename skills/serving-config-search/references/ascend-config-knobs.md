# Ascend Config Knob Catalog

What each launch knob actually does, with the measurement or source for every
claim. A knob entry without a source is a guess — do not add one.

Sources: `serve_a2.sh` / `run_test.sh` comments and measured tables in the
public V4.1 Ascend port (read 2026-10-15), plus this suite's own runs.

---

## Memory & capacity

### `GPU_UTIL` (HBM fraction)

- Default 0.92. **0.94 leaves only 6.11 GiB headroom and makes 8K prefill
  8.0–8.6 s instead of 1.14 s** — a 6–7× slowdown for ~1 GB more KV
  (source: serve_a2.sh measured table; KV 2,987,727 @ BAT=8192 without
  repack).
- 0.88 → 9.80 GiB headroom, prefill 1.28 s (mildly slower than 0.92).
- Not cosmetic: it trades KV capacity against activation headroom, and the
  curve has a cliff. Search it, but expect the answer to be near 0.92.

### `KV_CACHE_MEMORY_BYTES` / `--kv-cache-memory`

- Sets the KV pool directly; when present, the utilization fraction no
  longer governs the pool (see the capacity planner's parsing rules).
- Pairs with the KV32 slot repack on the A3 path: **changing one without the
  other either fails to boot or silently misreads** (source: the port repo's
  KV32 verdict doc).

## Chunked prefill

### `BAT_TOKENS`

- Default 8192 (raised from 2048 by a specific fix).
- **2048 causes long-context accuracy collapse**: ~0% pass at 260K context
  (127 chunks; ~2% deviation per chunk, accumulating). 8192 → 32 chunks,
  6/6 pass. (Source: the port repo's longctx-accuracy-fix report.)
- **This is the accuracy-coupled knob.** Any trial that changes it must
  re-pass the L1 gate before it can be declared the winner. A config search
  that treats it as a pure performance knob can silently select a
  configuration that answers wrongly at length.

## Concurrency & graph capture

### `MAX_SEQS`

- 4 = the quick/perf 口径; 32 = the production shape.
- Determines graph-capture buckets: 32 → max bucket 192, ~5 extra buckets vs
  4. More buckets = more graph memory (see the 1.02 GiB NPU-graph figure)
  and longer startup, but the production concurrency form.
- **口径 warning:** perf numbers at MAX_SEQS=4 and production numbers at 32
  are not comparable (the benchmark skill enforces this as a match key).

### `PREFIX` (prefix caching)

- 0 = perf 口径 (historical, byte-identical comparisons); 1 = production
  (decode queue interleaves new requests). The two measure different things.
- With PREFIX=1, TTFT comparisons must state it — a prefix hit changes TTFT
  by an order of magnitude.

## Parallelism

### `TP` / `EP` / `DP` degrees

- Counted in **devices**, not cards (A3: 8 cards = 16 devices).
- At TP8/EP8 on a large MoE, communication was **83% of device time**
  (measured, 229,642-row trace). Reducing or restructuring the parallel
  layout moves the ceiling far more than kernel tuning — which is exactly
  why the config search should consider parallelism axes, not just memory
  knobs.

## Speculative decode

### `DRAFT_GRAPH` / draft settings

- Default off **deliberately**: enabling it without understanding yields a
  bad accept length (A ≈ 1.07 observed) with **no error anywhere** — the
  classic silent degradation.
- Draft on/off also changes determinism (~5× fewer identical-logprob runs
  with draft). For accuracy trials, pin it and record it.
- Helps low concurrency, can hurt high concurrency. A search axis only with
  the accept-length anchor recorded.

## Graph mode

### `STATIC_KERNEL` / `NPUGRAPH_EX`

- Real HBM cost (1.02 GiB measured) for real speed. Part of the capacity
  budget, not a free flag.
- A model that only works because graph capture hides a host-sync bug is not
  Day-0 ready (see the day0 skill's G5 gate).

---

## Constraint types for the planner

| Type | Meaning | Example |
|---|---|---|
| `pair` | knobs must move together | KV32_REPACK ↔ BAT_TOKENS |
| `forbidden` | a known-bad region, with the measured reason | GPU_UTIL=0.94 |
| `accuracy_coupled` | changing it requires an L1 re-pass before winning | BAT_TOKENS |
| `口径` | changes the measurement regime; rows must record it and not mix | MAX_SEQS 4 vs 32, PREFIX 0 vs 1 |
