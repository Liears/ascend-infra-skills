# Measured Anchor — where the 430 µs per-collective latency comes from

The communication model (`scripts/comm_volume.py`) refuses to assume any
bandwidth or latency number. Its latency input is **derived from our own
measurement**, and this file shows the derivation so it can be checked.

## The measurement (2026-10-15, live deployment)

DeepSeek-V4.1-Flash W4A8, 8×910B3, TP8+EP8, single-stream decode at 8K
context:

- measured step time: **62.32 ms/step** (median, quote 口径, run_test)
- measured communication share: **83%** of device time
  (229,642-row kernel_details.csv, core-type attribution)
- ⇒ communication per step ≈ 0.83 × 62.32 ≈ **51.7 ms**

## The count (from the model structure)

TP8+EP8 layout, per layer: 1 attention allReduce + 1 MoE dispatch + 1 MoE
combine = **3 collectives**. 40 layers ⇒ **120 collectives per decode step**.

## The derivation

```
per-collective latency ≈ 51.7 ms / 120 ≈ 430 µs
```

## Why this is the latency-bound regime, not bandwidth-bound

Message sizes at batch-1 decode: one hidden-state copy is
5120 × 2 B = **10.24 KB**. A ring allReduce moves ~1.75× that per rank
(≈18 KB); the MoE alltoall sends ~4.4 copies (≈45 KB). At these sizes the
per-collective cost is dominated by **synchronization/launch latency, not
data transfer** — the same collective at 1 MB would cost barely more.

Two independent checks that this is real:

1. **Closure**: the model predicts 51.6 ms from 120 × 430 µs — the
   measurement it was derived from (51.7 ms). This is consistency, not
   validation; the next check is the real one.
2. **Structure**: the top trace ops are `AivKernel` (41.5%) and `allReduce`
   (39.4%) — collective wrappers, not large transfers. A bandwidth-bound
   profile would show comm time scaling with bytes; a latency-bound one
   shows it scaling with count. (Full confirmation: run two batch sizes and
   check comm ms/step stays ~flat while bytes/step scale — a cheap
   experiment left as the next measurement.)

## What the theory changes about lever ordering

At latency-bound small batch:

- **reducing BYTES does nothing** — `tp-ep-ag` (different byte pattern,
  same count) projects ±0%
- **reducing COUNT is the lever** — `dp-ep` (DP attention, no attn
  allReduce) and pure `tp` (no alltoall) both project **−33% comm time**
- projected step: 62.32 → ~45 ms ⇒ ~**1.4× tok/s** ceiling from layout
  alone, before any kernel work

Both layouts carry feasibility costs the model states but does not price:
`dp-ep` replicates attention weights 8×; pure `tp` touches a shard of every
activated expert (memory reads scale with distinct experts — comparable
total bytes to EP at small batch, but check before trusting).

## Using this file

When re-anchoring on a new deployment, replace the three inputs (step time,
comm share, collective count for your layout) and re-derive. Never carry
the 430 µs to a different hardware or topology without re-measuring.
