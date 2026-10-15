---
name: ascend-compute-simulation
description: Estimate per-op FLOPs and MFU for a model at a serving shape on Ascend, from its config.json. Use when the user asks how much compute an op or a model costs, what MFU a latency implies, whether a layer is compute-bound, or wants operator shapes and FLOPs for a parallelism what-if. Computes FLOPs always, and MFU only when the chip peak is supplied with a source.
---

# Ascend Compute Simulation

Static, operator-level FLOPs for a serving shape — the "what should this cost"
half that pairs with `npu-profiler-analysis`'s "what did it cost".

## The rule that shapes this skill

**The tool never assumes a peak FLOPS number.** Ascend per-SKU peaks are
reported inconsistently across public sources (the same 910B variant is quoted
anywhere from ~280 to ~400 TFLOPs FP16 depending on who is counting and
whether sparsity is included). An MFU computed against a guessed peak is a
confident wrong number, which the evidence rules forbid.

So: you get FLOPs always, and MFU only if you pass `--peak-tflops` **and**
`--peak-source`. If you do not have a defensible peak, you still get the FLOPs
and the per-op breakdown, which is usually what the question needs.

## Workflow

```bash
# FLOPs and per-op table for a decode step
python3 scripts/model_compute_sim.py \
    --config /mnt/public/models/DeepSeek-V4.1-Flash/config.json \
    --batch 1 --seq 1 --kv-len 131072

# Prefill at 4k
python3 scripts/model_compute_sim.py --config cfg.json --batch 1 --seq 4096

# MFU, only with a sourced peak
python3 scripts/model_compute_sim.py --config cfg.json --batch 1 --seq 1 \
    --kv-len 128000 --latency-ms 31.6 \
    --peak-tflops <n> --peak-source "vendor datasheet <exact table>"
```

## What it produces

- total FLOPs and FLOPs/token (a MAC counted as 2 FLOPs)
- a per-op table with tensor shapes and per-op GFLOPs
- MFU, if and only if a sourced peak and a measured latency are both given
- warnings for the cases where the estimate is an upper bound

## Reading the numbers honestly

| Situation | What the tool does |
|---|---|
| MLA model (q_lora/kv_lora present) | uses the latent projections, not the dense QKV form |
| MoE | per token activates exactly top-k experts — an **upper bound** on useful work |
| `--tp`/`--ep` given | records them but **does not divide** FLOPs by them — routers, norms and shared experts are replicated, so a flat division misstates per-device work |
| decode (`--seq 1`) | attention is dominated by context; pass `--kv-len` or the attention FLOPs are wrong |
| no peak | no MFU, with an explanation rather than a default |

## Why the per-op table is the useful part

A single total FLOPs number answers nothing actionable. The table shows that,
for a 1-token decode of V4.1-Flash, `lm_head` alone is ~1.3 GFLOP while an
attention projection is a few hundredths — which tells you where a fusion or a
parallelism change can matter, and where it cannot.

Pair it with a real `kernel_details.csv` from `npu-profiler-analysis`:
the per-op FLOPs give the theoretical floor, the trace gives the measured time,
and the gap between them is where the optimization work is.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/model_compute_sim.py` | Config → per-op FLOPs + optional MFU |
| `scripts/comm_volume.py` | Per-layout communication model: collective COUNT and bytes per step (latency-bound vs bandwidth-bound) |
| `scripts/theoretical_optimum.py` | Achievable bound: necessary work × measured op capability; gap decomposition vs the measured step |

## References

- `references/flops-model.md` — the FLOP formulas used, and their assumptions
- `../../references/ascend-hardware-specs.json` — chip HBM and core counts (peaks deliberately not included: see the rule above)
- `../../docs/EVIDENCE-RULES.md` — why an unsourced peak is not admissible
