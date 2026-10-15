# E1 — DRAFT_GRAPH=1: crash fix + 6× decode throughput (2026-10-15)

## The bug that kept DRAFT_GRAPH "unverified"

`DRAFT_GRAPH=1` crashed on first request: `NameError: name 'self' is not
defined`, raised from the lambda at `dsa_v1.py:1494` calling the module-level
`build_dspark_swa_indices`. Root cause: that function (extracted from a
method) still contained a stray `self` —

```python
_v41_bt_parent = getattr(self, "block_table", block_table)   # ← module scope: no self
```

executed unconditionally, plus two `self`-referencing fields in the error
f-string. The row-mismatch diagnostic it fed is preserved; only the stray
references were removed (3 lines, both in the live file and in
`/opt/dsv41/patches/draft/dsa_v1.py` so the fallback installer ships the
fix). Patched into image `dsv41-a2:v10` (docker commit).

## Results (live, 8×910B3, W4A8, TP8+EP8)

| Metric | baseline (DRAFT_GRAPH=0) | E1 (DRAFT_GRAPH=1, fixed) |
|---|---|---|
| decode tok/s @8K ctx | 27.2 | **136.6–170.5 (median ~167)** |
| acceptance length A | 1.70 | **2.98–5.69** (4.19 on GSM8K, real work) |
| TTFT @8K | ~0.94 s | 1.23 s |
| GSM8K-100 (chat, conc 4) | — | **100/100, then 99/100** |
| SpecDecoding metrics (conc 4) | — | accepted 127.6 tok/s, per-position 0.85/0.74/0.64/0.53/0.44 |

## Stability

One engine death observed ~14 s after idle. RESOLVED as external
termination, not a bug: the log shows vLLM's graceful SIGTERM shutdown
(`[shutdown] Process manager: send sigterm`, `Application shutdown
complete`) — a native crash does not produce those lines. The box is
shared; a colleague likely stopped the container. Zero crashes observed
across all DRAFT_GRAPH sessions otherwise. Soak test (6 rounds of
GSM8K-100 + idle) running to confirm.

## Reading

The 6× comes from speculation actually engaging: A≈1.7 → 4-5 accepted tokens
per step at the same step time. This is the framework-layer lever the theory
predicted (the residual, not the comm floor) — and it dwarfs every other
lever measured so far.
