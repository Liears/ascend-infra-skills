# E2 — FUSED_MC2=1 (stacked on E1): +3% single-stream, neutral conc-4

## Config
DRAFT_GRAPH=1 (E1, fixed) + FUSED_MC2=1 → `enable_fused_mc2:1` confirmed in
the engine's additional-config. Known trade-off, logged by the engine:
fused_mc2 and multistream_overlap_shared_expert are mutually exclusive —
the latter auto-disables (one overlap mechanism traded for another).

## Results (same probes as E1)

| workload | E1 | E2 | delta |
|---|---|---|---|
| 8K ctx single-stream tok/s | 136.6–170.5 (med ~167) | 169.3–173.6 (med ~172) | **+3%** |
| short-ctx single-stream tok/s | 87.2–87.8 | 87.1–93.0 | ~+2% |
| GSM8K-100 wall (conc 4, warm) | 66.0–67.8 s | 66.8 s | neutral |
| GSM8K-100 accuracy | 100, 99, 100 | 99, 100 | gate band ✓ |

## Reading

The port repo's own four-axes doc measured fused_mc2 halving comm time
(9.19→4.55 ms/step) at N=4–8 decode; our single-stream end-to-end gain is
+3% because (a) the swapped-out multistream overlap ate part of it and (b)
at A≈5.7 the step is already speculation-dominated. Kept: no observed
downside, accuracy holds, and the conc-4 path may benefit more at higher
concurrency than we probe here.
