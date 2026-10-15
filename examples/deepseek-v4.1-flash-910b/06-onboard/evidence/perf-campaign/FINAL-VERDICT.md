# Final verdict — optimized service (2026-10-15)

Winning config: DRAFT_GRAPH=1 (with stray-self fix, image dsv41-a2:v10)
+ FUSED_MC2=1, on the E1/E2 stack (GPU_UTIL 0.92, MAX_SEQS 4, PREFIX 0,
BAT 8192, TP8+EP8, W4A8).

## Final gates

| Gate | Result |
|---|---|
| GSM8K-200 (chat, conc 4) | **197/200 = 98.5%** — identical to the pre-optimization baseline run; PASS (drop 1 vs port baseline 198, ±1 noise) |
| Wall time, GSM8K-200 | **174 s vs 231 s baseline = 1.33× on real work** |
| Single-stream 8K decode (predictable-continuation 口径) | 27.2 → **~172 tok/s (6.3×)**, A up to 5.69 |
| Stability | soak 5/6 rounds clean (round 6 ended by external container removal — shared box); zero crashes attributable to the config across all sessions |

## Campaign ledger

| Lever | Verdict | Gain |
|---|---|---|
| E1 DRAFT_GRAPH=1 + bug fix | **kept** | 6.1× (continuation 口径) / ~1.3× (real work) |
| E2 FUSED_MC2=1 | **kept** | +3% single-stream, neutral conc-4 |
| E3 dynamic K | abandoned | dead end (3 walls, documented) |
| Operator-level (fusion/TileLang) | not reached | theory says ≤17% ceiling; deferred |

Theory cross-check: the win came from the residual (speculation), exactly
where the theoretical decomposition placed the framework headroom.
