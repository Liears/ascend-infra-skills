# L1b gate verdict — GSM8K-200 (real run, 2026-10-15)

```
# Accuracy gate: L1b GSM8K-200 chat — ✓ PASS

- result: **197/200**
- drop vs baseline: 1

## Reasons
- drop 1 vs baseline 198/200 (rule: >2 = FAIL, ±1 = noise)

## Evidence block
gate:     L1b GSM8K-200 chat
harness:  acc_eval.py @ chiro2001 port (8-shot, /v1/completions + official encoder)
dataset:  gsm8k test, 200 problems (AI-ModelScope/gsm8k parquet->jsonl)
sampling: chat mode, conc 4, effort 75, max_tokens 512
draft:    DSpark S=5 on (serve default), DRAFT_GRAPH=0
result:   197/200
artifact: gsm8k-result.json
```

Context: the port repo's own three runs were 198/199/197 out of 200 — this
run's 197 sits inside their observed range. Wall time 231 s at conc 4.
The "accuracy not established" gap in this example's README is now closed
for L1b; L2 (AIME 2026, avg@4, official sampling) remains open.
