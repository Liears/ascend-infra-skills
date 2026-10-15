# Harness Notes

Protocol details of each harness, verified where possible against the
installed source. The rule they all serve: **one harness per comparison, and
the harness identity goes into the evidence block.**

---

## The port repo's own runner (`acc_eval.py`)

Read from the public Ascend port's `tests/acc_eval.py` on 2026-10-15.

- **Endpoint:** `/v1/completions` (completion API, not chat), with the prompt
  built by the **official encoder** (`encoding/encode_messages` from the
  weight package) — this is what "chat mode" means in its acceptance docs
- **GSM8K construction:** 8-shot, shots taken from the first 8 rows of the
  train split, `Question: …\nAnswer: …` format, gold extracted from the
  `#### N` suffix
- **Flags that define the 口径:**
  - `--mode chat|thinking`
  - `--effort` (default 75) — the reasoning-effort knob
  - `--max-tokens` (default 512)
  - `--conc` (default 4)
  - `--gsm8k-data-dir` — local `train.jsonl`/`test.jsonl`, same 8-shot order,
    **no `datasets` dependency** (preferred on an offline box)
- **The serialization caveat (from its own wrapper's docstring):** `--conc 4`
  holds a global lock until first token — concurrent decode, **serialized
  prefill**. Its wrapper states this explicitly: passing GSM8K does not prove
  concurrent-batch or mixed prefill+decode correctness.
- Checkpoint/resume: supported (`.ckpt` files)

**Data on this box:** ModelScope mirror `AI-ModelScope/gsm8k` ships parquet;
convert to the expected JSONL with
`df[["question","answer"]].to_json(out, orient="records", lines=True)` —
field names must be exactly `question`/`answer`.

## evalscope

- Community-standard evaluation framework (ModelScope), OpenAI-compatible
  endpoints — connects directly to a vLLM-Ascend service
- AIME support; task config in YAML; supports repeats for avg@k
- **Comparability:** numbers are comparable to official/community reports
  only when sampling (temperature, top-p, thinking, effort) is matched.
  evalscope defaults will not match the official protocol by itself —
  set them explicitly.
- Not preinstalled on this box; install per-project, do not install globally
  (shared machine rule).

## aisbench

- Huawei's benchmark harness; `vllm-ascend` ships a wrapper
  (`tools/aisbench.py`) and its CI uses it
- Pulls its own datasets from ModelScope at runtime — an offline box needs a
  pre-populated cache
- **Comparability:** use when comparing against Huawei-published gates;
  do not mix its numbers into a comparison against community-harness numbers

---

## Choosing between them

| You want to compare against | Use |
|---|---|
| The port repo's published gates (GSM8K-200, vision 23) | the port repo's runner — same 口径 by construction |
| The official tech report (AIME 2026 etc.) | evalscope with official sampling pinned |
| Huawei/vllm-ascend published gates | aisbench |

If the comparison target has no published protocol, run your own baseline
twice and gate against your own run — never against a number you cannot
reproduce the conditions of.
