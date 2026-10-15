# Gate Definitions

The three gates in full. Each names what it catches, its pass rule, and where
the rule came from — a threshold without a source is a guess wearing a number.

---

## L1 — smoke gate (every build)

**Purpose:** catch port / quantization / chat-template bugs cheaply. Runs in
minutes, fails loudly.

### L1a — determinism check

- Fire the same prompt N times (N ≥ 5), `max_tokens=1`, `logprobs=1`
- **Pass:** the top-1 logprob at the same position is identical across all N
- Source: the port repo's `clean` criterion (its CORRECTNESS_STATUS.md)
- **Caveats, from its own findings:**
  - speculative decode reduces determinism ~5× — pin the draft setting
    before comparing
  - this checks determinism, not correctness; a model can be perfectly
    deterministic and wrong

### L1b — GSM8K-200 chat mode

- 200 problems, chat mode via the **official encoder directory** (the
  `encoding/` shipped in the weight package), conc 4, thinking per the
  port's accepted mode
- **Pass:** drop >2 correct vs the port repo's baseline (198/200) = FAIL;
  baseline ±1 is noise
- Source: the port repo's acceptance rule, verbatim
- **Known limitation (stated in its own runner):** `--conc 4` holds a global
  lock until first token — prefill is serialized. Passing L1b does **not**
  prove concurrent-batch or mixed prefill+decode correctness. Those need the
  repo's multibatch tests.

---

## L2 — sensitive detector (pre-release)

**Purpose:** quantify what quantization did to reasoning. Math reasoning is
the first axis W4A8 damages, and AIME 2026 is the official benchmark for it.

- Dataset: **AIME 2026** (30 problems)
- Sampling: **official protocol — temperature 1.0, top-p 0.95**, thinking on,
  **effort pinned and recorded** (effort 25→100 spans 94→100% officially;
  an unpinned effort makes the number uninterpretable)
- Repetition: **avg@4 minimum** (each problem = 3.3 points; a single pass
  cannot separate a real regression from two lucky guesses)
- `max_tokens`: ≥ 16k (official output length is 4.6k–11.4k per response
  depending on effort; too-small a cap truncates reasoning and silently
  scores correct reasoning as wrong)
- **Report:** mean and per-run spread, never the mean alone
- **Comparison rule:** only against numbers produced under the same protocol.
  Against the official MXFP8/MXFP4 baseline, a gap is the measurement, not
  a failure; the failure mode L2 exists to catch is a *sudden* drop vs the
  port repo's own recorded L2, or a spread so wide the mean is meaningless.

---

## L3 — coverage gate (pre-delivery)

**Purpose:** axes L1/L2 cannot see. AIME tests math reasoning only; a
deployment can pass L1+L2 and still be broken on other axes.

| Axis | Test | Why AIME can't see it |
|---|---|---|
| Vision | the port repo's 23-case image QA set | V4.1-Flash has a vision tower; text-only evals never touch it |
| Long context | long-context QA at the deployed max length (this family serves 1M) | 30 short problems never exercise KV at depth, where quantized attention degrades |
| Code | LiveCodeBench (official benchmark for this family) | execution-based, different failure surface |
| Chat/instruction | chat-mode checks beyond GSM8K extraction | template bugs surface in chat mode; completion-mode harnesses can bypass them |

Per-axis thresholds come from the port repo's gates where they exist
(vision 23/23) and are otherwise set from the first accepted run of this
deployment — recorded at that time, so later regressions have something to
regress against.

---

## What a gate result must carry

Every gate result is recorded with the extended evidence block (see the
SKILL.md and `scripts/check_accuracy_gate.py`):

```
gate:        L1b GSM8K-200 chat
harness:     acc_eval.py @ <commit> (port repo runner)
dataset:     gsm8k test split, 200 problems, 8-shot from train
sampling:    chat mode, conc 4, effort <n>, max_tokens 512
draft:       <on|off, pinned>
baseline:    198/200 (port repo CORRECTNESS_STATUS, 3-run median)
result:      <c>/200
verdict:     PASS | FAIL (drop >2 = FAIL)
artifact:    <path to raw json>
```

A result without harness, dataset version, sampling, and draft setting is
not a result — it is a number that cannot be compared to anything.
