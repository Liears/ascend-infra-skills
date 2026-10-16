# Campaign Playbook

The phase-by-phase playbook, model-agnostic. Every number below is from the
worked example (DeepSeek-V4.1-Flash W4A8, 8×910B3, TP8+EP8) and is cited so
the pattern — not the number — is what you transfer.

---

## P0 — Theory (no restarts)

Run the three tools in `ascend-compute-simulation`:

| Tool | Output in the example |
|---|---|
| `model_compute_sim.py` | ~39 GFLOP/token |
| `comm_volume.py` | 120 collectives/step; latency-bound at batch-6 (10 KB messages); model closed to 0.2% against the measured comm time |
| `theoretical_optimum.py` | 62.32 ms step = 51.6 comm floor + 10.7 residual |

**Deliverable: the lever table.** Example ranking that came out:

| lever | predicted basis |
|---|---|
| speculation engaging (the residual) | 10.7 ms residual, A was 1.7 vs a proven 4+ |
| collective count (layout) | comm floor is count×430 µs — fewer collectives/layer lowers the floor itself |
| bytes-variant comm (e.g. allgather) | ±0% — count unchanged, latency-bound |
| kernels | ≤17% ceiling, and partly shadowed |

The campaign then confirmed the ranking: the residual lever delivered 6.3×;
the bytes-variant delivered ±0%; the kernel ceiling held.

## P1 — Framework experiments

The experiment ledger format (keep one; it is the campaign's memory):

```
E1 <knob=value>  hypothesis → result → verdict (kept/abandoned) → gate result
E2 <knob=value>  ...
```

Worked example ledger:

| # | config | result | verdict |
|---|---|---|---|
| E1 | DRAFT_GRAPH=1 (+ a bug fix — see lessons #1) | 27.2→167 tok/s, A 1.70→4.19 | **kept** |
| E2 | +FUSED_MC2=1 | +3% single-stream, neutral conc-4, accuracy in band | **kept** |
| E3 | dynamic K (SP_SCHEDULE) | 3 distinct walls | **abandoned, documented** |

Rules that earned their place:

- **Gate every change** — E1's speed was only believable after GSM8K-100
  100/100; the earlier "garbage output" scare was a prompt-format artifact
  (see lessons #5).
- **Warm before probing** — 15.1 s cold vs 0.94 s warm on the same probe.
- **Early-stop** — E3's ceiling (~10–20%) did not justify a third restart
  after two walls; documented the escape hatch instead.
- **Dead levers get write-ups** — the E3 dead-end doc (3 walls + the v11
  path) is a deliverable, not an apology.

## P2 — Re-trace the winning config

The example's structural shift:

| core split | baseline | optimized |
|---|---:|---:|
| COMMUNICATION | 83.1% | **39.0%** |
| compute (all cores) | 16.9% | **61.0%** |

Operator decisions made on the baseline trace would have been aimed at the
wrong profile. **Never carry a trace across a config change** (already an
evidence rule; this is the concrete reason it matters most).

## P3 — Operator layer

The exact order, with the example's numbers:

1. **Library check**: `npu_add_rms_norm_dynamic_quant` existed → 4.97×
   eager, graph-safe. The no-add variant did NOT exist despite appearing in
   `dir()` (lessons #7).
2. **Model-flow check**: V4.1's mHC layers break the `Add→norm` adjacency →
   the validated add-variant could not drop in (lessons #6).
3. **Triton fallback**: chunked row kernel, UB-safe; ±1 step / 95% match /
   2.1× eager (lessons #8 has the UB numbers).
4. **Headroom arithmetic** (`fusion_headroom.py`): 60 KB tensors × ~60
   sites/step in a 33 ms step → **0.3–1.2% end-to-end < wiring cost →
   stopped, kernel shipped as an artifact** for the higher-batch case.

## P4 — Stabilize

- Soak = load/idle rounds with health checks. 5/6 clean; round 6 = external
  container removal (lessons #3: graceful SIGTERM logs are the tell).
- Final full gate on the final config: 197/200, identical to pre-optimization
  baseline — the campaign's completeness proof.

## P5 — Upstream track

The example's drift: pinned base 2026-09-10 vs upstream **v0.30.0rc2**
(2026-10-13, the newest tag — an earlier draft of this page said "v0.31.0",
which does not exist; verify the tag, don't trust the campaign's memory) =
**171 model-relevant commits**, including upstream's proper Triton fusion of
the exact function we had hand-patched, a mixed-quant integration for our
top compute kernel, and a config-only dynamic-spec (our abandoned E3 lever).

`upstream_drift.py` quantifies this in one command. Then the pricing rule:
**run upstream natively with your weights before rebasing** — the measured
gap IS your patches' value; rebase only what the gap proves necessary.

The example's priced verdict (full data:
`examples/deepseek-v4.1-flash-910b/06-onboard/evidence/perf-campaign/P5-UPSTREAM-PRICING.md`):

| | tok/s | accuracy |
|---|---:|---|
| our patched stack | 174.3 | 99/100 |
| upstream doc-faithful | 94.5 | — |
| upstream + our knobs | 96.3 | 99/100 |
| upstream + dynamic K | 90.2 | — |

→ upstream absorbed everything except the draft-graph patch (still **1.81×**
ours); rebase and carry exactly that one patch. Two pricing-session traps
worth naming: the quick hand-rolled accuracy gate scored ~30/100 on BOTH
stacks (protocol artifact — concurrent prefills drift the output format;
the original gate serializes prefill and extracts with fallback), and the
restored service was noisier than the morning anchor (graph-cache state) —
always re-anchor the baseline in the same session you measure the trial.

## P5b — Contribute back (after the rebase verdict)

When pricing proves your asset is real and upstream lacks it, the loop is:
minimal port onto **today's main** → single opt-in switch (default-off is
bit-identical) → strip every diagnostic → re-verify the CLEAN code on
hardware → PR with the re-verified numbers. The worked example:

| | tok/s | accuracy |
|---|---:|---|
| upstream main (eager draft) | 96.3 | 99/100 |
| clean port, one switch (PR code) | **116.9 (+21%)** | **100/100** |
| full internal stack (reference) | 174.3 | 99/100 |

The PR's number is the CLEAN port's number — never cite the internal stack.
Expect upstream's own unit tests to encode the old contract (they are the
behavior spec); update them with the change. The whole contribution
discipline (branch base, DCO, the CI fix loop, the maintainer label gate)
is its own skill: `upstream-pr-contribution`.
