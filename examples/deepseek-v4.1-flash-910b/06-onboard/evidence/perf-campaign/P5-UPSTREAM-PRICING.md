# P5 — Upstream pricing: v0.30.0rc2 native vs our patched stack (2026-10-15)

**Question**: 5 weeks of upstream drift (171 model-relevant commits between our
base 2026-09-10/dev5097 and v0.30.0rc2@8fde6d4) — how much of our patch stack's
value has upstream absorbed, and what does upstream give natively on
single-node 8×910B3 (a config upstream does not publish: their A2 recipe is
4-node W8A8, "retained unchanged, not revalidated")?

## Pre-registered protocol (written before any upstream number exists)

- **Our stack, re-anchored today** (fresh, same session):
  `decode_probe.py http://127.0.0.1:8100 longgen-prompt.json 512`,
  5 warmup discarded + 3 measured, median. Result: **174.3 tok/s** (172.2 /
  174.3 / 175.0), A=5.82, TTFT ≈ 1.25 s. Matches the recorded E2 final (~172).
- **Upstream runs**: identical probe against port 8200, identical warmup
  discipline, identical weights (/mnt/local/hbw-work/dsv41-w4a8, W4A8+Engram),
  identical serving params searched for THIS workload (TP8+EP8, MAX_SEQS 4,
  BAT 8192, UTIL 0.92, LEN 1M, no prefix caching, dspark S=5).
  Only the software differs (image + flags it supports).

| run | software | intent |
|---|---|---|
| U1 | v0.30.0rc2, A2 documented recipe (npugraph_ex/static off, no engram override, draft eager) | "what the docs give you" |
| U2 | v0.30.0rc2 + our flag-portable knobs (npugraph_ex/static on, fused_mc2, engram cpu_offload) | "our config minus our patches" |
| U3 (if U1/U2 healthy) | U2 + K=7 + `dynamic_spec_config:{method:dspark}` | E3's lever, now upstream-native |
| ours | dsv41-a2:v10 (dev5097 + 12 patches, DRAFT_GRAPH=1) | the priced asset |

**Decision rule**: gap(ours − best upstream) = the value our patches add on
top of upstream-today. If upstream-native ≥ ours → rebase immediately, drop
patches. If upstream-native < ours by a large factor → carry the DRAFT_GRAPH
patch forward; price the rebase as "port 3 files, keep everything else upstream".

## Known from source inspection (before running)

- Upstream dspark is **eager-only by construction**: `dspark_proposer.py:73
  self.use_cuda_graph = False` ("Ascend cudagraph unsupported on this path").
  Our E1 patch removes exactly this hard-disable. → expect the biggest gap here.
- Upstream HAS absorbed: mc2 fusion family, multistream overlaps, dsa_cp,
  flashcomm, static kernel, engram (new `--engram-config` iface, int8 storage
  default), Triton `dspark_swa_indices` (the function we hand-patched),
  `fuse_norm_quant` compile pass (default on), and its own
  `dynamic_spec_config` (the E3 lever, config-only).
- Upstream still does NOT have: DRAFT_GRAPH (deepseek dspark), our admission
  gate, the V4.1 frontend encoding fix (tbc at runtime).
- The norm-quant fusion patterns are all Add-prefixed (except an MX variant)
  → mHC conclusion (OP1) still holds upstream: no drop-in site on V4.1.
- Earlier docs said "v0.31.0" — wrong; the newest tag is **v0.30.0rc2**
  (2026-10-13). Corrected here.

## Results

| run | tok/s (median of 3) | A | TTFT | GSM8K-100, original gate* | notes |
|---|---|---|---|---|---|
| ours (today, re-anchored) | **174.3** | 5.82 | 1.25 s | **99/100** | dev5097+12 patches, DRAFT_GRAPH=1 |
| U1 doc-faithful | **94.5** (94.6/94.5/93.7) | 5.82 | 1.16 s | — (see reading 5) | draft eager, npugraph/static off |
| U2 + our knobs | **96.3** (96.4/96.2/96.3) | 5.82 | 1.16 s | **99/100** | +npugraph_ex/static/fused_mc2/engram offload = only +1.9% |
| U3 U2+K=10+dynamic spec | **90.2** (92.5/90.2/89.1) | ~5.65 | 1.16 s | — | **−6.3% vs U2** |

\* original gate = port-repo `acc_eval.py` (official encoder via
`/v1/completions`, serialized prefill, effort 75, conc 4, numeric-fallback
extraction). Both engines also ran a quick hand-rolled 8-shot chat gate that
scored ~30/100 on BOTH stacks — that protocol is an artifact (concurrent
prefills flip batch-shape numerics and drift the answer format; the original
gate serializes prefill and extracts with fallback). Recorded as a lesson:
never compare accuracies across gate protocols.

Readings:

1. **Upstream closed most of the OLD gap by itself**: identical probe on the
   dev5097 base was 27.2 tok/s (E1's baseline); upstream-today runs the same
   eager-draft dspark at 94.5 — the Triton `dspark_swa_indices` fusion (of
   the function we had hand-patched) plus 171 commits did that, with ZERO
   patches and our W4A8 checkpoint loading unmodified.
2. **Our remaining edge is almost entirely DRAFT_GRAPH**: 174.3 vs 96.3 =
   **1.81×**. Our flag-portable knobs on upstream code add only +1.9% (they
   were largely absorbed, or target the already-graphed decode path).
3. **E3 (dynamic K) confirmed dead, twice over**: upstream-native
   dynamic_spec (K=10, budget-adaptive) LOSES 6.3% here — acceptance is
   already at the K=5 ceiling (A=5.82/6) on predictable continuation, so K
   headroom only buys draft cost. Also: K must be a multiple of n_predict=5
   (K=7 is architecturally impossible on this draft head).
4. **Accuracy: no difference** — upstream-today serves our W4A8 checkpoint
   at 99/100 on the original gate, same as our patched stack.
5. The earlier in-session scare ("upstream loses the answer format under
   concurrency, 27-32/100") was a protocol artifact of the quick gate, not
   an upstream regression — our own stack scored 31/100 under that same
   quick gate. The accuracy-gate skill's rule (refuse protocol-mismatched
   comparisons) applies even when you wrote both sides of the comparison
   yourself.

## Verdict

**Rebase: yes — onto v0.30.0rc2, carrying exactly one asset.**

- Upstream has absorbed: the hand-patched function (now a proper Triton op),
  our flag surface (mc2/overlap/static-kernel/engram incl. int8 + host
  offload), the E3 lever (dynamic spec, config-only), and serves our W4A8
  checkpoint natively at equal accuracy.
- Still ours alone: **DRAFT_GRAPH** (`patches/draft/`: dsa_v1.py,
  dspark_proposer.py, llm_base_proposer.py — removes upstream's
  `use_cuda_graph = False` hard-disable for deepseek dspark). Measured value
  on top of upstream-today: **1.81× single-stream** (96.3 → 174.3 tok/s).
- Rebase cost: port 3 files onto v0.30.0rc2 + rerun the gates. Everything
  else (11 of 12 patches) is dropped, not ported.
- Contribute back: the draft-graph port is a clean upstream PR candidate
  (upstream already graphs the GLM dspark draft — the deepseek path is the
  missing one).

## Verdict

(to be written with the numbers)
