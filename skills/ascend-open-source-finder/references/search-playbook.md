# Search Playbook

How to construct queries and how to separate a real Ascend port from a README
claim. Written from what actually worked when searching for DeepSeek-V4.1-Flash
on 910B (2026-10-14).

---

## 1. The name has many spellings — try all of them

A single spelling finds a fraction of what exists. The search space:

| Axis | Variants |
|---|---|
| Version separator | `V4.1`, `V4_1`, `V4-1`, `V41` |
| Case | `DeepSeek` / `deepseek`, `Flash` / `flash` |
| Prefix | with `DeepSeek-` and without |
| Hardware | `910B`, `910B3`, `Ascend910B`, `ascend910b` |
| Server SKU | `A2`, `Atlas 800 A2`, `A3`, `Atlas 800 A3`, `950`, `A5` |
| Stack | `vllm-ascend`, `vllm_ascend`, `mindie`, `torchair` |

The finder generates these automatically. When doing it by hand, a query like
`DeepSeek-V4-1-Flash 910B` and `deepseek-v4.1-flash 910B` return **the same
repo**, while `DeepSeekV41Flash 910B` returns **nothing** — because the owner
named the repo with a hyphen. Missing one spelling can miss the only hit.

> Concrete: `deepseek-v4.1-flash 910B` → 1 result (the real port).
> `DeepSeekV41Flash 910B` → 0 results. Same repo, different query.

---

## 2. What makes a repo a real port, ranked by strength of evidence

Read the tree, not the description. In descending order of trust:

1. **A patch set against the upstream framework.** Files like
   `patches/vllm-ascend/0001-perf-*.patch` mean someone had to modify the
   framework to make the model run. That is the hard part, and it is present.
2. **A self-check or acceptance document that states a result with a date.**
   `EXPECTED_PERF.md`, `CORRECTNESS_STATUS.md`, `docs/*ACCEPTANCE*.md`. A doc
   with no date is weaker.
3. **A REPRO.md with the exact commands.** Reproducibility stated, not implied.
4. **A quant recipe** (`quant/`, `*best_practice.yaml`, a ModelSlim patch). On
   Ascend the quantized format is frequently the only one that fits, so a
   published recipe is a real contribution.
5. **Entry-point scripts** (`serve_*.sh`, `build_image.sh`). Useful, but a
   launch script alone can be copied from a generic tutorial.
6. **A weight page on ModelScope/HF.** Proves the weights exist, proves nothing
   about the code.

**The strongest single signal is #1.** A repo with 30 patches and no README
beats a repo with a beautiful README and no patches.

---

## 3. How to tell a claim from an implementation

| Looks like | Actually is | Why |
|---|---|---|
| "Supports Ascend NPU" in the description, tree has only `README.md` | a claim | nothing to run |
| "Based on vLLM Ascend" + `private` in the name | a fork, possibly empty | private forks are often just mirrors |
| A launch script calling `vllm serve` with no patch dir | a recipe, not a port | works only if upstream already supports the model |
| `REPRO.md` + patches + acceptance doc | an implementation | the artifacts a run leaves behind |

**The `private` marker is a red flag, not a bonus.** Repos named
`vllm-ascend-v41-private` usually hold no public work.

---

## 4. Hardware-name traps

- **`A2` = 8×910B3 (8 devices). `A3` = 8×910C (16 devices).** A repo saying "8
  NPU" on A3 means 4 physical cards. When a candidate states parallelism,
  check the unit before believing its capacity numbers.
- **`910B` vs `910B3` vs `910B4`.** Different HBM and core counts. A port for
  one is not automatically good for another.
- **`950` / `A5` is a different generation** (MXFP8/MXFP4 native). A V4-Flash
  port written for `950DT` may need no Ascend-specific quantization, whereas the
  `910B` port needs a W4A8/W8A8 recipe. The right repo depends on the chip.

---

## 5. Sources and their real behaviour

| Source | Behaviour observed on this machine |
|---|---|
| GitHub search API | works unauthenticated; finds personal ports reliably; rate-limits after ~10 rapid queries |
| GitCode search API | requires a `private-token` header; returns 400 without it |
| ModelScope JSON API | `GET /api/v1/models/{owner}/{name}` → 200 exists / 404 missing. The **HTML page returns 200 even for missing models** (SPA shell) — never judge existence by the page |
| ModelScope search | `/api/v1/dolphin/models` and `/api/v1/models?...` both 404 — no public search endpoint. Probe by name convention instead |
| HuggingFace | unreachable from this host without a proxy; an empty result is a network fact |

---

## 6. After you find it

Do not stop at the link. The next three questions decide whether it helps:

1. **Which upstream commit is the patch set based on?** A patch against a
   six-month-old `vllm-ascend` will not apply to today's.
2. **Which CANN/driver does its acceptance record belong to?** Numbers from
   CANN 9.1.0 do not carry to 9.2.0.
3. **Does its self-check actually pass on your machine?** Everything above is
   reading. Running is the only proof.

Hand these to `ascend-repo-audit` (`scripts/audit_repo.py`) — it answers 1 and
2 structurally and flags the ambiguity in 2 explicitly.
