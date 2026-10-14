---
name: ascend-open-source-finder
description: Find and rate existing open-source implementations of a model on Ascend hardware before writing anything from scratch. Use when the user asks "has anyone already done X on Ascend", "is there an open-source version of this", names a model plus a chip (910B/A2/910C/A3/950), or is about to port a model and wants prior art. Produces a ranked, evidence-rated candidate list across GitHub, GitCode, ModelScope and HuggingFace.
---

# Ascend Open-Source Finder

Answer "does an open implementation of this model already exist on Ascend?"
**before** the work of writing one starts. The output is a ranked list of
candidates, each rated by how much real evidence its repo carries.

## Why this exists

Porting a model to Ascend is expensive. A large fraction of the time the port
already exists — in a personal repo, a ModelScope weight page, or a deployment
script someone published after doing the work. Searching for it is the single
highest-leverage first step, and it is the step people skip.

## Workflow

1. **Take the pair as input.** A model name and a hardware target. If the user
   gives only one, ask for the other; "DeepSeek-V4.1-Flash" alone is not
   searchable — the interesting repos all say which chip they target.

2. **Run the finder.**
   ```bash
   python3 scripts/find_open_source.py --model "DeepSeek-V4.1-Flash" --hardware 910B
   ```
   It issues 8–12 query variants (the model name has many spellings: `V4.1`,
   `V4_1`, `V41`, `V4-1`; the hardware has many aliases: `910B`, `910B3`, `A2`,
   `Atlas 800 A2`) across the configured sources.

3. **Read the ratings, do not trust the descriptions.** Every candidate is rated
   `verified` / `partial` / `claimed` from metadata signals (patch files, REPRO,
   acceptance docs, evidence dirs, launch scripts, quant material). A repo that
   only *says* "supports Ascend" in its description is `claimed`, however
   confident the prose sounds.

4. **Audit the top candidate before citing it.**
   ```bash
   git clone --depth 1 <url> /tmp/cand
   python3 scripts/audit_repo.py /tmp/cand
   ```
   This reads the tree and reports patches-by-upstream, entry-point scripts,
   evidence docs, external weights, toolchain pins, tests, and red flags. The
   finder says *which* repos to read; the audit says *what is in one*.

5. **Report with the evidence block** defined in `docs/EVIDENCE-RULES.md`.

## What the rating means

| Rating | Meaning | What you may say |
|---|---|---|
| `verified` | Metadata shows implementation evidence (patches / acceptance docs / repro) | "this repo ships a port; read it before writing your own" |
| `partial` | Some signals, not a full evidence set | "worth reading; confirm what it actually runs" |
| `claimed` | Description asserts support, metadata shows no artifacts | "a lead only; do not cite as an implementation" |

**`verified` is not a claim that the code runs.** It means the repo carries the
artifacts a run would leave behind. Only executing its own self-check closes
that gap, and this skill never does that for you.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/find_open_source.py` | Multi-source search + metadata evidence rating |
| `scripts/audit_repo.py` | Read-only tree audit of a clone → evidence card |

## Sources and their limits

| Source | Needs | Limit |
|---|---|---|
| GitHub | nothing (rate-limited); `GITHUB_TOKEN` to raise it | search API caps total results; sort by stars |
| GitCode | `GITCODE_TOKEN` | silently skipped without a token |
| ModelScope | nothing | no working search endpoint; probed by name convention under owners already discovered |
| HuggingFace | network reachability | **often unreachable from CN hosts without a proxy** — a zero here is not evidence of absence |

A source that returns nothing is reported as "not queried" or "unreachable",
never as "nothing exists". The distinction matters: on a machine behind a
firewall, an empty HuggingFace result is a network fact, not an existence fact.

## References

- `references/search-playbook.md` — query construction, alias tables, how to tell a real port from a README claim
- `../../docs/EVIDENCE-RULES.md` — the evidence and reporting rules this skill obeys

## Worked example

`DeepSeek-V4.1-Flash` on `910B` returns, as the top hit:

```
## [verified] chiro2001/deepseek-v4.1-flash-ascend910B  (score 6)
- url: https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B
- source: github | lang: Python | stars: 24 | license: Apache-2.0
- evidence signals: acceptance/perf doc, quantization material
```

Auditing the clone reports 33 patches (12 against `vllm_ascend`, 4 against
`vllm`), a REPRO.md, an EXPECTED_PERF.md, entry-point scripts at
`a2/scripts/serve_a2_offload.sh`, ModelScope weights, and a red flag that the
toolchain is mentioned in more than one version — so the next step is to pin
which CANN baseline the numbers belong to, not to trust any single number.

That is the whole point: the skill converts "I wonder if someone did this"
into a specific repo, a specific patch list, and a specific open question.
