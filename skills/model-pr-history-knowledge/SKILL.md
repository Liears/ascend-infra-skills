---
name: model-pr-history-knowledge
description: A queryable knowledge base of how a model was built and optimized in the Ascend upstream repos (vllm-ascend, sglang, CANN). Use before changing a model path on NPU, to see which PRs already touched it, what kind of change each was, and which files they hit — so you do not call an optimization new or re-break a fixed bug. This is a knowledge source, not a task skill.
---

# Model PR History Knowledge

Before patching a model's Ascend path, read its history. Most "new" optimizations
have been tried in the upstream repo, and most regressions are a known fix being
inverted.

## What this contains

Per-model dossiers mined from an upstream repo's own git history:

- every commit that touched the model's files, with date and PR number
- a kind classification (perf / bugfix / feat / refactor / test / doc / ci)
- the exact files each commit changed
- the repo HEAD the dossier was built from

The dossier is **dated history**, not a statement about today's branch. A commit
tells you what upstream did; whether your branch has it is a separate question.

## Workflow

```bash
# Mine a dossier from a clone that has history (not a --depth 1 clone)
python3 scripts/mine_pr_history.py --repo /path/to/vllm-ascend --model deepseek-v4
python3 scripts/mine_pr_history.py --repo /path/to/vllm-ascend --model engram --limit 60
python3 scripts/mine_pr_history.py --repo /path/to/sglang --model deepseek --out dossiers/dsv4-sglang.md
```

Read `references/dossiers/` for the pre-built ones, or mine fresh.

## Which repos to mine

| Repo | What its history tells you |
|---|---|
| `vllm-project/vllm-ascend` | the NPU-specific model work: attention backends, MoE dispatch, fused ops, quant runtime |
| `sgl-project/sglang` | the reference model support and the optimization campaigns (filter to ascend-relevant commits) |
| the CANN / cannbot repos | operator and runtime-level changes |

A model's full history is the union: **what upstream vLLM does**, **what
vllm-ascend adds**, and **what the model repo itself ships**. Mine all three
before concluding a technique is new.

## Reading a dossier

| Signal | Meaning |
|---|---|
| many `perf` commits on one file | that file is where the NPU cost lives |
| a `bugfix` you are about to re-introduce | read it — the bug is probably subtle |
| a `refactor` renaming model metadata | the naming may have changed since any tutorial you read |
| `ci` commits adding a golden | there is an accuracy baseline you can reuse |
| commits clustered on one date | a burst from one integration effort; the files show its shape |

## Honesty rules for this knowledge base

1. **A commit subject is a claim; the diff is the fact.** Mine the subject for
   orientation, then read the diff before relying on it.
2. **Record the repo HEAD.** A dossier without the HEAD it was built from cannot
   be checked for staleness.
3. **A `--depth 1` clone has no history.** If the miner returns nothing, check
   the clone depth before concluding there is nothing to find.
4. **Keyword matching has false positives.** A wide-reaching doc commit can
   match a model keyword because it edited the model's doc page. The `files`
   column is what distinguishes a real code change from a doc edit.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/mine_pr_history.py` | Mine a dossier from an upstream clone |
| `scripts/mine_pr_history.py --json` | Machine-readable form |

## References

- `references/dossiers/` — pre-built dossiers (see the README there)
- `references/querying.md` — how to search a dossier and when to re-mine
- `../../docs/EVIDENCE-RULES.md` — the dating and sourcing rules this obeys
