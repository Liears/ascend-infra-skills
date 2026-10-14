# Querying a Dossier

How to search a mined dossier and decide when it needs re-mining.

---

## 1. Search patterns that work

The dossier is markdown with a timeline table and per-commit file lists, so
ordinary text search is enough:

```bash
# every commit that touched one file
grep -n "attention/dsa_v1.py" dossiers/vllm-ascend_deepseek.md

# all perf commits
grep -n "| perf |" dossiers/vllm-ascend_deepseek.md

# commits mentioning a subsystem
grep -n -i "engram" dossiers/vllm-ascend_deepseek.md

# the rebuild anchor: which HEAD this dossier came from
grep -n "repo HEAD" dossiers/vllm-ascend_deepseek.md
```

For structured queries, mine with `--json` and filter with `jq`:

```bash
python3 scripts/mine_pr_history.py --repo /path/to/vllm-ascend --model deepseek --json \
  | jq '.commits[] | select(.kind=="perf") | {date, pr, files}'
```

---

## 2. What a search result means

| You find | It means |
|---|---|
| 5 perf commits on one file | that file is the hot spot; read all five before optimizing it |
| a bugfix on a path you are about to change | read the fix; the bug is likely re-introducible |
| a rename/refactor over model metadata | any tutorial older than that date uses the old names |
| a CI commit adding a golden | an accuracy baseline exists; reuse it rather than inventing one |
| the model keyword only in a doc commit | not a code change; the `files` column shows it is a `.md` |

---

## 3. False positives

Keyword matching over commit history catches noise. Three common cases:

1. **Wide doc sweeps.** A commit like "Update Public FAQs links to relative
   paths" edits every model's tutorial page, so it matches every model keyword
   while changing no code. Filter by file extension: if every touched file is
   `.md`, it is documentation.

2. **CI/baseline edits.** A CI commit touching one `.yaml` under
   `tests/e2e/nightly/` is a test-config change, not a model change.

3. **Cross-model refactors.** A config refactor touching 30 files matches the
   keyword through one line. Read the file list to judge.

The miner reports the file count and the model keywords precisely so you can
apply these filters by eye.

---

## 4. When to re-mine

A dossier is a snapshot. Re-mine when:

- the upstream repo's HEAD has moved past the recorded HEAD by more than a few
  weeks, **and** you are about to cite the dossier for a decision;
- you switch the repo you are targeting (vllm-ascend vs sglang give different
  histories for the same model);
- a tutorial or a plan references the dossier and the model has since gained a
  new major capability.

Re-mining is cheap (a `git log` walk). Citing a stale dossier as current is not.

---

## 5. The rebuild anchor

Every dossier records the repo HEAD and last-commit date it was built from.
Always cite both:

> "per vllm-ascend `2b8da69474a7` (2026-10-14), DeepSeek V4 gained multimodal
> support in #18281"

A claim without the anchor cannot be checked later, and a reader cannot tell
whether the history it describes still applies.
