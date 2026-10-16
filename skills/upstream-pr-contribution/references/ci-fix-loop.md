# CI Fix Loop — the 8-round campaign, distilled

Every failure below came from one real PR (vllm-ascend #18967, DSpark draft
ACL-graph capture). Local checks passed before each push; the delta between
local tooling and CI tooling is the whole game.

| round | CI failure | root cause | fix pattern |
|---|---|---|---|
| 1 | ruff E402 ×3 | new module-level env definition placed before the import block | move the definition after ALL imports |
| 2 | ruff E501 + SIM105 | 121-char line; try/except/pass | wrap/`contextlib.suppress` |
| 3 | ruff format | one extra blank line after deleting a block | **apply the CI log patch verbatim** — local ruff 0.14.0 formatted differently from CI's "0.14.0" |
| 4 | mypy override ×2 | base-class signature gained a param; untouched subclasses now incompatible | add the defaulted param to every override (mechanical, expected) |
| 5 | DCO action_required | `git commit --amend -m` rewrote the message WITHOUT the sign-off | always `--amend -s --no-edit` |
| 6 | cpu-ut 5 failed | upstream test_dspark_proposer.py encodes the OLD executor-submit contract | rewrite tests to the new sync-run contract; mock must subclass `DeviceMetadataTaskProvider` for the isinstance gate |
| 7 | cpu-ut 6 failed | another test builds the proposer via `__new__`; new instance attribute missing | `getattr(self, "_draft_target_positions", None)` in the code — robust to partially-constructed instances |
| 8 | cpu-ut 6 failed | test's `check_metadata(**kw)` mock lacks the new kwarg | add `batch_descriptor=None` to the mock signature |

Then: pre-commit green, cpu-ut 11186/11186 green — and ci-gate STILL red,
with the annotation *"a maintainer adds exactly one label: ready-precise /
ready-all / main2main"*. That is the designed endpoint for outside
contributors, not a failure.

## Timing guidance

- pre-commit: ~1 min. cpu-ut: ~4 min per matrix (two matrices in parallel).
  Budget ~10 min per push→verdict cycle; do local passes first to keep the
  loop under 3 rounds.
- Do not push again while checks are green-but-gated; the label is the
  maintainer's move.
