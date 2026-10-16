#!/usr/bin/env python3
"""Pre-PR check: find existing upstream unit tests for the files you touched.

The tests under tests/ut/ are the behavior contract of the modules they
exercise. Changing behavior without updating them turns cpu-ut red. This
script maps touched source files to their test files BEFORE you open the PR.

Usage:
    check_touch.py <repo-root> vllm_ascend/spec_decode/dspark_proposer.py ...
    check_touch.py <repo-root> --diff upstream/main   # touch = files changed vs ref
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path


def touched_files(root: Path, ref: str | None, args: list[str]) -> list[Path]:
    if ref:
        out = subprocess.run(
            ["git", "diff", "--name-only", ref, "--", "vllm_ascend/"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout
        return [root / p for p in out.splitlines() if p.strip()]
    return [Path(a) for a in args]


def module_tokens(path: Path) -> set[str]:
    """Stem and hyphen-separated parts of a source path, e.g. dspark_proposer."""
    parts = set(path.stem.split("_"))
    parts.add(path.stem)
    return {p for p in parts if len(p) >= 5}


TEST_HINT_RE = re.compile(r"^test_(.+)\.py$")


def find_related_tests(root: Path, src: Path) -> list[Path]:
    tokens = module_tokens(src)
    hits: list[Path] = []
    ut = root / "tests" / "ut"
    if not ut.is_dir():
        return hits
    for tf in sorted(ut.rglob("test_*.py")):
        m = TEST_HINT_RE.match(tf.name)
        if not m:
            continue
        test_tokens = set(m.group(1).split("_"))
        test_tokens |= module_tokens(tf)
        if tokens & {t for t in test_tokens if len(t) >= 5}:
            hits.append(tf)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="upstream repo checkout root")
    ap.add_argument("files", nargs="*", help="touched source files (relative or absolute)")
    ap.add_argument("--diff", metavar="REF", help="derive touched files from git diff REF -- vllm_ascend/")
    ns = ap.parse_args()

    root = ns.root.resolve()
    files = touched_files(root, ns.diff, ns.files)
    if not files:
        print("no touched files given", file=sys.stderr)
        return 2

    any_test = False
    for f in files:
        p = f if f.is_absolute() else root / f
        rel = p.resolve().relative_to(root)
        tests = find_related_tests(root, p)
        print(f"\n== {rel}")
        if tests:
            any_test = True
            for t in tests:
                print(f"   contract: {t.relative_to(root)}")
        else:
            print("   (no existing unit tests found for this module)")

    print(
        "\nPR checklist:"
        "\n  [ ] base branch = today's upstream main (not a release tag)"
        "\n  [ ] single squashed commit, Signed-off-by present (git commit --amend -s)"
        "\n  [ ] opt-in switch, default-off keeps upstream behavior bit-identical"
        "\n  [ ] diagnostics stripped, comments in English explaining root causes"
        "\n  [ ] py_compile + pyflakes + AST function inventory vs main"
        + ("\n  [ ] update the behavior tests listed above to the new contract" if any_test else "")
        + "\n  [ ] local lint pass, then apply CI-reported format patches verbatim"
        "\n  [ ] re-verify the CLEAN code on hardware; PR cites those numbers"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
