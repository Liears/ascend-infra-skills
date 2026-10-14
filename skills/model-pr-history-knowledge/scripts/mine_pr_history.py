#!/usr/bin/env python3
"""
Mine model-optimization PR history from an Ascend upstream git repo.

Port of the idea behind BBuf's `model-pr-optimization-history`: a per-model
knowledge base built from the framework's own commit history, so that before
patching a model path you can see how it was actually built and what already
changed there.

Difference from the GPU version: the source repos are the Ascend ones
(`vllm-project/vllm-ascend`, `sgl-project/sglang`, the CANN repos), and the
dossier records the CANN/torch_npu context because an Ascend optimization is
only meaningful inside a stack version.

Usage:
    python3 mine_pr_history.py --repo /path/to/vllm-ascend --model deepseek-v4
    python3 mine_pr_history.py --repo /path/to/vllm-ascend --model deepseek-v4 --json
    python3 mine_pr_history.py --repo . --model engram --limit 50 --out out.md

Read-only: it runs `git log`/`git show` and writes nothing to the repo.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict


def git(repo: str, *args: str, timeout: int = 60) -> str:
    try:
        return subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True, text=True, timeout=timeout,
        ).stdout
    except Exception:  # noqa: BLE001
        return ""


@dataclass
class Commit:
    sha: str
    date: str
    subject: str
    pr: str = ""
    kind: str = ""          # perf | bugfix | feat | refactor | test | doc | ci | misc
    files: list = field(default_factory=list)
    added: int = 0
    removed: int = 0

    def models_touched(self) -> list[str]:
        toks: list[str] = []
        for f in self.files:
            low = f.lower()
            for kw in ("deepseek", "engram", "dsa", "mla", "moe", "v41", "v4",
                       "dsv4", "glm", "qwen", "kimi", "minimax"):
                if kw in low and kw not in toks:
                    toks.append(kw)
        return toks


KIND_PATTERNS = [
    ("perf", re.compile(r"\[?performance\]?|\[perf\]|perf\(", re.I)),
    ("bugfix", re.compile(r"bugfix|\[bug\]|fix[:(]", re.I)),
    ("feat", re.compile(r"\[feat|feat[:(]|\[feature\]", re.I)),
    ("refactor", re.compile(r"refactor|\[refactor\]", re.I)),
    ("test", re.compile(r"^\s*\[?test", re.I)),
    ("ci", re.compile(r"\[ci\]|^ci[:(]", re.I)),
    ("doc", re.compile(r"\[doc|docs?[:(]|\[misc\]\[doc\]", re.I)),
]


def classify(subject: str) -> str:
    for kind, rx in KIND_PATTERNS:
        if rx.search(subject):
            return kind
    return "misc"


def mine(repo: str, model: str, limit: int) -> list[Commit]:
    # Search commit subjects and the files they touch.
    log = git(repo, "log", f"--max-count={limit}", "--date=short",
              "--pretty=format:%H%x1f%ad%x1f%s", "--all", "--", "*")
    if not log:
        return []
    commits: list[Commit] = []
    for line in log.splitlines():
        if not line.strip():
            continue
        parts = line.split("\x1f")
        if len(parts) < 3:
            continue
        sha, date, subject = parts[0], parts[1], parts[2]
        # Match on the model keyword in the subject OR in the touched files.
        m = re.search(r"\(#(\d+)\)\s*$", subject)
        pr = m.group(1) if m else ""
        c = Commit(sha=sha[:12], date=date, subject=subject, pr=pr, kind=classify(subject))

        if model.lower() in subject.lower():
            c.files = _files(repo, sha)
            commits.append(c)
            continue
        # Otherwise check the file list (slower; only for candidate commits).
        files = _files(repo, sha)
        if any(model.lower() in f.lower() for f in files):
            c.files = files
            commits.append(c)
        if len(commits) >= limit:
            break
    return commits


def _files(repo: str, sha: str) -> list[str]:
    out = git(repo, "show", "--name-only", "--pretty=format:", sha)
    return [f for f in out.splitlines() if f.strip()]


def render(commits: list[Commit], model: str, repo: str) -> str:
    L: list[str] = []
    L.append(f"# PR history — `{model}` in {os.path.basename(repo.rstrip('/'))}")
    L.append("")
    head = git(repo, "rev-parse", "HEAD").strip()[:12]
    when = git(repo, "log", "-1", "--date=short", "--format=%ad").strip()
    L.append(f"- repo HEAD: `{head}` (last commit {when})")
    L.append(f"- commits matching `{model}`: {len(commits)}")
    L.append("")
    if not commits:
        L.append("No matching commits. Widen the keyword or check the clone depth "
                 "(`--depth 1` clones have no history to mine).")
        return "\n".join(L)

    by_kind: dict[str, int] = {}
    for c in commits:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
    L.append("By kind: " + ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))
    L.append("")

    L.append("## Timeline")
    L.append("")
    L.append("| date | PR | kind | files | subject |")
    L.append("|---|---|---|---:|---|")
    for c in sorted(commits, key=lambda c: c.date, reverse=True):
        files = len(c.files)
        subj = c.subject[:80].replace("|", "\\|")
        L.append(f"| {c.date} | #{c.pr} | {c.kind} | {files} | {subj} |")
    L.append("")

    L.append("## Per-commit files")
    L.append("")
    for c in sorted(commits, key=lambda c: c.date, reverse=True):
        L.append(f"### {c.date} — {c.subject}")
        if c.pr:
            L.append(f"- PR: #{c.pr}")
        L.append(f"- sha: `{c.sha}` | kind: {c.kind}")
        models = c.models_touched()
        if models:
            L.append(f"- model keywords: {', '.join(models)}")
        for f in c.files[:20]:
            L.append(f"    - `{f}`")
        if len(c.files) > 20:
            L.append(f"    - ... and {len(c.files)-20} more")
        L.append("")

    L.append("---")
    L.append("Mined from git history. A commit's subject is a claim; the diff is the")
    L.append("fact. Read the diff before treating anything here as current behavior,")
    L.append("and record the repo HEAD this dossier was built from.")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mine model PR history from an Ascend upstream repo.")
    ap.add_argument("--repo", required=True, help="path to a git clone (must have history)")
    ap.add_argument("--model", required=True, help="keyword, e.g. deepseek-v4 / engram / dsa")
    ap.add_argument("--limit", type=int, default=40, help="max commits to scan")
    ap.add_argument("--out", help="write markdown here instead of stdout")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    commits = mine(args.repo, args.model, args.limit)
    if args.json:
        text = json.dumps({"repo": args.repo, "model": args.model,
                           "commits": [asdict(c) for c in commits]},
                          indent=2, ensure_ascii=False)
    else:
        text = render(commits, args.model, args.repo)

    if args.out:
        # Create the parent dir: dossiers conventionally land in references/dossiers/
        # which may not exist yet in a fresh checkout.
        parent = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(parent, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out} ({len(commits)} commits)", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
