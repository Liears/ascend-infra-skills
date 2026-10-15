#!/usr/bin/env python3
"""
Upstream drift — quantify how far your pinned base has fallen behind, and
which of your local patches the upstream already has.

Rebasing a patched serving stack is expensive; this script prices the
decision BEFORE you spend a day on it:

  1. commits in the drift window (total + perf-relevant subset)
  2. which local patch keywords already exist upstream (overlap check)
  3. the price-the-rebase reminder: run upstream HEAD natively with your
     weights first — the gap to your optimized stack IS your patches' value.

Works against a local clone of upstream (offline, exact) or the GitHub API
(network, no clone needed). Both are read-only.

Usage:
    python3 upstream_drift.py --repo /path/to/vllm-ascend-clone \
        --base-date 2026-09-10 --keywords-file patches.txt
    python3 upstream_drift.py --api vllm-project/vllm-ascend \
        --base-date 2026-09-10 --keywords "engram_host,rope index_select"

keywords-file format: one keyword (or regex) per line; each is grepped in the
upstream tree — a hit means that local patch idea likely landed (or has an
upstream equivalent); a miss means it is still yours to carry.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta

# Substrings that mark a commit as perf-relevant for the drift subset.
PERF_HINTS = ("[Performance]", "[Perf]", "perf(", "Optimi", "[Attention]",
              "[Feature]", "fuse", "Fusion")


def _run(cmd: list[str], cwd: str | None = None) -> str:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=120).stdout


def drift_from_clone(repo: str, base_date: str) -> dict:
    since = f"--since={base_date}"
    log = _run(["git", "-C", repo, "log", since, "--pretty=format:%H%x1f%ad%x1f%s",
                "--date=short"])
    commits = []
    for line in log.splitlines():
        parts = line.split("\x1f")
        if len(parts) == 3:
            commits.append({"sha": parts[0][:9], "date": parts[1], "msg": parts[2]})
    head = _run(["git", "-C", repo, "log", "-1", "--pretty=format:%h %ad %s",
                 "--date=short"]).strip()
    return {"head": head, "commits": commits}


def drift_from_api(repo: str, base_date: str) -> dict:
    until = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
    url = (f"https://api.github.com/repos/{repo}/commits?"
           f"since={base_date}T00:00:00Z&until={until}T00:00:00Z&per_page=100")
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "upstream-drift"})
    with urllib.request.urlopen(req, timeout=30) as r:
        items = json.loads(r.read())
    commits = [{"sha": c["sha"][:9],
                "date": c["commit"]["committer"]["date"][:10],
                "msg": c["commit"]["message"].splitlines()[0]} for c in items]
    return {"head": f"{commits[0]['sha']} {commits[0]['date']} {commits[0]['msg']}"
            if commits else "(none)", "commits": commits}


def check_keywords(repo: str, keywords: list[str]) -> list[dict]:
    out = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        hits = _run(["grep", "-rlE", kw, "--include=*.py", repo]).splitlines()
        out.append({"keyword": kw, "upstream_hits": len(hits),
                    "sample": [os.path.relpath(h, repo) for h in hits[:2]]})
    return out


def render(drift: dict, overlap: list[dict], model_filter: str) -> str:
    commits = drift["commits"]
    perf = [c for c in commits if any(h in c["msg"] for h in PERF_HINTS)]
    model = [c for c in commits if model_filter.lower() in c["msg"].lower()] if model_filter else []

    L = []
    L.append("# Upstream drift report")
    L.append("")
    L.append(f"- upstream head: {drift['head']}")
    L.append(f"- commits in window: **{len(commits)}** "
             f"(perf-relevant {len(perf)}, model-matching '{model_filter or '—'}': {len(model)})")
    L.append("")
    if model:
        L.append(f"## Model-relevant commits ({len(model)}, newest first)")
        for c in model[:15]:
            L.append(f"- {c['date']} `{c['sha']}` {c['msg'][:80]}")
        L.append("")
    L.append("## Local-patch overlap")
    if overlap:
        for o in overlap:
            status = "UPSTREAM HAS IT (or an equivalent)" if o["upstream_hits"] else "still yours to carry"
            L.append(f"- `{o['keyword']}`: {o['upstream_hits']} file hit(s) -> {status}"
                     + (f"  e.g. {o['sample'][0]}" if o["sample"] else ""))
    else:
        L.append("- (no keywords given)")
    L.append("")
    L.append("## Price the rebase BEFORE doing it")
    L.append("Run upstream HEAD natively with your weights and the same probe")
    L.append("口径; the gap to your optimized stack is the proven value of")
    L.append("your local patches. Rebase only what the gap proves necessary.")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Quantify upstream drift and local-patch overlap.")
    ap.add_argument("--repo", help="local clone of upstream (offline mode)")
    ap.add_argument("--api", help="owner/repo for GitHub API mode")
    ap.add_argument("--base-date", required=True, help="your pinned base date, YYYY-MM-DD")
    ap.add_argument("--keywords", default="", help="comma-separated local-patch keywords")
    ap.add_argument("--keywords-file", help="one keyword per line")
    ap.add_argument("--model-filter", default="", help="model name to subset commits (e.g. deepseek)")
    a = ap.parse_args(argv)

    if not a.repo and not a.api:
        ap.error("give --repo (clone) or --api (owner/repo)")
    if a.repo and not os.path.isdir(os.path.join(a.repo, ".git")):
        print(f"not a git repo: {a.repo}", file=sys.stderr)
        return 2

    drift = (drift_from_clone(a.repo, a.base_date) if a.repo
             else drift_from_api(a.api, a.base_date))

    kws = [k for k in a.keywords.split(",") if k.strip()]
    if a.keywords_file and os.path.isfile(a.keywords_file):
        kws += [l.strip() for l in open(a.keywords_file) if l.strip()]
    overlap = check_keywords(a.repo, kws) if (a.repo and kws) else []

    print(render(drift, overlap, a.model_filter))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
