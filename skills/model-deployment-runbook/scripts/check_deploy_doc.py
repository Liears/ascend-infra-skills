#!/usr/bin/env python3
"""Deployment-runbook linter: catch machine-specific hardcoding and missing structure.

A deploy doc is written for SOMEONE ELSE'S machine. This checks the failure
classes that actually forced rewrites in practice:
  1. hardcoded local paths / image names / container names / IPs (127.0.0.1 ok)
  2. missing sections or unverifiable claims (TBD leftovers, bare numbers)
  3. a PR named without the command that gets its code (reader cannot act on
     "use PR #1234" when the PR is not merged)
  4. an image reference with no exact tag / digest (reader cannot reproduce)

Usage: check_deploy_doc.py <doc.md>
Exit 0 = PASS, 1 = FAIL (findings listed with line numbers).
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

REQUIRED_SECTIONS = [
    "权重",      # weights acquisition (or EN equivalent in the doc's language)
    "校验|sha256",
    "镜像|image|Image",
    "docker run|容器",
    "vllm serve|起服",
    "验收|门禁|gate",
]

HARDCODE_PATTERNS = [
    (re.compile(r"/mnt/(?:local|public|project)/\S+"), "machine-specific /mnt path"),
    (re.compile(r"/home/(?!user)\S+"), "machine-specific /home path"),
    (re.compile(r"/root/(?!npugraph_ex_cache)[^\s`)\]|\"]*"), "machine-specific /root path"),
    (re.compile(r"--name\s+(dsv41|upstream|deepseek)[\w-]*"), "local container name (use <NAME>)"),
    (re.compile(r"\b(?!127\.0\.0\.1\b|0\.0\.0\.0\b)\d{1,3}(?:\.\d{1,3}){3}\b"), "IP address (only 127.0.0.1/0.0.0.0 allowed)"),
    (re.compile(r"\bdocker\s+commit\s+\S+\s+(?!<)[\w:.-]+"), "concrete image tag from local commit (use <IMAGE_TAG>)"),
]

PLACEHOLDER_HINT = re.compile(r"<[A-Z_]+>")

# A PR reference only counts as actionable if the doc also says HOW to get the
# code (cherry-pick / fetch / git am / rebase), or states it is already merged.
PR_REF = re.compile(r"\bPR\s*#?\d+|\bpull/\d+\b|/pull/\d+\b")
PR_ACTION = re.compile(r"cherry-pick|git\s+am\b|git\s+fetch|pull/\d+/head|已合入|already merged|merged in")

# Image mentions must carry an exact tag or a digest. `repo:latest` and bare
# `repo` are not reproducible: the reader cannot tell what they will get.
IMAGE_REF = re.compile(
    r"\b((?:quay\.io|quay\.nju\.edu\.cn|quay\.dockerproxy\.net|docker\.io|swr\.[\w.]+)"
    r"/[\w./-]+)(:[\w.-]+)?(@sha256:[0-9a-f]{64})?"
)
FLOATING_TAGS = {"latest", "main", "master", "dev", "stable", "nightly"}


def check(doc: str) -> list[tuple[int, str, str]]:
    findings: list[tuple[int, str, str]] = []
    lines = doc.splitlines()

    has_pr_action = bool(PR_ACTION.search(doc))
    for i, line in enumerate(lines, 1):
        if PR_REF.search(line) and not PR_ACTION.search(line) and not has_pr_action:
            findings.append((i, "PR referenced but no way to obtain its code "
                                "(add cherry-pick/fetch command, or state it is merged)",
                             line.strip()[:70]))
        for m in IMAGE_REF.finditer(line):
            tag, digest = m.group(2), m.group(3)
            if digest:
                continue
            if not tag:
                findings.append((i, "image reference without a tag or digest "
                                    "(pin the exact version)", m.group(0)[:60]))
            elif tag.lstrip(":") in FLOATING_TAGS:
                findings.append((i, "floating image tag (use an exact version or digest)",
                                 m.group(0)[:60]))

    for i, line in enumerate(lines, 1):
        # inline code spans are the legitimate place for literal commands; but a
        # local PATH inside them is still local — check everything.
        for pat, msg in HARDCODE_PATTERNS:
            for m in pat.finditer(line):
                # allow the canonical port-repo example /home/user (upstream docs use it)
                if msg.startswith("machine-specific /home") and m.group(0).startswith("/home/user"):
                    continue
                # allow container-internal mount points that the template itself defines
                if "npugraph_ex_cache" in m.group(0) or m.group(0).startswith("/root/.cache"):
                    continue
                findings.append((i, msg, m.group(0)[:60]))
        if re.search(r"\bTBD\b", line) and "未验证" not in line and "unverified" not in line.lower():
            findings.append((i, "unresolved TBD (mark 未验证/未测 or resolve it)", line.strip()[:60]))

    # missing required sections
    for sec in REQUIRED_SECTIONS:
        if not re.search(sec, doc):
            findings.append((0, f"missing required section (pattern: {sec})", ""))

    # placeholders sanity: a deploy doc should actually USE placeholders
    if not PLACEHOLDER_HINT.search(doc):
        findings.append((0, "no <PLACEHOLDER> found — doc likely hardcodes machine specifics", ""))
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("doc", type=Path)
    ns = ap.parse_args()
    findings = check(ns.doc.read_text(encoding="utf-8"))
    if not findings:
        print("PASS — no hardcoding, all sections present")
        return 0
    print(f"FAIL — {len(findings)} finding(s):")
    for ln, msg, snippet in findings:
        loc = f"line {ln}" if ln else "document"
        print(f"  [{loc}] {msg}" + (f" :: {snippet}" if snippet else ""))
    return 1


if __name__ == "__main__":
    sys.exit(main())
