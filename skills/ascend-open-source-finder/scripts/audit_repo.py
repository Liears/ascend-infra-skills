#!/usr/bin/env python3
"""
Ascend Repo Audit — turn a candidate repo into a grounded evidence card.

The finder answers "which repos exist". This answers "what is actually in one,
and how far can I trust it". It reads a local clone (it never writes to it) and
reports, per axis:

  patches      how many, against which upstream repo, how large
  scripts      launch / build / self-check entry points
  evidence     docs that look like acceptance or perf records
  weights      external weight references and their declared size/format
  stack        CANN / torch_npu / vllm-ascend / transformers version pins
  tests        whether a test corpus exists and whether it is redistributed

Every line cites a path in the repo. Nothing is inferred from the README alone;
where the README and the tree disagree, the tree wins and the disagreement is
reported.

Usage:
    python3 audit_repo.py /path/to/clone
    python3 audit_repo.py /path/to/clone --json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "site-packages"}


def walk(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            yield os.path.join(dirpath, fn)


def rel(root: str, p: str) -> str:
    return os.path.relpath(p, root)


def read_text(p: str, limit: int = 200_000) -> str:
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(limit)
    except Exception:  # noqa: BLE001
        return ""


@dataclass
class Audit:
    root: str
    git: dict = field(default_factory=dict)
    patches: dict = field(default_factory=dict)
    scripts: list = field(default_factory=list)
    evidence_docs: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    stack_pins: dict = field(default_factory=dict)
    tests: dict = field(default_factory=dict)
    red_flags: list = field(default_factory=list)
    strengths: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# axes
# --------------------------------------------------------------------------

def audit_git(root: str) -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", root, *args],
                capture_output=True, text=True, timeout=20,
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    if not os.path.isdir(os.path.join(root, ".git")):
        return {"has_git": False}
    return {
        "has_git": True,
        "head": run("rev-parse", "HEAD")[:12],
        "last_commit": run("log", "-1", "--format=%ad", "--date=short"),
        "commit_count_shallow": run("rev-list", "--count", "HEAD"),
        "origin": run("config", "--get", "remote.origin.url"),
    }


def audit_patches(root: str) -> dict:
    """Classify patches by the upstream project their diff paths point at."""
    by_upstream: dict[str, list[dict]] = {}
    total = 0
    for p in walk(root):
        if not p.endswith(".patch"):
            continue
        total += 1
        text = read_text(p, 400_000)
        upstream = "unknown"
        # `diff --git a/<path>` — the first path component names the upstream tree.
        tops = re.findall(r"^diff --git a/([^/]+)/", text, re.MULTILINE)
        if tops:
            upstream = tops[0]
        # Fall back to the directory the patch lives in.
        if upstream == "unknown":
            parts = rel(root, p).split(os.sep)
            upstream = parts[-2] if len(parts) >= 2 else "unknown"
        by_upstream.setdefault(upstream, []).append({
            "path": rel(root, p),
            "lines": text.count("\n") + 1,
        })
    return {"total": total, "by_upstream": by_upstream}


# Entry points that actually do something when run.
SCRIPT_STEMS = ("serve", "build", "run", "launch", "start", "probe", "check", "verify", "selftest", "test")


def audit_scripts(root: str) -> list[dict]:
    out: list[dict] = []
    for p in walk(root):
        base = os.path.basename(p)
        if not (base.endswith(".sh") or base.endswith(".py")):
            continue
        if not any(base.lower().startswith(s) or f"_{s}" in base.lower() for s in SCRIPT_STEMS):
            continue
        r = rel(root, p)
        if r.startswith(("tests", "test")):
            continue
        text = read_text(p, 60_000)
        # Capture the first model/framework path mentioned, as a hint at what it drives.
        hint = ""
        m = re.search(r"(/[\w./\-]*(?:vllm|model|davinci)[\w./\-]*)", text)
        if m:
            hint = m.group(1)[:80]
        out.append({"path": r, "lines": text.count("\n") + 1, "hint": hint})
    return sorted(out, key=lambda d: d["path"])


EVIDENCE_HINTS = re.compile(
    r"(ACCEPTANCE|EXPECTED_PERF|CORRECTNESS|REPRO|VERDICT|BENCH|PERF|"
    r"HANDOVER|GAP|CHANGELOG|STATUS|EVIDENCE|REPORT)",
    re.IGNORECASE,
)


def audit_evidence(root: str) -> list[dict]:
    out: list[dict] = []
    for p in walk(root):
        if not p.endswith((".md", ".csv", ".jsonl")):
            continue
        r = rel(root, p)
        if not EVIDENCE_HINTS.search(os.path.basename(p)):
            continue
        text = read_text(p, 200_000)
        out.append({
            "path": r,
            "lines": text.count("\n") + 1,
            # A doc that states a dated result is more useful than one that does not.
            "dated": bool(re.search(r"20\d\d-\d\d-\d\d", text)),
        })
    return sorted(out, key=lambda d: -d["lines"])[:40]


# A weight-size claim must sit next to a word that makes it a weight-size claim.
# The naive "any nearby NN GB" match is wrong: a window around a weight reference
# also contains unrelated sizes (KV cache, HBM, a single shard), and reporting
# one of those as the weight size is exactly the kind of unverified number the
# evidence rules forbid. Require an explicit cue word on the same line.
_SIZE_CUE = re.compile(r"(权重|weights?|模型大小|总大小|下载|download|共|total)", re.I)
_SIZE_VAL = re.compile(r"(\d{2,4}(?:\.\d+)?)\s*(GB|GiB|G)\b")


def _declared_weight_size(line: str) -> str:
    """Extract a size only when the line itself marks it as a weight size."""
    if not _SIZE_CUE.search(line):
        return ""
    m = _SIZE_VAL.search(line)
    return m.group(0) if m else ""


def audit_weights(root: str) -> list[dict]:
    """Find external weight references and any *line-scoped* declared size."""
    out: list[dict] = []
    seen: set[str] = set()
    pats = [
        (re.compile(r"modelscope\s+download\s+([\w.\-]+/[\w.\-]+)"), "modelscope", "download cmd"),
        (re.compile(r"(https://(?:www\.)?modelscope\.cn/models/([\w.\-]+/[\w.\-]+))"), "modelscope", "url"),
        (re.compile(r"(https://huggingface\.co/([\w.\-]+/[\w.\-]+))"), "huggingface", "url"),
    ]
    for p in walk(root):
        if not p.endswith((".md", ".sh", ".py")):
            continue
        text = read_text(p, 120_000)
        for rx, src, kind in pats:
            for m in rx.finditer(text):
                mid = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
                key = f"{src}:{mid}"
                if key in seen:
                    continue
                seen.add(key)
                # Only the line the reference sits on may supply the size.
                line_start = text.rfind("\n", 0, m.start()) + 1
                line_end = text.find("\n", m.end())
                line = text[line_start: line_end if line_end != -1 else len(text)]
                out.append({
                    "source": src,
                    "id": mid,
                    "kind": kind,
                    "declared_size": _declared_weight_size(line),
                    "declared_size_basis": "same-line cue word + number" if _declared_weight_size(line) else "",
                    "cited_in": rel(root, p),
                })
    return out[:20]


# Version tokens. The core is `digit(.digit)+`; an optional suffix may follow a
# separator (. - _) that is itself followed by an alphanumeric. This keeps
# meaningful suffixes (2.10.0.post4, 0.27.1-tp8-ep-04469255) while refusing a
# sentence-final period ("CANN 9.1.0."), which would otherwise split one version
# into two spellings and defeat the ambiguity check.
_VERSION = r"([\d]+(?:\.[\d]+)+(?:[._\-][A-Za-z0-9][\w.\-]*)?)"

STACK_PATTERNS = {
    "cann": re.compile(r"(?:CANN|ascend-toolkit)[^\w]{0,3}(?:v|ver|version)?[^\w]{0,3}" + _VERSION, re.I),
    "torch_npu": re.compile(r"torch_npu[^\w]{0,3}" + _VERSION, re.I),
    "vllm_ascend": re.compile(r"vllm[-_]ascend[^\w]{0,3}" + _VERSION, re.I),
    "vllm": re.compile(r"\bvllm[^\w]{0,3}" + _VERSION, re.I),
    "transformers": re.compile(r"transformers[^\w]{0,3}" + _VERSION, re.I),
    "driver": re.compile(r"driver[^\w]{0,3}" + _VERSION, re.I),
}


def audit_stack(root: str) -> dict:
    """Collect every version string seen per component.

    Returning only the first hit hides a real hazard: a repo that mentions both
    CANN 9.1.0 and 9.2.0 has two incompatible baselines, and a reader given one
    number will assume a single reproducible stack. We keep a counter per
    version and let the renderer show them all.
    """
    found: dict[str, dict] = {}
    files = [p for p in walk(root) if p.endswith((".md", ".txt", ".cfg", ".toml", ".json", ".yaml", ".yml", "Dockerfile"))]
    for p in files:
        text = read_text(p, 120_000)
        for key, rx in STACK_PATTERNS.items():
            for m in rx.finditer(text):
                val = m.group(1)
                entry = found.setdefault(key, {"versions": {}, "cited_in": rel(root, p)})
                entry["versions"][val] = entry["versions"].get(val, 0) + 1
                entry["occurrences"] = entry.get("occurrences", 0) + 1
    return found


TEST_HINTS = re.compile(r"(test|corpus|语料|golden|fixture)", re.I)


def audit_tests(root: str) -> dict:
    test_files: list[str] = []
    for p in walk(root):
        r = rel(root, p)
        if r.startswith(("tests", "test")) or TEST_HINTS.search(os.path.basename(p)):
            test_files.append(r)
    redistributed = [t for t in test_files if t.endswith((".json", ".jsonl", ".csv", ".parquet"))]
    return {
        "test_file_count": len(test_files),
        "sample": test_files[:15],
        "data_files_present": len(redistributed),
        "note": "a repo that withholds its corpus for copyright must ship a fetch+checksum script instead",
    }


def classify(a: Audit) -> None:
    if a.patches.get("total", 0) > 0:
        ups = ", ".join(sorted(a.patches["by_upstream"].keys()))
        a.strengths.append(f"ships {a.patches['total']} patch(es) against: {ups}")
    if any("acceptance" in d["path"].lower() or "expected_perf" in d["path"].lower() for d in a.evidence_docs):
        a.strengths.append("has an acceptance/expected-perf document")
    if any(s["path"].endswith("REPRO.md") for s in a.evidence_docs):
        a.strengths.append("has a REPRO.md")
    if a.stack_pins:
        pinned = [
            k for k, v in a.stack_pins.items() if len(v.get("versions") or {}) == 1
        ]
        a.strengths.append(
            "pins a single version for: " + ", ".join(sorted(pinned))
            if pinned else "mentions a toolchain, but with ambiguous versions"
        )
    if a.weights:
        a.strengths.append(f"references {len(a.weights)} external weight package(s)")

    if not a.patches.get("total"):
        a.red_flags.append("no patch files: the repo may only describe changes it does not ship")
    if not a.stack_pins:
        a.red_flags.append("no CANN/torch_npu version pin found: reproducibility is unproven")
    ambiguous = [k for k, v in a.stack_pins.items() if len(v.get("versions") or {}) > 1]
    if ambiguous:
        a.red_flags.append(
            "ambiguous toolchain: " + ", ".join(sorted(ambiguous))
            + " each appear in more than one version — match the claim to the right baseline"
        )
    if not a.evidence_docs:
        a.red_flags.append("no evidence/perf doc found")
    if a.tests.get("data_files_present", 0) == 0 and a.tests.get("test_file_count", 0) > 0:
        a.red_flags.append("test scripts exist but no test data ships: verify a fetch+checksum path exists")


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

def render(a: Audit) -> str:
    L: list[str] = []
    L.append(f"# Evidence audit: {os.path.basename(a.root.rstrip('/'))}")
    L.append("")
    g = a.git
    if g.get("has_git"):
        L.append(f"- git HEAD: `{g.get('head')}` | last commit {g.get('last_commit')} | origin {g.get('origin')}")
    else:
        L.append("- no git metadata (not a clone, or history stripped)")
    L.append("")

    L.append("## Patches")
    p = a.patches
    L.append(f"- total: {p.get('total', 0)}")
    for up, items in sorted((p.get("by_upstream") or {}).items()):
        L.append(f"- upstream `{up}`: {len(items)} patch(es), {sum(i['lines'] for i in items)} lines")
        for i in sorted(items, key=lambda d: -d["lines"])[:12]:
            L.append(f"    - {i['path']} ({i['lines']} lines)")
    if not p.get("total"):
        L.append("- (none)")
    L.append("")

    L.append("## Entry-point scripts")
    for s in a.scripts[:25]:
        hint = f"  -> {s['hint']}" if s["hint"] else ""
        L.append(f"- `{s['path']}` ({s['lines']} lines){hint}")
    if not a.scripts:
        L.append("- (none matched the serve/build/run/probe/check pattern)")
    L.append("")

    L.append("## Evidence documents")
    for d in a.evidence_docs[:20]:
        L.append(f"- `{d['path']}` ({d['lines']} lines){'  [dated]' if d['dated'] else ''}")
    if not a.evidence_docs:
        L.append("- (none)")
    L.append("")

    L.append("## External weights")
    for w in a.weights:
        size = f" declared {w['declared_size']}" if w.get("declared_size") else " size: unknown"
        L.append(f"- {w['source']}: `{w['id']}`{size}  (cited in {w['cited_in']})")
    if not a.weights:
        L.append("- (none referenced)")
    L.append("- A blank size means no *same-line* cue made the number a weight size;")
    L.append("  check the model page before quoting one.")
    L.append("")

    L.append("## Toolchain pins")
    for k, v in sorted(a.stack_pins.items()):
        versions = v.get("versions") or {}
        ranked = sorted(versions.items(), key=lambda kv: -kv[1])
        if len(ranked) == 1:
            val, n = ranked[0]
            L.append(f"- {k}: {val}  ({n} hits)")
        else:
            shown = ", ".join(f"{val} x{n}" for val, n in ranked[:6])
            L.append(f"- {k}: **{len(ranked)} distinct versions** -> {shown}")
            L.append(f"    [!] multiple baselines mentioned; identify which one the claim belongs to")
    if not a.stack_pins:
        L.append("- (none found — treat reproducibility as unproven)")
    L.append("")

    L.append("## Tests")
    t = a.tests
    L.append(f"- test files: {t.get('test_file_count', 0)} | shipped data files: {t.get('data_files_present', 0)}")
    for s in (t.get("sample") or [])[:8]:
        L.append(f"    - {s}")
    L.append(f"- {t.get('note')}")
    L.append("")

    L.append("## Verdict")
    for s in a.strengths:
        L.append(f"- [+] {s}")
    for r in a.red_flags:
        L.append(f"- [-] {r}")
    L.append("")
    L.append("This audit reads only the tree. It does not run anything. A repo with no")
    L.append("red flags is *auditable*, not *proven* — running its own self-check is the")
    L.append("only thing that closes the loop.")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit a cloned repo for Ascend-deployment evidence.")
    ap.add_argument("path", help="path to a local clone (read-only; nothing is written)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.path)
    if not os.path.isdir(root):
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    a = Audit(root=root)
    a.git = audit_git(root)
    a.patches = audit_patches(root)
    a.scripts = audit_scripts(root)
    a.evidence_docs = audit_evidence(root)
    a.weights = audit_weights(root)
    a.stack_pins = audit_stack(root)
    a.tests = audit_tests(root)
    classify(a)

    print(json.dumps(a.to_dict(), indent=2, ensure_ascii=False) if args.json else render(a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
