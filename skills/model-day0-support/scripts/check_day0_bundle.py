#!/usr/bin/env python3
"""
Validate that a Day-0 bundle carries every required artifact.

A Day-0 claim ("model X is supported on Ascend") is falsifiable only if the
bundle contains the artifacts that would let someone else re-run it. This tool
checks for their presence and shape; it does not judge their quality, which
needs reading.

Usage:
    python3 check_day0_bundle.py /path/to/bundle
    python3 check_day0_bundle.py /path/to/bundle --json

Exit code 0 when every required artifact is present, 1 otherwise. Suitable for
blocking a release in CI.

A bundle is a directory. Files are matched by name pattern anywhere inside it,
except that launch scripts and configs are also matched by content.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict

SKIP_DIRS = {".git", "node_modules", "__pycache__"}


def walk(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            yield os.path.join(dirpath, fn)


def read(p: str, limit: int = 400_000) -> str:
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(limit)
    except Exception:  # noqa: BLE001
        return ""


@dataclass
class Requirement:
    key: str
    description: str
    found: bool = False
    where: list = field(default_factory=list)
    hint: str = ""


@dataclass
class Report:
    root: str
    requirements: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    passes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


# Each requirement: (key, description, matcher(files_map) -> list[path], hint)
def build_requirements():
    reqs: list[Requirement] = []

    def add(key, desc, hint=""):
        reqs.append(Requirement(key=key, description=desc, hint=hint))
        return reqs[-1]

    add("launch_command", "A launch/serve script or the exact server command",
        "a serve_*.sh / run*.sh, or a markdown block starting with `vllm serve`")
    add("stack_record", "CANN / driver / torch_npu / framework versions recorded",
        "grep for 'CANN' plus a version number somewhere in the bundle")
    add("accuracy_evidence", "An accuracy result and how it was produced",
        "a report mentioning accuracy/GSM8K/精度 with a number")
    add("fastpath_evidence", "Profiler evidence that the fused kernel ran",
        "a kernel_details.csv, or a report citing a fused op name")
    add("memory_figure", "Weights + KV budget at a stated context length",
        "a startup log line with 'KV cache size' or a documented KV budget")
    add("limitations", "Known limitations listed",
        "a section headed Limitations / Known issues / 已知问题")
    add("weight_source", "Weight source with version or checksum",
        "a modelscope/huggingface id, or a sha256 in a manifest")
    return reqs


def check(root: str) -> Report:
    rep = Report(root=root)
    reqs = build_requirements()

    all_files = list(walk(root))
    by_key = {r.key: r for r in reqs}

    # Pre-read text files once, bounded.
    text_cache: dict[str, str] = {}
    for p in all_files:
        if os.path.getsize(p) > 2_000_000:
            continue
        # .csv/.log must be here too: kernel_details.csv is the primary
        # fast-path artifact and is matched by filename inside this loop.
        if p.endswith((".md", ".txt", ".sh", ".py", ".json", ".yaml", ".yml",
                       ".log", ".cfg", ".csv")):
            text_cache[p] = read(p)

    def note(key: str, path: str):
        by_key[key].found = True
        by_key[key].where.append(os.path.relpath(path, root))

    for p, text in text_cache.items():
        base = os.path.basename(p)
        low = text.lower()

        # launch command
        if re.search(r"\b(serve|launch|start|run)[\w.]*\.sh$", base) or \
           re.search(r"^\s*vllm\s+serve\b", text, re.MULTILINE) or \
           re.search(r"entrypoints\.openai\.api_server", text):
            note("launch_command", p)

        # stack record
        if re.search(r"CANN\s*[v0-9]", text) and re.search(r"torch_npu\s*[v0-9]", text):
            note("stack_record", p)
        elif re.search(r"driver\s*(version)?\s*[:=]?\s*\d+\.\d+", low) and "cann" in low:
            note("stack_record", p)

        # accuracy evidence
        if re.search(r"(gsm8k|accuracy|精度|eval.*score|pass@)", low) and re.search(r"\d+\s*/\s*\d+|\d+\.\d+%", text):
            note("accuracy_evidence", p)

        # fastpath evidence
        if base == "kernel_details.csv" or \
           re.search(r"(fused|flash_attn|aclnn\w*_|op type).*(ran|executed|engaged|生效)", low):
            note("fastpath_evidence", p)

        # memory figure
        if re.search(r"kv cache size|available kv cache|kv_cache.*tokens|kv 预算|kv budget", low):
            note("memory_figure", p)

        # limitations
        if re.search(r"^#{1,4}\s*(known\s+)?(limitations?|issues?|已知问题|限制)", text, re.MULTILINE | re.IGNORECASE):
            note("limitations", p)

        # weight source — a model page URL, a `modelscope download <id>` CLI
        # invocation, or a sha256 checksum all count.
        if re.search(r"(modelscope\.cn|huggingface\.co/\S+/\S+)", text) or \
           re.search(r"modelscope\s+download\s+\S+/\S+", text) or \
           re.search(r"\bsha256[:=]?\s*[0-9a-f]{32,}", low):
            note("weight_source", p)

    for r in reqs:
        (rep.passes if r.found else rep.failures).append(r)
    rep.requirements = reqs
    return rep


def render(rep: Report) -> str:
    L: list[str] = []
    L.append(f"# Day-0 bundle check: {rep.root}")
    L.append("")
    L.append(f"{len(rep.passes)}/{len(rep.requirements)} required artifacts present")
    L.append("")

    if rep.failures:
        L.append("## Missing (blocking a Day-0 claim)")
        for r in rep.failures:
            L.append(f"- **{r.key}** — {r.description}")
            L.append(f"    expected: {r.hint}")
        L.append("")

    if rep.passes:
        L.append("## Present")
        for r in rep.passes:
            where = ", ".join(f"`{w}`" for w in r.where[:3])
            more = f" (+{len(r.where)-3} more)" if len(r.where) > 3 else ""
            L.append(f"- {r.key} — {r.description}")
            L.append(f"    found in: {where}{more}")
        L.append("")

    L.append("---")
    if rep.ok:
        L.append("Every required artifact is present. Presence is not quality:")
        L.append("read them before calling the model Day-0 supported.")
    else:
        L.append("At least one required artifact is missing. A Day-0 claim without")
        L.append("these cannot be re-run by anyone else, which makes it an assertion")
        L.append("rather than a result.")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check a Day-0 bundle for required artifacts.")
    ap.add_argument("path")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.path)
    if not os.path.isdir(root):
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    rep = check(root)
    if args.json:
        print(json.dumps(
            {"ok": rep.ok,
             "missing": [r.key for r in rep.failures],
             "present": [r.key for r in rep.passes],
             "details": [asdict(r) for r in rep.requirements]},
            indent=2, ensure_ascii=False))
    else:
        print(render(rep))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
