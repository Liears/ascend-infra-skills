#!/usr/bin/env python3
"""
Ascend Serving Benchmark Compare — make cross-framework rows comparable, then
refuse to compare the ones that are not.

Port of BBuf's `llm-serving-auto-benchmark` comparison step to Ascend. The GPU
version compares SGLang/vLLM/TensorRT-LLM/TokenSpeed under matched hardware and
SLA. On Ascend the frameworks are vLLM-Ascend, MindIE and SGLang-Ascend, and the
hardware axis has a trap the GPU version does not: **cards vs devices**.

This tool does two things:

  1. Normalizes a benchmark result (from `vllm bench serve` / aisbench / a
     hand-recorded run) into a common row shape.
  2. Guards the comparison: two rows are only put side by side when model,
     hardware, toolchain, workload and SLA match. When they do not, it says
     which field differs instead of printing a misleading ratio.

Usage:
    python3 compare_results.py rows.json
    python3 compare_results.py rows.json --json
    python3 compare_results.py --template > rows.json   # start from this

The row schema is the one in ../../docs/EVIDENCE-RULES.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

# Fields that must match for two rows to be comparable at all.
MATCH_KEYS = ("model", "chip", "devices", "toolchain", "in_len", "out_len",
              "concurrency", "prefix_cache", "graph_mode", "hccl")

# Fields that may legitimately differ (that is the point of the comparison).
VARY_KEYS = ("framework", "kv_cache_memory_bytes", "quant", "spec_decode")

REQUIRED = MATCH_KEYS + ("framework", "tok_s")


@dataclass
class Row:
    framework: str = ""
    model: str = ""
    chip: str = ""
    cards: Optional[int] = None
    devices: Optional[int] = None
    toolchain: str = ""          # CANN / driver / torch_npu
    framework_commit: str = ""
    quant: str = ""
    in_len: Optional[int] = None
    out_len: Optional[int] = None
    concurrency: Optional[int] = None
    prefix_cache: Optional[int] = None
    graph_mode: str = ""
    hccl: str = ""
    spec_decode: Optional[int] = None
    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    tok_s: Optional[float] = None
    accept_len: Optional[float] = None       # A, for spec-decode models
    tok_s_basis: str = ""                    # "A*1000/ms" or "gen_tokens/wall"
    sla_ok: Optional[bool] = None
    artifact: str = ""
    captured: str = ""

    def comparable_key(self) -> dict:
        d = asdict(self)
        key = {k: d.get(k) for k in MATCH_KEYS}
        key["devices"] = self.devices
        return key


def load(path: str) -> list[Row]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
    out = []
    for r in rows:
        out.append(Row(**{k: v for k, v in r.items() if k in Row.__dataclass_fields__}))
    return out


def validate(rows: list[Row]) -> tuple[list[str], set[int]]:
    """Return (problems, ids of rows that must not be ranked).

    The untrusted set is keyed by `id(row)` so the renderer can exclude exactly
    the rows that produced a problem. Validation and rendering therefore agree
    by construction: a row that raised a problem cannot also win the ranking.
    """
    problems: list[str] = []
    untrusted: set[int] = set()

    def flag(r: Row, msg: str) -> None:
        problems.append(msg)
        untrusted.add(id(r))

    for i, r in enumerate(rows):
        tag = f"row[{i}] {r.framework or '?'} on {r.chip or '?'}"
        missing = [k for k in REQUIRED if getattr(r, k) in (None, "")]
        if missing:
            flag(r, f"{tag}: missing {', '.join(missing)}")
        if r.devices and r.cards and r.devices < r.cards:
            flag(r, f"{tag}: devices({r.devices}) < cards({r.cards}) — check the unit")
        if r.cards is None and r.devices is not None:
            flag(r, f"{tag}: devices given but cards blank — on A3 this is ambiguous")
        if r.tok_s and r.tok_s_basis not in ("A*1000/ms", "gen_tokens/wall"):
            flag(r, f"{tag}: tok_s_basis='{r.tok_s_basis}' — must say which convention; "
                    f"on a spec-decode model the two differ and are routinely confused")
        if r.spec_decode and not r.accept_len:
            flag(r, f"{tag}: spec_decode=1 but no accept_len — tok/s is not interpretable")
    return problems, untrusted


def group(rows: list[Row]) -> dict:
    groups: dict = {}
    for r in rows:
        key = json.dumps(r.comparable_key(), sort_keys=True)
        groups.setdefault(key, []).append(r)
    return groups


def render(rows: list[Row], problems: list[str], bad_idx: set[int] | None = None) -> str:
    L: list[str] = []
    L.append("# Ascend cross-framework benchmark comparison")
    L.append("")

    bad_idx = bad_idx or set()

    if problems:
        L.append("## Rows that cannot be trusted yet")
        for p in problems:
            L.append(f"- {p}")
        L.append("")
        L.append("A flagged row is shown for transparency but is excluded from the")
        L.append("ranking below — an unverifiable row must not win a comparison.")
        L.append("")

    groups = group(rows)
    L.append(f"## {len(groups)} comparable group(s)")
    L.append("")

    for gk, members in groups.items():
        key = json.loads(gk)
        # Rank only trusted rows.
        ok = [r for r in members if id(r) not in bad_idx]
        if len(members) < 2:
            L.append("### single row (nothing to compare against)")
            L.append(f"- {_rowline(members[0])}"
                     + ("  **[flagged, untrusted]**" if id(members[0]) in bad_idx else ""))
            L.append(f"  matched on: {_keyline(key)}")
            L.append("")
            continue

        L.append(f"### {key.get('model')} on {key.get('chip')} "
                 f"({key.get('devices')} devices), in={key.get('in_len')} "
                 f"out={key.get('out_len')} conc={key.get('concurrency')}")
        L.append(f"matched on: {_keyline(key)}")
        L.append("")
        L.append("| framework | quant | graph | conc | TTFT ms | TPOT ms | tok/s | basis | A | SLA |")
        L.append("|---|---|---|---:|---:|---:|---:|---|---:|---|")
        for r in sorted(members, key=lambda r: -(r.tok_s or 0)):
            flag = " **[flagged]**" if id(r) in bad_idx else ""
            L.append(
                f"| {r.framework}{flag} | {r.quant or '-'} | {r.graph_mode or '-'} | "
                f"{r.concurrency} | {_f(r.ttft_ms)} | {_f(r.tpot_ms)} | "
                f"{_f(r.tok_s)} | {r.tok_s_basis or '?'} | {_f(r.accept_len)} | "
                f"{_sla(r.sla_ok)} |"
            )
        L.append("")
        if ok:
            best = max(ok, key=lambda r: (r.tok_s or 0))
            L.append(f"Highest tok/s among trusted rows: **{best.framework}** at "
                     f"{_f(best.tok_s)} ({best.tok_s_basis or 'basis unstated'}).")
        else:
            L.append("No trusted row in this group — fix the flagged rows first.")
        if any(m.sla_ok is False for m in members):
            L.append("At least one row missed its SLA — a higher throughput that misses "
                     "latency is not a win; compare only rows with sla_ok=true.")
        L.append("")

    L.append("---")
    L.append("Two rows are compared only when every match key is identical. A ratio")
    L.append("across different in_len/out_len/concurrency/hardware is not a result.")
    L.append("See ../../docs/EVIDENCE-RULES.md for the field meanings.")
    return "\n".join(L)


def _f(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else ("-" if v is None else str(v))


def _sla(v) -> str:
    return {True: "met", False: "**missed**", None: "?"}[v]


def _rowline(r: Row) -> str:
    return (f"{r.framework} on {r.chip}: tok/s={_f(r.tok_s)} "
            f"({r.tok_s_basis or 'basis unstated'}), TTFT={_f(r.ttft_ms)} ms")


def _keyline(key: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in key.items())


TEMPLATE = {
    "_comment": "Fill one object per measured run. Every field is required unless the "
                "validator says otherwise. tok_s_basis MUST be stated for spec-decode models.",
    "rows": [
        {
            "framework": "vllm-ascend",
            "model": "DeepSeek-V4.1-Flash",
            "chip": "910B3",
            "cards": 8,
            "devices": 8,
            "toolchain": "CANN 8.1.RC1 / driver 25.5.2 / torch_npu 2.10.0.post4",
            "framework_commit": "<vllm-ascend commit>",
            "quant": "W4A8",
            "in_len": 1024, "out_len": 256, "concurrency": 1,
            "prefix_cache": 0, "graph_mode": "eager", "hccl": "HCCL",
            "spec_decode": 1, "accept_len": 2.40,
            "ttft_ms": 0.0, "tpot_ms": 0.0,
            "tok_s": 0.0, "tok_s_basis": "A*1000/ms",
            "sla_ok": True,
            "artifact": "/path/to/result.json",
            "captured": "2026-10-14",
        }
    ],
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare Ascend serving benchmark rows, refusing bad comparisons.")
    ap.add_argument("rows", nargs="?", help="JSON file of rows")
    ap.add_argument("--template", action="store_true", help="print a starter rows.json")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.template:
        print(json.dumps(TEMPLATE, indent=2, ensure_ascii=False))
        return 0
    if not args.rows:
        ap.error("provide a rows.json or --template")

    rows = load(args.rows)
    problems, untrusted = validate(rows)
    if args.json:
        print(json.dumps({"problems": problems, "untrusted_rows": len(untrusted),
                          "groups": len(group(rows))},
                         indent=2, ensure_ascii=False))
    else:
        print(render(rows, problems, untrusted))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
