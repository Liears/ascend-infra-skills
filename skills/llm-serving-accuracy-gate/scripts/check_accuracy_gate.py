#!/usr/bin/env python3
"""
Accuracy gate verdict — turn an eval result into PASS/FAIL with an evidence
block, or refuse to judge when the protocol is not recorded.

The gates (see references/gate-definitions.md):
  L1b  GSM8K-200 chat mode — drop >2 correct vs baseline = FAIL
  L2   AIME — requires k>=4 repeats recorded; reports mean + spread

A result whose JSON lacks the protocol fields (harness, sampling, dataset,
draft) is judged "unrecorded" and fails the gate: a number that cannot be
compared to anything is not a result. This is deliberate — the gate exists
to stop unanchored numbers from becoming conclusions.

Usage:
    python3 check_accuracy_gate.py result.json --gate L1b --baseline 198/200
    python3 check_accuracy_gate.py result.json --gate L2 --official 94-100
    python3 check_accuracy_gate.py result.json --gate L1b --baseline 198/200 --json

Result JSON shape (either the port runner's raw output or a normalized one):
    {"correct": C, "total": T, ...}                        # single run
    {"runs": [{"correct": C, "total": T}, ...]}            # multi-run (L2)
Optional but required for a PASS verdict:
    "harness", "dataset", "sampling", "draft"

Exit codes: 0 = PASS, 1 = FAIL, 2 = usage error.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

# Protocol fields a result must carry to be judged at all.
REQUIRED_PROTOCOL = ("harness", "dataset", "sampling", "draft")


@dataclass
class Verdict:
    gate: str
    verdict: str = "UNRECORDED"     # PASS | FAIL | UNRECORDED
    correct: Optional[int] = None
    total: Optional[int] = None
    drop_vs_baseline: Optional[int] = None
    mean_pct: Optional[float] = None
    spread_pct: Optional[float] = None
    k: Optional[int] = None
    reasons: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


def load_result(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    # The port runner's wrapper writes {correct,total,acc}; its raw output
    # and hand-recorded results may nest runs. Normalize both.
    return d


def single_counts(d: dict) -> tuple[Optional[int], Optional[int]]:
    if d.get("correct") is not None and d.get("total") is not None:
        return int(d["correct"]), int(d["total"])
    return None, None


def runs_counts(d: dict) -> list[tuple[int, int]]:
    out = []
    for r in d.get("runs", []) or []:
        if r.get("correct") is None or r.get("total") is None:
            continue
        out.append((int(r["correct"]), int(r["total"])))
    return out


def parse_baseline(s: str) -> tuple[int, int]:
    try:
        num, den = s.replace(" ", "").split("/")
        return int(num), int(den)
    except Exception:
        # argparse-style error exit (2), so CI can distinguish usage errors
        # from gate failures (1).
        print(f"--baseline must look like 198/200, got: {s!r}", file=sys.stderr)
        raise SystemExit(2)


def missing_protocol(d: dict) -> list[str]:
    return [k for k in REQUIRED_PROTOCOL if not d.get(k)]


def gate_l1b(d: dict, base_c: int, base_t: int) -> Verdict:
    v = Verdict(gate="L1b GSM8K-200 chat")
    missing = missing_protocol(d)
    c, t = single_counts(d)
    if c is None and runs_counts(d):
        # multi-run L1b: judge the worst run (the gate is a floor)
        runs = runs_counts(d)
        c = min(r[0] for r in runs)
        t = runs[0][1]
        v.k = len(runs)
        v.reasons.append(f"multi-run input: judged on the worst of {len(runs)} runs")
    if c is None:
        v.verdict = "UNRECORDED"
        v.reasons.append("no correct/total in result")
        return v
    if missing:
        v.verdict = "UNRECORDED"
        v.reasons.append("protocol not recorded: " + ", ".join(missing))
        v.reasons.append("a number that cannot be compared is not a result")
        return v
    if t != base_t:
        v.verdict = "UNRECORDED"
        v.reasons.append(f"total {t} != baseline total {base_t}; denominators must match")
        return v

    v.correct, v.total = c, t
    v.drop_vs_baseline = base_c - c
    # Rule from the port repo's acceptance: drop >2 = unusable; ±1 is noise.
    if v.drop_vs_baseline > 2:
        v.verdict = "FAIL"
        v.reasons.append(
            f"dropped {v.drop_vs_baseline} vs baseline {base_c}/{base_t} "
            f"(rule: >2 = FAIL, ±1 = noise)")
    else:
        v.verdict = "PASS"
        v.reasons.append(
            f"drop {v.drop_vs_baseline} vs baseline {base_c}/{base_t} "
            f"(rule: >2 = FAIL, ±1 = noise)")
    return v


def gate_l2(d: dict, official_note: str) -> Verdict:
    v = Verdict(gate="L2 AIME")
    runs = runs_counts(d)
    if not runs:
        c, t = single_counts(d)
        if c is not None and t is not None:
            runs = [(c, t)]
    if not runs:
        v.verdict = "UNRECORDED"
        v.reasons.append("no runs with correct/total in result")
        return v
    if len(set(r[1] for r in runs)) != 1:
        v.verdict = "UNRECORDED"
        v.reasons.append("runs have differing totals; cannot average")
        return v
    v.k = len(runs)
    if v.k < 4:
        v.verdict = "FAIL"
        v.reasons.append(
            f"k={v.k} < 4: 30 problems means 1 problem = 3.3 points; a single "
            f"pass cannot separate regression from luck")
        return v
    missing = missing_protocol(d)
    if missing:
        v.verdict = "UNRECORDED"
        v.reasons.append("protocol not recorded: " + ", ".join(missing))
        return v

    totals = runs[0][1]
    pcts = [100.0 * c / totals for c, _ in runs]
    v.total = totals
    v.mean_pct = round(sum(pcts) / len(pcts), 2)
    v.spread_pct = round(max(pcts) - min(pcts), 2)
    # L2 does not FAIL against the official number (a quant gap is the
    # measurement). It FAILS on protocol violations, which are handled above.
    v.verdict = "PASS"
    v.reasons.append(
        f"mean {v.mean_pct}% over k={v.k}, spread {v.spread_pct}pp "
        f"(each problem = {round(100/totals, 1)}pp)")
    if official_note:
        v.reasons.append(
            f"official reference: {official_note} — a gap vs official is the "
            f"measurement (different weights), not a failure; compare "
            f"like-protocol runs for regressions")
    if v.spread_pct > 10:
        v.reasons.append(
            f"spread {v.spread_pct}pp is wide; treat the mean with caution "
            f"and consider more repeats")
    return v


def render(v: Verdict, d: dict) -> str:
    L = []
    icon = {"PASS": "✓", "FAIL": "✗", "UNRECORDED": "?"}[v.verdict]
    L.append(f"# Accuracy gate: {v.gate} — {icon} {v.verdict}")
    L.append("")
    if v.correct is not None:
        L.append(f"- result: **{v.correct}/{v.total}**"
                 + (f"  (worst of k={v.k})" if v.k and v.gate.startswith("L1") else ""))
    if v.drop_vs_baseline is not None:
        L.append(f"- drop vs baseline: {v.drop_vs_baseline}")
    if v.mean_pct is not None:
        L.append(f"- mean: {v.mean_pct}%  spread: {v.spread_pct}pp  k={v.k}")
    L.append("")
    L.append("## Reasons")
    for r in v.reasons:
        L.append(f"- {r}")
    L.append("")
    L.append("## Evidence block")
    L.append("```")
    L.append(f"gate:     {v.gate}")
    for k in REQUIRED_PROTOCOL:
        L.append(f"{k + ':':9s} {d.get(k) or '(not recorded)'}")
    if v.correct is not None:
        L.append(f"result:   {v.correct}/{v.total}")
    if v.mean_pct is not None:
        L.append(f"mean:     {v.mean_pct}% over k={v.k} (spread {v.spread_pct}pp)")
    art = d.get("artifact")
    if art:
        L.append(f"artifact: {art}")
    L.append("```")
    if v.verdict == "UNRECORDED":
        L.append("")
        L.append("The gate refuses to judge: record the protocol fields and re-run.")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verdict an accuracy result against a gate.")
    ap.add_argument("result", help="path to the result JSON")
    ap.add_argument("--gate", required=True, choices=["L1b", "L2"])
    ap.add_argument("--baseline", help="L1b: baseline as C/T, e.g. 198/200")
    ap.add_argument("--official", default="", help="L2: official reference note, e.g. '94-100 @ effort 25-100'")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    d = load_result(args.result)

    if args.gate == "L1b":
        if not args.baseline:
            ap.error("L1b requires --baseline C/T")
        base_c, base_t = parse_baseline(args.baseline)
        v = gate_l1b(d, base_c, base_t)
    else:
        v = gate_l2(d, args.official)

    if args.json:
        out = asdict(v)
        out["evidence"] = {k: d.get(k) for k in REQUIRED_PROTOCOL}
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(render(v, d))
    return {"PASS": 0, "FAIL": 1, "UNRECORDED": 1}[v.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
