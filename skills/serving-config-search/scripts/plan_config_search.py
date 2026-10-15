#!/usr/bin/env python3
"""
Plan a serving-config search: enumerate candidates, honour constraints,
extrapolate capacity offline, prune SLA-infeasible trials, order the
survivors, and budget the wall-clock — all without a single service restart.

Why the extrapolation carries an offset: on real data (DeepSeek-V4.1-Flash
W4A8, 8×910B3) the naive model `hbm×util − non_kv` predicts 12.59 GiB of KV
at GPU_UTIL 0.92 while the measured pool is 13.60 GiB — a 7% error **at the
anchor point itself**. The planner therefore calibrates the offset from the
Phase-0 measurement, making the model exact at the anchor and linear
elsewhere. Extrapolation still prunes only; measurement decides.

Usage:
    python3 plan_config_search.py search.json
    python3 plan_config_search.py search.json --out trials.json
    python3 plan_config_search.py search.json --json

Input schema (search.json):

    {
      "base":         {"GPU_UTIL": 0.92, "BAT_TOKENS": 8192, ...},
      "axes":         {"GPU_UTIL": [0.88, 0.90, 0.92, 0.94], "MAX_SEQS": [4, 32]},
      "constraints":  [
        {"type": "pair", "knobs": ["KV32_REPACK", "BAT_TOKENS"]},
        {"type": "forbidden", "when": {"GPU_UTIL": 0.94}, "reason": "..."},
        {"type": "accuracy_coupled", "knobs": ["BAT_TOKENS"]}
      ],
      "calibration": {
        "hbm_gib": 60.96, "non_kv_gib": 43.49, "base_util": 0.92,
        "kv_gib": 13.60, "kv_tokens": 2942071
      },
      "sla":          {"request_len": 131072, "min_concurrent": 8},
      "restart_cost_min": 15
    }

`calibration.kv_gib` is optional; when absent the offset is 0 and every
extrapolation is flagged `uncalibrated`.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

@dataclass
class Trial:
    config: dict
    knobs_changed: list
    order: int = 0
    status: str = "run"            # run | pruned | baseline
    prune_reason: str = ""
    accuracy_coupled: bool = False
    predicted: dict = field(default_factory=dict)


def load_search(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    for key in ("base", "axes", "calibration", "sla"):
        if key not in d:
            raise SystemExit(f"search.json missing required key: {key}")
    return d


# --------------------------------------------------------------------------
# capacity extrapolation
# --------------------------------------------------------------------------

class CapacityModel:
    """Linear-in-GPU_UTIL KV pool model, offset-calibrated at the anchor."""

    def __init__(self, cal: dict):
        self.hbm = float(cal["hbm_gib"])
        self.non_kv = float(cal["non_kv_gib"])
        self.base_util = float(cal.get("base_util", 0.9))
        self.kv_tokens = float(cal["kv_tokens"])
        measured = cal.get("kv_gib")
        naive = self.hbm * self.base_util - self.non_kv
        if measured is not None:
            self.offset = float(measured) - naive
            self.uncalibrated = False
        else:
            self.offset = 0.0
            self.uncalibrated = True
        self.anchor_kv_gib = naive + self.offset
        if self.anchor_kv_gib <= 0:
            raise SystemExit("calibration implies a non-positive KV pool")
        self.tokens_per_gib = self.kv_tokens / self.anchor_kv_gib

    def predict(self, util: float) -> dict:
        kv_gib = self.hbm * util - self.non_kv + self.offset
        if kv_gib <= 0:
            return {"kv_gib": round(kv_gib, 2), "kv_tokens": 0, "concurrency": 0,
                    "fits": False}
        tokens = self.tokens_per_gib * kv_gib
        return {"kv_gib": round(kv_gib, 2), "kv_tokens": int(tokens), "fits": True}

    def concurrency(self, util: float, request_len: int) -> int:
        p = self.predict(util)
        if not p["fits"]:
            return 0
        return int(p["kv_tokens"] // max(request_len, 1))


# --------------------------------------------------------------------------
# constraints
# --------------------------------------------------------------------------

def violates_pair(cfg: dict, base: dict, knobs: list) -> Optional[str]:
    """A pair must move together; moving one alone is invalid."""
    changed = [k for k in knobs if k in cfg and cfg.get(k) != base.get(k)]
    if len(changed) == 1:
        other = [k for k in knobs if k != changed[0]]
        return (f"pair violation: {changed[0]} moved without "
                f"{', '.join(other)} (must move together or not at all)")
    return None


def matches_when(cfg: dict, when: dict) -> bool:
    return all(cfg.get(k) == v for k, v in when.items())


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------

def plan(search: dict) -> dict:
    base = search["base"]
    axes = search.get("axes", {})
    cons = search.get("constraints", [])
    sla = search["sla"]
    restart_min = float(search.get("restart_cost_min", 15))

    model = CapacityModel(search["calibration"])
    request_len = int(sla["request_len"])
    min_conc = int(sla["min_concurrent"])

    # The util the trial runs at: the axis value if GPU_UTIL is being varied,
    # else the base value.
    base_util = float(base.get("GPU_UTIL", model.base_util))

    # Enumerate the cartesian product of the axes.
    knob_names = sorted(axes.keys())
    grids = [axes[k] if k in axes else [base.get(k)] for k in knob_names]
    trials: list[Trial] = []

    for combo in itertools.product(*grids):
        cfg = dict(base)
        cfg.update({k: v for k, v in zip(knob_names, combo) if v is not None})
        changed = sorted(k for k in knob_names if cfg.get(k) != base.get(k))
        t = Trial(config=cfg, knobs_changed=changed)

        if not changed:
            t.status = "baseline"
            t.predicted = model.predict(float(cfg.get("GPU_UTIL", base_util)))
            trials.append(t)
            continue

        # constraints: pair / forbidden
        pruned = False
        for c in cons:
            if c.get("type") == "pair":
                reason = violates_pair(cfg, base, c["knobs"])
                if reason:
                    t.status, t.prune_reason = "pruned", reason
                    pruned = True
                    break
            elif c.get("type") == "forbidden":
                if matches_when(cfg, c.get("when", {})):
                    t.status = "pruned"
                    t.prune_reason = ("forbidden region: " +
                                      (c.get("reason") or "no reason recorded"))
                    pruned = True
                    break
        if pruned:
            trials.append(t)
            continue

        # accuracy-coupled marking (never prunes — it defers the verdict)
        acc_knobs = set()
        for c in cons:
            if c.get("type") == "accuracy_coupled":
                acc_knobs.update(c.get("knobs", []))
        t.accuracy_coupled = bool(acc_knobs & set(changed))

        # capacity extrapolation + SLA prune
        util = float(cfg.get("GPU_UTIL", base_util))
        pred = model.predict(util)
        conc = model.concurrency(util, request_len)
        t.predicted = {**pred, "concurrency": conc,
                       "request_len": request_len}
        if not pred["fits"]:
            t.status = "pruned"
            t.prune_reason = (f"does not fit: predicted KV pool "
                              f"{pred['kv_gib']} GiB at util {util}")
        elif conc < min_conc:
            t.status = "pruned"
            t.predicted["concurrency"] = conc
            t.prune_reason = (
                f"SLA infeasible: predicted concurrency {conc} < {min_conc} "
                f"({int(pred['kv_tokens']):,} tokens / {request_len:,} "
                f"= {conc}; pool {pred['kv_gib']} GiB @ util {util})")
        trials.append(t)

    # The baseline must always be present: when an axis does not list the
    # base value (e.g. GPU_UTIL: [0.90] with base 0.92), the cartesian
    # product never contains the unchanged combo, and the summary's
    # baseline lookup would fail. Synthesize it if the product missed it.
    if not any(t.status == "baseline" for t in trials):
        trials.append(Trial(config=dict(base), knobs_changed=[],
                            status="baseline"))

    # Order survivors: nearest-neighbor from the baseline.
    runners = [t for t in trials if t.status == "run"]
    runners.sort(key=lambda t: (len(t.knobs_changed),
                                -t.predicted.get("concurrency", 0)))
    for i, t in enumerate(runners, 1):
        t.order = i

    return {
        "baseline": base,
        "sla": sla,
        "restart_cost_min": restart_min,
        "capacity_model": {
            "tokens_per_gib": int(model.tokens_per_gib),
            "offset_gib": round(model.offset, 2),
            "uncalibrated": model.uncalibrated,
        },
        "planned_trials": len(runners),
        "budget_min": round(len(runners) * restart_min, 1),
        "budget_hours": round(len(runners) * restart_min / 60, 2),
        "trials": [asdict(t) for t in
                   ([next(x for x in trials if x.status == "baseline")]
                    + runners
                    + [t for t in trials if t.status == "pruned"])],
    }


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

def render(p: dict) -> str:
    L = []
    L.append("# Serving config search plan")
    L.append("")
    cm = p["capacity_model"]
    L.append(f"- capacity model: {cm['tokens_per_gib']:,} tokens/GiB, "
             f"offset {cm['offset_gib']:+} GiB"
             + ("  ⚠ UNCALIBRATED (no kv_gib in calibration — offsets are 0)"
                if cm["uncalibrated"] else ""))
    L.append(f"- planned trials: {p['planned_trials']}  "
             f"budget ≈ {p['budget_min']} min ({p['budget_hours']} h) "
             f"@ {p['restart_cost_min']} min/restart")
    sla = p["sla"]
    L.append(f"- SLA: request_len {sla['request_len']:,} × "
             f"min_concurrent {sla['min_concurrent']}")
    L.append("")
    L.append("| # | changed knobs | config | pred. pool | pred. conc | status |")
    L.append("|---:|---|---|---:|---:|---|")
    for t in p["trials"]:
        cfg = " ".join(f"{k}={v}" for k, v in t["config"].items()
                       if k in t["knobs_changed"] or t["status"] == "baseline")
        pred = t.get("predicted", {})
        pool = f"{pred.get('kv_gib', '—')} GiB" if pred else "—"
        conc = str(pred.get("concurrency", "—")) if pred else "—"
        if t["status"] == "baseline":
            st = "baseline (Phase 0, already measured)"
        elif t["status"] == "run":
            st = "RUN" + ("  ⚠ accuracy-coupled (L1 re-pass before winning)"
                          if t["accuracy_coupled"] else "")
        else:
            st = "pruned — " + t["prune_reason"]
        L.append(f"| {t['order'] or '—'} | {', '.join(t['knobs_changed']) or '—'} "
                 f"| {cfg} | {pool} | {conc} | {st} |")
    L.append("")
    L.append("Extrapolation prunes; measurement decides. Run trials in order,")
    L.append("record each as a benchmark row (same workload for all), and stop")
    L.append("early when the ordering makes the answer deterministic.")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Plan a pruned, ordered serving-config search.")
    ap.add_argument("search", help="path to search.json")
    ap.add_argument("--out", help="write trials JSON here instead of stdout")
    ap.add_argument("--json", action="store_true",
                    help="print the plan as JSON instead of a table")
    args = ap.parse_args(argv)

    p = plan(load_search(args.search))
    text = json.dumps(p, indent=2, ensure_ascii=False) if args.json else render(p)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text if args.json else json.dumps(
                p, indent=2, ensure_ascii=False))
        print(f"wrote {args.out}: {p['planned_trials']} trials, "
              f"budget ≈ {p['budget_min']} min", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
