#!/usr/bin/env python3
"""
Fusion headroom — the should-I-wire calculator.

Answers the question that ends operator-layer work honestly: given a fused
kernel's EAGER micro-benchmark speedup, what is the expected END-TO-END gain
once it runs inside the serving graph? If that number is below the wiring
cost (restarts + gates + patch maintenance), the kernel ships as an artifact
and the wiring stops.

Why eager numbers overstate: inside a captured graph, kernel launches are
amortized — the per-call cost is memory traffic, not launch overhead. A
4.97x eager win on a 60 KB tensor can be worth <1% end-to-end. This script
makes that arithmetic explicit instead of vibed.

Model (stated so it can be checked):
  bytes_saved_per_site  ≈ one intermediate round-trip of the fused-away
                          tensor (read + write), or the eager delta if measured
  in_graph_us_per_site  ≈ bytes_saved / effective_bandwidth
  end_to_end_pct        = sites_per_step * in_graph_us_per_site
                          / (step_ms * 1000) * 100

The effective bandwidth defaults to 50 GB/s — a CONSERVATIVE stand-in for
small-tensor traffic on 910B-class parts, NOT a vendor number. Override with
a measured value when you have one; the output always states which was used.

Usage:
    python3 fusion_headroom.py --sites-per-step 60 --tensor-kb 60 \
        --step-ms 33 --eager-us 268 --fused-us 127
    python3 fusion_headroom.py --sites-per-step 60 --tensor-kb 60 \
        --step-ms 33 --bytes-saved-per-site-kb 120
"""
from __future__ import annotations

import argparse
import sys


def compute(sites: int, step_ms: float, bytes_saved_kb: float,
            eff_bw_gbs: float) -> dict:
    bytes_saved = bytes_saved_kb * 1024
    in_graph_us = bytes_saved / (eff_bw_gbs * 1e9) * 1e6
    step_us = step_ms * 1000
    total_us = sites * in_graph_us
    pct = total_us / step_us * 100
    return {
        "bytes_saved_per_site_kb": bytes_saved_kb,
        "in_graph_us_per_site": round(in_graph_us, 3),
        "sites_per_step": sites,
        "saved_us_per_step": round(total_us, 1),
        "step_ms": step_ms,
        "end_to_end_pct": round(pct, 2),
        "effective_bw_gbs_used": eff_bw_gbs,
        "bw_is_measured": False,
    }


def eager_delta(eager_us: float, fused_us: float) -> float:
    """Bytes-equivalent of an eager delta — only valid as an UPPER bound.

    The eager delta includes launch overhead that disappears in-graph; using
    it directly as saved-time would overstate the gain. We report it as the
    ceiling and the traffic model as the estimate.
    """
    return eager_us - fused_us


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Honest end-to-end headroom of a fused kernel, from traffic arithmetic.")
    ap.add_argument("--sites-per-step", type=int, required=True,
                    help="occurrences of the pattern per decode step")
    ap.add_argument("--step-ms", type=float, required=True,
                    help="measured decode step time")
    ap.add_argument("--tensor-kb", type=float, default=None,
                    help="fused-away tensor size in KB (one round-trip is saved)")
    ap.add_argument("--bytes-saved-per-site-kb", type=float, default=None,
                    help="explicit bytes saved per site (overrides --tensor-kb)")
    ap.add_argument("--eager-us", type=float, default=None,
                    help="eager benchmark of the unfused sequence (for the ceiling)")
    ap.add_argument("--fused-us", type=float, default=None,
                    help="eager benchmark of the fused kernel (for the ceiling)")
    ap.add_argument("--eff-bw-gbs", type=float, default=50.0,
                    help="effective small-tensor bandwidth GB/s (default 50, conservative; override with a measurement)")
    ap.add_argument("--wiring-cost-pct", type=float, default=1.0,
                    help="the end-to-end % below which wiring stops (default 1.0)")
    a = ap.parse_args(argv)

    if a.bytes_saved_per_site_kb is not None:
        saved = a.bytes_saved_per_site_kb
    elif a.tensor_kb is not None:
        saved = 2 * a.tensor_kb  # one intermediate round-trip: read + write
    else:
        ap.error("give --tensor-kb or --bytes-saved-per-site-kb")

    r = compute(a.sites_per_step, a.step_ms, saved, a.eff_bw_gbs)

    print("# Fusion headroom (the should-I-wire arithmetic)")
    print()
    print(f"- pattern sites/step:      {r['sites_per_step']}")
    print(f"- bytes saved/site:        {r['bytes_saved_per_site_kb']:.0f} KB "
          f"(one round-trip of the fused-away tensor)")
    print(f"- in-graph time/site:      {r['in_graph_us_per_site']} us  "
          f"@ {r['effective_bw_gbs_used']} GB/s "
          f"({'measured' if r['bw_is_measured'] else 'DEFAULT stand-in — override with a measurement'})")
    print(f"- saved/step:              {r['saved_us_per_step']} us of a "
          f"{r['step_ms']} ms step")
    print(f"- END-TO-END ESTIMATE:     **{r['end_to_end_pct']}%**")
    if a.eager_us is not None and a.fused_us is not None:
        ceiling = eager_delta(a.eager_us, a.fused_us) * a.sites_per_step / (a.step_ms * 1000) * 100
        print(f"- eager ceiling (launches NOT amortized): {ceiling:.2f}%  "
              f"← an overstatement; the traffic model above is the estimate")
    print()
    verdict = "WIRE" if r["end_to_end_pct"] >= a.wiring_cost_pct else "STOP — ship the kernel as an artifact"
    print(f"## Verdict vs wiring-cost threshold {a.wiring_cost_pct}%: {verdict}")
    if "STOP" in verdict:
        print()
        print("The computed payoff is below the wiring cost (restart + gates +")
        print("patch maintenance). Record the kernel as validated and revisit")
        print("at higher batch, where the tensors scale linearly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
