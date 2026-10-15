#!/usr/bin/env python3
"""
Communication-volume model — the theory half of framework-level tuning.

Ports the communication side of the theoretical bound that
`model_compute_sim.py` (FLOPs) does not cover. For a serving decode step it
answers two different questions, and at small batch the second one rules:

  1. BYTES per step — bounds comm time when bandwidth-bound (large batch)
  2. COLLECTIVE COUNT per step — bounds comm time when latency-bound
     (small batch: a ~10 KB allReduce costs the same ~0.4 ms as a large one)

The latency-bound regime is the one our real deployment sits in (see
references/measured-anchor.md): single-stream decode measured 51.7 ms/step of
communication over ~120 collectives ≈ 0.43 ms each — message sizes are ~10 KB,
far below the bandwidth ceiling. Reducing BYTES does nothing there; reducing
COUNT or hiding the latency does.

Layouts modelled (per layer, per token; R = ranks, E = experts, K = top-k):

  tp          pure TP: allReduce after attn + allReduce after MLP
              → 2 collectives/layer
  tp-ep       TP attention + EP MoE via alltoall dispatch/combine (our
              deployment) → 3 collectives/layer
  tp-ep-ag    MoE dispatch/combine as allgather/reduce-scatter (the port
              repo's patch-0001 variant) → 3 collectives/layer, different
              bytes
  dp-ep       DP attention (no attn allReduce — each rank holds the whole
              attention) + EP MoE alltoall → 2 collectives/layer, but
              attention weights are replicated R× (feasibility cost)

Formulas are stated inline; every constant is either derived from the config
or an explicit flag. Nothing is hidden.

Usage:
    python3 comm_volume.py --config config.json --layout tp-ep \
        --ranks 8 --positions-per-step 6
    python3 comm_volume.py --config config.json --layout tp-ep \
        --latency-us 430 --measured-comm-ms 51.7
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if "text_config" in d and isinstance(d["text_config"], dict):
        merged = dict(d)
        merged.update(d["text_config"])
        return merged
    return d


def g(cfg: dict, *names, default=None):
    for n in names:
        if n in cfg and cfg[n] is not None:
            return cfg[n]
    return default


@dataclass
class LayoutModel:
    name: str
    collectives_per_layer: int
    bytes_per_token_per_layer: float
    notes: list = field(default_factory=list)


def expected_distinct_ranks(ranks: int, topk: int) -> float:
    """E[number of distinct destination ranks] for topk experts uniform over ranks.

    P(a given rank holds >=1 of the topk) = 1 - (1-1/R)^K.
    """
    return ranks * (1.0 - (1.0 - 1.0 / ranks) ** topk)


def model_layouts(cfg: dict, ranks: int, dtype_bytes: int = 2) -> dict[str, LayoutModel]:
    H = g(cfg, "hidden_size", default=0)
    L = g(cfg, "num_hidden_layers", default=0)
    E = g(cfg, "n_routed_experts", "num_experts", default=0)
    K = g(cfg, "num_experts_per_tok", default=0)
    token_bytes = H * dtype_bytes  # one hidden-state copy
    # Ring allReduce moves ~2*(R-1)/R of the message per rank.
    ring_factor = 2.0 * (ranks - 1) / ranks
    dist = expected_distinct_ranks(ranks, K) if E else 0.0

    out = {}
    if E and K:
        out["tp-ep"] = LayoutModel(
            name="tp-ep (TP attention + EP MoE alltoall) — our deployment",
            collectives_per_layer=3,
            bytes_per_token_per_layer=ring_factor * token_bytes + 2 * dist * token_bytes,
            notes=[
                f"attn allReduce: ring ~{ring_factor:.2f}x msg = {ring_factor*token_bytes:,.0f} B",
                f"MoE dispatch+combine: 2 x E[distinct ranks]={dist:.2f} x {token_bytes:,} B "
                f"= {2*dist*token_bytes:,.0f} B",
                "at small batch this is LATENCY-bound: 3 collectives/layer regardless of bytes",
            ])
        out["tp-ep-ag"] = LayoutModel(
            name="tp-ep-ag (MoE via allgather/reduce-scatter, port patch-0001)",
            collectives_per_layer=3,
            bytes_per_token_per_layer=ring_factor * token_bytes + 2 * (ranks - 1) / ranks * ranks * token_bytes / ranks * ranks,
            notes=[
                "same collective count as tp-ep; different byte pattern",
                "allgather delivers all tokens to every rank: (R-1) x token_bytes received",
            ])
        out["dp-ep"] = LayoutModel(
            name="dp-ep (DP attention + EP MoE) — no attn allReduce",
            collectives_per_layer=2,
            bytes_per_token_per_layer=2 * dist * token_bytes,
            notes=[
                "attention runs replicated: ZERO attn collectives per layer",
                f"cost: attention weights replicated {ranks}x (feasibility, not comm)",
                "2 collectives/layer vs 3 — at latency-bound small batch this is the lever",
            ])
    out["tp"] = LayoutModel(
        name="tp (pure TP, experts sharded inside)",
        collectives_per_layer=2,
        bytes_per_token_per_layer=2 * ring_factor * token_bytes,
        notes=[
            "2 allReduce/layer (attn + MLP); no alltoall at all",
            "every rank touches a shard of EVERY activated expert — memory reads "
            "scale with distinct experts, same total bytes as EP for small batch",
        ])
    for m in out.values():
        m.notes.append(f"token_bytes(1 hidden copy, bf16) = {token_bytes:,} B")
    return out


def render(cfg: dict, models: dict, ranks: int, positions: int,
           latency_us: float | None, measured_ms: float | None) -> str:
    L = g(cfg, "num_hidden_layers", default=0)
    H = g(cfg, "hidden_size", default=0)
    E = g(cfg, "n_routed_experts", default=0)
    K = g(cfg, "num_experts_per_tok", default=0)
    lines = []
    lines.append("# Communication model — theory half of framework tuning")
    lines.append("")
    lines.append(f"- config: hidden={H} layers={L} experts={E} topk={K} ranks={ranks}")
    lines.append(f"- positions per decode step (spec-decode draft+verify): {positions}")
    lines.append("")
    lines.append("| layout | collectives/step | bytes/step/rank | notes |")
    lines.append("|---|---:|---:|---|")
    for name, m in models.items():
        cols = m.collectives_per_layer * L
        bts = m.bytes_per_token_per_layer * L * positions
        lines.append(f"| `{name}` | **{cols}** | {bts/1e6:.1f} MB | {'; '.join(m.notes[:2])} |")
    lines.append("")

    if latency_us:
        lines.append("## Latency-bound projection (the regime that matters at small batch)")
        lines.append("")
        lines.append(f"- per-collective latency: {latency_us:.0f} µs "
                     + ("(derived from our measured trace)" if measured_ms else "(input)"))
        lines.append("")
        lines.append("| layout | predicted comm ms/step | vs current |")
        lines.append("|---|---:|---:|")
        base = None
        preds = {}
        for name, m in models.items():
            pred = m.collectives_per_layer * L * latency_us / 1000.0
            preds[name] = pred
            if base is None:
                base = pred
        for name, pred in preds.items():
            cur = preds.get("tp-ep", base)
            lines.append(f"| `{name}` | {pred:.1f} | {(pred/cur-1)*100:+.0f}% |")
        lines.append("")
        if measured_ms:
            cur = preds.get("tp-ep")
            if cur is not None:
                lines.append(f"- check against measurement: predicted {cur:.1f} ms vs "
                             f"measured {measured_ms:.1f} ms comm/step — "
                             f"{'consistent (latency-bound confirmed)' if abs(cur-measured_ms)/measured_ms < 0.35 else 'gap >35%: revisit the per-collective latency or the collective count'}")
        lines.append("")
        lines.append("**Reading:** at small batch, the layout with fewer collectives per")
        lines.append("layer wins regardless of bytes. Bytes only start to matter when")
        lines.append("message sizes reach the bandwidth ceiling (large batch).")
    lines.append("")
    lines.append("---")
    lines.append("Formulas and the measured anchor are documented in")
    lines.append("references/measured-anchor.md. No bandwidth or peak numbers are")
    lines.append("assumed anywhere; the latency input is derived from measurement.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Model per-step communication volume and collective count by layout.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--layout", default=None, help="restrict to one layout (default: all)")
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--positions-per-step", type=int, default=6,
                    help="token positions per decode step (spec draft+verify)")
    ap.add_argument("--latency-us", type=float, default=None,
                    help="per-collective latency; ours is derived as 51.7ms/120 ~= 430us")
    ap.add_argument("--measured-comm-ms", type=float, default=None,
                    help="measured comm ms/step, for the consistency check")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_cfg(args.config)
    models = model_layouts(cfg, args.ranks)
    if args.layout:
        if args.layout not in models:
            ap.error(f"unknown layout {args.layout}; have {sorted(models)}")
        models = {args.layout: models[args.layout]}

    if args.json:
        L = g(cfg, "num_hidden_layers", default=0)
        out = {n: {**asdict(m),
                   "collectives_per_step": m.collectives_per_layer * L,
                   "bytes_per_step_rank": m.bytes_per_token_per_layer * L * args.positions_per_step}
               for n, m in models.items()}
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(render(cfg, models, args.ranks, args.positions_per_step,
                     args.latency_us, args.measured_comm_ms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
