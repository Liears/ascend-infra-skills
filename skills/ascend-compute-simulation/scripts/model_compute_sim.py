#!/usr/bin/env python3
"""
Ascend Compute Simulation — estimate per-op FLOPs and MFU for a serving shape.

Pure arithmetic: reads a model config, builds the representative operator
sequence, prints shapes and FLOPs, and (given a measured latency and a peak)
estimates MFU. Ports the intent of BBuf's `model-compute-simulation`.

**The peak-FLOPS number is never assumed.** Ascend per-SKU peaks are reported
inconsistently across public sources, and the evidence rules forbid inventing
one. You must pass `--peak-tflops` (with its source), or the tool reports FLOPs
and declines to compute MFU. This is deliberate: an MFU number built on a wrong
peak is worse than no MFU number.

Usage:
    python3 model_compute_sim.py --config /path/to/config.json --batch 1 --seq 1
    python3 model_compute_sim.py --config config.json --batch 1 --seq 4096 \\
        --latency-ms 31.6 --peak-tflops 313 --peak-source "vendor datasheet, 910B3 FP16"
    python3 model_compute_sim.py --config config.json --tp 8 --ep 8 --json

Config: a HuggingFace-style config.json. Text towers wrapped under `text_config`
are Unwrapped. Only the standard library is used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional


# --------------------------------------------------------------------------
# config loading
# --------------------------------------------------------------------------

def normalize_config(cfg: dict) -> dict:
    """Expand a wrapped HF config into a flat dict of the text model's fields.

    Multimodal VLM configs nest the language model under `text_config`. Model
    code reads the nested block; a simulator that reads only the top level sees
    zeros and produces silently wrong FLOPs.
    """
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        merged = dict(cfg)
        merged.update(cfg["text_config"])
        return merged
    return cfg


def g(cfg: dict, *names, default=None):
    for n in names:
        if n in cfg and cfg[n] is not None:
            return cfg[n]
    return default


# --------------------------------------------------------------------------
# the FLOP model
# --------------------------------------------------------------------------

@dataclass
class Op:
    name: str
    shape: str
    flops: float
    note: str = ""


@dataclass
class Result:
    model_type: str = ""
    batch: int = 1
    seq: int = 1
    layers: int = 0
    params_total_b: float = 0.0
    ops: list = field(default_factory=list)
    flops_total: float = 0.0
    flops_per_token: float = 0.0
    mfu_pct: Optional[float] = None
    peak_tflops: Optional[float] = None
    peak_source: str = ""
    tp: int = 1
    ep: int = 1
    warnings: list = field(default_factory=list)


def build_ops(cfg: dict, B: int, S: int) -> tuple[list[Op], dict]:
    ops: list[Op] = []
    facts: dict = {}

    H = g(cfg, "hidden_size", default=0)
    L = g(cfg, "num_hidden_layers", default=0)
    V = g(cfg, "vocab_size", default=0)
    n_heads = g(cfg, "num_attention_heads", default=0)
    n_kv = g(cfg, "num_key_value_heads", default=n_heads)
    head_dim = g(cfg, "head_dim", default=(H // n_heads if n_heads else 0))
    inter = g(cfg, "intermediate_size", default=0)

    # MoE
    n_experts = g(cfg, "n_routed_experts", "num_experts", "num_local_experts", default=0)
    topk = g(cfg, "num_experts_per_tok", "num_experts_per_token", default=0)
    moe_inter = g(cfg, "moe_intermediate_size", default=inter)
    n_shared = g(cfg, "n_shared_experts", default=0)

    # MLA-style latent projections (DeepSeek family)
    q_lora = g(cfg, "q_lora_rank", default=0)
    kv_lora = g(cfg, "kv_lora_rank", default=0)

    facts.update(dict(hidden=H, layers=L, vocab=V, heads=n_heads, kv_heads=n_kv,
                      head_dim=head_dim, experts=n_experts, topk=topk,
                      moe_inter=moe_inter, shared=n_shared,
                      q_lora=q_lora, kv_lora=kv_lora))

    if not H or not L:
        return ops, facts

    tokens = B * S

    # --- Attention projections (per layer) ---
    if q_lora:
        # q down-projection then up-projection
        ops.append(Op("attn.q_a_proj", f"[{tokens},{H}]x[{H},{q_lora}]",
                      2 * tokens * H * q_lora, "MLA q down"))
        ops.append(Op("attn.q_b_proj", f"[{tokens},{q_lora}]x[{q_lora},{n_heads*head_dim}]",
                      2 * tokens * q_lora * n_heads * head_dim, "MLA q up"))
    else:
        ops.append(Op("attn.q_proj", f"[{tokens},{H}]x[{H},{n_heads*head_dim}]",
                      2 * tokens * H * n_heads * head_dim))
    if kv_lora:
        ops.append(Op("attn.kv_a_proj", f"[{tokens},{H}]x[{H},{kv_lora}]",
                      2 * tokens * H * kv_lora, "MLA kv down"))
        ops.append(Op("attn.kv_b_proj", f"[{tokens},{kv_lora}]x[{kv_lora},{(n_heads+2)*head_dim}]",
                      2 * tokens * kv_lora * (n_heads + 2) * head_dim, "MLA kv up"))
    else:
        ops.append(Op("attn.kv_proj", f"[{tokens},{H}]x[{H},{2*n_kv*head_dim}]",
                      2 * tokens * H * 2 * n_kv * head_dim))
    ops.append(Op("attn.o_proj", f"[{tokens},{n_heads*head_dim}]x[{n_heads*head_dim},{H}]",
                  2 * tokens * n_heads * head_dim * H))

    # --- Attention scores (decode vs prefill differ) ---
    # QK^T + AV. Causal halves the prefill average; decode is S=1.
    if S > 1:
        kv_len = S
        scores = 2 * 2 * B * n_heads * head_dim * kv_len * S / 2  # 2 matmuls, causal /2
        ops.append(Op("attn.qk_av", f"[{B},{n_heads},{S},{head_dim}] causal",
                      scores, "causal average over prefill"))
    else:
        # decode attends to the full context; KV length supplied separately
        ctx = g(cfg, "_kv_len", default=S)
        ops.append(Op("attn.qk_av", f"[{B},{n_heads},{1},{head_dim}] x ctx {ctx}",
                      2 * 2 * B * n_heads * head_dim * ctx, "decode: KV length = context"))

    # --- FFN / MoE ---
    if n_experts and topk:
        routed = n_experts / max(topk, 1)
        ops.append(Op("ffn.experts.gate_up",
                      f"topk={topk} x [{tokens},{H}]x[{H},{2*moe_inter}]",
                      2 * tokens * H * 2 * moe_inter * topk, "per token: topk experts"))
        ops.append(Op("ffn.experts.down",
                      f"topk={topk} x [{tokens},{moe_inter}]x[{moe_inter},{H}]",
                      2 * tokens * moe_inter * H * topk, "per token: topk experts"))
        if n_shared:
            # Shared experts on MoE-only configs (DeepSeek family) size to
            # moe_intermediate_size; the top-level `intermediate_size` is often
            # absent there. Falling back to `inter` alone silently produced
            # zero-FLOP shared experts, which understated the model.
            shared_inter = inter or moe_inter
            ops.append(Op("ffn.shared.gate_up", f"[{tokens},{H}]x[{H},{2*shared_inter}]",
                          2 * tokens * H * 2 * shared_inter, f"{n_shared} shared expert(s)"))
            ops.append(Op("ffn.shared.down", f"[{tokens},{shared_inter}]x[{shared_inter},{H}]",
                          2 * tokens * shared_inter * H))
        ops.append(Op("moe.router", f"[{tokens},{H}]x[{H},{n_experts}]",
                      2 * tokens * H * n_experts, "routing gate"))
    else:
        ops.append(Op("ffn.gate_up", f"[{tokens},{H}]x[{H},{2*inter}]",
                      2 * tokens * H * 2 * inter))
        ops.append(Op("ffn.down", f"[{tokens},{inter}]x[{inter},{H}]",
                      2 * tokens * inter * H))

    return ops, facts


def summarize(cfg: dict, ops: list[Op], facts: dict, B: int, S: int,
              latency_ms: Optional[float], peak_tflops: Optional[float],
              peak_source: str, tp: int, ep: int) -> Result:
    r = Result(model_type=str(cfg.get("model_type", "")), batch=B, seq=S,
               peak_tflops=peak_tflops, peak_source=peak_source, tp=tp, ep=ep)
    r.layers = int(facts.get("layers") or 0)

    # Per-layer FLOPs = sum of the ops built above; total multiplies by layers,
    # except the lm_head which is counted once.
    per_layer = sum(o.flops for o in ops)
    r.flops_total = per_layer * r.layers
    # Copy: appending lm_head below must not mutate the caller's list, or a
    # second summarize() on the same ops would count lm_head twice.
    r.ops = list(ops)

    V = int(facts.get("vocab") or 0)
    H = int(facts.get("hidden") or 0)
    if V and H:
        lm = Op("lm_head", f"[{B*S},{H}]x[{H},{V}]", 2 * B * S * V * H, "once, not per layer")
        r.ops.append(lm)
        r.flops_total += lm.flops

    r.flops_per_token = (r.flops_total / (B * S)) if (B * S) else 0.0

    # Parameter estimate for a sanity cross-check (weights FLOPs / 2 per token).
    # Only an order-of-magnitude check; not a substitute for reading the safetensors index.
    if H and V:
        pass

    if latency_ms and peak_tflops:
        # MFU = achieved TFLOPS / peak. Achieved = FLOPs / time.
        achieved_tflops = r.flops_total / (latency_ms / 1000.0) / 1e12
        r.mfu_pct = 100.0 * achieved_tflops / peak_tflops
    elif latency_ms and not peak_tflops:
        r.warnings.append("latency given but no --peak-tflops: MFU not computed "
                          "(Ascend SKU peaks are not assumed by this tool)")
    elif peak_tflops and not latency_ms:
        r.warnings.append("peak given but no --latency-ms: MFU needs a measured time")

    if tp > 1 or ep > 1:
        r.warnings.append(
            f"TP={tp} EP={ep} are recorded but NOT applied to the FLOPs above. "
            f"Naively dividing FLOPs by TP/EP misstates per-device work because "
            f"routers, norms and shared experts are replicated, not sharded. "
            f"Use the per-op table against a real trace instead."
        )
    if facts.get("experts"):
        r.warnings.append(
            f"MoE with {facts['experts']} experts, top-{facts['topk']}/token: the "
            f"FLOPs assume every token activates exactly top-k experts. Real "
            f"dispatch, padding and capacity factors change this; treat as an "
            f"upper bound on useful work."
        )
    return r


def render(r: Result) -> str:
    L: list[str] = []
    L.append(f"# Compute estimate — {r.model_type or 'model'}")
    L.append("")
    L.append(f"- batch={r.batch} seq={r.seq} layers={r.layers}")
    L.append(f"- total FLOPs: {r.flops_total/1e12:.2f} TFLOPs "
             f"({r.flops_total/1e9:.1f} GFLOPs)")
    L.append(f"- FLOPs per token: {r.flops_per_token/1e9:.2f} GFLOP/token "
             f"(the 2x a weight-based estimate gives, i.e. a MAC = 2 FLOPs)")
    L.append("")

    L.append("## Per-op (per layer, except lm_head)")
    L.append("")
    L.append("| op | shape | GFLOPs | note |")
    L.append("|---|---|---:|---|")
    for o in r.ops:
        L.append(f"| {o.name} | `{o.shape}` | {o.flops/1e9:.2f} | {o.note} |")
    L.append("")

    L.append("## MFU")
    L.append("")
    if r.mfu_pct is not None:
        L.append(f"- peak: {r.peak_tflops} TFLOPs  ({r.peak_source or 'source not stated'})")
        L.append(f"- achieved MFU: **{r.mfu_pct:.1f}%**")
    else:
        L.append("- not computed. Ascend per-SKU peaks are reported inconsistently")
        L.append("  across public sources, so this tool will not assume one. Pass")
        L.append("  `--peak-tflops <n> --peak-source \"<where you read it>\"` along")
        L.append("  with `--latency-ms` to get MFU.")
    L.append("")

    if r.warnings:
        L.append("## Warnings")
        for w in r.warnings:
            L.append(f"- {w}")
        L.append("")

    L.append("---")
    L.append("FLOPs are a static, uniform-topology estimate. They are a *what-if*, "
             "not a measurement. For anything you will act on, compare the per-op "
             "table against a real `kernel_details.csv`.")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Estimate model FLOPs and MFU for an Ascend serving shape.")
    ap.add_argument("--config", required=True, help="path to config.json")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=1, help="sequence length (1 = decode)")
    ap.add_argument("--kv-len", type=int, default=None,
                    help="context length for decode attention (defaults to --seq)")
    ap.add_argument("--latency-ms", type=float, default=None)
    ap.add_argument("--peak-tflops", type=float, default=None,
                    help="per-chip peak; REQUIRED for MFU, never assumed")
    ap.add_argument("--peak-source", default="", help="where the peak came from")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        with open(args.config, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"cannot read config {args.config}: {e}", file=sys.stderr)
        return 2

    cfg = normalize_config(cfg)
    if args.kv_len is not None:
        cfg["_kv_len"] = args.kv_len

    ops, facts = build_ops(cfg, args.batch, args.seq)
    if not ops:
        print("config lacks hidden_size/num_hidden_layers; cannot build a topology",
              file=sys.stderr)
        return 2

    r = summarize(cfg, ops, facts, args.batch, args.seq, args.latency_ms,
                  args.peak_tflops, args.peak_source, args.tp, args.ep)

    if args.json:
        print(json.dumps(asdict(r), indent=2, ensure_ascii=False))
    else:
        print(render(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
