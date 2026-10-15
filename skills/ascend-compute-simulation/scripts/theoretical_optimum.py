#!/usr/bin/env python3
"""
Theoretical optimum — compose the model's necessary work with MEASURED
operator capability into an achievable per-step bound, and decompose the gap
to the measured step time.

The bound is achievable, not paper: instead of peak FLOPS (nobody reaches
it), compute time uses the BEST-OBSERVED throughput of each heavy op class
from a real trace; communication uses the measured per-collective latency;
memory uses the necessary weight bytes derived from the config.

    theoretical step ≈ max(compute, memory)          # perfect overlap
                       + comm                          # exposed, latency-bound
    gap = measured − theoretical, decomposed

Usage:
    python3 theoretical_optimum.py --config cfg.json \
        --kernel-details-csv kernel_details.csv \
        --measured-step-ms 62.32 --ranks 8 --positions-per-step 6
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from comm_volume import load_cfg, g, model_layouts, expected_distinct_ranks  # noqa: E402

# Heavy compute classes we look for in the trace (name substrings).
GEMM_PATTERNS = ("GroupedMatmul", "Matmul", "MatMul", "Gemm")
ATTN_PATTERNS = ("SparseFlashMla", "FlashAttention", "Attention", "Mla")


def best_op_capability(csv_path: str) -> dict:
    """Per heavy-op-class best observed throughput from a real trace.

    'Capability' = the fastest demonstrated instances in the trace, taken as
    the p10 duration per (class, shape) group — what the op has PROVEN it
    can do on this hardware, not a vendor peak.
    """
    groups: dict[tuple, list] = {}
    with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            name = row.get("Name") or ""
            cls = next((c for p, c in
                        [(p, "gemm") for p in GEMM_PATTERNS] +
                        [(p, "attn") for p in ATTN_PATTERNS]
                        if p in name), None)
            if not cls:
                continue
            try:
                dur = float(row.get("Duration(us)") or 0)
                shapes = (row.get("Input Shapes") or "").strip()
            except ValueError:
                continue
            groups.setdefault((cls, shapes), []).append(dur)
    out = {}
    for (cls, shapes), durs in groups.items():
        durs.sort()
        p10 = durs[max(0, len(durs) // 10)]
        out.setdefault(cls, []).append((p10, shapes, len(durs)))
    cap = {}
    for cls, items in out.items():
        items.sort()
        cap[cls] = {"best_p10_us": items[0][0],
                    "shape": items[0][1][:60], "n_instances": items[0][2],
                    "n_groups": len(items)}
    return cap


def necessary_memory_bytes(cfg: dict, ranks: int, positions: int,
                           w_bytes: float = 0.56) -> float:
    """Weight bytes a rank MUST read per decode step (W4A8 ≈ 0.56 B/param).

    Per layer: distinct local experts hit (E[hit] ≈ E[distinct]/R) read in
    full; shared expert + attention shard read once regardless.
    """
    H = g(cfg, "hidden_size", default=0)
    I = g(cfg, "moe_intermediate_size", default=0)
    L = g(cfg, "num_hidden_layers", default=0)
    E = g(cfg, "n_routed_experts", default=0)
    K = g(cfg, "num_experts_per_tok", default=0)
    if not (H and I and L):
        return 0.0
    expert_params = 3 * H * I
    if E and K:
        distinct = expected_distinct_ranks(ranks, K)  # across ranks
        local_hits = max(1.0, distinct)  # whole experts live on one rank
        moe = local_hits * expert_params * w_bytes
    else:
        moe = 0.0
    # attention + shared: sharded 1/ranks, read once per step
    attn_shared = (expert_params + 64 * 512 * H) * w_bytes / ranks
    return (moe + attn_shared) * L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--kernel-details-csv", required=True)
    ap.add_argument("--measured-step-ms", type=float, required=True)
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--positions-per-step", type=int, default=6)
    ap.add_argument("--per-collective-us", type=float, default=430.0,
                    help="derived from our trace: 51.7ms/120 (see measured-anchor.md)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_cfg(a.config)
    cap = best_op_capability(a.kernel_details_csv)
    L = g(cfg, "num_hidden_layers", default=0)

    # comm: current layout (tp-ep), latency-bound
    models = model_layouts(cfg, a.ranks)
    comm = models["tp-ep"]
    comm_ms = comm.collectives_per_layer * L * a.per_collective_us / 1000.0

    # memory floor: necessary bytes; effective BW derived, not assumed —
    # we report bytes and let the reader compare against a sourced BW.
    mem_bytes = necessary_memory_bytes(cfg, a.ranks, a.positions_per_step)

    # compute floor from measured op capability: best GEMM p10 x instance
    # count per step is trace-workload-specific; report capability itself.
    measured = a.measured_step_ms
    theoretical_overlap = comm_ms  # max(compute,mem) unknown w/o BW source;
    # with comm latency-bound and 83% share, comm dominates the bound.

    lines = []
    lines.append("# Theoretical optimum (achievable bound)")
    lines.append("")
    lines.append(f"- measured step: **{measured:.2f} ms**")
    lines.append(f"- comm bound: {comm_ms:.1f} ms = {comm.collectives_per_layer*L} collectives "
                 f"x {a.per_collective_us:.0f} us (latency-bound, measured anchor)")
    lines.append(f"- necessary weight reads/step/rank: {mem_bytes/1e9:.2f} GB "
                 f"(theory; needs a SOURCED HBM BW to convert to ms — not assumed)")
    if "gemm" in cap:
        c = cap["gemm"]
        lines.append(f"- best demonstrated GEMM instance: {c['best_p10_us']:.1f} us "
                     f"(p10 of {c['n_instances']} @ {c['shape']})")
    if "attn" in cap:
        c = cap["attn"]
        lines.append(f"- best demonstrated attention instance: {c['best_p10_us']:.1f} us "
                     f"(p10 of {c['n_instances']} @ {c['shape']})")
    gap = measured - comm_ms
    lines.append("")
    lines.append(f"## Gap decomposition (measured {measured:.1f} − comm bound {comm_ms:.1f})")
    lines.append(f"- residual for compute+memory+exposure: **{gap:.1f} ms "
                 f"({gap/measured*100:.0f}% of step)**")
    lines.append("")
    lines.append("Reading: at latency-bound small batch the comm count IS the")
    lines.append("theoretical floor under the current layout. Layouts with fewer")
    lines.append("collectives/layer (dp-ep, tp) lower the floor itself (−33% projected).")
    lines.append("Compute/memory sit inside the residual; price them with a sourced")
    lines.append("BW and per-op micro-benchmarks before kernel work.")
    print("\n".join(lines) if not a.json else json.dumps(
        {"measured_step_ms": measured, "comm_bound_ms": comm_ms,
         "necessary_mem_bytes": mem_bytes, "op_capability": cap}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
