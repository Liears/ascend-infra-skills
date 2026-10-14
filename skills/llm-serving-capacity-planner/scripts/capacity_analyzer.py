#!/usr/bin/env python3
"""
Ascend Serving Capacity Planner — explain where NPU HBM went, from a server log.

Ports the intent of BBuf's `llm-serving-capacity-planner` to Ascend. The log
wording on Ascend still says "GPU" because vllm-ascend inherits vLLM's strings,
but every number is NPU HBM. That naming mismatch is the trap this tool exists
to absorb.

What it reads (exact strings verified against vllm_ascend/worker/worker.py and
vllm/v1/core/kv_cache_utils.py on 2026-10-14):

  Free memory on device (F/T GiB) on startup. Desired GPU memory utilization is
    (U, R GiB). Actual usage: W GiB for weights, A GiB for peak activation,
    N GiB for non-torch memory, G GiB for NPU graph memory.
    ... Current KV cache memory: K GiB.
  Available KV cache memory: K GiB
  GPU KV cache size: T tokens
  Maximum concurrency for L tokens per request: Cx
  Initial free memory F GiB, reserved V GiB for KV Cache ...

Usage:
    python3 capacity_analyzer.py path/to/server.log
    python3 capacity_analyzer.py path/to/server.log --json
    python3 capacity_analyzer.py path/to/server.log --requests 8192 32768

Only the standard library is used.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional


# --------------------------------------------------------------------------
# log line patterns — each anchored on the real strings, not on a guess
# --------------------------------------------------------------------------

RE_FREE_MEMORY = re.compile(
    r"Free memory on device\s*\(\s*([\d.]+)\s*/\s*([\d.]+)\s*GiB\s*\)\s*on startup"
)

RE_UTILIZATION = re.compile(
    r"Desired GPU memory utilization is\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*GiB\s*\)"
)

RE_ACTUAL_USAGE = re.compile(
    r"Actual usage:\s*([\d.]+)\s*GiB for weights,\s*"
    r"([\d.]+)\s*GiB for peak activation,\s*"
    r"([\d.]+)\s*GiB for non-torch memory,\s*"
    r"([\d.]+)\s*GiB for NPU graph memory"
)

RE_AVAILABLE_KV = re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB")

RE_CURRENT_KV = re.compile(r"Current KV cache memory:\s*([\d.]+)\s*GiB")

RE_INITIAL_FREE = re.compile(
    r"Initial free memory\s*([\d.]+)\s*GiB,\s*reserved\s*([\d.]+)\s*GiB for KV Cache"
)

RE_KV_TOKENS = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")

RE_MAX_CONCURRENCY = re.compile(
    r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x"
)

# vllm-ascend also lets you set the KV budget directly instead of a fraction.
RE_KV_CACHE_MEM_ARG = re.compile(r"--kv-cache-memory[=\s]+(\d+)")

# npu-smi style HBM line, in case the log embeds one.
RE_NPU_SMI_HBM = re.compile(r"(\d+)\s*/\s*(\d+)\s*$")


@dataclass
class MemoryBreakdown:
    unit: str = "GiB"
    free_on_startup: Optional[float] = None
    total_on_startup: Optional[float] = None
    gpu_memory_utilization: Optional[float] = None
    requested_memory: Optional[float] = None
    weights: Optional[float] = None
    peak_activation: Optional[float] = None
    non_torch_memory: Optional[float] = None
    npugraph_memory: Optional[float] = None
    available_kv_cache: Optional[float] = None
    kv_cache_memory_bytes_arg: Optional[int] = None
    kv_cache_size_tokens: Optional[int] = None
    max_concurrency: Optional[float] = None
    max_model_len: Optional[int] = None

    def non_kv_total(self) -> Optional[float]:
        parts = [self.weights, self.peak_activation, self.non_torch_memory, self.npugraph_memory]
        if all(p is None for p in parts):
            return None
        return round(sum(p or 0.0 for p in parts), 2)


def parse_log(text: str) -> MemoryBreakdown:
    b = MemoryBreakdown()

    m = RE_FREE_MEMORY.search(text)
    if m:
        b.free_on_startup, b.total_on_startup = float(m.group(1)), float(m.group(2))

    m = RE_UTILIZATION.search(text)
    if m:
        b.gpu_memory_utilization, b.requested_memory = float(m.group(1)), float(m.group(2))

    m = RE_ACTUAL_USAGE.search(text)
    if m:
        b.weights = float(m.group(1))
        b.peak_activation = float(m.group(2))
        b.non_torch_memory = float(m.group(3))
        b.npugraph_memory = float(m.group(4))

    for rx, attr in (
        (RE_AVAILABLE_KV, "available_kv_cache"),
        (RE_CURRENT_KV, "available_kv_cache"),   # only used if the first did not match
    ):
        m = rx.search(text)
        if m and getattr(b, attr) is None:
            setattr(b, attr, float(m.group(1)))

    m = RE_INITIAL_FREE.search(text)
    if m and b.available_kv_cache is None:
        b.available_kv_cache = float(m.group(2))

    m = RE_KV_TOKENS.search(text)
    if m:
        b.kv_cache_size_tokens = int(m.group(1).replace(",", ""))

    m = RE_MAX_CONCURRENCY.search(text)
    if m:
        b.max_model_len = int(m.group(1).replace(",", ""))
        b.max_concurrency = float(m.group(2))

    m = RE_KV_CACHE_MEM_ARG.search(text)
    if m:
        b.kv_cache_memory_bytes_arg = int(m.group(1))

    return b


# --------------------------------------------------------------------------
# capacity math
# --------------------------------------------------------------------------

@dataclass
class Estimate:
    request_len: int
    concurrent_requests: Optional[float]
    basis: str


def estimate_concurrency(b: MemoryBreakdown, lengths: list[int]) -> list[Estimate]:
    """Estimate how many requests of a given length fit in the KV pool.

    There are two different quantities people conflate, so we report both and
    label which is which:

      A. pool capacity / request length — the hard arithmetic upper bound on how
         many *uniform* requests of that exact length fit simultaneously.
      B. reported max_concurrency scaled by length — what the scheduler allows
         at other lengths, derived from the tool's own reference point.

    They agree only when the reference length is the request length. For
    hybrid-attention models (SWA / compressed / local groups) neither is exact:
    per-request block counts are capped by windows, so lower bounds are safer
    than point values. We report A as the headline and B as a cross-check.
    """
    out: list[Estimate] = []
    if b.kv_cache_size_tokens is None:
        for L in lengths:
            out.append(Estimate(L, None, "no `GPU KV cache size` line in the log"))
        return out

    ref_len = b.max_model_len
    ref_conc = b.max_concurrency
    for L in lengths:
        pool_bound = b.kv_cache_size_tokens / L
        if ref_len and ref_conc and L == ref_len:
            # At the reference length the two bases coincide; say so.
            out.append(Estimate(
                L, round(ref_conc, 2),
                f"reported directly ({ref_conc}x at {ref_len} tokens)",
            ))
        elif ref_len and ref_conc:
            scaled = ref_conc * (ref_len / L)
            out.append(Estimate(
                L, round(min(pool_bound, scaled), 2),
                f"pool-bound {pool_bound:.1f} vs scaled {scaled:.1f} "
                f"(from {ref_conc}x @ {ref_len:,}); took the smaller",
            ))
        else:
            out.append(Estimate(
                L, round(pool_bound, 2),
                f"{b.kv_cache_size_tokens:,} tokens / {L} (linear; may be optimistic "
                f"for hybrid-attention models)",
            ))
    return out


def diagnose(b: MemoryBreakdown) -> list[str]:
    """Point at the likely pressure source. Each line cites the number it used."""
    notes: list[str] = []

    if b.weights is None and b.available_kv_cache is None:
        notes.append("no memory lines matched: is this a vllm-ascend/vLLM startup log?")
        return notes

    if b.weights is not None and b.available_kv_cache is not None:
        total = b.weights + (b.peak_activation or 0) + (b.non_torch_memory or 0) \
            + (b.npugraph_memory or 0) + b.available_kv_cache
        share = 100.0 * b.weights / total if total else 0
        notes.append(
            f"weights {b.weights} GiB = {share:.1f}% of the "
            f"{round(total, 2)} GiB accounted for"
        )
        if b.npugraph_memory and b.npugraph_memory > 0.5:
            notes.append(
                f"NPU graph memory is {b.npugraph_memory} GiB — graph capture is a "
                f"first-class consumer on Ascend, not a rounding error"
            )
        if b.non_torch_memory and b.weights and b.non_torch_memory > 0.25 * b.weights:
            notes.append(
                f"non-torch memory {b.non_torch_memory} GiB is high relative to weights "
                f"{b.weights} GiB — check driver/allocator reserves and any host-pinned tables"
            )

    if b.free_on_startup is not None and b.total_on_startup is not None:
        used = b.total_on_startup - b.free_on_startup
        if used > 0.1 * b.total_on_startup:
            notes.append(
                f"{used:.2f} GiB of {b.total_on_startup} GiB was already in use before "
                f"vLLM started — another process holds it; KV headroom is reduced by that much"
            )

    if b.gpu_memory_utilization is not None and b.gpu_memory_utilization > 0.95:
        notes.append(
            f"gpu_memory_utilization={b.gpu_memory_utilization} is very high; on Ascend the "
            f"driver and collective buffers need headroom, so OOM at this setting is expected"
        )

    if b.kv_cache_memory_bytes_arg is not None:
        notes.append(
            f"KV budget was set explicitly (--kv-cache-memory={b.kv_cache_memory_bytes_arg}); "
            f"the utilization fraction does not govern the pool in this mode"
        )

    return notes


def render(b: MemoryBreakdown, est: list[Estimate], notes: list[str]) -> str:
    L: list[str] = []
    L.append("# NPU HBM accounting (vllm-ascend startup log)")
    L.append("")
    L.append("> All figures are NPU HBM. The log wording says \"GPU\" because")
    L.append("> vllm-ascend inherits vLLM's strings; read them as NPU.")
    L.append("")

    def row(label: str, val: Optional[float], unit: str = "GiB") -> str:
        return f"| {label} | {('%.2f' % val) if val is not None else '—'} | {unit if val is not None else ''} |"

    L.append("| Quantity | Value | Unit |")
    L.append("|---|---:|---|")
    L.append(row("HBM free at startup", b.free_on_startup))
    L.append(row("HBM total", b.total_on_startup))
    L.append(row("gpu_memory_utilization", b.gpu_memory_utilization, ""))
    L.append(row("requested memory", b.requested_memory))
    L.append(row("weights", b.weights))
    L.append(row("peak activation", b.peak_activation))
    L.append(row("non-torch memory", b.non_torch_memory))
    L.append(row("NPU graph memory", b.npugraph_memory))
    L.append(row("non-KV subtotal", b.non_kv_total()))
    L.append(row("available KV cache", b.available_kv_cache))
    L.append("")

    if b.kv_cache_size_tokens is not None:
        L.append(f"- KV pool capacity: **{b.kv_cache_size_tokens:,} tokens**")
    else:
        L.append("- KV pool capacity: not stated in this log")
    if b.max_concurrency is not None:
        L.append(f"- Reported max concurrency for {b.max_model_len:,} tokens/request: **{b.max_concurrency}x**")
    L.append("")

    L.append("## Concurrency estimate")
    L.append("")
    L.append("| request length | est. concurrent requests | basis |")
    L.append("|---:|---:|---|")
    for e in est:
        val = f"{e.concurrent_requests:.2f}" if e.concurrent_requests is not None else "—"
        L.append(f"| {e.request_len:,} | {val} | {e.basis} |")
    L.append("")

    L.append("## Diagnosis")
    for n in notes:
        L.append(f"- {n}")
    L.append("")

    L.append("---")
    L.append("Estimates are derived from the log's own numbers. For hybrid-attention")
    L.append("models a linear tokens/length division understates capacity; trust the")
    L.append("reported `Maximum concurrency` line over any division you do yourself.")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Explain NPU HBM use from a vllm-ascend/vLLM startup log.")
    ap.add_argument("log", help="path to the server log (stdin with '-')")
    ap.add_argument("--requests", type=int, nargs="+", default=[2048, 8192, 32768],
                    help="request lengths (tokens) to estimate concurrency for")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.log == "-":
        text = sys.stdin.read()
    else:
        try:
            with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError as e:
            print(f"cannot read {args.log}: {e}", file=sys.stderr)
            return 2

    b = parse_log(text)
    est = estimate_concurrency(b, args.requests)
    notes = diagnose(b)

    if args.json:
        print(json.dumps(
            {"breakdown": asdict(b), "estimates": [asdict(e) for e in est], "diagnosis": notes},
            indent=2, ensure_ascii=False,
        ))
    else:
        print(render(b, est, notes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
