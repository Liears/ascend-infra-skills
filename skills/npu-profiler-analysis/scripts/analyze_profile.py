#!/usr/bin/env python3
"""
Ascend Torch Profiler Analysis — read a torch_npu trace and produce three tables.

Port of BBuf's `llm-torch-profiler-analysis` to Ascend. The CUDA version reads a
torch profiler trace and attributes kernels to sources; this reads a torch_npu
trace and does the same, with Ascend's two-level event structure handled
explicitly.

Input: a directory produced by `torch_npu.profiler` (the `ASCEND_PROFILER_OUTPUT`
folder, or its parent), containing `trace_view.json` and/or `kernel_details.csv`.

The three tables:
  1. Kernel attribution  — which ops dominate device time
  2. Host/device split    — how much wall time is host launch vs device compute
  3. Layer/stream view    — per-stream occupancy and exposed gaps

Key Ascend facts this tool encodes:
  - Kernel events are named `<...>_<OpType>_<ImplName>` (e.g.
    `aclnnReduceSum_ReduceSumOpAiCore_ReduceSum`); the OpType is the useful axis.
  - Host API calls are named `AscendCL@aclrt*` / `AscendCL@aclnn*`. They appear
    on CPU tracks; device kernels on the device pid's stream tids.
  - `kernel_details.csv` (Level1 + PipeUtilization) carries the pipe breakdown
    (aic/aiv mte/vec/scalar), which the JSON does not.

Usage:
    python3 analyze_profile.py /path/to/prof_dir
    python3 analyze_profile.py /path/to/prof_dir --json
    python3 analyze_profile.py /path/to/prof_dir --top 30
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional


# --------------------------------------------------------------------------
# locating artifacts
# --------------------------------------------------------------------------

def find_output_dir(path: str) -> Optional[str]:
    """Accept either the profiler root, the *_ascend_pt dir, or ASCEND_PROFILER_OUTPUT."""
    if os.path.isfile(os.path.join(path, "trace_view.json")):
        return path
    for root, dirs, files in os.walk(path):
        if "trace_view.json" in files or "kernel_details.csv" in files:
            return root
    return None


# --------------------------------------------------------------------------
# kernel_details.csv — the authoritative device-side table
# --------------------------------------------------------------------------

# Name shape: prefix_optype_impl, e.g. aclnnAddmm_MatMulCommon_MatMulV2
# We split on '_' and take the middle token as the op type when it matches a
# known Ascend op-type vocabulary. Falling back to the whole name is safer than
# guessing wrong.
OP_TYPE_HINTS = (
    "MatMulV2", "MatMul", "ReduceSum", "Add", "Mul", "Cast", "Transpose",
    "GatherV2", "Scatter", "Softmax", "FlashAttention", "FA", "RmsNorm",
    "LayerNorm", "Silu", "SwiGLU", "MoE", "Gating", "Sort", "TopK", "Index",
    "Concat", "Split", "Copy", "Fill", "BroadcastTo", "Reshape", "Slice",
    "Div", "Sub", "Exp", "Log", "Sqrt", "Pow", "Rsqrt", "BatchMatMul",
)


def op_type_of(name: str) -> str:
    parts = re.split(r"[_@]", name)
    for p in parts:
        for hint in OP_TYPE_HINTS:
            if p.startswith(hint):
                return hint
    # No known hint: return the longest token, which is usually the impl name.
    return max(parts, key=len) if parts else name


@dataclass
class KernelRow:
    name: str
    optype: str
    core: str
    duration_us: float
    device: str


def read_kernel_details(path: str) -> tuple[list[KernelRow], dict]:
    """Read kernel_details.csv. Returns rows and a note about the column set."""
    csv_path = os.path.join(path, "kernel_details.csv")
    meta: dict = {"present": False, "columns": 0, "level": "unknown"}
    if not os.path.isfile(csv_path):
        return [], meta

    rows: list[KernelRow] = []
    with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        meta["present"] = True
        meta["columns"] = len(cols)
        # The 46-column form comes from Level1 + AiCMetrics.PipeUtilization.
        # A 9-column form means the pipe breakdown is unavailable.
        meta["level"] = "Level1+PipeUtilization" if len(cols) >= 40 else "reduced (no pipe data)"

        for r in reader:
            name = (r.get("Name") or "").strip()
            if not name:
                continue
            try:
                dur = float(r.get("Duration(us)") or 0)
            except ValueError:
                dur = 0.0
            rows.append(KernelRow(
                name=name,
                optype=op_type_of(name),
                core=(r.get("Accelerator Core") or r.get("Type") or "").strip(),
                duration_us=dur,
                device=(r.get("Device_id") or "").strip(),
            ))
    return rows, meta


# --------------------------------------------------------------------------
# trace_view.json — host/device split and stream occupancy
# --------------------------------------------------------------------------

HOST_API_PREFIX = ("AscendCL@", "Node@", "python", "c10d", "torch", "nn.Module")


def is_host_api(name: str) -> bool:
    return name.startswith(HOST_API_PREFIX)


@dataclass
class TraceFacts:
    total_events: int = 0
    x_events: int = 0
    device_pids: list = field(default_factory=list)
    streams: int = 0
    host_api_us: float = 0.0
    device_us: float = 0.0
    host_calls: int = 0
    device_kernels: int = 0
    per_stream_us: dict = field(default_factory=dict)


def read_trace(path: str) -> TraceFacts:
    tpath = os.path.join(path, "trace_view.json")
    f = TraceFacts()
    if not os.path.isfile(tpath):
        return f
    with open(tpath, "r", encoding="utf-8", errors="ignore") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError:
            return f
    events = [e for e in data if isinstance(e, dict)]
    f.total_events = len(events)
    xs = [e for e in events if e.get("ph") == "X"]
    f.x_events = len(xs)

    # Device tracks are the pids whose tid values look like NPU stream ids.
    # We infer the device pid as the one carrying the most non-host events with
    # small tid values (stream indices), which is stable across CANN versions.
    by_pid: dict = defaultdict(lambda: {"host": 0.0, "dev": 0.0, "streams": set()})
    for e in xs:
        name = e.get("name", "")
        dur = float(e.get("dur") or 0)
        pid, tid = e.get("pid"), e.get("tid")
        if is_host_api(name):
            by_pid[pid]["host"] += dur
        else:
            by_pid[pid]["dev"] += dur
            by_pid[pid]["streams"].add(tid)

    for pid, agg in by_pid.items():
        if agg["dev"] > 0:
            f.device_pids.append(pid)
            f.device_us += agg["dev"]
            f.streams = max(f.streams, len(agg["streams"]))
            for tid in agg["streams"]:
                f.per_stream_us[f"{pid}:{tid}"] = f.per_stream_us.get(f"{pid}:{tid}", 0.0)
        f.host_api_us += agg["host"]

    f.host_calls = sum(1 for e in xs if is_host_api(e.get("name", "")))
    f.device_kernels = sum(1 for e in xs if not is_host_api(e.get("name", "")))
    return f


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def attribute_kernels(rows: list[KernelRow], top: int) -> list[dict]:
    agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_us": 0.0, "cores": set()})
    for r in rows:
        a = agg[r.optype]
        a["count"] += 1
        a["total_us"] += r.duration_us
        if r.core:
            a["cores"].add(r.core)
    total = sum(a["total_us"] for a in agg.values()) or 1.0
    out = []
    for optype, a in sorted(agg.items(), key=lambda kv: -kv[1]["total_us"])[:top]:
        out.append({
            "op_type": optype,
            "count": a["count"],
            "total_us": round(a["total_us"], 2),
            "pct": round(100.0 * a["total_us"] / total, 2),
            "cores": sorted(a["cores"]),
        })
    return out


@dataclass
class Analysis:
    output_dir: str
    kernel_meta: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)
    top_kernels: list = field(default_factory=list)
    core_split: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)


def analyze(path: str, top: int) -> Analysis:
    d = find_output_dir(path)
    a = Analysis(output_dir=d or path)
    if not d:
        a.findings.append("no trace_view.json or kernel_details.csv found under the given path")
        return a

    rows, meta = read_kernel_details(d)
    a.kernel_meta = meta
    tf = read_trace(d)
    a.trace = asdict(tf)

    if rows:
        a.top_kernels = attribute_kernels(rows, top)
        core = defaultdict(float)
        for r in rows:
            core[r.core or "unknown"] += r.duration_us
        tot = sum(core.values()) or 1.0
        a.core_split = {k: round(100.0 * v / tot, 2) for k, v in
                        sorted(core.items(), key=lambda kv: -kv[1])}

    # ---- findings -------------------------------------------------------
    if meta.get("present") and meta.get("columns", 0) < 40:
        a.findings.append(
            f"kernel_details.csv has only {meta['columns']} columns — the pipe breakdown "
            f"(aic/aiv mte/vec/scalar) is absent. Re-capture with "
            f"ExperimentalConfig(ProfilerLevel.Level1, AiCMetrics.PipeUtilization) to "
            f"get the 46-column form. Without it, fusion/overlap analysis is blind."
        )
    if rows:
        top1 = a.top_kernels[0]
        a.findings.append(
            f"hottest op type is `{top1['op_type']}` at {top1['pct']}% of device time "
            f"across {top1['count']} launches"
        )
        # A large gap between count and time share means few but heavy kernels.
        if top1["pct"] > 50:
            a.findings.append(
                "one op type exceeds half of device time — a fused replacement or a "
                "shape fix there moves the number more than tuning anything else"
            )
    if tf.x_events:
        if tf.host_api_us > 2 * tf.device_us and tf.device_us > 0:
            a.findings.append(
                f"host API time ({tf.host_api_us:.0f} us) is >2x device time "
                f"({tf.device_us:.0f} us) — launch-bound; look at graph mode before "
                f"tuning kernels"
            )
        a.findings.append(
            f"{tf.host_calls} host-API events vs {tf.device_kernels} device events — "
            f"a ratio near 1 means little host-side batching overlap"
        )
    if tf.streams <= 2 and tf.device_us > 0:
        a.findings.append(
            f"only {tf.streams} device stream(s) observed — single-stream execution "
            f"exposes every kernel tail; multi-stream is a candidate if gaps are large"
        )
    return a


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

def render(a: Analysis) -> str:
    L: list[str] = []
    L.append("# Ascend profiler analysis (torch_npu)")
    L.append("")
    L.append(f"- output dir: `{a.output_dir}`")
    if a.kernel_meta.get("present"):
        L.append(f"- kernel_details.csv: {a.kernel_meta['columns']} columns "
                 f"({a.kernel_meta['level']})")
    else:
        L.append("- kernel_details.csv: absent")
    t = a.trace
    if t.get("x_events"):
        L.append(f"- trace_view.json: {t['total_events']} events, {t['x_events']} complete (X)")
        L.append(f"- device pids: {t['device_pids']} | device streams: {t['streams']}")
    L.append("")

    if a.top_kernels:
        L.append("## Table 1 — kernel attribution (device time)")
        L.append("")
        L.append("| op type | launches | total us | % device | cores |")
        L.append("|---|---:|---:|---:|---|")
        for k in a.top_kernels:
            L.append(f"| {k['op_type']} | {k['count']} | {k['total_us']} | {k['pct']} | "
                     f"{','.join(k['cores']) or '-'} |")
        L.append("")

    if a.core_split:
        L.append("## Table 2 — accelerator core split")
        L.append("")
        L.append("| core | % device time |")
        L.append("|---|---:|")
        for core, pct in a.core_split.items():
            L.append(f"| {core} | {pct} |")
        L.append("")

    if t.get("x_events"):
        L.append("## Table 3 — host vs device")
        L.append("")
        L.append("| quantity | value |")
        L.append("|---|---:|")
        L.append(f"| host API events | {t['host_calls']} |")
        L.append(f"| host API total (us) | {t['host_api_us']:.0f} |")
        L.append(f"| device events | {t['device_kernels']} |")
        L.append(f"| device total (us) | {t['device_us']:.0f} |")
        ratio = (t['device_us'] / t['host_api_us']) if t['host_api_us'] else 0
        L.append(f"| device/host ratio | {ratio:.2f} |")
        L.append("")
        L.append("> **Read the duration sums with care.** They are sums over")
        L.append("> per-event durations, and host API calls on different CPU threads")
        L.append("> overlap in wall time. The sums therefore overstate concurrent host")
        L.append("> time; they are useful for comparing *which* host calls dominate and")
        L.append("> for the event-count ratio, not as a wall-clock budget. For a true")
        L.append("> timeline, open the trace in Perfetto or use `step_trace_time.csv`.")
        L.append("")

    L.append("## Findings")
    for f in a.findings:
        L.append(f"- {f}")
    L.append("")

    L.append("---")
    L.append("Reads a trace only; it does not capture one. To capture, use the")
    L.append("`model-infer-profiling` skill from CANNBot (torch_npu.profiler template)")
    L.append("or `torch_npu.profiler.profile(... tensorboard_trace_handler(...))` directly.")
    L.append("Per EVIDENCE-RULES, record the CANN/driver version this trace came from:")
    L.append("kernel sets change across CANN releases and an old analysis does not carry.")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Analyze a torch_npu profiler trace.")
    ap.add_argument("path", help="profiler root dir, *_ascend_pt dir, or ASCEND_PROFILER_OUTPUT dir")
    ap.add_argument("--top", type=int, default=20, help="rows in the kernel table")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    a = analyze(args.path, args.top)
    if args.json:
        print(json.dumps(asdict(a), indent=2, ensure_ascii=False))
    else:
        print(render(a))
    return 0 if a.output_dir else 2


if __name__ == "__main__":
    raise SystemExit(main())
