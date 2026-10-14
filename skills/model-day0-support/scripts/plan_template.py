#!/usr/bin/env python3
"""
Emit a Day-0 support plan skeleton for a model on Ascend.

The point is not to generate prose — it is to force the plan to name, for every
task, the file it touches, the evidence it needs, and what it blocks. A plan
that cannot name the file is not yet understood.

Usage:
    python3 plan_template.py --model DeepSeek-V4.1-Flash --chip 910B3 --framework vllm-ascend
    python3 plan_template.py --model Foo --chip 910C --framework sglang-ascend --json
"""
from __future__ import annotations

import argparse
import json
import sys

# A task DAG template. Each entry: id, title, touches, evidence, blocks.
# `touches` are real extension points — see references/ascend-extension-points.md.
TEMPLATE = [
    {
        "id": "T1", "title": "Model skeleton loads",
        "touches": ["vllm_ascend/models/<Model>.py", "models/__init__.py"],
        "evidence": "G1 load: weight load log + parameter count match",
        "blocks": ["T2", "T3"],
        "notes": "correctness of shapes before any perf work",
    },
    {
        "id": "T2", "title": "Attention path (eager)",
        "touches": ["vllm_ascend/attention/<backend>.py", "vllm_ascend/ops/rope_*.py"],
        "evidence": "G2 accuracy on the real path",
        "blocks": ["T4"],
        "notes": "selector wired in the forward context; no fused kernel yet",
    },
    {
        "id": "T3", "title": "MoE / FFN path",
        "touches": ["vllm_ascend/ops/fused_moe/token_dispatcher.py"],
        "evidence": "G2 accuracy",
        "blocks": ["T4"],
        "notes": "validate routing/scoring matches the reference",
    },
    {
        "id": "T4", "title": "Runnable eager end-to-end",
        "touches": ["vllm_ascend/models/<Model>.py"],
        "evidence": "G2 accuracy, G5 mode(eager)",
        "blocks": ["T5", "T6", "T10"],
        "notes": "the first honest baseline",
    },
    {
        "id": "T5", "title": "Quant recipe + runtime mapping",
        "touches": ["<msmodelslim>/recipe", "vllm_ascend quant config"],
        "evidence": "G4 memory + a real-path accuracy check on the quantized model",
        "blocks": ["G4 memory gate"],
        "notes": "record which layers are quantized; fail loudly on missing config",
    },
    {
        "id": "T6", "title": "Model-unique feature (e.g. Engram-style tables)",
        "touches": ["vllm_ascend/models/<Model>/"],
        "evidence": "G3 fast-path: prove the device-side path ran",
        "blocks": ["G4 memory", "G3 fastpath"],
        "notes": "if the feature has no analogue in the runtime, this is the schedule risk",
    },
    {
        "id": "T7", "title": "MTP / speculative head",
        "touches": ["vllm_ascend/models/<Model>_mtp.py"],
        "evidence": "accept-length reported + tok/s basis stated",
        "blocks": [],
        "notes": "perf only; independent of the eager baseline",
    },
    {
        "id": "T8", "title": "Vision tower + processor",
        "touches": ["vllm_ascend/models/", "multimodal processor"],
        "evidence": "task-based accuracy on an image set",
        "blocks": [],
        "notes": "only if the model is multimodal",
    },
    {
        "id": "T9", "title": "Fused fast paths",
        "touches": ["attention", "fused_moe", "rope_*"],
        "evidence": "G3 fast-path per fused op",
        "blocks": [],
        "notes": "each must be shown to engage; silent fallback is the trap",
    },
    {
        "id": "T10", "title": "Graph mode",
        "touches": ["graph-capture integration"],
        "evidence": "G5 mode(graph) + no accuracy change vs eager",
        "blocks": [],
        "notes": "after eager is stable; graph memory is a real HBM cost",
    },
]

GATES = [
    {"id": "G1", "name": "LOAD", "check": "weights load; parameter count == config",
     "artifact": "load log with the count"},
    {"id": "G2", "name": "ACCURACY", "check": "<task> >= <expected> on the real path, fast paths on",
     "artifact": "run output + exact command"},
    {"id": "G3", "name": "FASTPATH", "check": "profiler shows each claimed fused op executed",
     "artifact": "kernel_details.csv row per fused op"},
    {"id": "G4", "name": "MEMORY", "check": "fits the chip at ctx=<N>, KV budget stated",
     "artifact": "startup log + `GPU KV cache size` line"},
    {"id": "G5", "name": "MODE", "check": "runs in each claimed mode (eager/graph/PD)",
     "artifact": "one run per mode"},
]


def render(model: str, chip: str, framework: str) -> str:
    L: list[str] = []
    L.append(f"# Day-0 support plan — {model} on {chip}")
    L.append("")
    L.append(f"Framework target: `{framework}`")
    L.append("")
    L.append("> Fill every `touches` with a real path and every gate with a real artifact.")
    L.append("> A task that cannot name the file it changes is not ready to schedule.")
    L.append("")
    L.append("## Task DAG")
    L.append("")
    for t in TEMPLATE:
        L.append(f"### {t['id']} — {t['title']}")
        L.append(f"- touches: {', '.join('`'+x+'`' for x in t['touches'])}")
        L.append(f"- evidence: {t['evidence']}")
        L.append(f"- blocks: {', '.join(t['blocks']) or '(nothing)'}")
        L.append(f"- note: {t['notes']}")
        L.append("")
    L.append("## Release gates")
    L.append("")
    L.append("| gate | check | artifact required |")
    L.append("|---|---|---|")
    for g in GATES:
        L.append(f"| {g['id']} {g['name']} | {g['check']} | {g['artifact']} |")
    L.append("")
    L.append("## Stack record (fill before running)")
    L.append("")
    L.append("```")
    L.append("model:        %s @ <revision>" % model)
    L.append("framework:    %s @ <commit>" % framework)
    L.append("chip:         %s x <N> devices (cards: <M>)" % chip)
    L.append("toolchain:    CANN <ver> / driver <ver> / torch_npu <ver>")
    L.append("captured:     <date>")
    L.append("```")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Emit a Day-0 plan skeleton.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--chip", required=True)
    ap.add_argument("--framework", default="vllm-ascend")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.json:
        print(json.dumps({"model": args.model, "chip": args.chip,
                          "framework": args.framework,
                          "tasks": TEMPLATE, "gates": GATES},
                         indent=2, ensure_ascii=False))
    else:
        print(render(args.model, args.chip, args.framework))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
