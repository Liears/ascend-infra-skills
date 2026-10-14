# Day-0 support plan — DeepSeek-V4.1-Flash on 910B3

Framework target: `vllm-ascend`

> Fill every `touches` with a real path and every gate with a real artifact.
> A task that cannot name the file it changes is not ready to schedule.

## Task DAG

### T1 — Model skeleton loads
- touches: `vllm_ascend/models/<Model>.py`, `models/__init__.py`
- evidence: G1 load: weight load log + parameter count match
- blocks: T2, T3
- note: correctness of shapes before any perf work

### T2 — Attention path (eager)
- touches: `vllm_ascend/attention/<backend>.py`, `vllm_ascend/ops/rope_*.py`
- evidence: G2 accuracy on the real path
- blocks: T4
- note: selector wired in the forward context; no fused kernel yet

### T3 — MoE / FFN path
- touches: `vllm_ascend/ops/fused_moe/token_dispatcher.py`
- evidence: G2 accuracy
- blocks: T4
- note: validate routing/scoring matches the reference

### T4 — Runnable eager end-to-end
- touches: `vllm_ascend/models/<Model>.py`
- evidence: G2 accuracy, G5 mode(eager)
- blocks: T5, T6, T10
- note: the first honest baseline

### T5 — Quant recipe + runtime mapping
- touches: `<msmodelslim>/recipe`, `vllm_ascend quant config`
- evidence: G4 memory + a real-path accuracy check on the quantized model
- blocks: G4 memory gate
- note: record which layers are quantized; fail loudly on missing config

### T6 — Model-unique feature (e.g. Engram-style tables)
- touches: `vllm_ascend/models/<Model>/`
- evidence: G3 fast-path: prove the device-side path ran
- blocks: G4 memory, G3 fastpath
- note: if the feature has no analogue in the runtime, this is the schedule risk

### T7 — MTP / speculative head
- touches: `vllm_ascend/models/<Model>_mtp.py`
- evidence: accept-length reported + tok/s basis stated
- blocks: (nothing)
- note: perf only; independent of the eager baseline

### T8 — Vision tower + processor
- touches: `vllm_ascend/models/`, `multimodal processor`
- evidence: task-based accuracy on an image set
- blocks: (nothing)
- note: only if the model is multimodal

### T9 — Fused fast paths
- touches: `attention`, `fused_moe`, `rope_*`
- evidence: G3 fast-path per fused op
- blocks: (nothing)
- note: each must be shown to engage; silent fallback is the trap

### T10 — Graph mode
- touches: `graph-capture integration`
- evidence: G5 mode(graph) + no accuracy change vs eager
- blocks: (nothing)
- note: after eager is stable; graph memory is a real HBM cost

## Release gates

| gate | check | artifact required |
|---|---|---|
| G1 LOAD | weights load; parameter count == config | load log with the count |
| G2 ACCURACY | <task> >= <expected> on the real path, fast paths on | run output + exact command |
| G3 FASTPATH | profiler shows each claimed fused op executed | kernel_details.csv row per fused op |
| G4 MEMORY | fits the chip at ctx=<N>, KV budget stated | startup log + `GPU KV cache size` line |
| G5 MODE | runs in each claimed mode (eager/graph/PD) | one run per mode |

## Stack record (fill before running)

```
model:        DeepSeek-V4.1-Flash @ <revision>
framework:    vllm-ascend @ <commit>
chip:         910B3 x <N> devices (cards: <M>)
toolchain:    CANN <ver> / driver <ver> / torch_npu <ver>
captured:     <date>
```
