---
name: npu-profiler-analysis
description: Analyze a torch_npu (Ascend) profiler trace and produce kernel attribution, core split, and host/device tables. Use when the user has a torch_npu prof trace (kernel_details.csv / trace_view.json), asks which Ascend ops dominate device time, why a model is slow on NPU, how to read an Ascend profiler dump, or which op to fuse or fix first. Reads an existing trace; needs no NPU to run.
---

# NPU Profiler Analysis

Read an Ascend torch profiler trace and answer "which op eats the device time,
and is the run kernel-bound or launch-bound".

## Input

A directory from `torch_npu.profiler`. The tool accepts any of:
- the profiler root (`./prof/`),
- the `*_ascend_pt/` directory,
- the `ASCEND_PROFILER_OUTPUT/` directory itself.

It uses whichever of these it finds:

| File | What it gives |
|---|---|
| `ASCEND_PROFILER_OUTPUT/kernel_details.csv` | device kernels + (at Level1) pipe breakdown |
| `ASCEND_PROFILER_OUTPUT/trace_view.json` | Chrome-format timeline: host API + device streams |
| `ASCEND_PROFILER_OUTPUT/step_trace_time.csv` | per-step timing |
| `ASCEND_PROFILER_OUTPUT/op_statistic.csv` | per-op rollup from CANN |

## Workflow

```bash
python3 scripts/analyze_profile.py /path/to/prof --top 20
python3 scripts/analyze_profile.py /path/to/prof --json
```

Three tables: **kernel attribution** (by op type), **core split** (AI_CORE /
AI_VECTOR_CORE / MIX_AIV), and **host vs device**. Then explicit findings.

## How Ascend traces differ from CUDA traces

This is why the CUDA-era tooling does not port unchanged.

**1. Event names carry the op type and the implementation, joined by `_`:**

```
aclnnAddmm_MatMulCommon_MatMulV2          -> op MatMulV2, impl MatMulCommon
aclnnReduceSum_ReduceSumOpAiCore_ReduceSum -> op ReduceSum, on AICore
```

The analyzer splits on `_`/`@` and matches against an Ascend op-type vocabulary.
The useful axis is the **op type**, because that is what a fusion or a shape
change targets.

**2. There are two event families, and they live on different tracks:**

| Family | Naming | Lives on |
|---|---|---|
| Host API | `AscendCL@aclrtLaunchKernelWithHostArgs`, `AscendCL@aclnnReduceSum`, `Node@launch` | CPU tracks |
| Device kernel | `aclnn*_Op_Impl`, `*_Tiling` | device pid's stream tids |

A high host-API-to-device event ratio means launch-bound execution — no amount
of kernel tuning fixes that; graph mode does.

**3. `kernel_details.csv` has a column count that depends on your config.**

| Columns | Config | What you get |
|---|---|---|
| 46 | `ExperimentalConfig(ProfilerLevel.Level1, AiCMetrics.PipeUtilization)` | full pipe breakdown: `aic_mte1/mte2/mac/scalar`, `aiv_vec/mte2/mte3/scalar`, `cube_utilization(%)` |
| ~9 | default level | names and durations only — **no pipe data** |

If the analyzer reports fewer than 40 columns, it says so; **re-capture before
attempting fusion analysis**, because you cannot see MTE-bound vs MAC-bound
without the pipe columns.

**4. Core types mean different things.**

| Core | Meaning |
|---|---|
| `AI_CORE` | cube/matrix unit (matmul) |
| `AI_VECTOR_CORE` | vector unit (elementwise, reductions) |
| `MIX_AIV` / `MIX_AIC` | mixed cube+vector kernels |
| `AI_CPU` | AICPU kernels (some collectives, some shape ops) |

A large `AI_VECTOR_CORE` share on a model that should be matmul-bound usually
means an unfused elementwise chain, not that the model is "vector bound".

## Interpreting the findings

| Finding | Reading |
|---|---|
| One op type > 50% of device time | a fused replacement or shape fix there is the highest-leverage change |
| **Communication ≥ 40% of device time** | the run is collective-bound. This is the **normal shape** for a large MoE served with EP/TP on Ascend, not a defect — but it caps what kernel tuning can achieve (only the remaining share). Look at parallelism layout (EP vs TP), overlap, and HCCL first |
| host API time ≫ device time | launch-bound → graph mode (`torch.compile` / aclgraph / npugraph_ex) |
| device/host event ratio near 1:1 | little host/device batching overlap; more host events means more exposed latency |
| device/host event ratio high (≫5) | heavy batching — the expected shape under graph mode |
| 1–2 device streams only | single-stream exposes tails; multi-stream is a candidate |
| `MIX_AIV` dominant on a matmul-heavy model | the "matmul" is running in a mixed kernel with vector work attached — check tiling |

### The communication-dominance finding, concretely

On a real DeepSeek-V4.1-Flash trace (8×910B3, TP8/EP8, one decode workload),
the core split came out as:

| core | % device time |
|---|---:|
| **COMMUNICATION** | **83.06** |
| MIX_AIC | 7.95 |
| AI_VECTOR_CORE | 5.05 |
| AI_CORE | 2.64 |

The top ops were `AivKernel` (41.5%), `allReduce` (39.4%), `alltoallv`,
`allGather` — all collectives. **A GPU-shaped analyzer reports these as
unremarkable "other kernels" and then points the reader at a 2%-of-time matmul
as the hot spot.** That is the wrong conclusion by an order of magnitude. The
analyzer therefore detects a communication-majority split and says what it
implies: compute-kernel tuning can move at most the non-collective share.

See `examples/deepseek-v4.1-flash-910b/` for the full captured output.

## What this skill does NOT do

- **It does not capture traces.** Capture belongs to CANNBot's
  `model-infer-profiling` skill (`torch_npu.profiler` template with the correct
  Level1 + PipeUtilization config). This skill is the reader half.
- **It does not do pipe-level fusion analysis** on its own; for that, read the
  pipe columns in `kernel_details.csv` after confirming the 46-column form.
- **It does not compare across CANN versions.** Kernel sets change between CANN
  releases; per EVIDENCE-RULES, never reuse a trace for a new capture.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/analyze_profile.py` | Trace → 3 tables + findings |

## References

- `references/ascend-trace-format.md` — the trace JSON layout, event families, and the op-name grammar
- `../../docs/EVIDENCE-RULES.md` — record CANN/driver with every trace-derived claim
