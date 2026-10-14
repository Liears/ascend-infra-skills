# Ascend profiler analysis (torch_npu)

- output dir: `/tmp/dsv41-ascend/results/a2_20261015_031017/prof/dp0_pp0_tp0_dcp0_ep0_rank0_1002_20261014193857831_ascend_pt/ASCEND_PROFILER_OUTPUT`
- kernel_details.csv: 46 columns (Level1+PipeUtilization)
- trace_view.json: 2490669 events, 1769798 complete (X)
- device pids: [1002, 1026560, 1027040, 1026303, 1026496] | device streams: 14

## Table 1 — kernel attribution (device time)

| op type | launches | total us | % device | cores |
|---|---:|---:|---:|---|
| AivKernel | 8229 | 4288060.16 | 41.53 | COMMUNICATION |
| allReduce | 7473 | 4067704.48 | 39.39 | COMMUNICATION |
| aclnnGroupedMatmulSwigluQuantWeightNzV2 | 3612 | 245630.76 | 2.38 | MIX_AIC |
| MatMul | 13440 | 241614.18 | 2.34 | AI_CORE |
| HcPre | 7224 | 203236.75 | 1.97 | MIX_AIC |
| aclnnQuantMatmulWeightNz | 18984 | 197422.84 | 1.91 | AI_CORE,MIX_AIC |
| aclnnGroupedMatmulWeightNz | 3612 | 121459.85 | 1.18 | MIX_AIC |
| alltoallv | 168 | 112631.33 | 1.09 | COMMUNICATION |
| allGather | 420 | 105804.9 | 1.02 | COMMUNICATION |
| RmsNorm | 15624 | 77742.12 | 0.75 | AI_VECTOR_CORE |
| SparseFlashMla | 3360 | 60261.74 | 0.58 | MIX_AIC |
| HcPost | 7224 | 52083.59 | 0.5 | AI_VECTOR_CORE |
| aclnnMoeInitRoutingV3 | 3612 | 44842.75 | 0.43 | MIX_AIV |
| InplacePartialRotaryMul | 12180 | 42989.44 | 0.42 | AI_VECTOR_CORE |
| aclnnDynamicQuantV2 | 11844 | 40526.29 | 0.39 | AI_VECTOR_CORE |

## Table 2 — accelerator core split

| core | % device time |
|---|---:|
| COMMUNICATION | 83.06 |
| MIX_AIC | 7.95 |
| AI_VECTOR_CORE | 5.05 |
| AI_CORE | 2.64 |
| MIX_AIV | 0.77 |
| AI_CPU | 0.54 |

## Table 3 — host vs device

| quantity | value |
|---|---:|
| host API events | 385122 |
| host API total (us) | 4589775 |
| device events | 1384676 |
| device total (us) | 61540886 |
| device/host ratio | 13.41 |

> **Read the duration sums with care.** They are sums over
> per-event durations, and host API calls on different CPU threads
> overlap in wall time. The sums therefore overstate concurrent host
> time; they are useful for comparing *which* host calls dominate and
> for the event-count ratio, not as a wall-clock budget. For a true
> timeline, open the trace in Perfetto or use `step_trace_time.csv`.

## Findings
- hottest op type is `AivKernel` at 41.53% of device time across 8229 launches
- **communication dominates**: 83% of device time is collective ops (AivKernel, allReduce, alltoallv, allGather). On a large MoE with EP/TP this is the normal shape, not a defect — but it means kernel-level tuning of compute ops can move at most ~17% of the time. Look at parallelism layout (EP vs TP), overlap, and HCCL before anything else.
- hottest *compute* op (excluding collectives) is `aclnnGroupedMatmulSwigluQuantWeightNzV2` at 2.38% of total device time
- 385122 host-API events vs 1384676 device events (ratio 3.60:1) — moderate host/device batching

---
Reads a trace only; it does not capture one. To capture, use the
`model-infer-profiling` skill from CANNBot (torch_npu.profiler template)
or `torch_npu.profiler.profile(... tensorboard_trace_handler(...))` directly.
Per EVIDENCE-RULES, record the CANN/driver version this trace came from:
kernel sets change across CANN releases and an old analysis does not carry.
