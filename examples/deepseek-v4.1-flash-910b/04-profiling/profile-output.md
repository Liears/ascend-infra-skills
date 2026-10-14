# Ascend profiler analysis (torch_npu)

- output dir: `/tmp/prof-out/B06-05U-AT800T-BMS-NODE-13_1917_20261014155146794_ascend_pt/ASCEND_PROFILER_OUTPUT`
- kernel_details.csv: 46 columns (Level1+PipeUtilization)
- trace_view.json: 338 events, 243 complete (X)
- device pids: [1963456, 1964000] | device streams: 2

## Table 1 — kernel attribution (device time)

| op type | launches | total us | % device | cores |
|---|---:|---:|---:|---|
| ReduceSum | 8 | 47.96 | 41.13 | MIX_AIV |
| MatMul | 8 | 39.56 | 33.92 | AI_CORE |
| Add | 4 | 14.36 | 12.31 | AI_VECTOR_CORE |
| BroadcastTo | 6 | 10.86 | 9.31 | AI_VECTOR_CORE |
| Fill | 3 | 3.88 | 3.33 | AI_VECTOR_CORE |

## Table 2 — accelerator core split

| core | % device time |
|---|---:|
| MIX_AIV | 41.13 |
| AI_CORE | 33.92 |
| AI_VECTOR_CORE | 24.95 |

## Table 3 — host vs device

| quantity | value |
|---|---:|
| host API events | 155 |
| host API total (us) | 47928 |
| device events | 88 |
| device total (us) | 38029 |
| device/host ratio | 0.79 |

> **Read the duration sums with care.** They are sums over
> per-event durations, and host API calls on different CPU threads
> overlap in wall time. The sums therefore overstate concurrent host
> time; they are useful for comparing *which* host calls dominate and
> for the event-count ratio, not as a wall-clock budget. For a true
> timeline, open the trace in Perfetto or use `step_trace_time.csv`.

## Findings
- hottest op type is `ReduceSum` at 41.13% of device time across 8 launches
- 155 host-API events vs 88 device events — a ratio near 1 means little host-side batching overlap
- only 2 device stream(s) observed — single-stream execution exposes every kernel tail; multi-stream is a candidate if gaps are large

---
Reads a trace only; it does not capture one. To capture, use the
`model-infer-profiling` skill from CANNBot (torch_npu.profiler template)
or `torch_npu.profiler.profile(... tensorboard_trace_handler(...))` directly.
Per EVIDENCE-RULES, record the CANN/driver version this trace came from:
kernel sets change across CANN releases and an old analysis does not carry.
