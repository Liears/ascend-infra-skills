# Ascend Torch Profiler Trace Format

What `torch_npu.profiler` writes, and how to read it. Verified 2026-10-14 against
torch_npu 2.10.0.post4 on Ascend 910B3, CANN 8.1.RC1 / driver 25.5.2.

---

## 1. Directory layout

`tensorboard_trace_handler(save_path)` produces:

```
<save_path>/
└── <host>_<pid>_<timestamp>_ascend_pt/
    ├── ASCEND_PROFILER_OUTPUT/        <- the analysis-ready output
    │   ├── kernel_details.csv
    │   ├── trace_view.json
    │   ├── op_statistic.csv
    │   ├── api_statistic.csv
    │   ├── step_trace_time.csv
    │   ├── task_time.csv
    │   └── analysis.db / ascend_pytorch_profiler.db
    ├── PROF_000001_*/                 <- raw CANN data (host/device binary)
    ├── logs/                          <- parser logs
    └── profiler_info.json
```

Point the analyzer at **any** of the three levels; it finds
`ASCEND_PROFILER_OUTPUT` itself.

The `PROF_*/` tree holds the raw binary stream data. You never parse it by hand —
the `.csv`/`.json` in `ASCEND_PROFILER_OUTPUT` are the parsed forms, produced
during the profiling run by the CANN parser.

---

## 2. `trace_view.json` — Chrome trace format

A JSON array of events. `ph` values follow the Chrome trace-event spec:

| `ph` | Meaning | Has `ts`/`dur` |
|---|---|---|
| `M` | metadata (`process_name`, `thread_name`, `process_sort_index`) | no |
| `X` | complete event (a span) | yes |

### 2.1 Track topology

Events are keyed by `(pid, tid)`:

| Track | `pid` | `tid` | Carries |
|---|---|---|---|
| Host | the process pid | thread id (e.g. `1917`) | `AscendCL@*`, `Node@*`, python frames |
| Device | a distinct pid (e.g. `1963456`) | **NPU stream id** (small ints: `47`, `48`) | kernels |

The stream id in `tid` is the NPU stream index — this is how you tell which
device stream a kernel ran on, and therefore whether execution is single-stream.

### 2.2 Event name grammar

**Host API calls** are prefixed with the layer that emitted them:

```
AscendCL@aclrtLaunchKernelWithHostArgs    runtime launch
AscendCL@aclnnReduceSum                   aclnn op dispatch
AscendCL@ReduceSum_Tiling                 tiling computation
AscendCL@aclrtBinaryLoadFromData          kernel-binary load
AscendCL@aclrtGetStreamAttribute          stream query
Node@launch                               torch_npu graph launch
```

**Device kernels** take the form `<aclPrefix>_<OpCommon>_<OpType>`, e.g.:

```
aclnnAddmm_MatMulCommon_MatMulV2
aclnnReduceSum_ReduceSumOpAiCore_ReduceSum
aclnnAdd_AddAiCore_Add
<<<>>>_Add_Add   (direct-invoke form)
```

The **op type is the third token** in the common case, but not always — hence
the analyzer matches a vocabulary rather than always taking index 2.

### 2.3 `args` contents

Device events carry `args` with at least `{"level": "kernel"}`. Host API events
carry a richer set (`Thread Id`, `Mode`, `level`, `item_id`, `connection_id`).
`kernel_details.csv` is the better source for per-kernel detail; the JSON is
better for the **timeline** and the **host/device split**.

---

## 3. `kernel_details.csv` — the device-side table

Column count depends on `_ExperimentalConfig`:

### 3.1 The 46-column form (Level1 + `AiCMetrics.PipeUtilization`)

```
Device_id, Model ID, Task ID, Stream ID, Name, Type, OP State, Accelerator Core,
Start Time(us), Duration(us), Wait Time(us), Block Num, Mix Block Num,
HF32 Eligible, Input Shapes, Input Data Types, Input Formats,
Output Shapes, Output Data Types, Output Formats, Context ID,
aicore_time(us), aic_total_cycles, aic_mac_time(us), aic_mac_ratio,
aic_scalar_time(us), aic_scalar_ratio, aic_mte1_time(us), aic_mte1_ratio,
aic_mte2_time(us), aic_mte2_ratio, aic_fixpipe_time(us), aic_fixpipe_ratio,
aic_icache_miss_rate, aiv_time(us), aiv_total_cycles, aiv_vec_time(us),
aiv_vec_ratio, aiv_scalar_time(us), aiv_scalar_ratio, aiv_mte2_time(us),
aiv_mte2_ratio, aiv_mte3_time(us), aiv_mte3_ratio, aiv_icache_miss_rate,
cube_utilization(%)
```

The pipe columns are the point of this form:

| Column family | Unit | Tells you |
|---|---|---|
| `aic_mac_*` | AI Core matrix | cube-bound? |
| `aic_mte1/mte2_*` | AI Core memory | input-bound? (MTE2 = load) |
| `aic_fixpipe_*` | AI Core fixpipe | output-write bound? |
| `aiv_vec_*` | AI Vector | vector-op bound? |
| `aiv_mte2/mte3_*` | AI Vector memory | in/out bandwidth bound? |
| `*_ratio` | fraction of `*_time` | share of the kernel's own time |
| `cube_utilization(%)` | % | how hard the cube was driven |

**MTE2-bound** on a matmul means the weight/activation load dominates — a
layout or L2-reuse fix, not an arithmetic fix. **MAC-bound** means the cube is
the limit — a tiling/precision fix. This distinction is invisible without the
pipe columns.

### 3.2 Reduced form

Without Level1 + PipeUtilization you get names, durations, cores and shapes
only. The analyzer detects this (< 40 columns) and refuses to imply pipe
conclusions.

### 3.3 Capturing the full form

```python
from torch_npu.profiler import _ExperimentalConfig
import torch_npu

experimental_config = _ExperimentalConfig(
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
)
prof = torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU,
                torch_npu.profiler.ProfilerActivity.CPU],
    experimental_config=experimental_config,
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=N, repeat=1, skip_first=0),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(save_path),
)
```

Two rules that are easy to get wrong (from CANNBot's `model-infer-profiling`):

1. `prof.step()` must be called at least `warmup + active + 1` times — the extra
   call flushes the last step.
2. Compilation / graph-capture warmup must happen **outside** the profiler
   window, or the first step's numbers are garbage.

---

## 4. Reading a trace without this tool

Load `trace_view.json` in Perfetto (`ui.perfetto.dev`). The device streams
appear as swimlanes; host API calls sit above them. Use it when you need the
**timeline shape** (where the gaps are), and use this tool when you need the
**aggregate ranking** (which op dominates).

The two are complementary: the aggregate hides where a kernel sits in the step;
the timeline hides which op type totals the most.

---

## 5. Version sensitivity

The CSV schema and the event-name grammar are products of the CANN profiler.
They **change between CANN releases**. Every trace-derived claim must record:

- CANN toolkit version (`/usr/local/Ascend/ascend-toolkit/latest/version.cfg`)
- driver version (`npu-smi info` / `/usr/local/Ascend/driver/version.info`)
- torch_npu version
- the `ProfilerLevel` and `AiCMetrics` used

A trace taken on one CANN release does not validate an analysis on another.
