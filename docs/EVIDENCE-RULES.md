# Evidence Rules — Ascend AI-Infra Skills

Every claim these skills produce must be traceable to a recorded run, an inspected
source file, or a cited official document. This file defines what "traceable" means.

Ported from `BBuf/AI-Infra-Auto-Driven-SKILLS` evidence rules, with the hardware,
framework, and toolchain dimensions adapted to Ascend. The rules themselves are
unchanged; only what must be recorded changed.

---

## 1. The four rule families

### 1.1 Benchmark rows

A benchmark row is not admissible unless it records **all** of:

| Field | Why |
|---|---|
| Model (name + revision/commit) | A weight file change invalidates the number. |
| Framework (name + commit) | vLLM-Ascend and upstream vLLM move independently. |
| **Hardware (chip + card/device count + CANN + driver version)** | Ascend numbers are meaningless without the CANN/driver pair. |
| Workload (input len, output len, concurrency, prefix-cache state) | Ascend KV pressure is extremely sensitive to these. |
| SLA status (per-request latency budget, whether met) | Throughput without a latency bound is not a result. |
| Both commands (server launch + client load) | A number nobody can reproduce is a rumor. |
| Raw artifact path | The trace, csv, or jsonl that the number was computed from. |

**Ascend-specific additions** over the GPU version:
- `TP/EP/DP degrees` in **devices**, not cards. State the unit.
- `HCCL backend` and whether the run used `HCCL_DETERMINISTIC`.
- `torch_npu` and `CANN` versions (they gate which fused kernels exist).
- Whether the run used a **graph mode** (aclgraph / npugraph_ex / GE) or eager.

### 1.2 Profiler reports

- Keep **prefill and decode separate**. They are different regimes and averaging them hides everything.
- **Never reuse an older trace for a new capture.** Ascend kernel sets change with every CANN bump.
- Record the `ProfilerLevel` and `AiCMetrics` used, because the column set of
  `kernel_details.csv` depends on them (Level1 + PipeUtilization yields the 46-column
  form used here; a lower level yields a 9-column form and cannot support pipe analysis).
- Record whether `torch.compile` / `torchair` / JIT warmup happened **inside** the profiler window.

### 1.3 Performance claims

Scoped to the exact **model, chip, precision, workload, and framework revisions**.
A claim without a scope is not a claim.

On Ascend, the precision axis is richer than on GPU — a claim must say which of these it means:
- weight/activation dtype (BF16 / FP8 / W8A8 / W4A8 / MXFP4)
- whether quantized weights were produced by ModelSlim (`msmodelslim`) or another tool
- whether the attention path ran fused (`FA`, `MLA`, `DSA`) or fell back

**Accuracy is checked on the real path.** A kernel swap that passes an end-to-end
test is not proven until the real model executes the candidate kernel. Silent
fallbacks to a slower reference path are the most common false-positive on Ascend;
assert the fast path actually ran.

### 1.4 Historical PR evidence

- Historical evidence keeps its **audit date**. A source refresh is not a rerun.
- Upstream PR cards describe **what the upstream change did**, not what is currently
  true on this machine's branch.
- A generated dossier that was not manually diffed is marked **not a manual audit**
  and must not be cited as implemented behavior.

---

## 2. What counts as a source

| Trust | Source |
|---|---|
| Authoritative | `npu-smi` / CANN version files on the machine the claim is about |
| Authoritative | The source file itself, read at a stated commit |
| Strong | Official CANN / vllm-ascend / MindIE documentation at a stated version |
| Strong | A recorded run with its raw artifact |
| Weak | A README claim that the repo does not back with a run |
| Reject | Anything invented, rounded without notice, or copied from a different chip |

**Never infer an Ascend number from an NVIDIA number.** The architectures differ in
ways that change the answer (cube/vector split, HBM bandwidth, collective cost,
mandatory graph structure). Estimate is allowed; it must be labelled `estimate`.

---

## 3. Reporting format

Every skill output that makes a performance or capacity claim ends with an evidence block:

```
evidence:
  model:        <name> @ <revision>
  framework:    <vllm-ascend|sglang|mindie> @ <commit>
  hardware:     <chip> x <N> devices  (cards: <M>)
  toolchain:    CANN <ver> / driver <ver> / torch_npu <ver>
  workload:     in=<n> out=<n> conc=<n> prefix_cache=<0|1>
  graph_mode:   <eager|aclgraph|npugraph_ex|ge>
  hccl:         <backend>, deterministic=<0|1>
  sla:          <budget>  ->  <met|missed>
  artifact:     <path to raw trace / csv / jsonl>
  captured:     <date>
  scope_note:   <what this does NOT cover>
```

If a field is unknown, write `unknown` — do not omit the field. An omitted field
reads as "did not apply", which is a different and often false statement.

---

## 4. Anti-patterns

- **Card/device conflation.** "8 NPU" is ambiguous on A3. Write "8 devices = 4 cards".
- **Averaging prefill and decode.** Report them separately.
- **Quoting a vendor GB as a usable GiB.** 64 GB HBM is not 64 GiB addressable.
- **Reusing a trace across CANN versions.** The kernel set changes; the old analysis is void.
- **Citing a dossier as current behavior.** Dossiers are dated history.
- **Reporting tok/s without the acceptance-length convention.** On speculative-decoding
  models, say whether tok/s is `A x 1000 / ms_per_step` or raw generated tokens over
  wall time; the two differ and are routinely confused.
