---
name: llm-serving-capacity-planner
description: Explain NPU HBM use, KV cache budget, and request capacity from a vllm-ascend or vLLM startup log. Use when the user asks where the NPU memory went, what KV cache budget a config yields, how many concurrent requests fit at a given context length, why the server OOMs at startup, or how gpu_memory_utilization / kv-cache-memory changes capacity. Reads logs only; needs no NPU.
---

# Ascend Serving Capacity Planner

Turn a startup log into an HBM accounting table and a concurrency estimate.

## The naming trap this skill absorbs

On Ascend, **the numbers are NPU HBM but the log says "GPU"**. vllm-ascend
inherits vLLM's log strings verbatim:

```
Free memory on device (62.31/64.0 GiB) on startup. Desired GPU memory utilization is (0.92, 58.88 GiB).
Actual usage: 24.15 GiB for weights, 1.82 GiB for peak activation, 3.41 GiB for
non-torch memory, 2.07 GiB for NPU graph memory. ... Current KV cache memory: 27.43 GiB.
GPU KV cache size: 4,010,585 tokens
Maximum concurrency for 1,048,576 tokens per request: 3.82x
```

Every "GPU" here means NPU. Do not report these numbers as GPU numbers, and do
not tell the user the log is wrong — it is a known inheritance, and this skill
exists partly to translate it.

## Workflow

```bash
python3 scripts/capacity_analyzer.py /path/to/server.log
python3 scripts/capacity_analyzer.py /path/to/server.log --requests 8192 32768 131072
cat server.log | python3 scripts/capacity_analyzer.py -
python3 scripts/capacity_analyzer.py server.log --json
```

Output: an HBM accounting table, a concurrency estimate table, and a diagnosis
that cites the specific number each note is based on.

## What each field means on Ascend

| Log field | Ascend meaning |
|---|---|
| `Free memory on device (F/T GiB)` | HBM free / HBM total, as the driver reports it before vLLM allocates |
| `weights` | model parameters resident in HBM (for W4A8/W8A8, the quantized sizes) |
| `peak activation` | largest activation buffer during profiling |
| `non-torch memory` | HBM not tracked by the torch allocator — driver reserves, collective buffers, CANN runtime |
| **`NPU graph memory`** | graph-capture buffers. **Ascend-specific and non-trivial** — it has no exact CUDA-graph analogue in cost, and it is frequently the difference between fitting and OOM |
| `Available KV cache memory` | the pool budget after everything above |
| `GPU KV cache size: N tokens` | **the authoritative pool capacity**, already accounting for hybrid layouts |
| `Maximum concurrency for L tokens: Cx` | the scheduler's own answer at length L |

## Why the tool prefers the reported numbers over its own division

For hybrid-attention models (sliding-window, compressed, chunked-local groups —
common in DeepSeek-V4.x and GLM families) a naive `pool_tokens / request_len`
division is **wrong**: each request's block count is capped by its window, not
by its length. The log's `Maximum concurrency` line is computed with that
handled; a division is not. So the tool reports the pool-bound arithmetic and
the scheduler's scaled figure side by side, takes the smaller, and labels both.

When you must choose one to quote, quote the reported `Maximum concurrency`.

## Diagnosing common situations

| Symptom in the log | Meaning | Next step |
|---|---|---|
| `Free memory on device (F/T)` with F well below T | another process holds HBM before vLLM starts | `npu-smi info` to find it; it is not vLLM's fault and KV headroom is short by exactly that gap |
| `NPU graph memory` large (> 1–2 GiB) | graph capture is consuming real budget | decide whether graph mode earns its memory; see the graph-mode skill |
| `non-torch memory` high relative to weights | driver/runtime/collective reserves | on multi-card runs HCCL buffers scale with device count |
| OOM at `gpu_memory_utilization` near 1.0 | no headroom for driver + collectives | lower it; Ascend needs more headroom than the GPU-era default assumes |
| Explicit `--kv-cache-memory` present | the utilization fraction does not govern the pool | the tool flags this; the two knobs interact |

## Scripts

| Script | Purpose |
|---|---|
| `scripts/capacity_analyzer.py` | Parse a startup log → HBM table + concurrency estimate + diagnosis |
| `scripts/capacity_analyzer.py --json` | Machine-readable form for chaining into a report |

## References

- `references/npu-memory-lines.md` — the exact log strings this parser matches, and where each comes from in the source tree
- `../../references/ascend-hardware-specs.json` — chip HBM sizes and the card/device distinction
- `../../docs/EVIDENCE-RULES.md` — record CANN/driver version with any capacity claim
