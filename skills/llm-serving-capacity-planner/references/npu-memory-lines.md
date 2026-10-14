# NPU Memory Log Lines — Exact Strings and Provenance

Every pattern the analyzer matches, with the source file it was read from.
Verified 2026-10-14 against vllm-ascend `0.23.1.dev14` and vLLM `0.23.0` in the
`verl_hobo` container on this machine.

If a line does not match, **the fix is to re-read the source and update the
pattern here** — never to loosen the regex until it happens to match. A loose
regex on memory lines silently produces wrong capacity numbers, which is the
worst failure mode this skill has.

---

## 1. Startup memory lines

### 1.1 The full accounting line (verbose form)

Source: `vllm_ascend/worker/worker.py` (~line 750), the `log_memory_stats` path.

```
Free memory on device (62.31/64.0 GiB) on startup. Desired GPU memory utilization is
(0.92, 58.88 GiB). Actual usage: 24.15 GiB for weights, 1.82 GiB for peak activation,
3.41 GiB for non-torch memory, 2.07 GiB for NPU graph memory. Replace
gpu_memory_utilization with `--kv-cache-memory=26518185984` (24.70 GiB) to fit into
requested memory, or `--kv-cache-memory=31024021504` (28.89 GiB) to fully utilize NPU
free memory. Current KV cache memory: 27.43 GiB.
```

Patterns:
- `RE_FREE_MEMORY` → `Free memory on device\s*\(\s*([\d.]+)\s*/\s*([\d.]+)\s*GiB\s*\)\s*on startup`
- `RE_UTILIZATION` → `Desired GPU memory utilization is\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*GiB\s*\)`
- `RE_ACTUAL_USAGE` → the four-component `Actual usage:` sentence
- `RE_CURRENT_KV` → `Current KV cache memory:\s*([\d.]+)\s*GiB`

**Note the components.** Ascend splits non-KV memory into four buckets, one of
which (`NPU graph memory`) has no exact GPU analogue in cost. A capacity model
that omits it will over-predict how much is left for KV.

### 1.2 The short form

Source: `vllm_ascend/worker/worker.py` line ~594 (and vLLM
`v1/worker/gpu_worker.py` ~481).

```
Available KV cache memory: 27.43 GiB
```
Pattern: `RE_AVAILABLE_KV`.

### 1.3 The explicit-budget form

Source: `vllm_ascend/worker/worker.py` line ~543.

```
Initial free memory 62.31 GiB, reserved 27.43 GiB for KV Cache as specified by
kv_cache_memory_bytes, skipping memory profiling.
```
Patterns: `RE_INITIAL_FREE`, and `RE_KV_CACHE_MEM_ARG` on the argparse value.

When this line is present, **memory profiling was skipped** and the
utilization fraction does not govern the pool. The analyzer says so.

### 1.4 The failure form

Source: same file, the `ValueError` path (~line 473).

```
Free memory on device (61.20/64.0 GiB) on startup is less than desired GPU memory
utilization (0.98, 62.72 GiB). Decrease GPU memory utilization or reduce GPU memory
used by other processes.
```

This is an **OOM at startup**, not a running OOM. The gap `T - F` is what another
process already holds. Report it as such.

---

## 2. KV pool capacity lines

Source: `vllm/v1/core/kv_cache_utils.py` line ~1744.

```
GPU KV cache size: 4,010,585 tokens
Maximum concurrency for 1,048,576 tokens per request: 3.82x
```
Patterns: `RE_KV_TOKENS`, `RE_MAX_CONCURRENCY`.

The source comment above this call is worth knowing, because it explains why the
analyzer does not trust a division:

> "Sourcing this from the concurrency calculation handles hybrid layouts
> correctly: SWA / chunked-local groups have a per-request block count that's
> capped by their window, so a naive `num_blocks // num_groups * block_size`
> formula underestimates capacity for these models."

So on a hybrid-attention model, `pool_tokens / request_len` is an underestimate
for short requests and unreliable in general. Prefer the reported line.

---

## 3. What is NOT in the log (and must be supplied)

| Quantity | Where to get it |
|---|---|
| Chip HBM total | `npu-smi info`, or `references/ascend-hardware-specs.json` |
| HBM held by *other* processes | `npu-smi info` process table |
| HCCL buffer cost | grows with device count; not in the vLLM log |
| Whether the pool is in HBM or host DRAM | a KV-offload setup changes the answer entirely — check for offload flags |

A blank in the analyzer's table means "this log did not state it", which is a
different fact from "it is zero".

---

## 4. Keeping this file honest

When a pattern stops matching after a framework upgrade:

1. Find the log string in the **installed** source, not on the web.
2. Record the new string and the file/line here.
3. Add the version pair to the provenance line at the top.
4. Re-run the analyzer against a real log and confirm the numbers moved for a
   reason you can name.
