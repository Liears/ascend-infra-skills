# Onboarding evidence — DeepSeek-V4.1-Flash on 8×910B3

Real artifacts from a live run on this machine, 2026-10-15. Not synthetic.

## What was actually run

| Step | Result |
|---|---|
| Weights: `chiro2001/DeepSeek-V4.1-Flash-w4a8-Ascend` | 72/72 shards, 490 GB, ModelScope `Snapshot ready` |
| Engram reassembly (`engram_int8/reassemble_engram_weights.sh`) | both files rebuilt, **sha256 verified** (98.3 GB + 98.3 GB) |
| Image build (`scripts/build_image.sh`) | `dsv41-a2:v9`, 15 patched files byte-for-byte md5 verified |
| Model dir self-check (`tools/check_model_dir.sh`) | pass, 1 warning (vision rotation unconfirmed) |
| Service (`scripts/serve_a2.sh`, TP8, GPU_UTIL=0.92, bat=8192) | **up**, health HTTP 200 |
| Real inference (`/v1/chat/completions`) | correct output |

## The numbers from the live server

From `real-memory-lines.txt` (extracted from the server log):

```
Available KV cache memory: 13.60 GiB
GPU KV cache size: 2,942,071 tokens
Maximum concurrency for 1,048,576 tokens per request: 2.81x
Free memory on device (60.61/60.96 GiB) on startup. Desired GPU memory utilization
  is (0.92, 56.08 GiB). Actual usage: 38.7 GiB for weights, 3.21 GiB for peak
  activation, 0.56 GiB for non-torch memory, 1.02 GiB for NPU graph memory.
```

Note the four-bucket split: **weights 38.7 GiB** and **NPU graph memory 1.02 GiB**
are the two large consumers, and the graph-memory bucket is the Ascend-specific
one that a GPU-shaped accounting model would miss.

## Acceptance gate results (`acceptance-REPORT.md`)

| Gate | Result |
|---|---|
| `static_kernel` not silently downgraded | ✅ 0 hits at `static_kernel.py:650` |
| Engram local-owner self-check | ✅ `fast` |
| KV capacity > 2,800,000 (at GPU_UTIL=0.92) | ✅ 2,942,071 tokens |
| 8K quote decode | 27.2 tok/s, A=1.70, ms/step=62.32 |
| 32K quote decode | 47.7 tok/s, A=2.91, ms/step=60.91 |
| Vision 23 cases | ⏭ skipped — no official image dir given |
| GSM8K-200 | ⏭ not run (MODE=quick) |

## The one silent degradation the log shows

```
WARNING [patch_triton.py:332] NPU Triton causal_conv1d_update is unavailable
(cannot import name 'causal_conv1d_update_npu' ...); falling back to the PyTorch
implementation, which syncs per request and therefore stalls ACL graph capture at
decode-FULL.
```

**This is exactly the failure mode the `model-day0-support` skill's fast-path
gate (G3) exists to catch.** The model runs and produces correct output, but one
kernel is not the fused one, and the only signal is a single WARNING line. Any
Day-0 claim for this model must state whether this fallback applies to the
path being claimed.

## What the skills did with the real data

- `capacity_analyzer.py` on the live `serve.log` reproduced the HBM split and
  the 2,942,071-token pool, and estimated concurrency at three request lengths.
- Running it **found a bug in the skill**: vLLM prints
  `` Replace gpu_memory_utilization with `--kv-cache-memory=N` `` as a
  *suggestion*, and the parser was reporting it as an effective setting. Fixed:
  the parser now excludes suggestion lines. This is the second-level validation
  doing its job — a synthetic log would never have surfaced it.

## What this does NOT establish

- Performance parity with the vendor's published numbers. The measured 27.2
  tok/s @8K is well below the repo's `EXPECTED_PERF` A3 figures; A2 is a
  different, slower device and this run used `MODE=quick` with `GPU_UTIL=0.92`,
  `bat=8192`, no PGO (the PGO artifact was absent → the script logged
  `PYTHON_PGO=1 但没有可用的 PGO 产物 → 降级为不挂`).
- Accuracy. GSM8K was not run in `MODE=quick`.
- Vision. Skipped: no official image directory was provided.

Recording these as *not established* is the point of the evidence rules.

---

## A fail-open bug found while checking DRAM offload

`run_test.sh` explicitly does not verify DRAM offload, so we ran the repo's own
`a2/scripts/verify_dram_offload.sh` against the live service. Its closing line
was:

```
✅ 硬判据全过：DRAM 卸载**存取双向**都验证到了
```

**That line is wrong, and the criteria above it say so.** The same output shows
every hard criterion as `? 未验` (`Δ=None`) and H3 as `✗`. Reading the source
(`tests/verify_dram_offload.py`, the `line()` helper) explains why: `hard` is
only set to 1 on an explicit `False`, so a criterion that evaluates to `None`
("unverified") leaves `hard == 0`, and `hard == 0` prints the success banner. A
run where **nothing** could be measured reports success.

Why everything was `None` here — verified directly against the live service:

| Probe | Result |
|---|---|
| `POST /reset_prefix_cache` | **HTTP 404** |
| `vllm:kv_offload_total_bytes_total` in `/metrics` | **absent** |
| `cache_config_info` | `enable_prefix_caching="False"`, `kv_offloading_size="None"`, `kv_offloading_backend="native"` |

So two separate facts:

1. **This deployment has DRAM offload disabled.** We launched with
   `scripts/serve_a2.sh`, not `serve_a2_offload.sh`; the offload path is a
   separate launcher. Expecting offload counters from it was the mistake.
2. **The verifier fails open.** Given (1), it should have refused to conclude.
   Instead an unmeasurable run reported a pass.

Both belong in the skills:

- the **capacity planner** must report `kv_offloading_size=None` as
  "offload not enabled", not as an absent feature — it is now in
  `references/npu-memory-lines.md` as a field to read;
- the **Day-0 skill's G3 gate** is precisely about this: "the fast path ran" must
  be *observed*, never inferred from a wrapper's summary line. A tool that
  reports success when it measured nothing is the same class of hazard as a
  model that silently falls back to a slow kernel.

Per the evidence rules, the conclusion here is **"DRAM offload is not enabled in
this deployment and was therefore not verified"** — not "offload works".

