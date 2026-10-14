# Ascend Benchmark Tooling — Verified Commands

Read from the installed source on this machine (vllm-ascend `0.23.1.dev14`, vLLM
`0.23.0`, 2026-10-14). Do not copy benchmark flags from GPU tutorials; the
Ascend entry points and defaults differ.

---

## 1. `vllm bench serve`

Source: `vllm-ascend/tools/vllm_bench.py`, which shells out to:

```
vllm bench serve
    --backend openai-chat
    --trust-remote-code
    --served-model-name <name>
    --model <model_path>
    --tokenizer <model_path>
    --metric-percentiles 50,90,99
    --host <ip> --port <port>
    --save-result --result-filename <file>
    --endpoint <path>
```

Then the load shape (`--random-input-len`, `--random-output-len`,
`--max-concurrency`, `--num-prompts`, `--request-rate`). These are the fields
that must match across frameworks.

**Result file** is JSON with per-percentile TTFT/TPOT/ITL and an overall
throughput. It does **not** record the CANN/driver version — you must add that
yourself when writing the benchmark row.

---

## 2. `tools/aisbench.py`

Huawei's AISBench wrapper. It downloads its own dataset
(`maybe_download_from_modelscope` in the same file) and runs a fixed task set.
Use it when you want numbers comparable to Huawei's published ones; use
`vllm bench serve` when you want full control of the workload shape.

Its dependency on ModelScope means an offline machine cannot run it without a
pre-populated cache.

---

## 3. Launching the server on Ascend

The image may provide `vllm` on `PATH` with the Ascend plugin auto-loaded
(`vllm_ascend` registers itself), or require `python -m vllm.entrypoints...`.
Check before writing a plan:

```bash
which vllm && vllm --version
python3 -c "import vllm_ascend; print(vllm_ascend.__version__)"
```

Ascend-specific flags worth knowing (names differ from upstream vLLM):

| Flag | Effect |
|---|---|
| `--gpu-memory-utilization` | **still named `gpu`** on Ascend; governs NPU HBM |
| `--kv-cache-memory` | set the KV pool directly instead of via the fraction |
| `--additional-config '{"...":...}'` | vllm-ascend JSON config channel (torchair graph, etc.) |
| `--tensor-parallel-size` / `--data-parallel-size` | **degrees count devices**, not cards |

When `--kv-cache-memory` is set, the utilization fraction no longer governs the
pool. Report which knob you used.

---

## 4. Device selection

`ASCEND_RT_VISIBLE_DEVICES` selects devices, and on multi-die parts the value
counts **devices**:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7     # 8 devices
```

On an A3 server (2 dies/card) that is 4 physical cards, not 8. State the unit
in every row.

---

## 5. What the evidence block must contain

Per `docs/EVIDENCE-RULES.md`:

```
model, framework(+commit), chip, cards, devices, toolchain(CANN/driver/torch_npu),
quant, workload(in/out/conc/prefix_cache), graph_mode, hccl, SLA, both commands,
artifact path, capture date
```

The comparison script's row schema is exactly this set. Missing any required
field makes the row untrusted and it is excluded from ranking.

---

## 6. Traps observed in real Ascend benchmarks

| Trap | Detail |
|---|---|
| tok/s basis | Spec-decode models: `A × 1000 / ms_per_step` ≠ `gen_tokens / wall`. State which. |
| accept-length as a metric | On V4.1-family models A is a *draw per measurement*, not an optimizable amount. Do not report A as a KPI; use it as a "did the fast path engage" anchor. |
| proxy hijacking health checks | An HTTP proxy intercepting `127.0.0.1` turns a healthy server into "startup hang". Verify with `curl --noproxy '*'`. |
| CPU binding hang | CPU-pinning to a full NUMA node can hang `migratepages` indefinitely on a loaded box. If a multi-card launch never becomes ready, test with binding off. |
| prefix cache default | Recording `prefix_cache=0` when the server had it on silently inflates TTFT comparisons. Read the launch command, not the default. |
