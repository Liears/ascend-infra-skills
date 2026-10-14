# NPU HBM accounting (vllm-ascend startup log)

> All figures are NPU HBM. The log wording says "GPU" because
> vllm-ascend inherits vLLM's strings; read them as NPU.

| Quantity | Value | Unit |
|---|---:|---|
| HBM free at startup | 62.31 | GiB |
| HBM total | 64.00 | GiB |
| gpu_memory_utilization | 0.92 |  |
| requested memory | 58.88 | GiB |
| weights | 24.15 | GiB |
| peak activation | 1.82 | GiB |
| non-torch memory | 3.41 | GiB |
| NPU graph memory | 2.07 | GiB |
| non-KV subtotal | 31.45 | GiB |
| available KV cache | 27.43 | GiB |

- KV pool capacity: **4,010,585 tokens**
- Reported max concurrency for 1,048,576 tokens/request: **3.82x**

## Concurrency estimate

| request length | est. concurrent requests | basis |
|---:|---:|---|
| 8,192 | 488.96 | pool-bound 489.6 vs scaled 489.0 (from 3.82x @ 1,048,576); took the smaller |
| 32,768 | 122.24 | pool-bound 122.4 vs scaled 122.2 (from 3.82x @ 1,048,576); took the smaller |
| 131,072 | 30.56 | pool-bound 30.6 vs scaled 30.6 (from 3.82x @ 1,048,576); took the smaller |

## Diagnosis
- weights 24.15 GiB = 41.0% of the 58.88 GiB accounted for
- NPU graph memory is 2.07 GiB — graph capture is a first-class consumer on Ascend, not a rounding error
- KV budget was set explicitly (--kv-cache-memory=26518185984); the utilization fraction does not govern the pool in this mode

---
Estimates are derived from the log's own numbers. For hybrid-attention
models a linear tokens/length division understates capacity; trust the
reported `Maximum concurrency` line over any division you do yourself.
