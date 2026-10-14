# The FLOP Model

The formulas the simulator uses, and the assumptions baked into each.

Convention: a multiply-add counts as **2 FLOPs**. A matrix multiply
`[m,k] x [k,n]` is therefore `2·m·k·n` FLOPs.

---

## 1. Attention

### 1.1 Dense (standard) attention

| op | FLOPs | note |
|---|---|---|
| `q_proj` | `2·T·H·(n_h·d_h)` | T = tokens = batch·seq |
| `k_proj`, `v_proj` | `2·T·H·(2·n_kv·d_h)` | combined row in the table |
| `o_proj` | `2·T·(n_h·d_h)·H` | |
| `qk_av` | `2·(2·B·n_h·d_h·L_kv)·S` for prefill, halved for causality | decode: `L_kv` = context length |

**The decode case is where people go wrong.** At `S=1` the projections
shrink to almost nothing, and attention is entirely `qk_av` against the whole
KV context. A decode FLOPs number without `--kv-len` is off by the context
factor — which is why the tool takes it as a separate argument.

### 1.2 MLA (DeepSeek family)

When the config has `q_lora_rank` / `kv_lora_rank`, the projections are not
dense. The tool models:

| op | FLOPs |
|---|---|
| `q_a_proj` | `2·T·H·q_lora_rank` |
| `q_b_proj` | `2·T·q_lora_rank·(n_h·d_h)` |
| `kv_a_proj` | `2·T·H·kv_lora_rank` |
| `kv_b_proj` | `2·T·kv_lora_rank·((n_h+2)·d_h)` |

The `+2` in the kv up-projection is the RoPE key and the value head. Getting
the MLA path wrong (using the dense form) inflates attention FLOPs by roughly
the ratio of `d_h` to `kv_lora_rank`, which on DeepSeek models is a factor of
several.

**Caveat:** V4.1's config also carries `compress_ratios`, `index_*` and
`kv_source_layer_ids` — a compressed/sparse attention path. The static model
does **not** model the sparsity saving, so its attention FLOPs for V4.1 are an
upper bound relative to a run that uses the sparse path.

---

## 2. FFN / MoE

### 2.1 Dense FFN

| op | FLOPs |
|---|---|
| `gate_up` | `2·T·H·(2·I)` |
| `down` | `2·T·I·H` |

with `I` = `intermediate_size`. The `2·I` is a fused gate+up projection
(SwiGLU convention); a model with a plain MLP would use `I`, not `2I`.

### 2.2 MoE

Per token, exactly `topk` experts are activated:

| op | FLOPs |
|---|---|
| `experts.gate_up` | `2·T·H·(2·I_moe)·topk` |
| `experts.down` | `2·T·I_moe·H·topk` |
| `router` | `2·T·H·n_experts` |
| shared experts | as a dense FFN of `I_moe`, counted once |

**This is an upper bound on useful work.** Real MoE adds padding to a capacity
factor, may drop or re-route tokens, and the router's cost is real but the
expert FLOPs assume perfect balance with no padding.

**Shared-expert sizing trap:** on MoE-only configs (the DeepSeek family) there
is often **no top-level `intermediate_size`**; the shared experts size to
`moe_intermediate_size`. A simulator that reads only `intermediate_size` emits
zero-FLOP shared experts and understates the model. This tool falls back to
`moe_intermediate_size`.

---

## 3. Per-layer vs once

Attention and FFN are per-layer, so their FLOPs multiply by
`num_hidden_layers`. `lm_head` runs **once**, not per layer — the tool adds it
separately. Multiplying `lm_head` by the layer count is a common and large
error (40 layers × 1.3 GFLOP = 52 GFLOP of pure fiction on V4.1).

---

## 4. MFU

```
achieved_TFLOPS = total_FLOPs / (latency_ms / 1000) / 1e12
MFU% = 100 · achieved_TFLOPS / peak_TFLOPS
```

MFU is only as good as `peak_TFLOPS`. Ascend per-SKU peaks are quoted
inconsistently, so the tool requires you to supply it with a source. A number
like "0.5% MFU" for a single-token decode is not a defect — decode is
memory-bound and a low computed MFU is the expected result, not a bug. MFU is
a useful metric for **prefill and training-shaped** workloads, and a misleading
one for decode.

---

## 5. What this model deliberately does not do

- **Apply TP/EP division.** Replicated components (routers, norms, shared
  experts) do not shard, so per-device FLOPs is not `total / TP`. The tool
  records the degrees and warns instead of dividing.
- **Model the sparse/compressed attention saving.**
- **Count memory traffic.** Many decode ops are bandwidth-bound; FLOPs say
  nothing about them. Use the capacity planner and a real trace for that.
- **Assume a peak.** Never.
