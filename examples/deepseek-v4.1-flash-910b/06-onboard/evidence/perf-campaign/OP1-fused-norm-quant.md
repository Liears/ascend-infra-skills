# OP1 — operator layer, phase 1: target selected, kernel validated

## Fresh trace on the OPTIMIZED config (DRAFT_GRAPH=1 + FUSED_MC2=1)

The structural shift vs the baseline trace is the campaign's key operator-
level fact:

| core split | baseline config | optimized config |
|---|---:|---:|
| COMMUNICATION | 83.06% | **39.04%** |
| MIX_AIC | 7.95% | **33.58%** |
| AI_VECTOR_CORE | 5.05% | **15.79%** |
| AI_CORE | 2.64% | 6.45% |

Compute is now 60% of device time — the operator ceiling grew from ≤17%
(baseline theory) to a much larger share. Top compute ops: SparseFlashMla
11.09% (vendor), GEMM family ~19% (vendor), and the elementwise chain:

| op | launches (per 4-req trace) | share |
|---|---:|---:|
| RmsNorm (5120/1280/512) | 34,224 | 2.14% |
| aclnnDynamicQuantV2 | 25,944 | 1.11% |
| InplacePartialRotaryMul | 26,680 | 0.95% |
| Cast (incl. 18.5k scalar (6,) int32) | 48,180 | 0.90% |
| Add | 11,420 | 0.80% |

## Target: Add + RmsNorm + DynamicQuant → one kernel

The per-layer pattern is `residual Add → RMSNorm(5120) → 2× per-token int8
dynamic quant` (each normed output feeds two quantized linears — the 2:1
quant:norm ratio in the trace).

**No need to write a kernel: CANN already ships it.**
`torch_npu.npu_add_rms_norm_dynamic_quant(x1, x2, gamma, epsilon, output_mask,
y_dtype)` fuses Add + RMSNorm + up to two DynamicQuant paths.

Validated on the live NPU (2026-10-15, shape (6,5120) bf16, eps 1e-20):

| check | result |
|---|---|
| output semantics | (q1, q2, new_residual, scale1, scale2) |
| residual exactness | maxdiff **0.0** vs reference |
| int8 vs reference | ±1 quant step (rounding order — pre-bf16-round vs post) |
| micro-benchmark | **296.6 → 59.7 µs = 4.97×** per call |
| graph capture | **SAFE** — NPUGraph capture, replay, outputs follow inputs |

## Wiring plan (phase 2, next unit of work)

Patch `deepseek_v41/model.py` layer sites (`x = self.input_layernorm(x)` at
:851/:872 + post_attention_layernorm) to call the fused op, and thread the
pre-quantized `(q, s)` into the two quantized linears by having
`w8a8_dynamic.apply` accept a pre-quantized input (skip its internal
`npu_dynamic_quant`). Dual-path output covers both consumers in one call.
Gate after wiring: GSM8K-100 quick, then 200, then end-to-end probe.

## Also spotted (next targets, ranked)

1. `npu_rope_quant_kvcache` exists — RoPE+quant+KV-write fusion for the
   26,680 InplacePartialRotaryMul launches (+ kv writes)
2. 18,540 scalar `(6,)` INT32 casts — pure launch overhead, worth hunting
3. SparseFlashMla (11%) and GEMM family (19%) — vendor kernels, only
   upstream-track improvements apply
