# Ascend Runtime Extension Points

Where each kind of Day-0 work lands in a vLLM-Ascend tree.
Read from the installed source (vllm-ascend `0.23.1.dev14`) on 2026-10-14.

Paths are relative to the `vllm_ascend/` package unless noted.

---

## 1. Model implementations

```
vllm_ascend/models/<ModelName>.py          # the model class
vllm_ascend/models/<ModelName>_mtp.py      # MTP / speculative head, if the model has one
vllm_ascend/models/<ModelName>/            # for architecture-heavy models, a package
    model.py                               #   the transformer
    engram_*.py                            #   model-specific submodules
    indexer.py                             #   sparse-attention indexer
```

Registration goes through `vllm_ascend/models/__init__.py` and the model
registry. A model added as a package (`models/deepseek_v41/`) keeps its
model-specific complexity out of the shared runtime — do this when the
architecture has features no other model shares.

**Observation:** reference ports of V4.1-class models keep `engram_*` in a
model-local package precisely because the feature is unique to that family.

---

## 2. Attention

```
vllm_ascend/attention/<backend>.py         # attention backends (DSA, MLA, FA variants)
vllm_ascend/ops/rope_*.py                  # rotary variants
vllm_ascend/ascend_forward_context.py      # per-step runtime context
```

A new attention variant (sparse / compressed / sliding-window hybrids) needs a
backend file and a selection rule in the forward context. Do the **eager,
correctness-first** version before any fused version.

---

## 3. MoE / FFN

```
vllm_ascend/ops/fused_moe/token_dispatcher.py   # dispatch/combine over the collective
vllm_ascend/ops/fused_moe/                      # other MoE plumbing
```

New routing or scoring functions (e.g. a different top-k selection or a scaled
softplus gate) usually do **not** need a new dispatcher — they need a change in
how the expert mask and gate are computed, then validated that the fused path
still engages.

**Real example:** a MoE expert-mask that compares a contiguous range instead of
building a full mask is a local change in this area with a measurable win.

---

## 4. Quantization

Two sides, and both are needed for a quantized Day-0:

```
<msmodelslim>/…                            # the offline recipe that produces the weights
vllm_ascend/…/quant_config.py or equivalent # the runtime that consumes them
```

The offline side is a separate repo (ModelSlim). A published W4A8/W8A8 model
needs its recipe recorded, because "which layers were quantized and how" is not
recoverable from the weight file alone.

**Rule:** the runtime must fail loudly if a quantized layer's config is missing,
not silently dequantize.

---

## 5. Graph mode

Graph capture on Ascend is a first-class, memory-consuming feature (roughly 1–2
GiB of HBM in the runs observed here). It is not a free optimization.

```
vllm_ascend/…    # graph capture integration, torch.compile / aclgraph / npugraph_ex
```

Graph mode comes **after** the eager path is correct. A model that only works
because graph capture hides a host-sync bug is not Day-0 ready.

---

## 6. Scheduler / serving

```
vllm/v1/core/sched/scheduler.py            # upstream vLLM — modifying this is a bigger claim
```

Some Ascend features need a scheduler change (e.g. protecting decode from
prefill starvation, or admission control). These are **upstream-vLLM** patches,
not vllm-ascend patches, and carry a higher review bar: they affect every
platform. Prefer a vllm-ascend-local solution if one exists.

---

## 7. Where to look when a fused path "works"

If output is correct but slow, the fused kernel likely did not run. Check, in
order:

1. The profiler (`npu-profiler-analysis`) — is the fused op name present in
   `kernel_details.csv`?
2. The dispatch condition — what shape/dtype/flag gates the fast path?
3. The quant config — a mismatched format can force a dequant fallback.

**Never conclude "the fused path is used" from a correct output.** Assert the
op ran.
