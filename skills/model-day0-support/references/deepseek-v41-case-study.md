# Case Study — DeepSeek-V4.1-Flash on 910B (A2)

A worked Day-0 gap analysis. Sources are cited; nothing here is invented.

**Status of this case study:** analysis of public artifacts, dated 2026-10-14.
It describes *what the work consists of*, not a completed run on this machine.

---

## 1. The gap, stated precisely

| Layer | V4-Flash | V4.1 | Source |
|---|---|---|---|
| vLLM-Ascend support matrix | ✅ (V4-Flash, V4-Pro) | ❌ not listed | `vllm-ascend/docs/.../supported_models.md` |
| Model classes in vllm-ascend | `models/deepseek_v4.py`, `deepseek_v4_mtp.py` | none | tree listing |
| `deepseek_v41` in transformers 5.3.0 | — | ❌ only v2/v3/vl configs | installed tree |
| Upstream SGLang V4.1 | — | ✅ active (Roadmap #42170 + ~40 PRs) | GitHub API |
| Public Ascend port | — | ✅ `chiro2001/deepseek-v4.1-flash-ascend910B` (W4A8, A2/A3) | repo audit |

**Conclusion:** Day-0 for V4.1 on Ascend is *not* greenfield. A public port
exists and is the correct starting point. The Day-0 plan is therefore
**adopt-and-verify**, not build-from-zero. That is the normal and good outcome —
finding this before writing a port is the value of running the finder first.

---

## 2. What V4.1 adds over V4 (from the model config)

Read from `/mnt/public/models/DeepSeek-V4.1-Flash/config.json`:

| Feature | Config evidence | Why it matters on Ascend |
|---|---|---|
| 384 routed experts, 6/token | `n_routed_experts: 384`, `num_experts_per_tok: 6` | MoE dispatch is collective-heavy; EP/TP layout dominates perf |
| MLA with `head_dim: 512`, `kv_lora_rank` | `head_dim: 512`, `q_lora_rank: 1280`, `o_lora_rank: 1024` | attention memory + fused-kernel shape constraints |
| Compressed/sparse attention | `compress_ratios`, `index_*`, `kv_source_layer_ids` | needs an Ascend attention backend for the sparse indexer |
| **Engram** | `engram_layer_ids: [1,14]`, `engram_num_embeddings: [...384M...]` | unique feature; enormous embedding tables that must live in host DRAM |
| MTP / speculative | `num_nextn_predict_layers: 3` | separate MTP head + scheduler wiring |
| Vision tower | `vision_config` present | multimodal processor |

**Engram is the distinctive Day-0 risk.** Its embedding tables are hundreds of
millions of rows; on Ascend they must be host-resident with device-side lookup,
which is a mechanism no other model in the runtime has. Public port patches
confirm this is where the work concentrated: `engram_hbm.py`,
`engram_device_index.py`, `engram_jit_kernel.py`, `engram_gate.py`.

---

## 3. The PR DAG this implies

```
[1] model skeleton (deepseek_v41 package)          BLOCKS all
[2] attention: MLA + sparse indexer (eager)        BLOCKS [4],[6]
[3] MoE: 384-expert dispatch/combine               BLOCKS [4]
[4] runnable eager end-to-end                      BLOCKS [5],[6]
[5] quant recipe (W4A8) + runtime mapping          BLOCKS memory gate
[6] Engram: host-resident tables + device lookup   BLOCKS memory + fastpath
[7] MTP head + scheduler wiring                    independent (perf)
[8] vision tower + processor                       independent (if multimodal)
[9] fps: fused attention/MoE/RoPE                  independent (perf)
[10] graph mode                                    after [4]-[9] stable
```

Node [6] has no analogue in other models and is the schedule risk.

---

## 4. Evidence the public port carries (from its audit)

Audited with `audit_repo.py`: 33 patches (12 against `vllm_ascend`, 4 against
`vllm`), a REPRO.md, an EXPECTED_PERF.md, `a2/scripts/serve_a2_offload.sh`, and
ModelScope weights. Red flag: the toolchain appears in multiple versions
(CANN 9.1.0 and 9.2.0; driver 25.5.5 and 26.1.1), so **which CANN the numbers
belong to must be resolved before citing any of them.**

That red flag is the single most important output of the audit — it is exactly
the question Day-0 verification has to answer.

---

## 5. What a Day-0 plan for this must gate on

| Gate | V4.1-specific content |
|---|---|
| G1 Load | 384 experts + engram tables load; param count matches |
| G2 Accuracy | GSM8K-style check at the expected rate, fast paths on |
| G3 Fast-path | profiler shows the fused attention/MoE ops, *and* the engram device lookup |
| G4 Memory | fits 8×910B3 at the target context with the KV budget stated; engram tables accounted for |
| G5 Mode | eager first; graph mode only after eager is stable |

G4 is the binding constraint on A2: Engram's host-resident tables plus a 1M
context KV budget on 64 GB/card is where the arithmetic either closes or does not.
