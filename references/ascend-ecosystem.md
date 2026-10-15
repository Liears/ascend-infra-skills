# Ascend Ecosystem Index

Curated, verified index of the Ascend ecosystem relevant to this skill suite.
Every entry was checked directly (GitHub API or clone) on 2026-10-15 — star
counts and repo existence are point-in-time facts, not folklore.

Purpose: when a skill needs a capability that lives outside this suite, it
should point here rather than duplicating. The index also records **what each
repo taught this suite** (absorbed ideas are marked).

---

## 1. Skill / agent ecosystems

| Repo | What it is | Relationship to this suite |
|---|---|---|
| `gitcode.com/cann/cannbot` (+ `cannbot-skills`, `cannbot-knowledge`, `cannbot-sentry`) | Official CANN agent layer: operator dev, model migration, inference optimization (14 `model-infer-*` skills) | **The execution layer we deliberately do not duplicate.** Our skills point to it for profiling capture, KV cache, fusion, quantization, graph mode |
| `gitcode.com/cann/cann-bench` | Evaluates AI models on Ascend-C operator-code generation (4 complexity levels, three-layer eval framework) | Harness-structure reference; evaluates coding agents, not serving |
| `github.com/ascend-ai-coding/awesome-ascend-skills` ★173 | Community Ascend skills knowledge base: 7 domains, 46+ skills (inference domain has 13: `vllm-ascend`, `vllm-bench-serve`, `ais-bench`, `inference-precision`, `msmodelslim`, …), external-sync mechanism, bundle installs | **Closest peer.** Encyclopedic (one skill per tool) vs our methodological (one flow + evidence discipline). Complementary: its tool skills are reference targets; absorbed two ideas (§4) |
| `github.com/Krusty84/triton-ascend-agent-dev-kit` ★106 | 24 task-level agent skills for Triton-Ascend kernel work (build-fused-attention, tune-kernels, validate-accuracy, UB-overflow prevention…) | Operator-layer granularity exemplar; task-level naming (verb+object) worth imitating |
| `github.com/anthropics/skills` ★181k | Anthropic's official Agent Skills repo | SKILL.md format conventions |
| `github.com/obra/superpowers` ★299k | Agentic skills framework organized as a methodology | Skills-as-workflow philosophy (same as ours) |
| `github.com/BBuf/AI-Infra-Auto-Driven-SKILLS` | The NVIDIA/CUDA ancestor of this suite | Methodology source (evidence rules, skill shapes) |
| `github.com/BBuf/KDA-Pilot` ★297 | BBuf's sibling repo: kernel-development agent loops | Boundary reference for the operator layer |

## 2. Official Ascend stack (GitHub mirrors of gitcode.com/Ascend)

| Repo | Role |
|---|---|
| `Ascend/pytorch` ★587 | torch_npu adapter — the authoritative source for the profiler formats our `npu-profiler-analysis` documents |
| `Ascend/MindSpeed-LLM` ★62 / `MindIE-LLM` ★28 | Training / inference framework layers; MindIE is one of the frameworks our benchmark skill compares |
| `Ascend/msprobe`, `msinsight`, `msopprof` | Official precision/perf probing family |
| `Ascend/docs` ★18 | Docs & FAQ |
| `Ascend/TransferQueue` ★164 | Async streaming data management |

## 3. GitCode `cann` org (85 repos, by layer)

- **Recipes & samples** — `cann-recipes-infer` (official inference recipes, `integration/vllm/<model>/` format; **contains a DSV4.1-Flash recipe for a different hardware path**: 1× 950PR + x86 CPU MoE offload, MXFP4, CANN 9.2.0-beta.2 — parallel to the 8×910B3 W4A8 path our example uses), `cann-recipes-train`, `cann-samples`, `cann-learning-hub`, `cann-ops-competitions`
- **Operator libraries** — `ops-transformer` (LLM ops), `ops-nn`, `ops-math`, `ops-cv`, `ops-gnn`, `ops-fft`, `ops-collections`
- **Programming layers** — `asc-devkit`, `pyasc`/`pypto`/`pto-isa`, `catlass`, `xla-npu` (JAX/XLA backend), `torchtitan-npu`
- **System layers** — `ge` (graph engine), `runtime`, `hccl`/`hcomm`/`hixl`/`shmem` (communication), `driver`, `amct` (model compression)
- **CANNBot family** — see §1

## 4. DeepSeek official Ascend stack (the V4 family's upstream)

| Repo | Why it matters to us |
|---|---|
| `deepseek-ai/DeepGEMM-Ascend` ★578 | The MoE GEMM kernels behind the `aclnnGroupedMatmul*` ops in our traces |
| `deepseek-ai/DeepEP-Ascend` ★255 | The EP communication library — the upstream of the **83% communication share** our real trace measured |

## 5. Community tooling

| Repo | Use |
|---|---|
| `triton-lang/triton-ascend` ★192 | Triton backend for Ascend |
| `youyve/nputop` ★139 | Interactive NPU process viewer (htop-style) |
| `omni-ai-npu/omni-infer` ★131 | Ascend inference acceleration suite |
| `linzm1007/nano-vllm-ascend` ★95 | Minimal vLLM-Ascend for learning the serving pipeline |
| `TencentARC/mllm-npu` ★95 | Multimodal LLM training on Ascend |
| `LMCache/LMCache-Ascend` ★92 | LMCache plugin for Ascend (KV-cache layer, same domain as DRAM offload) |

## 6. Serving frameworks & harnesses

| Repo | Role in our suite |
|---|---|
| `vllm-project/vllm-ascend` ★2979 | The serving framework our example deploys; support matrix + model tutorials are the upstream of default configs; our PR-history skill mines it |
| `sgl-project/sglang` ★37k | Reference implementation for V4.1-class models; its cookbook is the upstream of the deployment configs our dossier records |
| `modelscope/evalscope` ★3542 | L2 harness of choice for the accuracy gate (OpenAI-compatible backend) |
| `open-compass/opencompass` ★7515, `EleutherAI/lm-evaluation-harness` ★14223, `huggingface/lighteval` ★2560, `stanford-crfm/helm` ★2944 | Evaluation-framework references |
| `triton-inference-server/perf_analyzer` (genai-perf) ★164 | Standard TTFT/ITL metric definitions |

## 7. Model-weight ecosystem

| Source | Note |
|---|---|
| ModelScope `Eco-Tech` org | Publishes Ascend-quantized official weights (e.g. `DeepSeek-V4-Flash-w8a8-mtp`) |
| ModelScope `chiro2001/*` | W4A8 Ascend weights for V4.1-Flash (used in our worked example) |
| Weight dirs on shared boxes | Always run the model-dir self-check (`check_model_dir.sh` pattern) before serving |

---

## Ideas absorbed from this ecosystem (recorded 2026-10-15)

1. **Two-layer config search** + **phase-scaled SLO search** (from
   `awesome-ascend-skills`'s `vllm-bench-serve` auto-optimization): launch
   configs are restart-costly (our phased planner); runtime load
   (concurrency/request-rate) is cheap to probe (warmup-abort → exponential →
   binary → validation, with sample counts scaled per phase). Absorbed into
   `serving-config-search`.
2. **Tensor-level precision localization** (from its `inference-precision`
   tensor-dump-compare): when end-to-end accuracy fails, layer-wise tensor
   comparison answers *where* it diverges — including the operational gotcha
   that dumps require `TORCHDYNAMO_DISABLE=1`. Absorbed as the escalation
   ladder in `llm-serving-accuracy-gate`.
3. **Rich SLO vocabulary** (`p99_ttft`, `mean_tpot`, `success_rate`,
   `goodput_ratio` rather than a bare boolean) — absorbed into
   `llm-serving-auto-benchmark`'s recording discipline.

## Maintaining this index

- Verify a repo exists before adding it (GitHub API or a clone); record the
  check date.
- Star counts are context, never quality claims.
- An entry must state **what it offers this suite**; a repo we would never
  point to does not belong here.
