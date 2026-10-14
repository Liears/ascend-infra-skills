# PR history — `deepseek` in vllm-ascend-full

- repo HEAD: `2b8da69474a7` (last commit 2026-10-14)
- commits matching `deepseek`: 15

By kind: bugfix=6, ci=2, doc=4, feat=1, perf=1, refactor=1

## Timeline

| date | PR | kind | files | subject |
|---|---|---|---:|---|
| 2026-10-14 | #17263 | perf | 3 | [Performance][DSpark] Stack DeepSeek V4 context WKV projections (#17263) |
| 2026-10-14 | #18662 | doc | 1 | [Doc]: Document DeepSeek V4 TurboQuant serving options (#18662) |
| 2026-10-14 | #18628 | bugfix | 1 | [BugFix] Request huge pages before Engram host registration (#18628) |
| 2026-10-13 | #18418 | bugfix | 3 | [Bugfix] Validate DeepSeek V4 tensor parallel limits (#18418) |
| 2026-10-13 | #18538 | doc | 58 | [Doc] Translated Doc files 2026-10-13 (#18538) |
| 2026-10-13 | #18493 | doc | 16 | [Doc][Misc] Add note in the model tutorials (#18493) |
| 2026-10-12 | #18392 | bugfix | 42 | [BugFix][Doc] Delete the usage of jemalloc (#18392) |
| 2026-10-12 | #18322 | bugfix | 10 | [BugFix][Model] Backport SWA bounded replay to vLLM 0.30 (#18322) |
| 2026-10-12 | #18422 | ci | 12 | [CI]  Remove old-version model test cases (#18422) |
| 2026-10-12 | #17500 | refactor | 15 | [Refactor][Worker] Remove legacy speculative PP compatibility (#17500) |
| 2026-10-12 | #17870 | ci | 64 | [CI][Misc] mian2main vllm 0928 (#17870) |
| 2026-10-12 | #18313 | doc | 11 | [Doc][Misc] Model offline modification (#18313) |
| 2026-10-12 | #18281 | bugfix | 6 | [BugFix][PCP] Support multimodal inputs for DeepSeek V4.1 (#18281) |
| 2026-10-11 | #18257 | bugfix | 2 | [BugFix][Model] Avoid false buffer mutation guard in DeepSeek forward (#18257) |
| 2026-10-11 | #18112 | feat | 11 | [Feature][PCP] Support fixed-K speculative decode request sharding (#18112) |

## Per-commit files

### 2026-10-14 — [Performance][DSpark] Stack DeepSeek V4 context WKV projections (#17263)
- PR: #17263
- sha: `1e1541098561` | kind: perf
- model keywords: deepseek, v4
    - `tests/ut/models/test_deepseek_v4_dspark_context_wkv.py`
    - `tests/ut/ops/test_rope_proxy.py`
    - `vllm_ascend/models/deepseek_v4/dspark.py`

### 2026-10-14 — [Doc]: Document DeepSeek V4 TurboQuant serving options (#18662)
- PR: #18662
- sha: `aaf7978067fd` | kind: doc
- model keywords: deepseek, v4
    - `docs/source/tutorials/models/DeepSeek-V4-Flash.md`

### 2026-10-14 — [BugFix] Request huge pages before Engram host registration (#18628)
- PR: #18628
- sha: `d11968781fc7` | kind: bugfix
- model keywords: deepseek, engram, v41, v4
    - `vllm_ascend/models/deepseek_v41/engram/npu.py`

### 2026-10-13 — [Bugfix] Validate DeepSeek V4 tensor parallel limits (#18418)
- PR: #18418
- sha: `d1da39542603` | kind: bugfix
    - `tests/ut/test_platform.py`
    - `vllm_ascend/device/hardware_profile.py`
    - `vllm_ascend/platform.py`

### 2026-10-13 — [Doc] Translated Doc files 2026-10-13 (#18538)
- PR: #18538
- sha: `af9f80032348` | kind: doc
- model keywords: deepseek, v4, glm, kimi, minimax, qwen
    - `docs/source/locale/zh_CN/LC_MESSAGES/community/versioning_policy.po`
    - `docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/Design_Documents/add_custom_aclnn_op.po`
    - `docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/Design_Documents/index.po`
    - `docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/Design_Documents/model_runner_v2_eplb.po`
    - `docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/Design_Documents/non_contiguous_kv_cache.po`
    - `docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/Design_Documents/persistent_csrc_build_cache.po`
    - `docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/contribution/doc_writing.po`
    - `docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/contribution/index.po`
    - `docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/contribution/multi_node_test.po`
    - `docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/contribution/nightly_ci_test.po`
    - `docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/contribution/testing.po`
    - `docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/performance_and_debug/optimization_and_tuning.po`
    - `docs/source/locale/zh_CN/LC_MESSAGES/faqs.po`
    - `docs/source/locale/zh_CN/LC_MESSAGES/getting_started/installation/install_vllm_ascend.inc.po`
    - `docs/source/locale/zh_CN/LC_MESSAGES/tutorials/features/pd_disaggregation_mooncake_multi_node.po`
    - `docs/source/locale/zh_CN/LC_MESSAGES/tutorials/features/pd_disaggregation_mooncake_single_node.po`
    - `docs/source/locale/zh_CN/LC_MESSAGES/tutorials/models/DeepSeek-R1.po`
    - `docs/source/locale/zh_CN/LC_MESSAGES/tutorials/models/DeepSeek-V3.1.po`
    - `docs/source/locale/zh_CN/LC_MESSAGES/tutorials/models/DeepSeek-V3.2.po`
    - `docs/source/locale/zh_CN/LC_MESSAGES/tutorials/models/DeepSeek-V4-Flash.po`
    - ... and 38 more

### 2026-10-13 — [Doc][Misc] Add note in the model tutorials (#18493)
- PR: #18493
- sha: `d6ce55f96a01` | kind: doc
- model keywords: deepseek, v4, glm, kimi, minimax, qwen
    - `docs/source/tutorials/models/DeepSeek-V4-Flash.md`
    - `docs/source/tutorials/models/DeepSeek-V4-Pro.md`
    - `docs/source/tutorials/models/DeepSeek-V4.1-Flash.md`
    - `docs/source/tutorials/models/GLM5.2.md`
    - `docs/source/tutorials/models/GLM5.3-Flash.md`
    - `docs/source/tutorials/models/GLM5.3.md`
    - `docs/source/tutorials/models/GLM5.md`
    - `docs/source/tutorials/models/Kimi-K2.6.md`
    - `docs/source/tutorials/models/Kimi-K3.md`
    - `docs/source/tutorials/models/MiniMax-M3.md`
    - `docs/source/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md`
    - `docs/source/tutorials/models/Qwen3.5-397B-A17B.md`
    - `docs/source/tutorials/models/Qwen3.6-35B-A3B.md`
    - `docs/source/tutorials/models/Qwen3.8-2.4T-A95B.md`
    - `docs/source/tutorials/models/Qwen3.8-27B.md`
    - `docs/source/tutorials/models/Qwen3.8-Flash-Next.md`

### 2026-10-12 — [BugFix][Doc] Delete the usage of jemalloc (#18392)
- PR: #18392
- sha: `06d7a24415c1` | kind: bugfix
- model keywords: deepseek, v4, kimi, minimax, qwen
    - `.github/workflows/_build_csrc_cache.yaml`
    - `Dockerfile`
    - `Dockerfile.310p`
    - `Dockerfile.310p.openEuler`
    - `Dockerfile.a3`
    - `Dockerfile.a3.openEuler`
    - `Dockerfile.a5`
    - `Dockerfile.a5.openEuler`
    - `Dockerfile.openEuler`
    - `docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md`
    - `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md`
    - `docs/source/tutorials/models/DeepSeek-V4-Flash.md`
    - `docs/source/tutorials/models/DeepSeek-V4-Pro.md`
    - `docs/source/tutorials/models/DeepSeek-V4.1-Flash.md`
    - `docs/source/tutorials/models/Dots3-Note.md`
    - `docs/source/tutorials/models/Kimi-K2.6.md`
    - `docs/source/tutorials/models/MiniMax-M3.md`
    - `docs/source/tutorials/models/PaddleOCR-VL.md`
    - `docs/source/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md`
    - `docs/source/tutorials/models/Qwen3.5-397B-A17B.md`
    - ... and 22 more

### 2026-10-12 — [BugFix][Model] Backport SWA bounded replay to vLLM 0.30 (#18322)
- PR: #18322
- sha: `b59bfc57c717` | kind: bugfix
- model keywords: deepseek, v4
    - `docs/source/tutorials/models/DeepSeek-V4.1-Flash.md`
    - `tests/e2e/models/swa_bounded_replay_probe.py`
    - `tests/ut/core/test_kv_cache_interface.py`
    - `tests/ut/patch/platform/test_patch_swa_bounded_replay_legacy.py`
    - `tests/ut/patch/platform/test_prefix_cache_cp_patches.py`
    - `vllm_ascend/core/kv_cache_interface.py`
    - `vllm_ascend/patch/__init__.py`
    - `vllm_ascend/patch/platform/__init__.py`
    - `vllm_ascend/patch/platform/patch_swa_bounded_replay_legacy.py`
    - `vllm_ascend/platform.py`

### 2026-10-12 — [CI]  Remove old-version model test cases (#18422)
- PR: #18422
- sha: `8c50d945e83c` | kind: ci
- model keywords: deepseek, kimi, minimax, qwen
    - `.github/workflows/configs/nightly_config.yaml`
    - `tests/e2e/cases/models/configs/DeepSeek/DeepSeek-R1-0528-W8A8.yaml`
    - `tests/e2e/cases/models/configs/DeepSeek/Deepseek_R1_W8A8_A3.yaml`
    - `tests/e2e/cases/models/configs/DeepSeek/Deepseek_R1_W8A8_Reasoning_output_A3.yaml`
    - `tests/e2e/cases/models/configs/DeepSeek/MTPX-DeepSeek-R1-0528-W8A8.yaml`
    - `tests/e2e/cases/models/configs/DeepSeek/Prefix-Cache-DeepSeek-R1-0528-W8A8.yaml`
    - `tests/e2e/cases/models/configs/Kimi/Kimi-K2-Thinking.yaml`
    - `tests/e2e/cases/models/configs/Kimi/Kimi-K2.5.yaml`
    - `tests/e2e/cases/models/configs/Kimi/kimi25_w4a8_step_3_7.yaml`
    - `tests/e2e/cases/models/configs/MiniMax/MiniMax-M2.5-w8a8-QuaRot-A2.yaml`
    - `tests/e2e/cases/models/configs/Qwen/Qwen3-VL-235B-A22B-Instruct-W8A8.yaml`
    - `tests/e2e/cases/models/configs/Qwen/Qwen3-VL-235B-disagg-pd.yaml`

### 2026-10-12 — [Refactor][Worker] Remove legacy speculative PP compatibility (#17500)
- PR: #17500
- sha: `654d19cff5ea` | kind: refactor
- model keywords: kimi, deepseek
    - `tests/ut/models/test_kimi_k3_pp_sp.py`
    - `tests/ut/patch/platform/test_patch_use_v2_model_runner.py`
    - `tests/ut/patch/worker/test_patch_deepseek_v2.py`
    - `tests/ut/patch/worker/test_patch_dspark_pp.py`
    - `tests/ut/patch/worker/test_patch_spec_pp.py`
    - `tests/ut/worker/test_model_runner_v2.py`
    - `tests/ut/worker/test_pcp_manager_v2.py`
    - `tests/ut/worker/v2/test_pp_transport.py`
    - `vllm_ascend/patch/__init__.py`
    - `vllm_ascend/patch/platform/patch_use_v2_model_runner.py`
    - `vllm_ascend/patch/worker/patch_deepseek_v2.py`
    - `vllm_ascend/patch/worker/patch_v2/patch_dspark.py`
    - `vllm_ascend/patch/worker/patch_v2/patch_spec_pp.py`
    - `vllm_ascend/worker/v2/model_runner.py`
    - `vllm_ascend/worker/v2/pp_transport.py`

### 2026-10-12 — [CI][Misc] mian2main vllm 0928 (#17870)
- PR: #17870
- sha: `e00cb64a4469` | kind: ci
- model keywords: kimi, mla, deepseek, v41, v4, engram, minimax, dsa, moe
    - `.github/vllm-main-verified.commit`
    - `tests/e2e/pull_request/four_card/test_kimi_k3.py`
    - `tests/e2e/pull_request/one_card/test_model_runner_v1_with_device.py`
    - `tests/e2e/pull_request/one_card/test_thinking_budget.py`
    - `tests/e2e/pull_request/two_card/spec_decode/test_spec_decode.py`
    - `tests/ut/_310p/test_model_runner_v2_310p.py`
    - `tests/ut/attention/test_gqa_dcp_split.py`
    - `tests/ut/attention/test_mla_cp.py`
    - `tests/ut/core/test_recompute_scheduler_short_request_first.py`
    - `tests/ut/distributed/eplb/test_state.py`
    - `tests/ut/distributed/kv_transfer/sparse_kv_offload/test_sparse_kv_offload_manager.py`
    - `tests/ut/kv_offload/test_preempt_offload_connector.py`
    - `tests/ut/models/test_deepseek_v41_release_config.py`
    - `tests/ut/models/test_kimi_k3_adapter.py`
    - `tests/ut/models/test_kimi_k3_pp_sp.py`
    - `tests/ut/ops/test_gdn_attn_builder.py`
    - `tests/ut/patch/platform/test_patch_engram_config.py`
    - `tests/ut/patch/platform/test_patch_kv_cache_coordinator.py`
    - `tests/ut/patch/platform/test_patch_mamba_block_aligned_split.py`
    - `tests/ut/spec_decode/test_eagle_proposer.py`
    - ... and 44 more

### 2026-10-12 — [Doc][Misc] Model offline modification (#18313)
- PR: #18313
- sha: `bc35d8bba4de` | kind: doc
- model keywords: deepseek, glm, kimi, minimax, qwen
    - `docs/source/tutorials/models/DeepSeek-R1.md`
    - `docs/source/tutorials/models/DeepSeek-V3.1.md`
    - `docs/source/tutorials/models/DeepSeek-V3.2.md`
    - `docs/source/tutorials/models/GLM4.x.md`
    - `docs/source/tutorials/models/Hy3-preview.md`
    - `docs/source/tutorials/models/Kimi-K2-Thinking.md`
    - `docs/source/tutorials/models/Kimi-K2.5.md`
    - `docs/source/tutorials/models/MiniMax-M2.md`
    - `docs/source/tutorials/models/Qwen3-235B-A22B.md`
    - `docs/source/tutorials/models/Qwen3-VL-235B-A22B-Instruct.md`
    - `docs/source/user_guide/support_matrix/supported_models.md`

### 2026-10-12 — [BugFix][PCP] Support multimodal inputs for DeepSeek V4.1 (#18281)
- PR: #18281
- sha: `6b8b2f3f6511` | kind: bugfix
- model keywords: deepseek, v41, v4
    - `tests/ut/models/test_deepseek_v41_vision.py`
    - `tests/ut/ops/test_vocab_parallel_embedding.py`
    - `vllm_ascend/models/deepseek_v41/vl_model.py`
    - `vllm_ascend/ops/vocab_parallel_embedding.py`
    - `vllm_ascend/worker/v2/model_states/default.py`
    - `vllm_ascend/worker/v2/pcp_manager.py`

### 2026-10-11 — [BugFix][Model] Avoid false buffer mutation guard in DeepSeek forward (#18257)
- PR: #18257
- sha: `527a1bb2aa73` | kind: bugfix
- model keywords: deepseek
    - `tests/ut/patch/worker/test_patch_deepseek_v2.py`
    - `vllm_ascend/patch/worker/patch_deepseek_v2.py`

### 2026-10-11 — [Feature][PCP] Support fixed-K speculative decode request sharding (#18112)
- PR: #18112
- sha: `15f24d5e6f5e` | kind: feat
- model keywords: dsa, v41, v4, mla, deepseek
    - `tests/ut/attention/test_dsa_v41_cp.py`
    - `tests/ut/spec_decode/test_mla_dspark.py`
    - `tests/ut/worker/test_dcp_speculator_metadata_v2.py`
    - `tests/ut/worker/test_mtp_pcp_speculator_v2.py`
    - `tests/ut/worker/test_pcp_manager_v2.py`
    - `tests/ut/worker/v2/test_speculator.py`
    - `vllm_ascend/attention/context_parallel/common_cp.py`
    - `vllm_ascend/models/deepseek_v4/model.py`
    - `vllm_ascend/worker/v2/pcp_manager.py`
    - `vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py`
    - `vllm_ascend/worker/v2/spec_decode/dspark/speculator.py`

---
Mined from git history. A commit's subject is a claim; the diff is the
fact. Read the diff before treating anything here as current behavior,
and record the repo HEAD this dossier was built from.