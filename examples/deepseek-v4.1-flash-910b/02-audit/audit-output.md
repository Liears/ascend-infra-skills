# Evidence audit: dsv41-ascend

- git HEAD: `889a6a92797d` | last commit 2026-10-14 | origin https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B.git

## Patches
- total: 33
- upstream `20261003-dynspec`: 2 patch(es), 131 lines
    - fixes/20261003-dynspec/patch_cudagraph.dynspec.patch (87 lines)
    - fixes/20261003-dynspec/serve_a2.dynspec.patch (44 lines)
- upstream `20261004-hcfuse`: 1 patch(es), 114 lines
    - fixes/20261004-hcfuse/model.py.hcfuse.patch (114 lines)
- upstream `ced`: 2 patch(es), 170 lines
    - experimental/ced/core_scheduler_replay.patch (102 lines)
    - experimental/ced/core_scheduler_prefill_hit.patch (68 lines)
- upstream `config`: 2 patch(es), 3407 lines
    - patches/msmodelslim/0001-feat-dsv41-DeepSeek-V4.1-W4A8.patch (1719 lines)
    - quant/patches/msmodelslim_v41_w4a8.patch (1688 lines)
- upstream `lab_practice`: 1 patch(es), 152 lines
    - patches/msmodelslim/0002-feat-dsv41-hiaux-variant-of-the-W4A8-quant-recipe.patch (152 lines)
- upstream `ops`: 1 patch(es), 130 lines
    - experimental/v41-dcp/ops/dcp-mount.patch (130 lines)
- upstream `patches`: 8 patch(es), 297 lines
    - upstream/pr/patches/cpu-binding-hang-fix.patch (104 lines)
    - docs/patches/dcp8-variantE.patch (54 lines)
    - docs/patches/dcp8-variantC.patch (36 lines)
    - docs/patches/dcp8-v1-csa_kernel.patch (26 lines)
    - a2/patches/0003-kv8-fused-rebuild-triton.patch (24 lines)
    - docs/patches/dcp8-v1-csa_block_vector.patch (20 lines)
    - docs/patches/dcp8-variantB.patch (17 lines)
    - docs/patches/dcp8-variantA.patch (16 lines)
- upstream `vllm`: 4 patch(es), 870 lines
    - patches/vllm/0001-feat-scheduler-admission-gate-to-protect-decode-from.patch (394 lines)
    - patches/admission_gate.patch (380 lines)
    - evidence/ced_tiny_n1fix_20260924/core_scheduler_replay.patch (57 lines)
    - experimental/ced/core_config_dynamic_sd_gate.patch (39 lines)
- upstream `vllm_ascend`: 12 patch(es), 5232 lines
    - patches/vllm-ascend/0009-feat-engram-device-side-table-lookup-over-host-mappe.patch (1769 lines)
    - patches/vllm-ascend/0007-feat-engram-host-resident-INT8-tables-with-local-own.patch (1605 lines)
    - patches/vllm-ascend/0008-perf-engram-numba-JIT-for-hash-and-plan-kernels.patch (644 lines)
    - patches/vllm-ascend/0006-perf-engram-chunked-gate-without-2048-row-padding.patch (259 lines)
    - patches/vllm-ascend/0010-feat-engram-A3-Engram-A2.patch (257 lines)
    - experimental/ced/core_model_runner_dynamic_spec.patch (219 lines)
    - patches/vllm-ascend/0002-perf-moe-range-compare-expert-mask-optional-invalid-.patch (111 lines)
    - patches/vllm-ascend/0005-perf-attention-2D-wo_a-matmul-and-dummy-shape-guard.patch (111 lines)
    - patches/vllm-ascend/0003-perf-rope-fuse-cos-sin-table-index-selection.patch (105 lines)
    - patches/vllm-ascend/0004-perf-indexer-fast-path-when-QLI-has-no-candidate.patch (67 lines)
    - experimental/ced/core_model_runner_prompt_tail.patch (44 lines)
    - patches/vllm-ascend/0001-perf-moe-dispatch-combine-over-AllGather-when-TP-EP.patch (41 lines)

## Entry-point scripts
- `a2/patches/engram-true-tokens/model_runner_v1.patched.py` (1401 lines)  -> /vllm/vllm/worker/gpu_model_runner.py
- `a2/patches/engram-true-tokens/tests/run_all.sh` (44 lines)
- `a2/patches/engram-true-tokens/tests/test_callsite_contract.py` (170 lines)
- `a2/patches/engram-true-tokens/tests/test_repair_plan.py` (105 lines)
- `a2/patches/engram-true-tokens/tests/test_true_tokens_repair.py` (290 lines)
- `a2/patches/kv8-graphsafe/adapt_runner.py` (95 lines)  -> /pkg-kv8pf/shadow/vllm_ascend
- `a2/patches/kv8-graphsafe/patch_serve_sg.sh` (46 lines)
- `a2/scripts/a2_multiproc_reg_probe.sh` (304 lines)
- `a2/scripts/a2_one_shot_probe.sh` (440 lines)
- `a2/scripts/check_4axis_acceptance.py` (361 lines)
- `a2/scripts/check_artifact_identity.sh` (131 lines)  -> /kv8-int8-pkg/vllm_ascend/core/deepseek_v41.py
- `a2/scripts/check_image_fingerprint.sh` (103 lines)  -> /vllm-workspace/vllm-ascend/vllm_ascend
- `a2/scripts/check_merged_fresh.sh` (108 lines)  -> /model_merged.py
- `a2/scripts/check_mount_list.py` (100 lines)
- `a2/scripts/check_publish_links.sh` (60 lines)
- `a2/scripts/check_same_run_replay.py` (126 lines)
- `a2/scripts/check_semantic_nodiff.py` (84 lines)
- `a2/scripts/run_4axis_arm.sh` (305 lines)  -> /tmp/20260922/merge4axis/model.img.py
- `a2/scripts/selftest_apc_align.py` (111 lines)
- `a2/scripts/selftest_collect_evidence.sh` (239 lines)
- `a2/scripts/selftest_deploy_a3.sh` (215 lines)  -> /check_model_dir.sh
- `a2/scripts/selftest_make_shadow_pkg.sh` (95 lines)  -> /deepseek_v41/model.py
- `a2/scripts/selftest_serve_a2_offload.sh` (281 lines)  -> /repo/a2/patches/kv8-int8-pkg/vllm_ascend/attention
- `a2/scripts/serve_a2_offload.sh` (987 lines)  -> /ascend/vllm-ascend
- `a2/scripts/serve_a3_offload.sh` (42 lines)  -> /path/to/model

## Evidence documents
- `CHANGELOG.md` (1393 lines)  [dated]
- `quant/REPRO_W4A8_QUANT.md` (1022 lines)  [dated]
- `docs/V41-LINE-A-ABSPERF-BUDGET-20261001.md` (691 lines)  [dated]
- `EXPECTED_PERF.md` (598 lines)  [dated]
- `docs/V41-DCP8-CORRECTNESS-FIX-AND-PERF-PLAN-20261001.md` (588 lines)  [dated]
- `docs/V41-DCP8-DECODE-PERF-20261002.md` (374 lines)  [dated]
- `REPRO.md` (362 lines)
- `docs/V41-DCP-HANDOVER-20260929.md` (349 lines)  [dated]
- `docs/V41-UPSTREAM-STATUS-20260930.md` (344 lines)  [dated]
- `experiments/delivery_matrix/STATUS.md` (333 lines)  [dated]
- `docs/CED-PD-ACCEPTANCE.md` (332 lines)  [dated]
- `docs/CED-PD-PERF-20260925.md` (309 lines)  [dated]
- `CORRECTNESS_STATUS.md` (290 lines)  [dated]
- `docs/V41-TP8-DSPARK-MULTISTREAM-PERF-20261003.md` (285 lines)  [dated]
- `upstream/HANDOVER-2.md` (272 lines)  [dated]
- `PACKAGING_REPORT.md` (253 lines)  [dated]
- `upstream/HANDOVER.md` (233 lines)  [dated]
- `docs/HANDOVER-20261001-DSPARK-DCP.md` (232 lines)  [dated]
- `docs/PERF-FOUR-AXES-STATUS-20261004.md` (231 lines)  [dated]
- `a2/logs/135-20260923-handover-context.md` (230 lines)  [dated]

## External weights
- modelscope: `https://www.modelscope.cn/models/chiro2001/DeepSeek-V4.1-Flash-w4a8-Ascend` declared 212.6 GB  (cited in REPRO.md)
- modelscope: `chiro2001/DeepSeek-V4.1-Flash-w4a8-Ascend` size: unknown  (cited in README.md)
- huggingface: `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash` size: unknown  (cited in upstream/docs/KV-CACHE-ACCOUNTING.md)
- A blank size means no *same-line* cue made the number a weight size;
  check the model page before quoting one.

## Toolchain pins
- cann: **6 distinct versions** -> 9.1.0 x5222, 9.1 x10, 9.2.0 x7, 9.1.0_Ascend910_9382.json x2, 9.1.0_Ascend910_9382.lock x1, 9.2.0.pt2.13.0_dsv4.1_aarch_a5_image_custom_20260930 x1
    [!] multiple baselines mentioned; identify which one the claim belongs to
- driver: **2 distinct versions** -> 26.1.1 x21, 25.5.5 x8
    [!] multiple baselines mentioned; identify which one the claim belongs to
- torch_npu: **4 distinct versions** -> 2.10.0.post4 x21, 2.10 x2, 2.10.0 x2, 2.10.post4 x1
    [!] multiple baselines mentioned; identify which one the claim belongs to
- transformers: 5.14.1  (1 hits)
- vllm: **18 distinct versions** -> 0.27.1 x178, 0.27.1-tp8-ep-04469255 x29, 0.27.1-tp8-ep-4e822518 x28, 0.27.1-tp8-ep-0b5664ce x24, 0.27.1-tp8-ep-ca374a5d x22, 0.27.1-tp8-ep-a0c074a8 x16
    [!] multiple baselines mentioned; identify which one the claim belongs to
- vllm_ascend: **2 distinct versions** -> 0.1.dev5097 x2, 0.23.0-a3-s50057815 x1
    [!] multiple baselines mentioned; identify which one the claim belongs to

## Tests
- test files: 351 | shipped data files: 108
    - a2/patches/engram-true-tokens/tests/test_true_tokens_repair.py
    - a2/patches/engram-true-tokens/tests/test_repair_plan.py
    - a2/patches/engram-true-tokens/tests/test_callsite_contract.py
    - a2/scripts/selftest_serve_a2_offload.sh
    - a2/scripts/selftest_make_shadow_pkg.sh
    - a2/scripts/selftest_apc_align.py
    - a2/scripts/selftest_collect_evidence.sh
    - a2/scripts/selftest_deploy_a3.sh
- a repo that withholds its corpus for copyright must ship a fetch+checksum script instead

## Verdict
- [+] ships 33 patch(es) against: 20261003-dynspec, 20261004-hcfuse, ced, config, lab_practice, ops, patches, vllm, vllm_ascend
- [+] has an acceptance/expected-perf document
- [+] has a REPRO.md
- [+] pins a single version for: transformers
- [+] references 3 external weight package(s)
- [-] ambiguous toolchain: cann, driver, torch_npu, vllm, vllm_ascend each appear in more than one version — match the claim to the right baseline

This audit reads only the tree. It does not run anything. A repo with no
red flags is *auditable*, not *proven* — running its own self-check is the
only thing that closes the loop.
