# DeepSeek-V4.1-Flash（1017 权重）单机部署指导书

> 适用：一台 **8×Ascend 910B3（A2，8×64GB HBM）**，从零部署 1017 版权重并以已验证的最优配置运行。
> 全部数字来自 2026-10-18/19 真机部署：**GSM8K-100 = 100/100，A=5.89/6，AISBench(8K/256/3/并发1) E2E 78.25 tok/s**。
> 文中 `<占位符>` 需按你的机器替换；除特别注明外命令均以 root 执行。

---

## 0. 一页决策表

| 问题 | 答案 | 详见 |
|---|---|---|
| 权重从哪来 | ModelScope `chiro2001/DeepSeek-V4.1-Flash-1017-w4a8-Ascend`（304.5GB） | §1 |
| 要自己量化吗 | **不要**（已是 W4A8_DYNAMIC）；仅需 engram 6 分片拼接（~10 分钟） | §2 |
| vLLM 版本 | **vllm-ascend 0.1.dev5097**（= `deepseek-v4.1-flash` 系列镜像内置，vllm 0.27.1） | §3 |
| 镜像 | 官方基底 + 端口仓补丁链自建（两条命令）；**勿直接用上游 v0.30.x 官方镜像跑本配置** | §3 |
| 要合入 PR 吗 | **不需要**。补丁全部烘焙进自建镜像 | §3、§8 |
| 起服命令 | §5.2（一条 `vllm serve`，61 个参数逐字可用） | §5 |
| 验收 | 健康 + 预热探针 + GSM8K 门禁（基线 100/100） | §6 |

---

## 1. 权重获取

```bash
pip install modelscope            # 未装时
modelscope download \
  --model chiro2001/DeepSeek-V4.1-Flash-1017-w4a8-Ascend \
  --local_dir <WEIGHTS>           # 例：/data/models/DeepSeek-V4.1-Flash-1017-w4a8-Ascend
```

- 体量约 304.5GB 主体 + engram 分片（实测 ~84 分钟 @30MB/s，**支持断点续传**，中断重跑同一命令即可）。
- 下载完成后**必须校验**（仓库自带清单，150 项）：

```bash
cd <WEIGHTS> && sha256sum -c SHA256SUMS | grep -vc ': OK$'   # 期望输出 0
```

目录内容：`config.json`、86× `quant_model_weights-*.safetensors`、4× `mtpq-*`（DSpark draft 权重）、
`engram_extra.safetensors`、`engram_int8/`（分片形态，见 §2）、`encoding/`（官方编码器，门禁会用到）、`inference/`（官方极简参考实现，非生产引擎）。

**为什么选这份**（单机 8×64GB 下其余路线实测不可行）：

| 候选 | 结论 |
|---|---|
| 1017 官方原始 FP8+FP4（476GB） | 单机装不下：476/8≈59.5GB/卡，64GB 卡放完权重无 KV 余量 |
| `Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8`（781GB） | 需 4 台 A2 服务器（上游官方教程口径） |
| 自做 msmodelslim W4A8 | 可行但非必要；复现指南见端口仓 `quant/REPRO_W4A8_QUANT.md` |

## 2. 部署前唯一处理：engram 分片拼接

仓库的 engram 是 6 分片形态，引擎要求单文件（纯 `cat` 拼接，已验证拼接结果与官方合并件逐字节一致）：

```bash
E=<WEIGHTS>/engram_int8
for L in layers_1 layers_14; do
  cat $E/${L}_engram_embed.weight.safetensors.part-*-of-06 \
    > $E/${L}_engram_embed.weight.safetensors &
done; wait
# 产出两个 ~98GB 文件（98,308,270,704 / 98,305,579,120 字节）
```

缺这一步的症状：worker 启动即报
`FileNotFoundError: .../engram_int8/layers_1_engram_embed.weight.safetensors`。

## 3. 代码与镜像

### 3.1 取端口仓并构建镜像

```bash
git clone https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B.git <PORT_REPO>
# 基底镜像（二选一）：
#   官方源：  quay.io/ascend/vllm-ascend:deepseek-v4.1-flash
#   国内加速：quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler
# 烘焙端口仓补丁链（含 dsa_v1 等 15 项，细节与排障见 <PORT_REPO>/a2_package/RUNBOOK.md）
cd <PORT_REPO> && bash scripts/build_image.sh     # 以仓库文档为准
```

### 3.2 打 DRAFT_GRAPH 前置修复（必须，3 行）

draft 模型入图依赖 `dsa_v1.py` 游离 `self` 修复，补丁取自
[ascend-infra-skills 仓库](https://github.com/Liears/ascend-infra-skills)：
`examples/deepseek-v4.1-flash-910b/06-onboard/evidence/perf-campaign/fix-dsa-v1-stray-self.patch`

```bash
docker cp fix-dsa-v1-stray-self.patch <容器>:/tmp/
docker exec <容器> bash -c 'cd /vllm-workspace/vllm-ascend && patch -p1 < /tmp/fix-dsa-v1-stray-self.patch'
docker commit <容器> <IMAGE_TAG>        # 例：dsv41-opt:v1，作为你的部署镜像
```

> 版本事实（供核对）：部署验证时镜像内为 vllm-ascend `0.1.dev5097+ge43cf1e9f`、vllm `0.27.1+empty`、
> CANN 9.1.0、transformers 5.14.1。**不要**改用上游 `vllm-ascend ≥ v0.30` 镜像跑本配置——
> 那条线的 dspark draft 默认 eager，且补丁体系不同（见 §8）。

## 4. 起容器

```bash
W=<WEIGHTS>; C=<缓存根目录>          # C 首次为空目录即可（npugraph/静态核编译缓存会自动累积）
mkdir -p $C/{npugraph,skcache/install,skcache/compile_outputs,vllm,numba}
docker run -d --name <NAME> --net=host --shm-size=512g --privileged=true \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v $W:$W:ro -v $W/engram_int8:$W/engram_int8:rw \
  -v $C/npugraph:/root/npugraph_ex_cache \
  -v $C/skcache/install:/workspace/static_kernel_install \
  -v $C/skcache/install:/vllm-workspace/vllm-ascend/static_kernel_install \
  -v $C/skcache/compile_outputs:/workspace/static_kernel_compile_outputs \
  -v $C/skcache/compile_outputs:/vllm-workspace/vllm-ascend/static_kernel_compile_outputs \
  -v $C/skcache/compile_outputs:/vllm-workspace/vllm/static_kernel_compile_outputs \
  -v $C/vllm:/root/.cache/vllm -v $C/numba:/numba_cache \
  <IMAGE_TAG> sleep infinity
```

要点：权重目录与 engram_int8 分别挂载（engram 需写权限）；缓存目录**跨容器复用**可把起服时间从 ~20 分钟缩到 ~8 分钟。

## 5. 起服

### 5.1 容器内环境

```bash
export DRAFT_GRAPH=1        # 必须：draft 模型入 ACL 图（单流 ~1.8× 的来源）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=${ATB_CXX_ABI:-1}
```

### 5.2 起服命令（E2 终验最优配置，逐字可用）

```bash
MODEL=<WEIGHTS>
exec vllm serve $MODEL \
  --host 0.0.0.0 --port 8100 --served-model-name deepseek-v41 \
  --tensor-parallel-size 8 --enable-expert-parallel --trust-remote-code \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --max-model-len 1048576 \
  --max-num-seqs 4 --max-num-batched-tokens 8192 --block-size 128 \
  --gpu-memory-utilization 0.92 \
  --additional-config '{"enable_engram":true,"enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":true},"multistream_overlap_shared_expert":true,"multistream_dsv4_dsa_overlap":true,"enable_prefill_mc2":false,"enable_mc2_hierarchy_comm":false,"engram_storage":"int8","enable_fused_mc2":1}' \
  --quantization ascend \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,3,4,6,12,18,24,30,32]}' \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":false}' \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --safetensors-load-strategy lazy \
  --limit-mm-per-prompt '{"image": 4}' \
  --tokenizer-mode=deepseek_v41 --reasoning-parser=deepseek_v41 --tool-call-parser=deepseek_v41 \
  --enable-auto-tool-choice --default-chat-template-kwargs='{"enable_thinking":false}'
```

起服 ~8-20 分钟（权重 ~4.5min + 图捕获；首次静态核编译更久）。

## 6. 验收（缺一不可）

1. **健康**：`curl 127.0.0.1:8100/v1/models` 返回列表；引擎日志出现 `dspark-capture-audit` 行（证明 draft 图真的捕获了，没有它 A 会掉到 ~1 且无报错——静默退化）。
2. **预热后探针**（冷探针首条 TTFT 可达 15s，必须丢弃预热）：

```bash
# 任一安装了 python3 的机器；<W>/config-search 探针脚本可从 ascend-infra-skills 案例目录取
python3 decode_probe.py http://127.0.0.1:8100 longgen-prompt.json 512   # ×5 丢弃，×3 取中位
# 判读：A≈5.89、chunks≈87 为健康；A<2 = draft 图未生效，回查 §5.1 环境变量与 §3.2 补丁
```

3. **精度门禁**（新权重必过）：

```bash
# 评测脚本：<PORT_REPO>/tests/acc_eval.py；gsm8k test.jsonl 从 HF/ModelScope openai/gsm8k 下载
# 官方编码器：<WEIGHTS>/encoding（权重仓自带）
python3 <PORT_REPO>/tests/acc_eval.py --task gsm8k --limit 100 --mode chat \
  --out gsm8k.json --tag deploy --conc 4 --base-url http://127.0.0.1:8100 \
  --enc-dir <WEIGHTS>/encoding --gsm8k-data-dir <GSM8K_DIR>
```

**1017 基线：100/100**（<97 必查）。

**性能锚点**（口径：8K 上下文 512-token 强制长生成，预热 5 丢 3 取中位）：A=5.89/6；冷稳态 ~122 tok/s（深热态更高）；AISBench 8K/256/3/并发1：E2EL 3,272ms、输出吞吐 78.25 tok/s（含 prefill 的 E2E 口径）。

## 7. 已知坑（真机踩过）

| 症状 | 原因 → 处置 |
|---|---|
| 起服报 `HFValidationError: Repo id must be...` | 权重目录没挂进容器（路径不可见被当 repo id）→ 核对 §4 `-v` |
| `FileNotFoundError .../engram_int8/layers_1_...` | §2 的分片拼接没做 |
| 引擎重启即挂 `cv2 / libxcb.so.1` | 容器里装了 `opencv-python`（非 headless）→ 只装 `opencv-python-headless`（装评测工具时常见） |
| 换引擎后 `Free memory ... less than desired` | 旧 worker 残留占卡；worker 被 setproctitle 改名，`pkill -f vllm` 杀不到 → 按 npu-smi 报的 PID 杀：`npu-smi info \| grep VLLMWorker \| awk -F'\|' '{gsub(/ /,"",$3);print $3}' \| xargs kill -9` |
| 首条探针 TTFT ~15s | 冷探针假象 → 预热 5 条再测 |
| A≈1 且无报错 | draft 图静默退化：`DRAFT_GRAPH` 未导出，或 §3.2 补丁未打 |

## 8. 与上游 vllm-ascend 的关系（防混淆）

- 本指导书 = **端口仓栈**（dev5097 + 烘焙补丁，`DRAFT_GRAPH` 为私有实现）。**零上游 PR 依赖**。
- 上游 main 目前 dspark draft 仅 eager（单流 ~94.5 tok/s）。等价能力已提交
  **PR #18967**（`VLLM_ASCEND_ENABLE_DSPARK_DRAFT_GRAPH=1`，
  https://github.com/vllm-project/vllm-ascend/pull/18967，CI 全绿待维护者标签）。
  合入后上游线可用同配置思路部署，届时以 `--engram-config` 替代 `additional-config` 里的 engram 键。
- 定价依据：ascend-infra-skills 仓库 `examples/deepseek-v4.1-flash-910b/06-onboard/evidence/perf-campaign/P5-UPSTREAM-PRICING.md`。

## 9. 参考资料

- 端口仓：https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B （补丁链、build_image.sh、量化指南、acc_eval.py）
- 技能仓：https://github.com/Liears/ascend-infra-skills （E1 修复补丁、战役全部证据、验收协议）
- 上游教程：vllm-ascend 仓 `docs/source/tutorials/models/DeepSeek-V4.1-Flash.md`（多机 W8A8 路线）
