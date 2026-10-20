# DeepSeek-V4.1-Flash（1017 权重）单机部署指导书

> 适用：一台 **8×Ascend 910B3（A2，8×64GB HBM）**，从零部署 1017 版权重并以已验证配置运行。
> 正文给出**两条部署路径**，都实测过：A = 端口仓优化栈（快，但要自建镜像）；B = 上游镜像 + 一个 PR（慢一些，但全部来自公开来源）。
> 数字来自 2026-10-18/19（A 路径门禁）与 2026-10-20（同机同权重 A/B）：**GSM8K-100 = 99-100/100**、单流 decode **A 路径 163-171 tok/s / B 路径 119 tok/s**。
> 路径 A 的区间是**同一配置在多次起服的散布**（实测 158.5 / 163.3 / 168.4 / 170.6 都出现过），
> 不是多个优化档位；报数字时请连同工况一起写，单点数字不可比。
> 文中 `<占位符>` 需按你的机器替换；除特别注明外命令均以 root 执行。

---

## 0. 一页决策表

| 问题 | 路径 A（推荐，性能最优） | 路径 B（全公开来源） |
|---|---|---|
| 权重从哪来 | 同 B（ModelScope `chiro2001/DeepSeek-V4.1-Flash-1017-w4a8-Ascend`，304.6GB） | 同左 |
| 要自己量化吗 | **不要**（已是 W4A8_DYNAMIC）；仅需 engram 6 分片拼接（~10 分钟） | 同左 |
| 基础镜像 | 端口仓基底 `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler`，跑 `build_image.sh` 烘焙 **15 项补丁** | **`quay.io/ascend/vllm-ascend:v0.30.0rc2`**（官方 tag，不改造） |
| 镜像上的额外动作 | `docker commit` 出你自己的 tag（DRAFT_GRAPH 等已烘焙） | **cherry-pick PR #18967**（1 个 commit），`docker commit` 出你的 tag |
| vllm-ascend 版本 | `0.1.dev5097+ge43cf1e9f`（vllm `0.27.1+empty`，CANN 9.1.0） | `0.30.0rc2`（vllm `0.30.0+empty`，CANN 9.2.0） |
| 要合入 PR 吗 | **不需要**（补丁全部烘焙进自建镜像） | **需要**：PR #18967 尚未合入，从 fork 分支 cherry-pick |
| 起服开关 | `DRAFT_GRAPH=1` + 45 个 `V41_*`（镜像内已设）+ `additional-config` 里的 engram 键 | `VLLM_ASCEND_ENABLE_DSPARK_DRAFT_GRAPH=1` + **`--engram-config`**（新接口） |
| 单流 decode | **163–171 tok/s**（A=5.89，同配置散布） | **119.1 tok/s**（A=5.89） |
| 精度 | GSM8K-100 = 100/100 | GSM8K-100 = 99/100 |
| 起服命令 | §5.2 | §5.3 |

**怎么选**：要最快 → A（多 1.37×）。要"不依赖私有镜像、全部可追溯到公开 commit" → B。
两条路径的 `A`（投机接受长度）都是 5.89/6，即**接受率已到顶**，差异纯是每步耗时。

---

## 1. 权重获取（两条路径相同）

```bash
pip install modelscope            # 未装时
modelscope download \
  --model chiro2001/DeepSeek-V4.1-Flash-1017-w4a8-Ascend \
  --local_dir <WEIGHTS>           # 例：/data/models/DeepSeek-V4.1-Flash-1017-w4a8-Ascend
```

- 体量 304.6GB（根目录文件）+ engram 分片 417.8GB；实测 ~84 分钟 @30MB/s，**支持断点续传**（中断重跑同一命令）。
- 下载完成后**必须校验**（仓库自带清单 150 项）：

```bash
cd <WEIGHTS> && sha256sum -c SHA256SUMS | grep -vc ': OK$'   # 期望输出 0
```

目录内容：`config.json`、80× `quant_model_weights-*.safetensors`（主干，292.2GB）、
4× `mtpq-*`（DSpark draft 权重）、`engram_extra.safetensors`、`engram_int8/`（6 分片形态，见 §2）、
`release/`（msmodelslim 量化配方、报告、`tools/validate_product.py`）。

> **`encoding/` 不在权重仓里**。精度门禁（§6.3）需要官方编码器，从基座
> [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/files)
> 的 `encoding/` 目录单独取。

**为什么选这份**（单机 8×64GB 下其余路线实测不可行）：

| 候选 | 结论 |
|---|---|
| 1017 官方原始 FP8+FP4（476GB） | 单机装不下：476/8≈59.5GB/卡，64GB 卡放完权重无 KV 余量 |
| `Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8`（781GB） | 需 4 台 A2 服务器（上游官方教程口径） |
| 自做 msmodelslim W4A8 | 可行但非必要；配方在权重仓 `release/msmodelslim/` 与端口仓 `quant/REPRO_W4A8_QUANT.md` |

## 2. 部署前唯一处理：engram 分片拼接

仓库的 engram 是 6 分片形态（前 5 片各 16GiB，第 6 片为余量），引擎要求单文件：

```bash
E=<WEIGHTS>/engram_int8
for L in layers_1 layers_14; do
  cat $E/${L}_engram_embed.weight.safetensors.part-*-of-06 \
    > $E/${L}_engram_embed.weight.safetensors &
done; wait
# 期望产出两个文件，精确字节数：
#   layers_1_engram_embed.weight.safetensors   98,305,579,120
#   layers_14_engram_embed.weight.safetensors  98,308,270,704
```

拼接结果与官方合并件逐字节一致（权重仓自带 `reassemble_engram_weights.sh`，
会做逐片 SHA256 + 完整表校验 + 原子替换，也可以直接用它）。
磁盘需为两张还原表额外预留 ~184GiB；拼接成功后若加了 `--delete-parts` 可删分片。

缺这一步的症状：worker 启动即报
`FileNotFoundError: .../engram_int8/layers_1_engram_embed.weight.safetensors`。

## 3. 代码与镜像

### 3.1 路径 B：上游镜像 + PR #18967（全公开来源，先讲这条因为步骤更短）

**基于什么版本的镜像**：

```bash
docker pull quay.io/ascend/vllm-ascend:v0.30.0rc2
# 国内直连 quay.io 会断，换镜像站同一 tag：
#   docker pull quay.dockerproxy.net/ascend/vllm-ascend:v0.30.0rc2
# 版本事实（镜像内实测）：
#   vllm-ascend 0.30.0rc2 (editable /vllm-workspace/vllm-ascend, tag v0.30.0rc2 = 8fde6d4e3)
#   vllm 0.30.0+empty    CANN 9.2.0    torch 2.10.0+cpu    transformers 5.14.1    python 3.12.13
```

**在镜像基础上合什么 PR**：

- **PR #18967** — `[Feature][SpecDecode] Enable ACL-graph capture for the DSpark draft model (opt-in)`
  https://github.com/vllm-project/vllm-ascend/pull/18967
- 状态（2026-10-20）：**OPEN，未合入**。CI 全绿（pre-commit ✓ / cpu-ut 11,186 ✓ / DCO ✓），
  ci-gate 红是维护者的 `ready-*` 标签门，不是代码问题。所以要**自己 cherry-pick**。
- 单一 commit：`3ec7ae2cc24ed0ed09e13b032751ca54f7d6cdc0`，分支 `Liears:dspark-draft-graph`
  （基点 = 上游当日 main `b61c9b207`，不是 rc2；cherry-pick 到 rc2 会正常自动合并）。
- 改动：7 文件 +476/−37，只动 draft 相关路径，**默认关闭**（不开开关时行为逐字节不变）。

```bash
docker run -d --name <NAME> --net=host --shm-size=512g --privileged=true \
  <设备与驱动注入参数见 §4> \
  quay.io/ascend/vllm-ascend:v0.30.0rc2 sleep infinity

docker exec <NAME> bash -c '
  cd /vllm-workspace/vllm-ascend &&
  git fetch https://github.com/<上游>/vllm-ascend.git dspark-draft-graph &&
  git cherry-pick -n FETCH_HEAD &&      # -n：只落到工作区，不建 commit（DCO 不适用）
  git status --short'                   # 期望 7 个 M，无冲突

docker commit <NAME> <IMAGE_TAG>        # 例：dsv41-upstream:v1
```

> 为什么 `git fetch` 用上游 URL 而不是 `Liears`：读者要用的是**可公开获取**的 commit。
> `3ec7ae2cc` 在上游 `refs/pull/18967/head` 里也存在，可直接
> `git fetch https://github.com/vllm-project/vllm-ascend.git pull/18967/head`。
> 如果哪天 PR 合入了（`git log` 里能找到该 commit），这一步整个省略——直接 `git pull`。

### 3.2 路径 A：端口仓优化栈（性能最优，需自建镜像）

```bash
git clone https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B.git <PORT_REPO>
cd <PORT_REPO> && bash scripts/build_image.sh
# 默认 BASE_IMAGE=quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler
# 官方源可用时：BASE_IMAGE=quay.io/ascend/vllm-ascend:deepseek-v4.1-flash bash scripts/build_image.sh
# 烘焙 15 项补丁（清单+md5+门控 env+实测收益见 <PORT_REPO>/patches/README.md）
```

版本事实（镜像内实测）：vllm-ascend `0.1.dev5097+ge43cf1e9f`、vllm `0.27.1+empty`、
CANN 9.1.0、transformers 5.20.1、torch 2.10.0+cpu、python 3.12.13。

> **路径 A 的 163–171 tok/s 依赖 15 项补丁 + 45 个 `V41_*` 运行开关**（非默认的 19 个逐项见 §3.2.1），其中多项是端口仓私有实现
> （engram host 常驻/JIT、MoE AllGather、o_proj 2D、rope idxsel、QLI no-candidate……）。
> 这些**不在** PR #18967 里——PR 只覆盖 draft 图捕获，实测单独给到 119 tok/s。

#### 3.2.1 路径 A 依赖的 19 个非默认开关（缺一个就掉性能）

端口仓镜像已把这些写进 ENV，但**读者自建镜像后必须逐项核对**——
少设一个不会报错，只会变慢。这是本路径最容易踩的坑。

```bash
# 核对：起服后比对容器内实际生效值
docker exec <NAME> bash -c 'tr "\0" "\n" < /proc/$(pgrep -f "vllm serve" | head -1)/environ \
  | grep -E "^V41_|^DRAFT_GRAPH" | sort'
```

| 开关 | 值 | 作用 | 实测收益 |
|---|---|---|---|
| `DRAFT_GRAPH` | `1` | draft 模型入 ACL 图 | 本路径的大头（单流 ~1.8×） |
| `V41_ENGRAM_JIT` | `1` | engram hash/plan 走 numba JIT | hash 0.427→0.076 ms/步 |
| `V41_ENGRAM_HOST_RESIDENT` | `1` | engram 表 host 常驻 + local-owner fast | route 少一次 all_gather + all_to_all |
| `V41_ENGRAM_WKV_TP` | `1` | wkv 投影按输出维分片 | −0.57 ms/步，KV 容量 +5.9% |
| `V41_ENGRAM_PAD_SKIP` | `1` | 只清零会被读到的行 | 消 1.86 次 ZerosLike/步（−58 µs） |
| `V41_ENGRAM_REUSE_EP_GROUP` | `1` | 复用 EP 通信组 | 省组创建 |
| `V41_ENGRAM_DEVICE_INDEX` | `auto` | A3 开启 / A2 自动回退 host 路径 | A2 上必须允许回退，否则 `ret=207001` |
| `V41_ENGRAM_GATE_MAX_TOKENS` | `8192` | 分块 gate 上界（须 ≥ BAT） | 防广播失败 |
| `V41_MOE_COMM_ALLGATHER` | `1` | MoE 走 AllGather | 128K −4.25 ms、32K −1.35、8K −1.23 |
| `V41_MOE_MASK_RANGE` | `1` | 范围比较替代掩码链 | −0.51 ms，且去掉 80 次/步 Cast |
| `V41_O_PROJ_2D` | `1` | `wo_a` 2D matmul | −0.31~0.76 ms/步 |
| `V41_ROPE_IDXSEL` | `1` | cos/sin 取表 6 kernel→2 | −0.45~0.62 ms/pass |
| `V41_QLI_NO_CANDIDATE` | `1` | QLI 无候选快速路径 | 99.3→50.3 µs（−0.49 ms） |
| `V41_CED_SWA_CLIP` | `1` | CED 滑窗裁剪 | 见端口仓 `docs/` |
| `V41_DECODE_API_GUARD` | `1` | decode 期 API 守卫 | 稳定性 |
| `V41_GATE_MAX_PREFILL` | `8` | prefill gate 上界 | — |
| `VLLM_ADMISSION_GATE` | `1` | prefill 独占步门 | 长文精度门要求 |
| `V41_MOE_ZERO_INVALID_FILE` | `/tmp/v41_moe_zero_file` | 非法值置零标记文件 | 诊断 |
| `V41_MOE_ZERO_NONFINITE_FILE` | `/tmp/v41_moe_nf` | 非有限值置零标记文件 | 诊断 |

> 其余 26 个 `V41_CED_*` / `V41_FORCE_*` / `V41_IDS64_*` 等默认关，是排障与实验用的，
> 生产不需要动。完整的默认值清单见端口仓 `patches/README.md`。

### 3.3 两条路径的版本对照（防止混用）

| 项 | 路径 A | 路径 B |
|---|---|---|
| vllm-ascend | `0.1.dev5097+ge43cf1e9f` | `0.30.0rc2` |
| vllm | `0.27.1+empty` | `0.30.0+empty` |
| CANN | 9.1.0 | 9.2.0 |
| engram 配置接口 | `additional-config` 里的 `enable_engram`/`engram_storage` | 独立的 `--engram-config` |
| 投机开关 | `DRAFT_GRAPH=1` | `VLLM_ASCEND_ENABLE_DSPARK_DRAFT_GRAPH=1` |
| 单流 decode（8K/512） | 163–171 tok/s（同配置多次测量） | 119.1 tok/s |
| GSM8K-100 | 100/100 | 99/100 |

**两者不能混用**：`--engram-config` 在 0.27.x 上不存在；`DRAFT_GRAPH` 在 0.30.x 上是无效变量。

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

### 5.1 容器内环境（两条路径都要）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=${ATB_CXX_ABI:-1}
# 路径 A 追加（CANN 9.1.0 专有）：
source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh
```

### 5.2 路径 A 起服命令（163–171 tok/s，逐字可用）

```bash
export DRAFT_GRAPH=1        # 必须：draft 模型入 ACL 图（这条路径的大头收益）
# 其余 45 个 V41_* 开关已烘焙进端口仓镜像的 ENV，无需手工导出
MODEL=<WEIGHTS>
exec vllm serve $MODEL \
  --host 0.0.0.0 --port 8100 --served-model-name deepseek-v41 \
  --tensor-parallel-size 8 --enable-expert-parallel --trust-remote-code \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --max-model-len 1048576 \
  --max-num-seqs 4 --max-num-batched-tokens 8192 --block-size 128 \
  --gpu-memory-utilization 0.92 \
  --additional-config '{"enable_engram":true,"enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":true},"multistream_overlap_shared_expert":true,"multistream_dsv4_dsa_overlap":true,"enable_prefill_mc2":false,"enable_mc2_hierarchy_comm":false,"engram_storage":"int8","enable_fused_mc2":1}' \
  --quantization ascend \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,6,12,18,24,30,32]}' \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":false}' \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --safetensors-load-strategy lazy \
  --limit-mm-per-prompt '{"image": 4}' \
  --tokenizer-mode=deepseek_v41 --reasoning-parser=deepseek_v41 --tool-call-parser=deepseek_v41 \
  --enable-auto-tool-choice --default-chat-template-kwargs='{"enable_thinking":false}'
```

### 5.3 路径 B 起服命令（119.1 tok/s，逐字可用）

```bash
export VLLM_ASCEND_ENABLE_DSPARK_DRAFT_GRAPH=1   # PR #18967 的唯一开关
MODEL=<WEIGHTS>
exec vllm serve $MODEL \
  --host 0.0.0.0 --port 8200 --served-model-name deepseek-v41 \
  --tensor-parallel-size 8 --enable-expert-parallel --trust-remote-code \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --max-model-len 1048576 \
  --max-num-seqs 4 --max-num-batched-tokens 8192 --block-size 128 \
  --gpu-memory-utilization 0.92 --quantization ascend \
  --no-enable-prefix-caching --safetensors-load-strategy lazy \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  --engram-config '{"cpu_offload":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,6,12,18,24,30,32]}' \
  --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":true},"enable_fused_mc2":1,"multistream_dsv4_dsa_overlap":true}'
```

两条路径起服都约 8-20 分钟（权重 ~4.5min + 图捕获；首次静态核编译更久）。

## 6. 验收（缺一不可）

1. **健康**：`curl 127.0.0.1:<PORT>/v1/models` 返回列表；引擎日志出现 draft 图捕获行
   （路径 A：`[dspark-capture-audit] capture #N 提供的 metadata`；路径 B：`[dspark-graph-capture] built capture metadata`）。
   **没有这行而 A≈1 且无报错 = 静默退化**（draft 没入图，attention 走了空 metadata 分支）。

2. **预热后探针**（冷探针首条 TTFT 可达 15s，必须丢弃）：

```bash
# 探针脚本：examples/deepseek-v4.1-flash-910b 的 decode_probe.py（8K 上下文，强制 512-token 生成）
python3 decode_probe.py http://127.0.0.1:<PORT> longgen-prompt.json 512   # ×5 丢弃，×3 取中位
```

判读：**A≈5.89** 为健康（接受率已到顶，A 再高不可能）；路径 A 期望 ~163 tok/s、路径 B ~119 tok/s
（±5% 内属正常机器差异）。**A<2 = draft 图未生效**，回查 §5 开关与 §3 的补丁/PR 是否真的落到容器里。

3. **精度门禁**：

```bash
# 评测脚本与门禁协议：ascend-infra-skills 的 llm-serving-accuracy-gate skill / 端口仓 tests/acc_eval.py
# gsm8k test.jsonl：HF 或 ModelScope 的 openai/gsm8k
# 官方编码器：基座 deepseek-ai/DeepSeek-V4.1-Flash-1017 的 encoding/ 目录（见 §1 注意事项）
python3 acc_eval.py --task gsm8k --limit 100 --mode chat \
  --out gsm8k.json --tag deploy --conc 4 --base-url http://127.0.0.1:<PORT> \
  --enc-dir <ENCODING_DIR> --gsm8k-data-dir <GSM8K_DIR>
```

**基线：路径 A 100/100、路径 B 99/100**（<97 必查）。
注意门禁必须走**官方 encoder + `/v1/completions` + 串行 prefill + 数字兜底抽取**——
换协议（如并发 prefill 的 chat 简易门禁）会得到 ~30/100 的假阴性，两条路径都一样。

**性能锚点汇总**（口径：8K 上下文、512-token 强制长生成、预热 5 丢 3 取中位，2026-10-20 同机同权重）：

| 路径 | tok/s（中位） | A | TTFT | GSM8K-100 |
|---|---:|---:|---:|---:|
| A 端口仓优化栈 | **163–171**（实测 158.5/163.3/168.4/170.6） | 5.89 | 1.29–1.31 s | 100/100 |
| B rc2 + PR #18967 | 119.1 | 5.89 | 1.17 s | 99/100 |

AISBench 口径（`--input_len 8000 --output_len 256 --data_num 3 --concurrency 1`，含 prefill 的 E2E）：
路径 A 在 1017 权重上 E2EL 3,272ms / 输出吞吐 **78.25 tok/s**。

## 7. 已知坑（真机踩过）

| 症状 | 原因 → 处置 |
|---|---|
| 起服报 `HFValidationError: Repo id must be...` | 权重目录没挂进容器（路径不可见被当 repo id）→ 核对 §4 `-v` |
| `FileNotFoundError .../engram_int8/layers_1_...` | §2 的分片拼接没做 |
| 路径 B 起服报 `unrecognized arguments: --engram-config` | 用错镜像（0.27.x 线）→ 确认 `vllm-ascend 0.30.0rc2` |
| 路径 A 设了 `VLLM_ASCEND_ENABLE_DSPARK_DRAFT_GRAPH=1` 但没效果 | 0.27.x 线不认这个变量，要用 `DRAFT_GRAPH=1` |
| cherry-pick PR 后 import 失败 `No module named 'vllm_ascend.ops.triton.dspark_swa_indices'` | rc2 该模块在 `ops/triton/spec_decode/` 下，PR 的 import 走新路径 → 用 `git cherry-pick`（三方合并会自动保留 rc2 的旧 import 行）；直接 `git apply` 打补丁会失败 |
| 引擎重启即挂 `cv2 / libxcb.so.1` | 容器里装了 `opencv-python`（非 headless）→ 只装 `opencv-python-headless` |
| 换引擎后 `Free memory ... less than desired` | 旧 worker 残留占卡；worker 被 setproctitle 改名，`pkill -f vllm` 杀不到 → 按 npu-smi 报的 PID 杀：`npu-smi info \| grep VLLMWorker \| awk -F'\|' '{gsub(/ /,"",$3);print $3}' \| xargs kill -9` |
| 首条探针 TTFT ~15s | 冷探针假象 → 预热 5 条再测 |
| A≈1 且无报错 | draft 图静默退化：开关未导出，或补丁/PR 未真正落到容器里的 `/vllm-workspace/vllm-ascend` |

## 8. 与上游 vllm-ascend 的关系（防混淆）

- **路径 A**（端口仓栈）= 15 项补丁烘焙，**零上游 PR 依赖**；抬速来源是补丁 + 45 个 `V41_*` 开关。
- **路径 B** = 上游 `v0.30.0rc2` + **仅一个 PR（#18967）**；抬速来源只有 draft 图捕获。
- 两条线的差距（163–171 vs 119.1）**不能**归给 PR：PR 单独只把上游从 96.3 抬到 119.1（1.24×），
  剩下的 1.37× 在路径 A 的私有补丁与开关里。测量依据：
  `evidence/perf-campaign/ab-2026-10-20/README.md`（同机同权重同时段）。
- PR #18967 合入后，路径 B 可退化为"直接用含该功能的上游 release"：
  `git log` 能找到该 commit 就跳过 cherry-pick，开关名不变。
- 定价实验的完整推理：`evidence/perf-campaign/P5-UPSTREAM-PRICING.md`。

## 9. 参考资料

- 上游 PR #18967：https://github.com/vllm-project/vllm-ascend/pull/18967
- 端口仓（路径 A）：https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B
  （`patches/README.md` 有 15 项补丁清单、md5、门控 env 与实测收益）
- 技能仓：https://github.com/Liears/ascend-infra-skills （验收协议、战役证据、本指导书来源）
- 上游教程：vllm-ascend 仓 `docs/source/tutorials/models/DeepSeek-V4.1-Flash.md`（多机 W8A8 路线）
- 权重仓：https://www.modelscope.cn/models/chiro2001/DeepSeek-V4.1-Flash-1017-w4a8-Ascend
