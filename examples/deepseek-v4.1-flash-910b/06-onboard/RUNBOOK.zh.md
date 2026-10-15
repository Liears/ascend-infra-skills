# 上板手册 — DeepSeek-V4.1-Flash 在 8×910B3（A2）

[English](RUNBOOK.md) | 中文

第二级验证：把模型真正跑起来，让技能吃到的不再是合成 trace 而是真实的。

本手册对照公开移植仓
`chiro2001/deepseek-v4.1-flash-ascend910B`（其 `scripts/serve_a2.sh`）
编写，每个变量名和默认值都从该脚本里读出——不是从 README 抄的。凡标注
"默认"的值都来自脚本本身。

> **状态：已执行。** 2026-10-15 在本机完整走通：起服成功、推理正确、
> 真 trace 已采集。产物在 [`evidence/`](evidence/)。

---

## 0. 前置条件

| 项 | 要求 | 在哪 |
|---|---|---|
| 硬件 | 8 × 910B3（A2），每卡 64 GB HBM | 本节点 |
| 基础镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler` | 单独拉取 |
| 权重 | `chiro2001/DeepSeek-V4.1-Flash-w4a8-Ascend`，约 304 GB | ModelScope |
| 补丁镜像 | `scripts/build_image.sh` 构建 → `dsv41-a2:v9` | 本地构建 |

**Engram 重组是必做步骤。** 下载的权重不能直接起服：必须先跑随权重
发布的 Engram 重组脚本，否则加载就会失败。

---

## 1. 拿权重

```bash
modelscope download chiro2001/DeepSeek-V4.1-Flash-w4a8-Ascend \
    --local-dir /mnt/local/hbw-work/dsv41-w4a8
```

304 GB。下完后先跑 `engram_int8/reassemble_engram_weights.sh
--check-only` 验分片 sha256，通过再跑不带参数的正式重组（脚本自带
校验，不过就中止，不会留残缺文件）。

**磁盘：** 放 `/mnt/local`（~4 TB 可余）。不要放 `/`（438 GB，已用
85%）。

---

## 2. 构建补丁镜像

```bash
# BASE_IMAGE 与 IMAGE_TAG 默认值来自 build_image.sh
bash scripts/build_image.sh
```

它探测镜像布局（不写死路径），把已验证的补丁烘焙进镜像，被改文件留
`.a2orig` 备份，并逐文件校验 md5 + `py_compile`。产出
`dsv41-a2:v9`。

**tag 必须三处一致** —— `build_image.sh`、`serve_a2.sh`、
`run_test.sh`。包内自检发现不一致会拒发；它真的抓到过一次。

---

## 3. 起服前检查（不要跳过）

```bash
bash tools/check_model_dir.sh         # 模型目录完整性，~10 秒，不起容器
bash tools/list_chips.sh              # 哪些 device 空闲、属主是谁
```

共用机器上 `list_chips.sh` 很重要：占卡之前先确认 8 个 device 都空。

---

## 4. 起服

```bash
MODEL=/mnt/local/hbw-work/dsv41-w4a8 \
DEVS="0 1 2 3 4 5 6 7" \
bash scripts/serve_a2.sh
```

变量（默认值来自 `serve_a2.sh`）：

| 变量 | 默认 | 含义 |
|---|---|---|
| `MODEL` | *（必填）* | 权重目录 |
| `IMAGE` | `dsv41-a2:v9` | 补丁镜像 |
| `NAME` | `dsv41-a2` | 容器名 |
| `PORT` | `8100` | 服务端口 |
| `TP` | `8` | 张量并行 **device 数** |
| `DEVS` | `0 1 2 3 4 5 6 7` | device 列表 |
| `SERVED_NAME` | `deepseek-v41` | 对外模型名 |
| `GPU_UTIL` | `0.92` | HBM 占比。**0.94 只剩 6.11 GiB，8K prefill 从 1.14 s 掉到 8.0–8.6 s** —— 这个旋钮不是摆设 |
| `MAX_LEN` | `1048576` | 最大上下文（1M） |
| `MAX_SEQS` | `32` | 影响 graph 捕获的桶数 |
| `PREFIX` | 开（`NO_PREFIX=1` 关） | prefix caching |
| `DRAFT_GRAPH` | `0` | 投机草稿图；**默认关是刻意的** —— 不理解就开会拿到坏的接受长度且无任何报错 |
| `V41_PROFILE` | `0` | 设 `1` 才挂出 `/start_profile` `/stop_profile` |

### 脚本已经防好的两个坑

1. **代理劫持 localhost。** 代理拦截 `127.0.0.1` 会给 `/health` 回错误
   页，健康的服务看起来像起服挂死。脚本已设 localhost 的 `no_proxy`
   （除非 `KEEP_PROXY_FOR_LOCALHOST=1`）。验证用
   `curl -s --noproxy '*' http://127.0.0.1:8100/health`。

2. **CPU 绑核挂死。** `CPU_BIND=1` 在目标 NUMA 节点占满时会无限卡在
   `migratepages` —— 不是慢，是不收敛。多卡起服一直不就绪就换
   `CPU_BIND=0` 重试。

---

## 5. 确认真的起来了

```bash
curl -s --noproxy '*' http://127.0.0.1:8100/health
curl -s --noproxy '*' http://127.0.0.1:8100/v1/models
```

然后查四件各有产物的事：

| 检查 | 命令 / 来源 | 什么算证明 |
|---|---|---|
| KV 容量 | 启动日志 | `GPU KV cache size: N tokens` 行 |
| Engram 快路径 | profiler | `kernel_details.csv` 里出现设备侧查表算子 |
| 投机解码生效 | 响应元数据 | 报出接受长度，不是推断 |
| 无静默降级 | profiler | 融合算子确实在 |

---

## 6. 采 trace 喂给技能

起服时带 `V41_PROFILE=1`，然后：

```bash
curl -s --noproxy '*' -X POST http://127.0.0.1:8100/start_profile
# ...发几条请求...
curl -s --noproxy '*' -X POST http://127.0.0.1:8100/stop_profile
```

注意：**容器内**的 prof 落在 `/opt/dsv41/results/<run_id>/prof`（宿主
对应 `/tmp/dsv41-ascend/results/<run_id>/prof`）。API 采出来的是 raw
数据，还要在容器里跑一次离线解析才有 `kernel_details.csv`：

```bash
python3 -c "import torch_npu.profiler.profiler as P; P.analyse('<rank 目录>')"
```

然后喂给技能：

```bash
python3 skills/npu-profiler-analysis/scripts/analyze_profile.py <ASCEND_PROFILER_OUTPUT 目录>
python3 skills/llm-serving-capacity-planner/scripts/capacity_analyzer.py <serve.log>
```

闭环至此完成：原先只在合成和小 trace 上验证过的技能，现在跑在真实的
1M 上下文 MoE 服务上。

---

## 7. 回滚

容器即抛即弃。停掉它，机器回到空闲：

```bash
docker rm -f dsv41-a2
```

补丁镜像和权重留在盘上，下次直接用。static kernel 编译产物也有缓存
（`skcache`），重启不再冷编译。
