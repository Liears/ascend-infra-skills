# 上板证据 — DeepSeek-V4.1-Flash 在 8×910B3

[English](README.md) | 中文

2026-10-15 本机真实运行的产物。不是合成数据。

> 本目录下的 `real-*.log` / `*.txt` / `acceptance-REPORT.md` 是**原始
> 产物，保持原样不翻译** —— 它们是证据。中文叙述在本文件。

## 实际跑了什么

| 步骤 | 结果 |
|---|---|
| 权重：`chiro2001/DeepSeek-V4.1-Flash-w4a8-Ascend` | 72/72 分片，490 GB，ModelScope `Snapshot ready` |
| Engram 重组（`engram_int8/reassemble_engram_weights.sh`） | 两个文件重建，**sha256 校验通过**（各 98.3 GB） |
| 镜像构建（`scripts/build_image.sh`） | `dsv41-a2:v9`，15 个补丁文件逐字节 md5 校验 |
| 模型目录自检（`tools/check_model_dir.sh`） | 通过，1 个警告（vision 旋转版无法判定） |
| 起服（`scripts/serve_a2.sh`，TP8，GPU_UTIL=0.92，bat=8192） | **起来了**，health HTTP 200 |
| 真实推理（`/v1/chat/completions`） | 输出正确 |
| 真 trace 采集 + 离线解析 | **229,642 行 kernel_details.csv，486 MB trace** |

## 在线服务报出的数字

摘自 `real-memory-lines.txt`（从服务日志提取）：

```
Available KV cache memory: 13.60 GiB
GPU KV cache size: 2,942,071 tokens
Maximum concurrency for 1,048,576 tokens per request: 2.81x
Free memory on device (60.61/60.96 GiB) on startup. Desired GPU memory utilization
  is (0.92, 56.08 GiB). Actual usage: 38.7 GiB for weights, 3.21 GiB for peak
  activation, 0.56 GiB for non-torch memory, 1.02 GiB for NPU graph memory.
```

注意四项分账：**权重 38.7 GiB** 和 **NPU graph 显存 1.02 GiB** 是两个
大头，而 graph 显存这一项是昇腾特有的、按 GPU 思路建的账会漏掉它。

## 验收门结果（`acceptance-REPORT.md`）

| 门 | 结果 |
|---|---|
| `static_kernel` 未被静默降级 | ✅ `static_kernel.py:650` 零命中 |
| Engram local-owner 自检 | ✅ `fast` |
| KV 容量 > 2,800,000（GPU_UTIL=0.92 下） | ✅ 2,942,071 tokens |
| 8K quote 解码 | 27.2 tok/s，A=1.70，ms/step=62.32 |
| 32K quote 解码 | 47.7 tok/s，A=2.91，ms/step=60.91 |
| 视觉 23 例 | ⏭ 跳过 —— 未提供官方图片目录 |
| GSM8K-200 | ⏭ 未跑（MODE=quick） |

## 日志里出现的一处静默降级

```
WARNING [patch_triton.py:332] NPU Triton causal_conv1d_update is unavailable
(cannot import name 'causal_conv1d_update_npu' ...); falling back to the PyTorch
implementation, which syncs per request and therefore stalls ACL graph capture at
decode-FULL.
```

**这正是 `model-day0-support` 技能 fast-path 门禁（G3）要抓的失效形
态。** 模型能跑、输出正确，但有一个 kernel 不是融合版本，唯一的信号是
一条 WARNING。对这个模型的任何 Day-0 主张都必须说明这条降级是否影响
所主张的路径。

## 技能拿真数据做了什么

- `capacity_analyzer.py` 吃在线 `serve.log`，复现了 HBM 分账和
  2,942,071 token 的池，并按三个请求长度估了并发。
- 这一跑**抓出了技能自身的一个 bug**：vLLM 打印
  `` Replace gpu_memory_utilization with `--kv-cache-memory=N` `` 是
  *建议*，而解析器把它报成了实际生效的配置。已修：解析器现在排除
  建议行。这就是二级验证的意义 —— 合成日志永远暴露不了它。
- `analyze_profile.py` 吃 229,642 行的真 trace，暴露了第二个 bug
  （硬编码的结论文案与实际数值矛盾 + 没识别通信主导）。同样已修并加
  回归测试。

## 查 DRAM 卸载时发现的一个 fail-open bug

`run_test.sh` 明确不验证 DRAM 卸载，所以我们跑了该仓自己的
`a2/scripts/verify_dram_offload.sh`。它的收尾行是：

```
✅ 硬判据全过：DRAM 卸载**存取双向**都验证到了
```

**那行是错的，它上面的判据自己就能证明。** 同一份输出里每条硬判据都
是 `? 未验`（`Δ=None`），H3 是 `✗`。读源码
（`tests/verify_dram_offload.py` 的 `line()` 辅助函数）就知道为什么：
`hard` 只在显式 `False` 时置 1，所以判据求值为 `None`（"未验"）时
`hard` 保持 0，而 `hard == 0` 就打印成功横幅。一次**什么都没测到**的
运行报了成功。

为什么这里全是 `None` —— 直接对在线服务验证过：

| 探测 | 结果 |
|---|---|
| `POST /reset_prefix_cache` | **HTTP 404** |
| `/metrics` 里的 `vllm:kv_offload_total_bytes_total` | **不存在** |
| `cache_config_info` | `enable_prefix_caching="False"`、`kv_offloading_size="None"`、`kv_offloading_backend="native"` |

所以是两个独立的事实：

1. **本次部署没有启用 DRAM 卸载。** 我们用 `scripts/serve_a2.sh` 起
   服，不是 `serve_a2_offload.sh`；卸载路径是另一个启动器。指望它有
   卸载计数器本身就是个错误。
2. **验证器 fail-open。** 给定事实 1，它应该拒绝下结论；结果一次测不
   到任何东西的运行报了通过。

两条都归技能管：

- **容量规划器**必须把 `kv_offloading_size=None` 报成"未启用卸载"，
  而不是当字段不存在 —— 已写进 `references/npu-memory-lines.md`；
- **Day-0 技能的 G3 门禁**说的就是这类事："快路径跑了"必须*观察到*，
  绝不从包装器的总结行推断。一个测不到任何东西却报成功的工具，和一
  个静默退回慢 kernel 的模型，是同一类危险。

按证据规则，这里的结论是**"本部署未启用 DRAM 卸载，因此未验证"** ——
不是"卸载生效"。

## 本案例不主张什么

- 不主张与厂商公开数字的性能对齐。实测 27.2 tok/s @8K 远低于该仓
  `EXPECTED_PERF` 的 A3 数字；A2 是更慢的设备，且这次跑的是
  `MODE=quick`、`GPU_UTIL=0.92`、`bat=8192`、无 PGO（PGO 产物缺失，
  脚本记了 `PYTHON_PGO=1 但没有可用的 PGO 产物 → 降级为不挂`）。
- 不主张精度。`MODE=quick` 没跑 GSM8K。
- 不主张视觉通路。跳过了：没给官方图片目录。

把没做到的如实记为"没做到"，这就是证据规则存在的意义。
