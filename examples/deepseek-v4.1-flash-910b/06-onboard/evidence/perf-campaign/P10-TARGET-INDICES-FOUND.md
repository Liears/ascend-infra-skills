# P10 — 重大更正与突破：target 路径的 top-512 索引从来都是显式传入的（2026-10-27）

> 承接 P9 §5 与 PR #19852 的"范围说明"。那里写"target 层的 top-512 索引藏于
> `sas_metadata` 不透明 buffer，不可及"——**这个结论是错的**。
> 本轮读通了 V4.1 的专属实现，索引通道完全可见，替换的最后一道门已经打开。

## 0. 结论

**V4.1 的 target attention 走的是 `dsa_v41.py` 的专属实现（不是之前分析的
`dsa_v1.py` 通用路径），top-512 索引通过 `cmp_sparse_indices` 参数显式传入。**
PyTorch 替换的全部输入都在 vllm 侧可及。P9 的 3.8–6.4% 预期收益路径完全打通，
剩余工作是让替换实现在 ACL 图捕获下兼容（本轮已定位到具体卡点）。

## 1. 之前为什么看错

| 误判 | 实际 |
|---|---|
| 我分析的 `dsa_v1.py` 通用路径 `attn_op(q, **attn_kwargs)` | V4.1 用 `dsa_v41.py` 的 `DeepseekV41EagerAttentionImpl._native_attention`，直接调 `torch.ops._C_ascend.npu_sparse_flash_mla`（vllm_ascend 自编译扩展，非 cann_ops 版） |
| "target 不传 ori_sparse_indices ⇒ 索引进不透明 metadata" | target 走 **cmp 稀疏**：`cmp_sparse_indices=cmp_indices` 显式传入（`pad_sparse_indices(topk_indices, 512)` → `[T,1,512]` int32，-1 填充） |
| "cmp_ratio=128 HCA 锁死" | `_C_ascend` 版支持真压缩比（1/2）；aclnn 报的 128 约束属于 cann_ops 版的 A2 兼容路径 |

## 2. 完整索引链（每环有代码出处）

```
index 源层（index_source_layer_ids = [2,8,14,20,24,28,32,36]，8 层）
  └─ AscendIndexerOps.select_topk                        (indexer.py:185)
       npu_vllm_quant_lightning_indexer(..., sparse_count=512)
       → topk_idxs（KV 池全局 slot id）
  └─ shared.topk_indices[:T].copy_(selected)             (dsa_v41.py:476-481)
同组消费层（每 4 层一组，skip_topk=True）
  └─ return shared.topk_indices[:T]                      (dsa_v41.py:460)
_native_attention
  └─ cmp_indices = pad_sparse_indices(indices, 512)      (dsa_v41.py:229)
  └─ npu_sparse_flash_mla(..., cmp_sparse_indices=cmp_indices,
        cmp_ratio=ratio(1/2), cmp_mask_mode=3, topk_value_mode=1, ...)
```

`DeepseekV41SharedAttentionState`（model.py:478）持有持久 buffer，
"keeping the storage intact ... preserves a fixed address for ACL Graph"——
**索引 buffer 本来就是图捕获地址稳定设计**。

## 3. 替换实现已写好并挂上（dsa_v41.py 的运行时 hook）

`_torch_smla41`（两源拼接 attention）：

- **ori 源**：每 token 因果滑窗 `[pos-127, pos]`，经 `ori_block_table` gather
- **cmp 源**：`cmp_indices` 视为 cmp 池逻辑位置 → `cmp_block_table` 映射 → gather；
  cmp 因果掩码：cmp 位置 `c` 覆盖原 token `[c·ratio, c·ratio+ratio)` ≤ 当前 pos
- 两源 score 拼接 + sinks + softmax + 双源 V 加权
- 无效列 -1e4（同 draft 版的 NaN 防护）

hook 形态：文件开关 `/tmp/v41_smla41`（`1`=替换 / `parity`=对拍），调用点在
`_native_attention` 的官方算子前，`_p41args` 收集全部入参。

## 4. 本轮实测（交替协议之外的注意点）

| 状态 | 结果 |
|---|---|
| hook 代码在位、开关关 | **引擎健康**，A=5.89，基线 146.5 tok/s（8 次中位，137.6–152.6） |
| 开关开（mode=1，起服前设置） | **worker 在图捕获期静默退出**（"all workers exited gracefully"，无 traceback） |

**卡点定位**：`_torch_smla41` 里的动态形状算子（`repeat_interleave`、
`arange`/`div` 组合的 per-token 位置计算）在 npugraph_ex 捕获下不兼容。
draft 路径的实现（P9）能被捕获，差别正是这些动态形状算子——draft 版的
indices 是现成 `[T,1,W]` buffer，无需运行时构造索引。

## 5. 下一步（明确的收尾路径）

1. **捕获友好化**：decode 每请求 `q_len = 1+S = 6` 固定 ⇒ 预分配
   `[maxT, W+K, D]` 的 KV 拼接 buffer + 位置网格 buffer（常量形状），
   把 `repeat_interleave/arange` 移到图外（capture 前一次性构建，
   每步只 `copy_` 刷新）——与端口仓 engram buffer 同一套地址稳定手法。
2. 通过后按既定三件套验收：A=5.89、GSM8K-100、预热交替 A/B。
3. 上游化：沿用 PR #19852 的开关命名（`VLLM_ASCEND_ENABLE_TORCH_SPARSE_MLA`），
   在其分支上扩展 target 路径（同 PR 或 follow-up）。

## 6. 给"更正记录"的一课

P9 §5 的"索引不可及"结论写进了 PR #19852 的 Scope note——**该说明需要随
target 路径打通后一并更正**（在新 PR 里说明 cmp_sparse_indices 路径已覆盖）。
教训：**"我没找到传入点" ≠ "没有传入点"**；下结论前先确认自己看的是不是
模型实际执行的那条实现（V4.1 专属文件 vs 通用文件）。
## 7. 攻坚日志（v1→v7，每轮失败都有定位与修复）

| 版本 | 卡点 | 定位手段 | 修复 |
|---|---|---|---|
| v1 | worker 静默退出（无 traceback） | — | v3 加异常回退后才看见错误 |
| v2(未装) | 补丁复制被 kill 打断，静默装回 v1 | 读文件实际内容核对 | 整函数替换（final_fn.py），杜绝碎片 patch |
| v3 | `AttributeError: cmp_idx.shape`——**compress_ratio=0 的层无 cmp 源**（cmp_indices=None） | 异常回退 + traceback 打印 | 支持 None 单源路径 |
| v4 | OOM 3.3GiB：物化 `[T,512,512]` gathered KV，prefill T=8176 时 4.3GB/层 | OOM 分配量反推 | **decode-only 门**（T≤64 才替换） |
| v5 | 同 OOM：T=18 仍爆——**warmup 捕获期显存余量仅 3.16GB**（56.37/60.96 已预留），官方融合算子零中间张量、我的有 | P41-DBG 打印 T/K + OOM 上下文 | `--gpu-memory-utilization 0.86` 腾 3.7GB |
| v6(=v5+DBG) | **引擎完整启动、零 P41-ERR**，捕获通过 | — | ✅ 图捕获问题全部解决 |
| v7(=v6 真数据) | 第一个真实请求触发 **AI Core Error 507011**（设备侧 fault，Python 层无异常）；warmup dummy 数据可过 | npu-smi core dump（待分析） | ⛔ 下一轮起点 |

**v6 是里程碑**：PyTorch 替换在全部 36 层 target 路径上通过了 ACL 图捕获——
捕获友好化（静态形状索引数学）+ 边界防护 + None 支持 + decode-only 门 + 显存余量
五项修复全部到位。剩余最后一个问题在**数值/索引语义**：真实数据下某个 gather
的索引越界或 cmp 索引空间假设错误（逻辑位置 vs 池 slot），导致设备侧 fault。
dummy 数据形状合法所以能过，真数据暴露语义错位。

下一轮入口：
1. 从 npu-smi / plog 取 core dump，定位 fault 的具体算子与索引值
2. 或在 hook 里 dump 一次真请求的 cmp_idx/oslot 统计（min/max/负值比例），
   与 cmp 池行数对比，直接验证"逻辑位置 vs 池 slot"假设
3. 修对后按三件套验收（A=5.89 / GSM8K-100 / 交替 A/B），预期 3.8–6.4%

## 8. 工程教训（本轮新增）

- **异常回退是调试图捕获的必备基建**：`try/except + print + traceback` 三行，
  把"worker 静默退出"变成可读错误。没有它，v3 的 None 坑永远看不见。
- **碎片 patch 必须核对落盘内容**：kill 打断 `cp` 后 install 用了旧 helper，
  两轮"修复"其实没装上。整函数/整块替换 + `grep` 验证关键字。
- **OOM 要看"already allocated"**：T=18 时 3.3GB 分配失败不是我的张量大，
  是捕获期总余量只剩 3GB。修复杠杆在 gmu，不在实现。
- **dummy 过 ≠ 真数据过**：warmup dummy 的 block table/索引形状合法，
  语义错误（索引空间假设）只有真数据暴露。
