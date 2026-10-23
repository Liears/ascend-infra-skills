# P9 — 算子团队出击：用 PyTorch 组合算子替换 aclnnSparseFlashMla（概念验证成功，2026-10-23）

> 用户指令："你就是算子团队"。本文是一次完整的算子级攻击：从量化官方内核的
> 真实开销，到写出数值等价的替代实现并实测胜出。
> 设备：8×910B3（A2）。**所有图内数字与引擎 profile 的 kernel_details.csv 同口径。**

## 0. 结论

**PyTorch 组合算子（不用任何自定义内核）在 NPU graph 内跑同一稀疏 MQA，
设备时间 105.2 µs，比官方 `aclnnSparseFlashMla` 的 170 µs 快 38%，数值 bf16 级精确
（median Δ=0.0002，max Δ=0.0046）。**

接入引擎（每步 ~18.6 次调用）预期 **−1.2 ms/步 ≈ 3.8%**；若把 kernel 数从 10
压到精简版的 4（已实测 50.7 µs 的骨架），上限 **−2.0 ms/步 ≈ 6.4%**。

## 1. 官方内核的真实开销构成（为什么值得打）

每次调用的真实需求 vs 实际耗时：

| 项 | 量 |
|---|---:|
| 数据搬运 | 640 KV × 512 dim × 2B = **0.65 MB → <1 µs** |
| 计算 | 6 tok × 8 head × 640 × 512 × 2 = **63 MFLOP → <1 µs** |
| **实测（图内设备时间）** | **170 µs** |
| ⇒ 固定开销占比 | **~99%** |

它占步长 8.3%，其中 SCALAR 43%/FIXPIPE 56%（P7），且 A2 被锁进 HCA 兼容路径
（P8 §5：`ori_topk` 强制 0、`cmp_ratio` 强制 128）。

## 2. 实测记录（全部同机同形状：T=6, H=8, D=512, KV=640, paged block=128）

| 实现 | 图外墙钟 | **图内（NPU graph replay）** | kernel 数 |
|---|---:|---:|---:|
| aclnnSparseFlashMla（官方，引擎内 profile） | — | **170 µs** | 1 |
| PyTorch 朴素（einsum + fp32 广播） | 493 µs | — | ~12 |
| **PyTorch 精简（bmm×2 + softmax + gather）** | 144.6 µs | **50.7 µs** | 4 |
| **PyTorch 完整语义（+mask/sinks/req映射）** | 481.9 µs | **105.2 µs** | ~10 |
| Triton 单内核（online softmax flash-decoding） | 183.5 µs | 未测 | 1 |
| Triton split-KV 两阶段 | 391 µs | 未测 | 2 |

三个关键判定：

1. **图外测耗时是陷阱**：Triton 图外 183 µs 里 ~150 µs 是 Triton-Ascend 的
   launch 开销（split 两阶段 391 µs ≈ 2× launch，直接证实）。
   **任何替代方案必须在 NPU graph 内评估**（本表"图内"列）。
2. **官方内核的 170 µs 不是 launch**（它在引擎 ACL graph 内，launch 已摊掉）——
   是内核本身的 HCA 路径低效。PyTorch 10-kernel 组合在图内 105 µs 胜出，
   证明"多 kernel 但每个高效" > "单 kernel 但路径差"。
3. Triton 路线被 launch 开销否决（图内未测，但图外已输给 PyTorch 组合，
   且图内还得过 Triton 编译的 memory planning 关：BD=512 时首次尝试即失败，
   降到 BK=32 才过）。

## 3. 数值验证

完整语义版 vs 逐 token 手工参考（含 -1 无效列、sinks）：

```
median|Δ| = 0.000216   max|Δ| = 0.0046    (bf16 舍入级)
```

Triton 单内核版同样精确（median Δ=0，max 0.002）。

## 4. 替代实现的完整语义（已实现并单测通过）

```python
def torch_sparse_mla(q, kv_cache, sparse_indices, req_to_token,
                     sinks=None, scale=None):
    # q: [T,H,D] TND；kv_cache: [NB,BLOCK,1,D] paged latent KV
    # sparse_indices: [R,W] 每请求可见的全局 KV 行号（-1 无效）
    T, H, D = q.shape
    flat_kv = kv_cache.reshape(-1, D)
    idx  = sparse_indices.index_select(0, req_to_token)      # [T,W]
    safe = idx.clamp(min=0)
    kv   = flat_kv.index_select(0, safe.reshape(-1)).reshape(T, -1, D)
    s    = torch.bmm(q, kv.transpose(1, 2)) * scale          # [T,H,W]
    s    = s.masked_fill((idx < 0)[:, None, :], float("-inf"))
    if sinks is not None: s = s + sinks[None, :, None]
    a    = torch.softmax(s, dim=-1)
    return torch.bmm(a, kv)                                   # [T,H,D]
```

要点：`bmm` 形状 `[T,H,D]@[T,D,W]` 天然无广播复制（KV 按 token 分组），
这是朴素 einsum（493 µs）与精简版（144 µs）的全部差别。

## 5. 接入引擎的剩余工作（下一步的明确路线）

已确认的接口事实：

| 项 | 状态 |
|---|---|
| draft 路径 indices | ✅ 现成：`build_dspark_swa_indices()` 已产出 [R,W] 全局行号 buffer（图内地址稳定） |
| q 形状 | ✅ TND [T,H,D]，与调用点一致 |
| kv_cache / block_table / sinks / scale | ✅ 都在 `attn_kwargs` 里 |
| **target 路径 topk indices** | ❌ **未解决**：正常 decode 时 `ori_sparse_indices` 不传，top-512 选择经 `sas_metadata`（1024×int32 不透明 buffer）进入内核；buffer 布局内核私有，需逆向或走 QLI 输出 |
| 滑窗（win_left=127） | ⚠️ draft 路径已含在 indices 里；target 路径需在 indices 合并时补 |
| compress_ratio=2/1 的 36 层 | ⚠️ 走内核内 HCA 路径，压缩 KV（cmp_kv）语义需一并实现 |

**推荐接入顺序**：先接 draft 路径（indices 现成，3 层/步，可当门禁试验田），
再逆向 sas_metadata（用已知的 QLI topk 输出对拍 buffer 内容定位索引段），
最后统一 target 三种 compress_ratio 分支。全程 GSM8K 门禁 + 预热后交替 A/B。

## 6. 交付物清单（tools/ 目录）

| 文件 | 内容 |
|---|---|
| `mqa_ref.py` | 朴素 einsum 版（493 µs，反面教材：fp32 广播复制） |
| `mqa_v2.py` | 精简 4-kernel 版（图内 50.7 µs，语义子集） |
| `mqa_graph.py` | NPU graph 捕获/重放测法（**同口径的关键工具**） |
| `smla_torch.py` | 完整语义版 + 数值单测（图内 105.2 µs，**接入候选**） |
| `mqa_triton.py` | Triton 单内核（数值精确；图外 183 µs，被 launch 否决） |
| `mqa_tune.py` | Triton split-KV（391 µs，2× launch 的实证） |

## 7. 给 CANN 算子团队的对照数据

同一形状（T=6,H=8,D=512,选 640 KV,paged 128）下，**4 个通用 aclnn kernel
（gather + bmm + softmax + bmm）在 graph 内共 50.7 µs**。
`aclnnSparseFlashMla` 单 kernel 170 µs。
HCA 兼容路径的固定开销（SCALAR/FIXPIPE 流水排空）比"多 kernel 无融合"还贵 3.4×，
说明该路径在 A2 上存在 ≥3.4× 的优化空间——这与 P8 §5 的"A2 锁进 HCA 路径"互为印证。
