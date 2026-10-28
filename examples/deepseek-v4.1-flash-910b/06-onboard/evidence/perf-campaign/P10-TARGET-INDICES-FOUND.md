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
## 9. v8 战况（2026-10-27 深夜轮）：替换已真实运行，剩一个纯数值语义 bug

两项决定性实证（本轮新增）：

1. **索引空间判定（prefill dump，真数据）**：
   `cmp_idx min=0 max=4077 uniq=4078`，`cmp_seq_lens=[4078]`，`cmp_pool_rows=1,728,384`，
   `cmp_bt_max=64` ⇒ **cmp_idx = 压缩池逻辑位置**（max == cmp_seq-1），
   经 cmp_bt 映射到池 slot。v7 的映射方向本来正确。
2. **AI fault 根因（v8 修复并验证）**：v7 只对 gather slot 钳了**下界**；
   捕获期 dummy block table 可含垃圾大值 → `bt_value*128+off` 超出池行数 →
   index_select 设备侧越界 → AI Core Error 507011。
   v8 对 oslot/cslot 补了 **max=pool_rows-1 上界** → fault 消失。

**v8 运行结果**（gmu=0.86，异常回退兜住捕获期一次 OOM）：

| 观测 | 值 | 含义 |
|---|---|---|
| 服务 | 启动成功、200 OK、T41-ERR 仅捕获期 1 次 | 图捕获 + decode 替换路径**真正在跑** |
| 接受长度 A | **5.69 / 5.82（应 5.89）** | attention 输出数值有偏 → 接受率受损 |
| 吞吐 | 129–133 tok-s 交替 **2.6 tok-s 崩坏请求** | 周期性输出崩坏 |

⇒ **结构性问题全部解决（捕获/映射/越界），剩一个纯数值语义 bug。**

### 候选（按嫌疑排序，下一轮逐个二分，每轮一次重启）

1. **cmp 因果边界**：`cpos0 = c*ratio` vs `(c+1)*ratio - 1`（压缩 token 覆盖区间的
   起点还是终点），以及 `cpos0 <= pos` 的比较方向（cmp token 是否对"未来"可见）
2. **ratio 取整**：dump 显示 seq=8157 → cmp_seq=4078（向上取整），而
   `cmp_residual=seq%2`——最后一个不满 ratio 的 cmp 位置的覆盖区间处理
3. **ori 滑窗 off-by-one**：`win_left=127` 的窗口是 [pos-127, pos] 还是 [pos-128, pos-1]
4. **topk_value_mode=1**（v1 语义）在我实现里被忽略——值域模式可能影响 cmp 分数的
   缩放或 residual 的分数贡献

对拍定位法（比二分快）：把 v8 的 gate 放宽到 T≤256 并在 prefill 尾块对拍一次
官方输出 diff——prefill 尾块小且 eager，物化安全。一次对拍直接给出数值偏差的
结构（整体偏移 vs 局部坏列 vs 幅度错误）。

### v8 补丁存档
`p41-target-hook-WIP.patch` 已更新为 v8（LOGICAL-POS 实证注释 + 全界钳制 +
decode-only 门 + 异常回退）。生产已恢复干净源码（A=5.89 基线不受影响）。
## 10. 对拍判决（2026-10-27 收官）：算法逐位正确，bug 在 decode 图内特有条件

对拍设置：v9 的 parity 模式——短 prompt（~200 tok）触发小 T eager prefill，
T=24、ratio=2、R=1，`_torch_smla41` 与官方算子同输入双算。

**判决：`med=0.00000 max=0.0000 match=100%`——逐位一致。**

算法本体（滑窗因果 + LOGICAL-POS 的 cmp 映射 + 因果掩码 + sinks + softmax）
**全部正确**。§9 列的四个语义候选（cmp 边界/ratio 取整/滑窗 off-by-one/
topk_value_mode）全部排除。

**A 掉（5.69/5.82）与周期性崩坏请求的根因因此收窄到 decode 图内特有的条件**：

| 候选 | 机理 |
|---|---|
| 1（首位） | **R 不一致**：图捕获按 bucket 最大 R（如 4）固化 shape；replay 时实际
  R=1~4 变化。我的 `R = seq_lens.shape[0]` 与 `cu_q[:R]` 若在图内被 padding
  （cu_q 固定 4 行、seq_lens 只填 1 行），`req` 广播比较会算出错误请求归属
  → gather 错行 |
| 2 | 图内 `pad_sparse_indices` 每次 replay 的地址/内容刷新时序（官方算子
  可能有专门的 buffer 同步约定） |
| 3 | draft verify 块（q_len=6/请求）与对拍 prefill（q_len=24 单请求）的
  `pos` 网格差异——但数学同构，嫌疑低 |

**下一轮入口**：在 parity 模式里放开 `is_current_stream_capturing` 限制改为
"捕获期算 diff 存 device buffer，replay 后外部读"（或直接对比图内 R 的
padding 约定——查 `_build_qli_metadata` 的 `cu_seqlens_q` 在捕获期如何 padding）。
一次修对即三件套验收 → 3.8–6.4%。

## 11. 真凶落网（2026-10-28 凌晨三轮仪器）：decode 索引在"原始 token 空间"

§10 的 R-padding 假设被证伪，随后三轮"图内仪器"（每轮一次重启）把真凶
逼到铁证：

### 三轮仪器与判决

| 轮 | 仪器 | 判决 |
|---|---|---|
| v10 graphparity | 图内 mine+official 双算、diff 全 device 侧累计进持久 buffer（max/sum/计数/分层 one-hot），返回 official（行为=基线）；eager prefill 处打印+清零 | 真实 decode 流量 **45% 元素偏差**（big=42.2万/93.4万，max≈2.3）；捕获期 dummy eager 对拍反而逐位一致（含 T=24 R=4） |
| v11 gp2z | diff 按"真实/padding 行"分路（`toks < cu_q[R]`，cu_q[R]=实际总 token 数，buffer 内平 padding 值 device 侧可读） | **maxr=2.56 全在真实行；nelemp=0——根本没有 padding 行**（探针流量 R=1 走 T=6 bucket，nreal=6/调用）。padding 假设死 |
| v12 chk | 图内**输入校验和**（cu_q/seq_lens/bt/idx/q/sinks 的 max/sum/NaN 计数）+ **official 跑两次自检** | offself=0（无显存池污染，我的算子存在不影响官方）；cu_q=[0,6]、seq_lens≈8673、bt/q/sinks 全部合理——输入全是新鲜的。**唯独 idx_max=8667 ≈ seq_lens-6，远超 cmp_seq_lens≈4337** |

### 根因

**cmp_sparse_indices 的索引空间在 prefill 和 decode 是两套约定**：

- prefill（v9 对拍实证）：压缩池逻辑位置，`max = cmp_seq_lens-1`，直接经
  cmp_bt 映射。逐位一致。
- decode（本轮 checksum 实证）：**原始 token 位置**（max≈seq，含奇数值
  8667，压缩空间不可能），需先 `c = idx // ratio` 再映射。

我的组合把 decode 索引当压缩位置 → `idx//128` 块号超出 cmp_bt 列数 →
被 clamp 到错误块 → gather 错 KV → 真实行 45% 偏差 → v8 的 A 掉
（5.69/5.82）与周期性崩坏请求。官方算子两种空间都认（内部按
cmp_ratio/metadata 换算）。

### v13 修复

调用级 device 侧空间检测（全静态形状，捕获安全）：

```python
_vmax = torch.where(cvalid, csafe, -1).max()      # 有效索引最大值
is_orig = _vmax >= cmp_seq_lens.max()             # ≥ cmp_seq ⇒ 原始空间
c    = torch.where(is_orig, csafe // ratio, csafe)         # 压缩池逻辑位置
cov0 = torch.where(is_orig, csafe,     csafe * ratio)      # 覆盖区间起点
cvalid = cvalid & (cov0 <= pos) & (c < cseq_row)
```

检测边界：decode 索引 max≈seq 恒 ≥ cmp_seq=ceil(seq/2)（seq≥2 时）；
prefill max=cmp_seq-1 恒 < cmp_seq。seq≤1 时两种解释等价。

模式 `1mv`：返回修复版 mine，同时图内双算 official 累计 diff——一次重启
同时验证修复（A 值）与测量残差（maxr）。原语空间重复条目（同一组的两个
原始 token 都被选中 → 同一 cmp 行两次）是否需去重的语义问题，由 v13 的
maxr 残差直接裁决。

### 仪器工程（可复用）

- **图内累计器模式**：replay 里 Python 不跑，但 diff/max/计数可全用
  device 侧算子（`torch.maximum`、`+=`、one-hot 加权）录进捕获期分配的
  持久 buffer（vllm 图输出同款模式，host 随时可读）；**打印窗口选 eager
  prefill**（FULL_DECODE_ONLY 下 prefill 永远 eager，每个请求开头必经）。
- **official-official 自检**：同一图内把官方算子跑两遍 diff——一行判据
  分辨"我的算子污染共享状态"与"我的算子算错"。offself=0 排除了显存池
  重叠类整族假设。
- **输入校验和**：与其继续猜哪个 buffer 陈旧，直接在图内对每个输入记
  max/sum/NaN，一次拿到全部输入的健康画像。
- 补丁防叠加教训：installer 的 `assert` 关键字要选安装后必然出现的标识
  （如 `_torch_smla41`），`_TORCH41` 这类连字符标记永不匹配，三轮 patch
  会顺序叠加（碰巧模式字符串互斥、惰性跳过，未出错）。

## 12. 语义破解（2026-10-28 上午）：离线拟合台 bf16 级复现官方算子

v13（÷ratio 映射）/v14（cseq 单位检测+÷ratio）先后失败后，v15 idxab
（图内官方算子 ×3：原索引/÷2 索引/全-1 索引）给出关键排除：**索引被
大量使用（dac 98.6%），且不是 ÷2 语义（dab 44%）**。

随即停止引擎内试错，转**离线逆向**——csrc 源码 + `cann_ops_transformer`
命名空间离线调用（`ASCEND_CUSTOM_OPP_PATH` + pip 包注册的
`torch.ops.cann_ops_transformer.sparse_flash_mla`，与引擎 `_C_ascend.npu_*`
同 schema 同内核），搭出秒级迭代拟合台（`tools/smla_semantics_fit.py`）。

### 阶梯实验与判决

| 实验 | 设置 | 判决 |
|---|---|---|
| E0a | 全 -1 索引（无 cmp） | **med=0.0005 big=0%**：ori 窗+sinks 数学全对 |
| E0c/E0d | 单条目 / 双重复条目 | **逐位一致**：每条目数学正确，重复条目独立计权重 |
| E0b | 512 条目全同行 | 偏差大；倍数拟合收敛 **m=256** |
| E0g | 300 条目同行 | 有效 m=256 而非 300 ⇒ **不是倍数缩放是截断** |
| E0i | 仅有的条目放在位次 300 | **输出=无 cmp 逐位一致** ⇒ 位次在界外的条目被丢弃 |
| E1 | 随机 512 索引 + 前缀版 | **med=0.002 big=1.5%**：前缀版复现官方 |

### 官方算子 decode 语义（最终）

1. **行号 = 索引值直用**（`sparseBlockSize=1` 硬编码，无 ratio 除法）
2. **只消费前 `(pos+1)//ratio` 个条目**——位置前缀因果（算子假定索引器
   输出按位置有序，用"前多少个条目"实现因果裁剪，不做逐条目因果判断）
3. 重复条目各自独立进 softmax；sinks 加分；ori 滑窗 [pos-127, pos]
4. **prefill 无前缀截断、索引=压缩行号**（v9 引擎内逐位一致实证）——
   prefill/decode 走不同调度路径（"decode 调度由 metadata 驱动"）

引擎 decode 侧的索引单位混乱（原始空间 + cseq 原始单位，违背算子契约
"索引 ∈ [0, Lc)"）是**上游既有状态**——基线 A=5.89 就是在这套喂法上
测得的。替换实现的目标是匹配官方行为本身（离线已 bf16 级复现），而非
"语义正确"的想象版本。

### v16 修复（双语义按调用级检测切换）

```python
is_orig = cseq_max * ratio > sl_max + ratio        # cseq 原始单位检测
nvis = (pos + 1) // ratio                          # 前缀可见条目数
kvis = arange(K)[None,:] < nvis[:,None]
cvalid = (idx >= 0) & (kvis | ~is_orig)            # 压缩空间不截断
c = idx                                            # 行号直用(两空间同式)
cvalid &= (c*ratio <= pos) & (c < cseq_row)
```

（v13/v14 的教训固化在检测器里：idx 与 cseq 永远同空间，互比较不出
翻转；必须拿空间固定的 seq_lens 当参照。）

### 方法论沉淀

- **离线拟合台 >> 引擎内试错**：一次重启 25 分钟只测一个假设；离线
  `cann_ops_transformer` 直调一秒一轮，阶梯实验（无 cmp → 单条目 →
  双条目 → 全同行 → 位次越界）五步把语义钉死。
- **csrc 就是文档**：`csrc/attention/sparse_flash_mla/docs/ratio2_a2a3.md`
  给出因果公式 `visible_cmp = clamp((p+1)/r, 0, Lc)`——"前缀截断"正是
  它的逐条目退化形式；作者示例 `test_aclnn_sparse_flash_mla.cpp:264`
  给出索引契约 `idx ∈ [0, cmpKvLen)`。
- **E2 复现引擎喂法直接 AI Core Error**（窄 block table 下）——引擎里
  4096 列零填充宽表掩盖了越界，这是上游单位混乱的第三重证据。

## 13. 收官状态（2026-10-28 晚）：离线全链路逐位复现，引擎内最后 5% 未闭环

### 已证明（全部离线 bf16/逐位级）

| 环节 | 结论 | 证据 |
|---|---|---|
| 组合数学 = official | **med=0.00000 big=0.0%**（随机/降序索引、引擎尺度 L=8676、stale/0 块表、有限垃圾池） | `tools/smla_final_fit.py` |
| decode 语义 | 行号=索引直用、全部条目、无逐条目因果（值界/位次界/前缀均被否证）；prefill=v9 压缩语义 | 同上矩阵 |
| NaN 机制 | `0 权重 × NaN V = NaN`（我的 softmax-bmm 结构固有；官方 flash 核逐块跳过掩码列）；修法=isfinite 掩码+无效列 V 清零 | v17 nan=47% → v18 nan=0% |
| 图捕获执行 = eager | med=0.000000（含 buffer 刷新后 replay） | `tools/smla_capture_test.py` |
| 垃圾现场 | bt 尾列=stale 真实块 id(852-863)+大量 0（`torch.zeros_like` 初始化）；无 -1 | 引擎 BTDUMP |

### 未闭环

v17 语义代码装进引擎（1mv 双算）实测：非 NaN 部分与 official 0.03% 一致，
但 A=4.3-5.6（应 5.89）——**引擎内我的组合仍与 official 有 ~45% 元素级差异，
而同样代码离线逐位一致**。剩余嫌疑（按序）：

1. **DSA 多流重叠**（`multistream_dsv4_dsa_overlap`）：索引器在辅助流写
   `shared.topk_indices`，我的组合与官方算子对流的依赖记录可能不同，
   replay 时读到不同步的索引
2. **metadata 单次消费**：1mv 模式对 official 的额外调用可能污染
   smla_metadata device task（v19 崩溃 "previous submission has not been
   released" 是同一机制的显形）——即 45% 的测量本身可能有假
3. 静态内核对我算子组合的编译差异（捕获测试已排除普通 npugraph，
   但引擎用的是 npugraph_ex + static_kernel 路径）

**下一轮入口**：在 1mv 里加"输入指纹"（每层调用的 q/kv/bt/idx/cseq 校验和
累计）+ 同图双份 mine（fp32 路径对照），一次区分"输入不同步"vs"数值路径不同"。

### 本轮副产物

- 磁盘抢救：根分区 100% 满 → 清 13 天前旧 results 释放 9.3G（docker 335G
  是他人容器未动）
- 工具沉淀：`smla_semantics_fit.py`（小尺度阶梯）、`smla_final_fit.py`
  （引擎尺度终局拟合）、`smla_capture_test.py`（捕获等价性）、
  `p41_v1x` 全系 installer（v17+v22 存档于 tools/）
- 工程教训 ×2：installer 的每次 replace 必须 assert（本轮 3 次"以为装上了"：
  v16 前缀分支、v18 isfinite、v20 BTDUMP 全部静默失败过）；引擎内测量
  （1mv 双算）本身可能扰动被测系统（metadata 单次消费）
