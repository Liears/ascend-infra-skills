# P13 — DeepInfer 官方 950 栈全图 vs 我们的 910B3 栈：边界地图

> 2026-10-29 勘探。对象：deepseek-ai/DeepInfer（2026-09-30 开源，Apache-2.0，
> 24k stars，vLLM 衍生的 DeepSeek 生产推理框架）+ 6 个 vendor 子模块。
> 本地克隆（含子模块）：/mnt/local/hbw-work/DeepInfer。
> 结论一句话：**官方昇腾栈整体只验证 950 + CANN 9.2 + 毕昇编译器；910B3
> 一行跑不了，但源码/文档/常量可 harvest，已在 P12 落地。**

## 1. 官方栈构成与硬件边界

| 组件 | 内容 | 硬件/工具链边界 | 910B3 可用性 |
|---|---|---|---|
| DeepInfer 引擎 | DSpark MTP varlen（DEEPINFER_ENABLE_MTP_VARLEN）、SWA bounded replay（compact 模式 KEEP_TOKENS=140）、异步调度、EPLB、Attention DP/CP、MoE EP | SM100 / 950，torch 2.13+cpu + torch_npu 2.13rc1，CANN 9.2 | ✗ |
| FlashMLA `csrc/ascend_kernels` | V4.1 稀疏注意力 prefill+decode 的 AscendC 核（.asc，kernel_body.h 1701 行），410/360 TFlops = 95%/83% 峰值 | dav-3510（950），**毕昇 bisheng 编译器**（非 CANN 自带 ccec），SIMD VF/SS buffer/gather2/每核双 AIV | ✗（源码可读，P12 已收割） |
| DeepGEMM-Ascend | DeepGEMM 全 API 移植（BF16/FP8/FP4 GEMM、MQA logits、MegaMoE），近峰值；对昇腾 MAD 原语的轻量抽象（隐藏分形布局） | 950 验证，CANN 9.20 提供 bisheng | ✗（AscendC 写法参考价值高） |
| DeepEP-Ascend | EP all-to-all（FP8 dispatch、延迟 epilogue）、PP/CP/DP Bucket 集合通信、Engram 远程内存（开发中）；EP8 dispatch 373GB/s | **950 专属互联 UBMEM/URMA**，README 明言其他世代未验证 | ✗（Engram = P11 判死方向在 950 上的官方路线） |
| TileLang 0.1.15 ascend 方言 | `tilelang.ascend.language`，38 个生产算子 cuda/asc 双后端（deepinfer/model_executor/layers/tilelang_ops/） | README 标题即 "Ascend 950 Backend"，默认 dav-3510，需 bisheng | ✗ |
| TileKernels | TileLang 之上的算子库（deepseek-ai/TileKernels） | 同上 | ✗ |

编译器事实：950 栈全部走**毕昇（bisheng）**，CANN 9.2 起提供；910B3 的
CANN 9.1.0 只有 ccec/aic。这是比指令集更硬的边界（.asc 源码无法在本机
工具链编译）。

## 2. 与我们三层次工作的对位

**（a）P12 融合 decode 核：参照物已拿到（详见 P12 收获清单）**
- 静态分派（每 q token 一核，min(total_q, GRID)）官方同构
- skip-scale 阈值 6、双 max 滞后结构、uint32 索引技巧、UB 施工图
- 屋顶线定性反转：950@64head 算力 bound vs 910B3@16head 访存 bound
  → 我们的第一优先级是 gather 读效率，不是 cube 流水深度

**（b）次级杠杆（P12 存档的 22.5% 通信 + 19.2% MoE GEMM）**：官方的
DeepEP-Ascend（HcPre 对位物）与 DeepGEMM-Ascend（GMM 链对位物）都不可
移植（互联/编译器双绑定）。910B3 上这两条杠杆仍只能走 vLLM-Ascend 自带
优化 + 我们的算子层。**判决：官方栈不改变次级杠杆的执行路径。**

**（c）语义基准**：FlashMLA 接口文档（-1=唯一无效值、topk_length 左前缀、
attn_sink 使 out×exp(lse)/(exp(lse)+exp(sink))、无上界检查、FP8 KV
528B/token 与 FP4 288B/token 布局）与 P10 三重验证逆向出的
npu_sparse_flash_mla 语义吻合——独立来源交叉印证，P10 规格（及未来 KV
量化杠杆的格式标准）可信度升级。

## 3. 对 ascend-infra-skills 的定位意义

1. **官方栈把 910B 系排除在外**（引擎、注意力核、GEMM、通信、TileLang
   五条线全是 950 起）。910B 系用户跑 V4.1-Flash 的唯一路线仍是
   vLLM-Ascend——本仓库补的正是这条路上的算子/调优层。DeepInfer 的存在
   反向确认了本仓库的生态位。
2. **若团队获得 950 硬件**：部署目标直接切 DeepInfer（Q3 商用 HDK +
   CANN 9.2 + bisheng），本仓库的经验（DSA 语义、metadata 链、调参方法）
   平移，但算子层工作不再需要（官方核 83-95% 峰值）。
3. **工程方法可平移**：DeepInfer 的 DSpark MTP varlen、SWA bounded
   replay（保留后缀覆盖 SWA 窗+回溯长度）是引擎级技术，vLLM-Ascend 侧
   对应能力的差距清单可作为后续 skill 的素材。

## 4. 死路登记（防重复勘探）

- TileLang ascend 在 910B3 上试跑：✗ 已核实 950+bisheng 专属，不装
- FlashMLA .asc 直接编译：✗ bisheng/dav-3510 双绑定
- DeepEP-Ascend UBMEM/URMA：✗ 950 互联硬件
- DeepInfer 引擎本体：✗ torch 2.13/torch_npu 2.13rc1 + 950 驱动
