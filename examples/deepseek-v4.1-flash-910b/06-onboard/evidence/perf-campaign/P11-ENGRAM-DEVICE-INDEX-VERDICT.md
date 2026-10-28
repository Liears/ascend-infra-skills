# P11 — Engram 设备直读：910B3 硬件判死（2026-10-28）

> 承接 P10 §14 后的最后一根"廉价杠杆"：驱动升级解锁
> `V41_ENGRAM_DEVICE_INDEX=1`（Engram 查找设备直读 + 省 25.75 GB/rank/层 DRAM）。
> 本轮结论：**该路径在 910B3 上被硬件拓扑判死，与驱动版本无关。**

## 0. 判决

| 项 | 值 |
|---|---|
| 本机 NPU | Atlas 910B3，PCI `19e5:d802`（CPU↔NPU 走 **PCIe**） |
| 驱动/固件 | 25.5.2 / 7.8.0.7.220 |
| `host_mem_pool`（/proc/svm/dev{0..7}/feature） | **全 0** |
| 判决依据 | `engram_device_index.py::host_mem_pool_supported()` 的 2026-09-20 A2 真机定论 |

## 1. 机制（代码内真机实测注释，非公开文档）

`aclrtHostRegister(MAPPED)` 整表注册（206 GiB × 8 rank）走哪条路由
**驱动 probe 时按 PCI device id 定死**的 `host_mem_pool` 决定：

- **A3（910C，`d803`，HCCS 互联）**：`host_mem_pool=1` → 池化快路径 →
  8 rank × 206 GiB 注册 **133 秒**完成 → `V41_ENGRAM_DEVICE_INDEX=1` 可用
- **A2（910B3，`d802`，PCIe 互联）**：`host_mem_pool=0` → 逐页建元数据
  慢路径（每 4 KiB 页 64 B）→ 单次 `devmm_kvzalloc` 需 ~2.06 GiB 连续内核
  内存 → 实测 **17 分钟后 `ret=207001` 失败**（宿主 MemAvailable 仍有 703 GiB）

三个不可绕过：procfs 写入是 no-op；大页无效（页数按 VA÷4 KiB 硬编码）；
4 KiB 小探针通过≠整表可注册（auto 模式为此先查 flag）。

## 2. 为什么驱动升级救不了

池化路径依赖 **HCCS 拓扑的共享虚拟内存硬件能力**；PCIe 上每页必须做
DMA 映射——这恰恰就是"慢路径"本身。flag 按 PCI id 在 probe 期定死，
不是版本开关。公开渠道（hiascend/support/社区）检索不到任何
"PCIe A2 开启 host_mem_pool"的先例。驱动 25.5.2 无已知性能相关缺陷。

## 3. 对战役的意义

- **910B3（本机）**：Engram 查找维持宿主路径（`engram_hbm.py` 的 P18
  优化版 + JIT plan kernel 已是宿主路径的最优形态）。此杠杆关闭。
- **910C/A3 机器**：`V41_ENGRAM_DEVICE_INDEX=1` 是现成的、已实现并
  探针自适配的开关——换硬件即得，无需改代码。
- 剩余开放路径：target 路径融合算子（AscendC，语义规格见 P10 §12-13）。

## 4. 佐证

```
$ cat /proc/svm/dev0/feature/host_mem_pool
0
$ lspci -nn | grep 19e5:
01:00.0 Processing accelerators [1200]: Huawei ... [19e5:d802] (rev 20)
```

检索（2026-10-28）：[昇腾驱动固件安装指南](https://www.hiascend.com/developer/blog/details/02176215666023825269)、
[GPU 进阶笔记：华为昇腾910B](https://arthurchiao.art/blog/gpu-advanced-notes-2-zh)、
[Huawei Ascend 910 驱动升级](https://support.huawei.com/enterprise/en/doc/EDOC1100149962/76a340e3/upgrading-the-ascend-910-ai-processor-driver)
——均未涉及该 flag 的开启方法。
