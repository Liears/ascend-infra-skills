# A2 验收报告 — a2_20261015_023728

- 时间：2026-10-15 03:01:20
- 模型：`/mnt/local/hbw-work/dsv41-w4a8`
- local-owner：**fast**
- 镜像指纹：
  - base_image=quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler
  - ascend_pkg=/vllm-workspace/vllm-ascend/vllm_ascend
  - vllm_root=/vllm-workspace/vllm
  - built_at=2026-10-14T17:07:01+00:00
  - python=Python 3.12.13
  - pgo_python_md5=
  - pgo_libpython_md5=
  - e805b6e83c6cdde21820f03dfcbe424f  models/deepseek_v41/engram_hbm.py
  - 56dea733d9df82cd773d98b4ba1a7e10  models/deepseek_v41/engram_hash.py
  - 6668d3fe3c6333b47e9dde3df7cfe03a  models/deepseek_v41/engram_jit_kernel.py
  - 0be62d7775374b0167a54f5b393a65ac  models/deepseek_v41/engram_plan_kernel.py
  - 41e7f012c596cc7539162e5f7ed8475d  models/deepseek_v41/engram_device_index.py
  - bee1bdb20491ce0dfb2df1c3be1bf1f4  models/deepseek_v41/engram_graph.py
  - 146010cac42261e9dc4380699e156252  models/deepseek_v41/engram_gate.py
  - f6009ff0eba7505285c7667d837d5c63  models/deepseek_v41/model.py
  - f61f242df4f060106ce1bf4500ff5844  models/deepseek_v41/indexer.py
  - 6cccd4259bd65c907ef9d9dd42a83dca  ascend_forward_context.py
  - 9a36e709b0937589eab05c5316a62591  attention/dsa_v1.py
  - a91fbc48350530d987ef3bb1ff15f1cb  ops/fused_moe/token_dispatcher.py
  - 6a19890850ac7cb41c535b070c2dfbf6  ops/rope_dsv4.py
  - 3eda7c063816959b0b67cd16938a0c09  worker/block_table.py

## 1. 关键指标

| 指标 | 值 | 门槛 | 判定 |
|---|---|---|---|
| GPU KV cache tokens | 2942071 | > 2800000（默认 0.92 ⇒ 约 2.82M；0.94 ⇒ 3.09M） | ✅ PASS |
| 8k ms/step | 62.32 | — | — |
| 8k 接受长度 A | 1.702 | — | — |
| 8k decode tok/s | 27.2 | — | — |

## 2. 服务日志关键行

```
Available KV cache memory: 13.60 GiB
GPU KV cache size: 2,942,071 tokens
Maximum concurrency for 1
```

## 3. 已知预期（来自 A3 实测，A2 需自行比对）

| 项 | A3(A3-node2 front8, 910C) | 说明 |
|---|---|---|
| 8K ms/step | 35.8 | static_kernel=1 + mtpq + gate(CHUNK=0) + local-owner |
| 128K ms/step | 36.1–37.1 | 同上 |
| 128K A | 2.87–2.96 | Engram-on |
| 128K tok/s | 78–82 | 单流 |
| KV tokens | 3,557,471 | util=0.94 |
| Vision | 23/23 | qrot 修复后的 vision 分片 |
| GSM8K-200 | 197–199 | chat 模式 |

> A2 是 8×910B3，设备侧更慢但 verify 更快（A2 报告：无投机 18.9 ms/round）；
> **不要期待与 A3 完全一致**，重点是看趋势与门槛。

## 4. 原始产物

    total 376
    drwxr-xr-x 2 root root   4096 Oct 15 03:01 .
    drwxr-xr-x 4 root root   4096 Oct 15 02:37 ..
    -rw-r--r-- 1 root root   1171 Oct 15 02:59 BUILD_INFO.txt
    -rw-r--r-- 1 root root   4382 Oct 15 03:01 driver.log
    -rw-r--r-- 1 root root    169 Oct 15 03:00 env.txt
    -rwxr-xr-x 1 root root   3477 Oct 15 02:40 inner.sh
    -rw-r--r-- 1 root root      5 Oct 15 03:00 local_owner_effective.txt
    -rw-r--r-- 1 root root   2730 Oct 15 03:01 p42_t4_quote_32768_quote_32k.jsonl
    -rw-r--r-- 1 root root   2718 Oct 15 03:00 p42_t4_quote_8192_quote_8k.jsonl
    -rw-r--r-- 1 root root   2479 Oct 15 03:01 REPORT.md
    -rw-r--r-- 1 root root   1149 Oct 15 02:40 serve_cmd.txt
    -rw-r--r-- 1 root root 328538 Oct 15 03:01 serve.log
