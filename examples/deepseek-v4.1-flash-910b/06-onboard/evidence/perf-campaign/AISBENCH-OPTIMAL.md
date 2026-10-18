# AISBench — 最优方案官方工具评测（2026-10-18）

工具：ais_bench 3.1.0（官方 benchmark 框架，AISBench/benchmark@master，nltk 补丁）
环境：dsv41-a2 容器（E2 终验配置栈，DRAFT_GRAPH=1 + FUSED_MC2=1，:8100）
配置映射：input_len 8000 / output_len 256（ignore_eos）/ data_num 3 / concurrency 1（batch_size），temp=0 非流式

| 指标 | 平均 | 最小 | 最大 | 中位 | P99 |
|---|---:|---:|---:|---:|---:|
| E2EL (ms) | 3,499.9 | 3,468.4 | 3,524.0 | 3,507.2 | 3,523.7 |
| OutputTokenThroughput (tok/s) | 73.15 | 72.64 | 73.81 | 72.99 | 73.79 |

汇总：Input 24,000 tok · Output 768 tok · 3/3 成功 · 并发 0.9999 · 时长 10.5 s
口径注意：73.15 为 E2E（含 8K prefill）；纯 decode ≈116 tok/s（与探针带 113-134 一致）。

## 原始产物（已固化 /mnt/local/hbw-work/aisbench/）
- `aisbench-run-console.log` —— 完整控制台输出
- `20261018_144757/performances/dsv41-optimal/synthetic-8k-256.csv|_details.jsonl|_plot.html` —— 汇总/逐请求明细/图
- `20261018_144757/configs/*.py` —— 本次运行冻结配置
- `vllm_api_dsv41.py / synthetic_8k_256.py` —— 模型与数据集配置
