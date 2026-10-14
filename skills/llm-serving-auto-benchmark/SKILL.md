---
name: llm-serving-auto-benchmark
description: Compare LLM serving frameworks on Ascend hardware (vLLM-Ascend, MindIE, SGLang-Ascend) under the same model, workload and SLA, and refuse comparisons that are not valid. Use when the user wants throughput/latency numbers for a model on 910B/910C, asks which framework is faster on NPU, needs a benchmark plan, or has result files to compare. Records every field the evidence rules require.
---

# Ascend LLM Serving Auto-Benchmark

Produce comparable serving numbers across Ascend frameworks, and refuse the
comparisons that are not comparable.

## Why "refuse" is the core feature

On GPU, a careless comparison is a bad number. On Ascend it is worse, because
three extra axes silently differ:

1. **cards vs devices** — 8×910C is 16 devices. A row saying "8 NPU" is ambiguous.
2. **the CANN/driver pair** — a kernel set changed between CANN versions changes
   throughput. Two rows on different CANN are two different machines.
3. **tok/s convention on spec-decode models** — `A × 1000 / ms_per_step` and
   `gen_tokens / wall_time` are both called "tok/s" and differ by the acceptance
   length (often 2–3×).

This skill's comparison step will decline to rank rows that differ on these, and
will mark any row with a missing required field as untrusted and exclude it from
the ranking rather than let it win.

## Workflow

### 1. Pin the stack

```bash
npu-smi info                      # chip, HBM, driver-visible free memory
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg   # CANN component versions
npu-smi info -t board -i 0        # firmware
```

Record all three. A benchmark row without them is not admissible.

### 2. Launch the server

On Ascend the launcher is `vllm-ascend`'s own wrapper rather than bare `vllm`.
Confirm which entry point the image provides before writing a plan:

- `vllm serve <model> ...` (works when vllm-ascend is installed and on `PATH`)
- `python -m vllm.entrypoints.openai.api_server ...`
- the repo's own `serve_*.sh`, if benchmarking a patched deployment

### 3. Load it

vllm-ascend ships two drivers:

| Tool | Use |
|---|---|
| `vllm bench serve` (a.k.a. `tools/vllm_bench.py`) | standard path; `--backend openai-chat`, `--metric-percentiles 50,90,99` |
| `tools/aisbench.py` | Huawei's AISBench wrapper; pulls its own dataset |

Match `--random-input-len` / `--random-output-len` / `--max-concurrency` /
`--num-prompts` across frameworks or the rows are not comparable.

### 4. Normalize and compare

```bash
python3 scripts/compare_results.py --template > rows.json   # start here
# ... fill one object per run ...
python3 scripts/compare_results.py rows.json
```

The script:
- validates every required field,
- rejects ambiguous device/card and missing tok/s-basis rows,
- groups rows by match key, and
- ranks **only trusted rows** within a group.

Exit code is 1 when any row is untrusted — usable in CI to block a report built
on an incomplete row.

## The match keys

Two rows are comparable only if these are identical:

`model`, `chip`, `devices`, `toolchain`, `in_len`, `out_len`, `concurrency`,
`prefix_cache`, `graph_mode`, `hccl`

These may differ — that is the comparison:

`framework`, `quant`, `kv_cache_memory_bytes`, `spec_decode`

## Reading a result honestly

| Situation | What to say |
|---|---|
| Higher tok/s but SLA missed | not a win; the two are not the same measurement |
| Two rows on different CANN | two machines; report both stacks or rerun |
| Spec-decode on | state A and the tok/s basis, or the number means nothing |
| One framework at conc=1, other at conc=8 | not a comparison; regroup |
| `graph_mode` differs | that is often the real variable — say so explicitly |

## Scripts

| Script | Purpose |
|---|---|
| `scripts/compare_results.py` | Validate + group + rank benchmark rows; refuses invalid comparisons |
| `scripts/compare_results.py --template` | Print a starter `rows.json` |

## References

- `references/ascend-benchmark-tooling.md` — the exact commands and flags for vllm-ascend and AISBench, verified against the installed source
- `../../docs/EVIDENCE-RULES.md` — the required fields and their meaning
- `../../references/ascend-hardware-specs.json` — chip specs and the card/device rule
