# Dossiers

Pre-built model PR-history dossiers. Each is a snapshot; every one records the
repo HEAD it was mined from, so its staleness is checkable.

| Dossier | Repo | Keyword | Built from | Commits |
|---|---|---|---|---|
| `vllm-ascend_deepseek.md` | vllm-project/vllm-ascend | `deepseek` | `2b8da69474a7` (2026-10-14) | 15 |

## Refreshing

```bash
# The clone must have history (not --depth 1)
git clone --depth 200 https://github.com/vllm-project/vllm-ascend.git /tmp/vllm-ascend

python3 ../scripts/mine_pr_history.py \
    --repo /tmp/vllm-ascend \
    --model deepseek \
    --limit 120 \
    --out vllm-ascend_deepseek.md
```

## What the DeepSeek dossier already shows

Two findings worth noting, both from the 2026-10-14 snapshot:

- **#18281 — "Support multimodal inputs for DeepSeek V4.1"** is a merged
  bugfix in vllm-ascend. The "V4.1 not supported on Ascend" gap is closing
  upstream; re-check the support matrix before assuming a from-scratch port is
  needed.
- **#18628 — "Request huge pages before Engram host registration"** confirms
  Engram's host-resident tables are the area where NPU memory management does
  real work, and that it has already produced a bugfix.

This is the value of the dossier: it turns "V4.1 on NPU is unsupported" into
"here are the specific commits where it is being supported, and here is the
file each touched".
