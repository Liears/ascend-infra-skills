# Runbook — bring DeepSeek-V4.1-Flash up on 8×910B3 (A2)

The second-level validation: actually serving the model, so the skills can be
fed a real trace instead of a synthetic one.

This runbook is written against the public port
`chiro2001/deepseek-v4.1-flash-ascend910B` (its `scripts/serve_a2.sh`), with
every variable name and default read from that script — not from a README.
Where a value is a default from the script, it is labelled as such.

> **Status:** prepared, not yet executed end-to-end. The weight download
> (304 GB) and base image pull are the gating steps.

---

## 0. Prerequisites

| Item | Requirement | Where it is |
|---|---|---|
| Hardware | 8 × 910B3 (A2), 64 GB HBM each | this node |
| Base image | `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler` | pulled separately |
| Weights | `chiro2001/DeepSeek-V4.1-Flash-w4a8-Ascend`, ~304 GB | ModelScope |
| Patched image | built by `scripts/build_image.sh` → `dsv41-a2:v9` | built locally |

**Engram reorganization is mandatory.** The downloaded weights cannot serve
directly: an Engram reassembly step must run first (the script ships with the
weights). Skipping it fails at load.

---

## 1. Get the weights

```bash
modelscope download chiro2001/DeepSeek-V4.1-Flash-w4a8-Ascend \
    --local_dir /mnt/local/hbw-work/dsv41-w4a8
```

304 GB. Then run the Engram reorganization script that came with the weights.

**Disk:** `/mnt/local` has ~4 TB free, which is the right place. Do not use
`/` (438 GB, 80% full).

---

## 2. Build the patched image

```bash
# BASE_IMAGE and IMAGE_TAG defaults come from build_image.sh
bash scripts/build_image.sh
```

It probes the image layout (it does not hardcode paths), bakes in the verified
patches, keeps `.a2orig` backups of every file it touches, and verifies md5 +
`py_compile` per file. Output: `dsv41-a2:v9`.

**The tag must match in three places** — `build_image.sh`, `serve_a2.sh`,
`run_test.sh`. The package's own self-check refuses to run if they disagree;
that check caught a real mismatch once.

---

## 3. Pre-flight checks (do not skip)

```bash
bash tools/check_checksums.sh     # ~10 s, no container boot
bash tools/list_chips.sh          # which devices are free, and whose they are
```

`list_chips.sh` matters on a shared box: confirm all 8 devices are free before
claiming them.

---

## 4. Serve

```bash
MODEL=/mnt/local/hbw-work/dsv41-w4a8 \
DEVS="0 1 2 3 4 5 6 7" \
bash scripts/serve_a2.sh
```

Variables read from `serve_a2.sh` (defaults shown):

| Var | Default | Meaning |
|---|---|---|
| `MODEL` | *(required)* | weight directory |
| `IMAGE` | `dsv41-a2:v9` | patched image |
| `NAME` | `dsv41-a2` | container name |
| `PORT` | `8100` | service port |
| `TP` | `8` | tensor-parallel **devices** |
| `DEVS` | `0 1 2 3 4 5 6 7` | device list |
| `SERVED_NAME` | `deepseek-v41` | served model name |
| `GPU_UTIL` | `0.92` | HBM fraction. **0.94 leaves only 6.11 GiB and makes 8K prefill 8.0–8.6 s vs 1.14 s at 0.92** — this knob is not cosmetic |
| `MAX_LEN` | `1048576` | max context (1M) |
| `MAX_SEQS` | `32` | affects graph-capture bucket count |
| `PREFIX` | on (`NO_PREFIX=1` to disable) | prefix caching |
| `DRAFT_GRAPH` | `0` | speculative draft graph; **default off deliberately** — turning it on without understanding yields a bad accept-length with no error |

### Two traps the script already defends against

1. **Proxy hijacking localhost.** A proxy intercepting `127.0.0.1` returns an
   error page for `/health`, so a healthy server looks like a startup hang.
   The script sets `no_proxy` for localhost unless `KEEP_PROXY_FOR_LOCALHOST=1`.
   Verify with `curl -s --noproxy '*' http://127.0.0.1:8100/health`.

2. **CPU-binding hang.** `CPU_BIND=1` on a box whose target NUMA node is full
   hangs in `migratepages` indefinitely — not slow, non-converging. If a
   multi-card launch never becomes ready, retry with `CPU_BIND=0`.

---

## 5. Verify it is really up

```bash
curl -s --noproxy '*' http://127.0.0.1:8100/health
curl -s --noproxy '*' http://127.0.0.1:8100/v1/models
```

Then check the four things that matter, each with its artifact:

| Check | Command / source | What proves it |
|---|---|---|
| KV capacity | startup log | `GPU KV cache size: N tokens` line |
| Engram fast path | profiler | the device-side lookup op in `kernel_details.csv` |
| Spec-decode engaged | response metadata | accept length reported, not inferred |
| No silent fallback | profiler | the fused ops are present |

---

## 6. Capture the trace for the skills

Once serving, capture with `torch_npu.profiler` at **Level1 + PipeUtilization**
(see `skills/npu-profiler-analysis/references/ascend-trace-format.md` for the
exact config), then feed the output to:

```bash
python3 skills/npu-profiler-analysis/scripts/analyze_profile.py <prof_dir>
python3 skills/llm-serving-capacity-planner/scripts/capacity_analyzer.py <server.log>
```

This closes the loop: the skills that were validated on synthetic and small
traces now run on a real 1M-context MoE service.

---

## 7. Roll back

The container is disposable. Stop it and the box returns to idle:

```bash
docker rm -f dsv41-a2
```

The patched image and the weights stay on disk for the next run.
