# Lessons Learned — the trap catalog

Ten traps from a real campaign, each with its symptom and its tell. Written
to be recognized, not re-hit.

---

## 1. Stray `self` in a module-level function

**Symptom**: `NameError: name 'self' is not defined` at a call site whose
source looks correct — the lambda/caller line is blamed, not the callee.
**Cause**: a helper extracted from a method still references `self`; the
line often executes unconditionally, so the path dies on first use.
**Tell**: the traceback's last frame is a *different* file line than any
`self` you can see; grep the callee function body for `\bself\b`.
**Worked example**: this single bug is why DRAFT_GRAPH had been marked
"unverified" — fixing 3 lines unlocked the campaign's 6.3×.

## 2. Patch-mode layering reintroduces fixed bugs

**Symptom**: a bug you fixed and baked into the image reappears in one
launch mode but not another.
**Cause**: baked images carry fixes in-place; mount-mode launch copies the
repo's *unpatched* source over the live tree at start, or applies a git
patch series whose ordering overwrites your fix.
**Tell**: md5 the live file after launch (`9a36e709...` vs your fixed hash);
know which mode the launcher used.
**Rule**: fix the *source of the copy*, both in the image and in the repo
the mount flow reads — or bake a new image.

## 3. External kill vs crash — read the shutdown logs

**Symptom**: engine "died" with no Worker traceback.
**Tell**: graceful shutdown lines (`[shutdown] Process manager: send
sigterm`, `Application shutdown complete`) = something sent SIGTERM. A
native crash does NOT produce them. On a shared box, suspect the humans
first when the logs are graceful. A vanished container (`docker rm`) is
proof of external action; a crashed engine leaves the container behind.

## 4. Cold-service probes lie

**Symptom**: first probe of a fresh service shows TTFT 15 s where warm runs
show 0.94 s.
**Cause**: the first requests absorb graph capture / JIT warmup.
**Rule**: discard ~5 warmup requests; never compare a warm baseline to a
cold trial.

## 5. "Garbage output" may be a prompt-format artifact

**Symptom**: raw `/v1/completions` on a bare prompt returns repetition
("1+1=3
1+1=3").
**Cause**: no chat template / official encoder — completion-mode prompts are
out-of-distribution for chat models; some prompts also EOS instantly.
**Tell**: run the accuracy gate with the official encoder before crying
corruption. In the example, GSM8K-100 = 100/100 on the same service the raw
probe called "broken". Also the reverse class: a config that "runs fine"
(A≈1.07, no errors) can be silently wrong — the gate is the only detector.

## 6. Non-standard layer structure breaks fused-kernel assumptions

**Symptom**: a validated CANN fused op (add+norm+quant) has no drop-in site.
**Cause**: models with mHC (hyper-connections) or other pre/post-mixing put
the residual add inside the mixing matrices — not adjacent to the norm.
**Tell**: read the model's layer forward before planning wiring; the
standard `Add → norm → attn` picture does not hold everywhere.

## 7. `dir(torch_npu)` lists APIs that fail lazy import

**Symptom**: a fused op appears in the inventory, then `AttributeError` on
use.
**Cause**: the dir listing reflects a registry; the build may not ship the
symbol.
**Rule**: probe with `getattr(torch_npu, name)` before designing around it;
check sibling variants (a *static*-scale version existing does not imply the
*dynamic* one does).

## 8. Ascend UB limits break naive row kernels

**Symptom**: Triton compile error `ub overflow, requires N bits while
1572864 bits available`.
**Cause**: a full-row BLOCK (e.g. next_pow2(5120)=8192) × fp32 × several
buffers exceeds the ~192 KB unified buffer.
**Fix**: chunked loops (BLOCK 512–4096 all perform similarly at small
shapes; the kernel is launch-bound there anyway).

## 9. Eager micro-benchmarks overstate in-graph gains

**Symptom**: 4.97× in a micro-benchmark, <1% end-to-end after wiring
estimates.
**Cause**: eager numbers include launch overhead that graphs amortize; at
small batch the tensors are tens of KB and traffic, not launches, dominates.
**Rule**: run `fusion_headroom.py` before wiring — the traffic model, not
the eager ratio, decides.

## 10. Your pinned base ages fast

**Symptom**: a local bug you hand-patched gets properly fixed upstream
weeks later; perf PRs accumulate against your frozen base.
**Cause**: 5 weeks of drift in the example = 171 model-relevant commits.
**Rule**: P5 — quantify drift (`upstream_drift.py`), price the rebase by
running upstream natively, rebase only what the measured gap justifies.
