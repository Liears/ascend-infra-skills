# The Day-0 Bundle Contract

What a public "we support model X on Ascend" claim must contain, and what it
must not.

---

## 1. Required artifacts

A Day-0 bundle is falsifiable only if it carries all of these. `check_day0_bundle.py`
enforces the list.

| Artifact | Why it is required |
|---|---|
| Model config + weight source (with version/checksum) | without it, "we ran model X" is unverifiable |
| Exact launch command (all flags) | the flags *are* the configuration; a default hides the real setting |
| Stack record: CANN, driver, torch_npu, framework commit | a result belongs to a stack, not to a chip |
| Accuracy result on the real path | proves the model computes, not just loads |
| Fast-path evidence (profiler showing the fused op) | proves it runs *as claimed*, not as a slow fallback |
| Memory figure: weights + KV budget at a stated context | proves it fits, and says where the limit is |
| Known limitations | a bundle with no stated limits has not been tested enough |

## 2. The forbidden contents

| Must not contain | Why |
|---|---|
| Private fork names, internal branch names | leaks development structure; also useless to the reader |
| Internal hostnames, paths, tickets | leaks infrastructure |
| Corpus with unclear redistribution rights | a copyright problem you inherit |
| Numbers from a different chip/CANN stack, unlabelled | misleads; the rules forbid it |
| "Works" with no artifact | not a claim, a hope |

If development happened in a private tree, the public bundle is a **clean-room
reconstruction**: same technique, freshly applied to the public tree, with the
private commits left behind. See `ascend-open-source-finder`'s audit for how to
tell whether an existing bundle did this.

## 3. Evidence dated, not eternal

Every artifact carries its capture date and stack version. A Day-0 bundle from
CANN 9.0 does not validate a CANN 9.2 deployment. When the stack moves,
re-validate the gates that depend on kernels (accuracy, fast-path, memory);
loading and shape checks rarely change, kernel behavior often does.

## 4. Gate definitions (copy into the plan)

```
G1 LOAD    weights load; param count == config
           artifact: load log line with the parameter count
G2 ACCURACY <task> >= <expected> on the real path, fast paths on
           artifact: run output + the command
G3 FASTPATH profiler shows <fused op> executed
           artifact: kernel_details.csv row for the fused op
G4 MEMORY  fits <chip> at ctx=<N>, KV budget=<M> tokens
           artifact: startup log + kv capacity line
G5 MODE    runs in <eager|graph|PD> as claimed
           artifact: a run in each claimed mode
```

Each gate is advisory until its artifact exists. A plan with gates defined but
no artifacts is a plan, which is fine — it just cannot be called done.
