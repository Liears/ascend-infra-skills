# E3 — dynamic K (SP_SCHEDULE): dead end on this stack, documented

Goal: raise single-stream speculative K from 5 to 7 (A=5.69 was near the
K=5 ceiling of 6).

Three attempts, three distinct walls:

1. **baked image**: SP_SCHEDULE requires PATCH_MODE=mount — the dynamic-spec
   patch is not baked into v9/v10.
2. **mount mode**: dynamic SD downgrades cudagraph_mode to PIECEWISE; V4.1's
   cache init only supports eager / FULL_DECODE_ONLY → dies at model
   construction (~2 min in). Only escape hatch:
   `V41_CED_DYNAMIC_SPEC_FULL_GRAPHS=1` (explicitly risky per the script).
3. **mount + escape hatch**: survives construction and compiles (120/120
   static kernels), but the first request crashes with the SAME
   `NameError: self` class as E1's bug — the mount-mode patch layering
   (repo files copied in + git-apply series + draft install) produces an
   inconsistent `dsa_v1.py`. The repo's own mount source is clean and the
   v10 image is clean; the inconsistency comes from the layering order,
   not from any single file.

Verdict: abandoned. Ceiling ~10–20% does not justify untangling the
mount-mode patch ordering. If pursued later: fix the layering order so the
DRAFT_GRAPH install runs LAST, or bake the dynamic-spec patch into a v11
image.
