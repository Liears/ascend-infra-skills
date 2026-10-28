"""M2 冒烟: SparseFlashMlaDecode(改名副本) vs 原版 —— 应逐位一致 + 同口径计时。"""
import time
import torch, torch_npu
import cann_ops_transformer
from cann_ops_transformer.ops import sparse_flash_mla as _smla_mod  # noqa
from torch.utils.cpp_extension import load

PIPCOT = '/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/cann_ops_transformer'
VENDOR = '/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer'
ext = load(
    name='smla_decode_wrap',
    sources=['/tmp/smla_decode_wrap.cpp'],
    extra_include_paths=[
        PIPCOT + '/common/inc',
        VENDOR + '/op_api/include',
        '/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/include',
        '/usr/local/Ascend/cann-9.1.0/include',
        '/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/include/third_party/acl/inc',
    ],
    extra_ldflags=['-L' + VENDOR + '/op_api/lib', '-lcust_opapi',
                   '-L/usr/local/Ascend/cann-9.1.0/aarch64-linux/devlib', '-lascendcl',
                   '-L/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/lib',
                   '-ltorch_npu'],
    verbose=False)

OP = torch.ops.cann_ops_transformer.sparse_flash_mla
MD = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata
dev = "npu:0"
torch.manual_seed(11)
D, BLOCK, RATIO, K, W, SEQ = 512, 128, 2, 512, 128, 8676
H, NB_O, NB_C, BTW = 16, 80, 900, 256
SCALE = D ** -0.5
SINKS = (torch.randn(H, device=dev) * 0.1).float()


def call_orig(q, ori_kv, cmp_kv, idx, obt, cbt, cu_q, seq, resid, md):
    return OP(q, ori_kv=ori_kv, cmp_kv=cmp_kv, cmp_sparse_indices=idx,
              ori_block_table=obt, cmp_block_table=cbt, cu_seqlens_q=cu_q,
              seqused_ori_kv=seq, seqused_cmp_kv=seq, cmp_residual_kv=resid,
              sinks=SINKS, metadata=md, softmax_scale=SCALE, cmp_ratio=RATIO,
              ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=W - 1, ori_win_right=0,
              layout_q="TND", layout_kv="PA_BBND", topk_value_mode=1,
              return_softmax_lse=False)[0]


def call_decode(q, ori_kv, cmp_kv, idx, obt, cbt, cu_q, seq, resid, md):
    return ext.sparse_flash_mla_decode(
        q, ori_kv, cmp_kv, None, idx, obt, cbt, cu_q, None, None, None, seq, seq,
        resid, None, None, SINKS, md, SCALE, RATIO, 4, 3, W - 1, 0,
        "TND", "PA_BBND", 1, False)[0]


def bench_graph(fn, n_inner=30, n_outer=20):
    g = torch.npu.NPUGraph()
    s = torch.npu.Stream(); s.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(s):
        for _ in range(3):
            fn()
    torch.npu.current_stream().wait_stream(s)
    with torch.npu.graph(g):
        for _ in range(n_inner):
            fn()
    g.replay(); torch.npu.synchronize()
    ts = []
    for _ in range(n_outer):
        t0 = time.perf_counter()
        g.replay(); torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) / n_inner * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


for T, R in ((6, 1), (24, 4)):
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev) * 0.3
    ori_kv = torch.randn(NB_O, BLOCK, 1, D, dtype=torch.bfloat16, device=dev)
    cmp_kv = torch.randn(NB_C, BLOCK, 1, D, dtype=torch.bfloat16, device=dev)
    obt = torch.zeros(R, BTW, dtype=torch.int32, device=dev)
    cbt = torch.zeros(R, BTW, dtype=torch.int32, device=dev)
    for r in range(R):
        obt[r, :68] = torch.arange(820 + r * 68, 820 + r * 68 + 68) % NB_O
        cbt[r, :34] = torch.arange(820 + r * 34, 820 + r * 34 + 34) % NB_C
        for c in range(34, 46):
            cbt[r, c] = (818 + (c - 34) + r) % NB_C
    cu_q = torch.arange(0, R * (T // R) + 1, T // R, dtype=torch.int32, device=dev)
    seq = torch.full((R,), SEQ, dtype=torch.int32, device=dev)
    resid = torch.zeros(R, dtype=torch.int32, device=dev)
    idx = torch.randint(0, SEQ, (T, 1, K), device=dev, dtype=torch.int32)
    md = MD(H, 1, D, cu_seqlens_q=cu_q, seqused_ori_kv=seq, seqused_cmp_kv=seq,
            cmp_residual_kv=resid, batch_size=R, max_seqlen_q=T // R,
            max_seqlen_ori_kv=SEQ, max_seqlen_cmp_kv=SEQ,
            ori_topk=0, cmp_topk=K, cmp_ratio=RATIO, ori_mask_mode=4,
            cmp_mask_mode=3, ori_win_left=W - 1, ori_win_right=0,
            layout_q="TND", layout_kv="PA_BBND", has_ori_kv=True, has_cmp_kv=True)
    args = (q, ori_kv, cmp_kv, idx, obt, cbt, cu_q, seq, resid, md)
    try:
        o = call_orig(*args)
        m = call_decode(*args)
        torch.npu.synchronize()
    except Exception as e:
        print(f"T={T} DECODE-CALL-ERR: {type(e).__name__}: {str(e)[:200]}", flush=True)
        continue
    d = (m.float() - o.float()).abs()
    print(f"T={T:2d} R={R}: bit-diff max={d.max().item():.6f} "
          f"nonzero={int((d > 0).sum())} | orig={bench_graph(lambda: call_orig(*args)):6.1f}us "
          f"decode-copy={bench_graph(lambda: call_decode(*args)):6.1f}us", flush=True)
print("SMOKE-DONE", flush=True)
