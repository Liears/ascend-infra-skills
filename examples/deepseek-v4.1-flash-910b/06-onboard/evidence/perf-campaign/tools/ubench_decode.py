"""M1 基线 + M3 对决台: 官方 npu_sparse_flash_mla 在 decode 精确形状的图内时间。
用法: python3 ubench.py [official|kernel_path]
"""
import sys, time
import torch, torch_npu
import cann_ops_transformer
from cann_ops_transformer.ops import sparse_flash_mla as _smla_mod  # noqa
OP = torch.ops.cann_ops_transformer.sparse_flash_mla
MD = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata

dev = "npu:0"
torch.manual_seed(11)
D, BLOCK, RATIO, K, W = 512, 128, 2, 512, 128
import os
H = int(os.environ.get("UB_H", "16"))   # 128 heads / TP8
SEQ = 8676                  # 引擎实测 decode 上下文
NB_O, NB_C, BTW = 80, 900, 256
sinks = (torch.randn(H, device=dev) * 0.1).float()
scale = D ** -0.5
resid = None  # per-case (须为 [R])


def make_case(T, R):
    """R 请求 × 每请求 T//R 个 draft token, 引擎 dspark 形态。"""
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev) * 0.3
    ori_kv = torch.randn(NB_O, BLOCK, 1, D, dtype=torch.bfloat16, device=dev)
    cmp_kv = torch.randn(NB_C, BLOCK, 1, D, dtype=torch.bfloat16, device=dev)
    ori_bt = torch.zeros(R, BTW, dtype=torch.int32, device=dev)
    cmp_bt = torch.zeros(R, BTW, dtype=torch.int32, device=dev)
    for r in range(R):
        ori_bt[r, :68] = torch.arange(820 + r * 68, 820 + r * 68 + 68) % NB_O
        cmp_bt[r, :34] = torch.arange(820 + r * 34, 820 + r * 34 + 34) % NB_C
        for c in range(34, 46):
            cmp_bt[r, c] = (818 + (c - 34) + r) % NB_C
    cu_q = torch.arange(0, R * (T // R) + 1, T // R, dtype=torch.int32, device=dev)
    seq_o = torch.full((R,), SEQ, dtype=torch.int32, device=dev)
    seq_c = torch.full((R,), SEQ, dtype=torch.int32, device=dev)   # 引擎: 原始单位
    idx = torch.randint(0, SEQ, (T, 1, K), device=dev, dtype=torch.int32)
    resid = torch.zeros(R, dtype=torch.int32, device=dev)
    md = MD(H, 1, D, cu_seqlens_q=cu_q, seqused_ori_kv=seq_o, seqused_cmp_kv=seq_c,
            cmp_residual_kv=resid, batch_size=R, max_seqlen_q=T // R,
            max_seqlen_ori_kv=SEQ, max_seqlen_cmp_kv=SEQ,
            ori_topk=0, cmp_topk=K, cmp_ratio=RATIO,
            ori_mask_mode=4, cmp_mask_mode=3,
            ori_win_left=W - 1, ori_win_right=0,
            layout_q="TND", layout_kv="PA_BBND", has_ori_kv=True, has_cmp_kv=True)

    def call():
        return OP(q, ori_kv=ori_kv, cmp_kv=cmp_kv,
                  cmp_sparse_indices=idx, ori_block_table=ori_bt,
                  cmp_block_table=cmp_bt, cu_seqlens_q=cu_q,
                  seqused_ori_kv=seq_o, seqused_cmp_kv=seq_c,
                  cmp_residual_kv=resid, sinks=sinks, metadata=md,
                  softmax_scale=scale, cmp_ratio=RATIO,
                  ori_mask_mode=4, cmp_mask_mode=3,
                  ori_win_left=W - 1, ori_win_right=0,
                  layout_q="TND", layout_kv="PA_BBND",
                  topk_value_mode=1, return_softmax_lse=False)[0]
    return call, (q, ori_kv, cmp_kv, ori_bt, cmp_bt, cu_q, seq_o, seq_c, idx, md)


def bench_graph(fn, n_inner=30, n_outer=20):
    """NPUGraph 捕获 n_inner 次调用, replay 计时 -> 每次设备时间 us。"""
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


def bench_eager(fn, n=200):
    for _ in range(20):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(); torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "official"
    for T, R in ((6, 1), (12, 2), (18, 3), (24, 4)):
        call, _ = make_case(T, R)
        out = call(); torch.npu.synchronize()          # 正确性烟测
        print("finite%:", round(torch.isfinite(out.float()).float().mean().item()*100,1), flush=True)
        tg = bench_graph(call)
        te = bench_eager(call)
        print(f"{which} T={T:2d} R={R}: graph={tg:7.1f} us/call   eager={te:7.1f} us/call", flush=True)
    print("BASELINE-DONE", flush=True)
