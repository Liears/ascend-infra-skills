"""v2: 每 token 一个 program, tl.dot 立方体路径 (kv 按行共享跨头复用)。
score = dot(q[t] [H,512], kv.T [512,BK]) -> [H,BK]; acc += dot(p [H,BK], kv [BK,512])。
"""
import time
import torch, torch_npu
import triton
import triton.language as tl
import cann_ops_transformer
from cann_ops_transformer.ops import sparse_flash_mla as _smla_mod  # noqa

OP = torch.ops.cann_ops_transformer.sparse_flash_mla
MD = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata

dev = "npu:0"
D, BLOCK, RATIO, K, W, SEQ = 512, 128, 2, 512, 128, 8676
H, NB_O, NB_C, BTW = 16, 80, 900, 256
SCALE = D ** -0.5
_LAST = {}


@triton.jit
def _decode_csa_v2(
    q_ptr, okv_ptr, ckv_ptr, obt_ptr, cbt_ptr,
    idx_ptr, sink_ptr, out_ptr,
    pos_ptr, req_ptr,
    scale,
    BK: tl.constexpr, BD: tl.constexpr,
    KW_TOT: tl.constexpr, KSEL: tl.constexpr, SBLK: tl.constexpr,
    BTW_MAX: tl.constexpr, H: tl.constexpr,
):
    t = tl.program_id(0)
    h_off = tl.arange(0, H)
    d_off = tl.arange(0, BD)
    pos = tl.load(pos_ptr + t).to(tl.int32)
    req = tl.load(req_ptr + t).to(tl.int32)

    q = tl.load(q_ptr + (t * H + h_off[:, None]) * BD + d_off[None, :])  # [H,BD] bf16
    sink = tl.load(sink_ptr + h_off)                                       # [H] f32

    m_i = tl.full([H], float("-inf"), tl.float32)
    l_i = tl.zeros([H], tl.float32)
    acc = tl.zeros([H, BD], tl.float32)

    k_off = tl.arange(0, BK)
    idx_row = idx_ptr + t * KSEL
    cbt_row = cbt_ptr + req * BTW_MAX
    obt_row = obt_ptr + req * BTW_MAX

    # ---- ori 滑窗块 ----
    for wb in range(0, KW_TOT, BK):
        cols = pos - (KW_TOT - 1) + wb + k_off
        valid = cols >= 0
        cc = tl.where(valid, cols, 0)
        oblk = cc // SBLK
        ooff = cc - oblk * SBLK
        ob_v = tl.load(obt_row + oblk, mask=valid, other=-1)
        valid = valid & (ob_v >= 0)
        oslot = tl.where(valid, ob_v * SBLK + ooff, 0)
        kv = tl.load(okv_ptr + oslot[:, None] * BD + d_off[None, :],
                     mask=valid[:, None], other=0.0)                       # [BK,BD] bf16
        s = tl.dot(q, tl.trans(kv)) * scale                                # [H,BK] f32
        vmask = tl.where(valid[None, :], s + sink[:, None], float("-inf"))
        m_new = tl.maximum(m_i, tl.max(vmask, axis=1))
        p = tl.exp(vmask - m_new[:, None])
        p = tl.where(valid[None, :], p, 0.0)
        corr = tl.exp(m_i - m_new)
        l_i = l_i * corr + tl.sum(p, axis=1)
        acc = acc * corr[:, None] + tl.dot(p.to(tl.bfloat16), kv).to(tl.float32)
        m_i = m_new

    # ---- cmp 条目块 ----
    for kb in range(0, KSEL, BK):
        kk = kb + k_off
        ci = tl.load(idx_row + kk).to(tl.int32)
        cv = ci >= 0
        c_safe = tl.where(cv, ci, 0)
        cblk = c_safe // SBLK
        coff = c_safe - cblk * SBLK
        cb_v = tl.load(cbt_row + cblk, mask=cv, other=-1)
        cv = cv & (cb_v >= 0)
        slot = tl.where(cv, cb_v * SBLK + coff, 0)
        kv = tl.load(ckv_ptr + slot[:, None] * BD + d_off[None, :],
                     mask=cv[:, None], other=0.0)
        s = tl.dot(q, tl.trans(kv)) * scale
        vmask = tl.where(cv[None, :], s + sink[:, None], float("-inf"))
        vmask = tl.where(vmask == vmask, vmask, float("-inf"))            # NaN 列防护
        m_new = tl.maximum(m_i, tl.max(vmask, axis=1))
        p = tl.exp(vmask - m_new[:, None])
        p = tl.where(cv[None, :], p, 0.0)
        corr = tl.exp(m_i - m_new)
        l_i = l_i * corr + tl.sum(p, axis=1)
        acc = acc * corr[:, None] + tl.dot(p.to(tl.bfloat16), kv).to(tl.float32)
        m_i = m_new

    o = acc / l_i[:, None]
    tl.store(out_ptr + (t * H + h_off[:, None]) * BD + d_off[None, :],
             o.to(tl.bfloat16))


def decode_csa_v2(q, ori_flat, cmp_flat, ori_bt, cmp_bt, cu_q, seq_lens, idx,
                  sinks, scale, R, BK=64):
    T = q.shape[0]
    out = torch.empty_like(q)
    cu = cu_q.long()
    toks = torch.arange(T, device=q.device)
    hits = ((toks[None, :] >= cu[:-1][:, None]) & (toks[None, :] < cu[1:][:, None])).to(torch.int64)
    req_t = hits.argmax(0)
    cu_start = cu[:-1].gather(0, req_t)
    q_len = (cu[1:] - cu[:-1]).gather(0, req_t)
    pos_t = (seq_lens.long().gather(0, req_t) - q_len + toks - cu_start).to(torch.int32)
    req_t = req_t.to(torch.int32)
    _LAST["pos"], _LAST["req"] = pos_t, req_t
    _decode_csa_v2[(T,)](
        q, ori_flat, cmp_flat, ori_bt, cmp_bt, idx, sinks, out, pos_t, req_t,
        scale,
        BK=BK, BD=D, KW_TOT=W, KSEL=K, SBLK=BLOCK, BTW_MAX=BTW, H=H,
        num_warps=4,
    )
    return out


def build_case(T=6, R=1, seed=11):
    torch.manual_seed(seed)
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev) * 0.3
    ori_flat = torch.randn(NB_O * BLOCK, D, dtype=torch.bfloat16, device=dev)
    cmp_flat = torch.randn(NB_C * BLOCK, D, dtype=torch.bfloat16, device=dev)
    ori_kv3 = ori_flat.view(NB_O, BLOCK, 1, D)
    cmp_kv3 = cmp_flat.view(NB_C, BLOCK, 1, D)
    ori_bt = torch.zeros(R, BTW, dtype=torch.int32, device=dev)
    cmp_bt = torch.zeros(R, BTW, dtype=torch.int32, device=dev)
    for r in range(R):
        ori_bt[r, :68] = torch.arange(820 + r * 68, 820 + r * 68 + 68) % NB_O
        cmp_bt[r, :34] = torch.arange(820 + r * 34, 820 + r * 34 + 34) % NB_C
        for c in range(34, 46):
            cmp_bt[r, c] = (818 + (c - 34) + r) % NB_C
    cu_q = torch.arange(0, R * (T // R) + 1, T // R, dtype=torch.int32, device=dev)
    seq_o = torch.full((R,), SEQ, dtype=torch.int32, device=dev)
    resid = torch.zeros(R, dtype=torch.int32, device=dev)
    idx = torch.randint(0, SEQ, (T, K), device=dev, dtype=torch.int32)
    md = MD(H, 1, D, cu_seqlens_q=cu_q, seqused_ori_kv=seq_o, seqused_cmp_kv=seq_o,
            cmp_residual_kv=resid, batch_size=R, max_seqlen_q=T // R,
            max_seqlen_ori_kv=SEQ, max_seqlen_cmp_kv=SEQ,
            ori_topk=0, cmp_topk=K, cmp_ratio=RATIO, ori_mask_mode=4,
            cmp_mask_mode=3, ori_win_left=W - 1, ori_win_right=0,
            layout_q="TND", layout_kv="PA_BBND", has_ori_kv=True, has_cmp_kv=True)
    return dict(q=q, ori_flat=ori_flat, cmp_flat=cmp_flat, ori_kv=ori_kv3,
                cmp_kv=cmp_kv3, ori_bt=ori_bt, cmp_bt=cmp_bt, cu_q=cu_q,
                seq_o=seq_o, idx=idx, md=md, resid=resid)


def official(c):
    return OP(c["q"], ori_kv=c["ori_kv"], cmp_kv=c["cmp_kv"],
              cmp_sparse_indices=c["idx"].view(c["q"].shape[0], 1, K),
              ori_block_table=c["ori_bt"], cmp_block_table=c["cmp_bt"],
              cu_seqlens_q=c["cu_q"], seqused_ori_kv=c["seq_o"],
              seqused_cmp_kv=c["seq_o"], cmp_residual_kv=c["resid"],
              sinks=_SINKS, metadata=c["md"], softmax_scale=SCALE, cmp_ratio=RATIO,
              ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=W - 1, ori_win_right=0,
              layout_q="TND", layout_kv="PA_BBND", topk_value_mode=1,
              return_softmax_lse=False)[0]


_SINKS = None


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


if __name__ == "__main__":
    for T, R in ((6, 1), (12, 2), (18, 3), (24, 4)):
        c = build_case(T, R)
        import __main__; __main__._SINKS = (torch.randn(H, device=dev) * 0.1).float()
        o = official(c)
        try:
            m = decode_csa_v2(c["q"], c["ori_flat"], c["cmp_flat"], c["ori_bt"],
                              c["cmp_bt"], c["cu_q"], c["seq_o"], c["idx"],
                              _SINKS, SCALE, R)
            torch.npu.synchronize()
        except Exception as e:
            _e = e
            while _e.__cause__:
                _e = _e.__cause__
            print(f"T={T} V2-ERR: {type(_e).__name__}: {str(_e)[:300]}", flush=True)
            break
        fin = torch.isfinite(o.float())
        d = torch.where(fin, (m.float() - o.float()).abs(), torch.zeros_like(fin, dtype=torch.float32))
        nfin = fin.float().mean().item() * 100
        tg_m = bench_graph(lambda: decode_csa_v2(c["q"], c["ori_flat"], c["cmp_flat"],
                                                 c["ori_bt"], c["cmp_bt"], c["cu_q"],
                                                 c["seq_o"], c["idx"], _SINKS, SCALE, R))
        tg_o = bench_graph(lambda: official(c))
        print(f"T={T:2d} R={R}: med={d.median().item():.5f} max={d.max().item():.4f} "
              f"big%={(d > 0.01).float().mean().item()*100:.1f} offnan%={100-nfin:.0f} | "
              f"v2={tg_m:6.1f}us official={tg_o:6.1f}us ({(1-tg_m/tg_o)*100:+.0f}%)",
              flush=True)
    print("V2-DONE", flush=True)
