"""v4 终版: prep + attn(列分裂) + merge 三核, 全部落盘前已单项验证的结构。
数值目标: 与官方 bf16 级一致; 性能目标: 图内 < 官方 42-61us。
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
NCOL = W + K                       # 640
SPLITS = 5                         # 640/5=128, 每split 2块BK=64
SINKS = (torch.randn(H, device=dev) * 0.1).float()


@triton.jit
def _prep_kernel(
    obt_ptr, cbt_ptr, cu_ptr, sl_ptr, idx_ptr,
    slot_ptr, valid_ptr,
    R,
    RMAX: tl.constexpr, BK: tl.constexpr,
    KW_TOT: tl.constexpr, KSEL: tl.constexpr, SBLK: tl.constexpr,
    BTW_MAX: tl.constexpr, NCOL_T: tl.constexpr,
):
    t = tl.program_id(0)
    req = tl.zeros((), tl.int32)
    cu_start = tl.zeros((), tl.int32)
    q_len = tl.zeros((), tl.int32)
    for r in tl.static_range(RMAX):
        cs = tl.load(cu_ptr + r).to(tl.int32)
        ce = tl.load(cu_ptr + r + 1).to(tl.int32)
        hit = (r < R) & (t >= cs) & (t < ce)
        req += tl.where(hit, r, 0).to(tl.int32)
        cu_start += tl.where(hit, cs, 0)
        q_len += tl.where(hit, ce - cs, 0)
    sl = tl.load(sl_ptr + req).to(tl.int32)
    pos = sl - q_len + (t - cu_start)

    obt_row = obt_ptr + req * BTW_MAX
    cbt_row = cbt_ptr + req * BTW_MAX
    idx_row = idx_ptr + t * KSEL
    srow = slot_ptr + t * NCOL_T
    vrow = valid_ptr + t * NCOL_T
    k_off = tl.arange(0, BK)

    for wb in tl.static_range(KW_TOT // BK):
        cols = pos - (KW_TOT - 1) + wb * BK + k_off
        v = cols >= 0
        cc = tl.where(v, cols, 0)
        blk = cc // SBLK
        off = cc - blk * SBLK
        bt = tl.load(obt_row + blk, mask=v, other=-1)
        v = v & (bt >= 0)
        st = tl.where(v, bt * SBLK + off, 0)
        tl.store(srow + wb * BK + k_off, st)
        tl.store(vrow + wb * BK + k_off, v.to(tl.int32))

    for kb in tl.static_range(KSEL // BK):
        kk = kb * BK + k_off
        ci = tl.load(idx_row + kk).to(tl.int32)
        v = ci >= 0
        c_safe = tl.where(v, ci, 0)
        blk = c_safe // SBLK
        off = c_safe - blk * SBLK
        bt = tl.load(cbt_row + blk, mask=v, other=-1)
        v = v & (bt >= 0)
        st = tl.where(v, bt * SBLK + off, 0)
        tl.store(srow + KW_TOT + kb * BK + k_off, st)
        tl.store(vrow + KW_TOT + kb * BK + k_off, v.to(tl.int32))


@triton.jit
def _attn_kernel(
    q_ptr, okv_ptr, ckv_ptr, sink_ptr,
    slot_ptr, valid_ptr, pm_ptr, pl_ptr, pacc_ptr,
    scale,
    BK: tl.constexpr, BD: tl.constexpr, H: tl.constexpr,
    KW_TOT: tl.constexpr, NCOL_T: tl.constexpr,
    SPLIT: tl.constexpr,
):
    t = tl.program_id(0)
    sp = tl.program_id(1)
    h_off = tl.arange(0, H)
    d_off = tl.arange(0, BD)
    q = tl.load(q_ptr + (t * H + h_off[:, None]) * BD + d_off[None, :])
    sink = tl.load(sink_ptr + h_off)

    m_i = tl.full([H], float("-inf"), tl.float32)
    l_i = tl.zeros([H], tl.float32)
    acc = tl.zeros([H, BD], tl.float32)
    k_off = tl.arange(0, BK)
    srow = slot_ptr + t * NCOL_T
    vrow = valid_ptr + t * NCOL_T
    span = NCOL_T // SPLIT
    lo = sp * span

    for b in range(lo, lo + span, BK):
        kk = b + k_off
        v = tl.load(vrow + kk) > 0
        st = tl.load(srow + kk)
        if b < KW_TOT:
            kv = tl.load(okv_ptr + st[:, None] * BD + d_off[None, :])
        else:
            kv = tl.load(ckv_ptr + st[:, None] * BD + d_off[None, :])
        s = tl.dot(q, tl.trans(kv)) * scale
        vm = tl.where(v[None, :], s + sink[:, None], float("-inf"))
        vm = tl.where(vm == vm, vm, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(vm, axis=1))
        p = tl.exp(vm - m_new[:, None])
        corr = tl.exp(m_i - m_new)
        l_i = l_i * corr + tl.sum(p, axis=1)
        acc = acc * corr[:, None] + tl.dot(p.to(tl.bfloat16), kv).to(tl.float32)
        m_i = m_new

    tl.store(pl_ptr + (t * SPLIT + sp) * H + h_off, l_i)
    tl.store(pm_ptr + (t * SPLIT + sp) * H + h_off, m_i)
    tl.store(pacc_ptr + ((t * SPLIT + sp) * H + h_off[:, None]) * BD + d_off[None, :],
             acc)


@triton.jit
def _merge_kernel(out_ptr, pm_ptr, pl_ptr, pacc_ptr,
                  SPLIT: tl.constexpr, BD: tl.constexpr, H: tl.constexpr):
    t = tl.program_id(0)
    h = tl.arange(0, H)
    d = tl.arange(0, BD)
    sp_off = tl.arange(0, SPLIT)
    m_all = tl.load(pm_ptr + (t * SPLIT + sp_off[:, None]) * H + h[None, :])
    l_all = tl.load(pl_ptr + (t * SPLIT + sp_off[:, None]) * H + h[None, :])
    m_g = tl.max(m_all, axis=0)
    w = tl.exp(m_all - m_g[None, :]) * l_all
    l_g = tl.sum(w, axis=0)
    acc = tl.zeros([H, BD], tl.float32)
    for sp in tl.static_range(SPLIT):
        ms = tl.load(pm_ptr + (t * SPLIT + sp) * H + h)
        ls = tl.load(pl_ptr + (t * SPLIT + sp) * H + h)
        w_s = tl.exp(ms - m_g) * ls
        a = tl.load(pacc_ptr + ((t * SPLIT + sp) * H + h[:, None]) * BD + d[None, :])
        acc += a * w_s[:, None]
    tl.store(out_ptr + (t * H + h[:, None]) * BD + d[None, :],
             (acc / l_g[:, None]).to(tl.bfloat16))


_WS = {}


def decode_csa_v4(q, ori_flat, cmp_flat, ori_bt, cmp_bt, cu_q, seq_lens, idx,
                  sinks, scale, R, BK=64, SPLIT=5):
    T = q.shape[0]
    out = torch.empty_like(q)
    key = (T, SPLIT)
    ws = _WS.get(key)
    if ws is None:
        ws = (
            torch.empty(T * NCOL, dtype=torch.int32, device=q.device),
            torch.empty(T * NCOL, dtype=torch.int32, device=q.device),
            torch.empty(T * SPLIT * H, dtype=torch.float32, device=q.device),
            torch.empty(T * SPLIT * H, dtype=torch.float32, device=q.device),
            torch.empty(T * SPLIT * H * D, dtype=torch.float32, device=q.device),
        )
        _WS[key] = ws
    slots, valid, pm, pl, pacc = ws
    cu_pad = torch.zeros(6, dtype=torch.int32, device=q.device)
    cu_pad[:cu_q.shape[0]] = cu_q
    _prep_kernel[(T,)](ori_bt, cmp_bt, cu_pad, seq_lens, idx, slots, valid, R,
                       RMAX=4, BK=BK, KW_TOT=W, KSEL=K, SBLK=BLOCK,
                       BTW_MAX=BTW, NCOL_T=NCOL, num_warps=1)
    _attn_kernel[(T, SPLIT)](q, ori_flat, cmp_flat, sinks, slots, valid,
                             pm, pl, pacc, scale,
                             BK=BK, BD=D, H=H, KW_TOT=W, NCOL_T=NCOL,
                             SPLIT=SPLIT, num_warps=4)
    _merge_kernel[(T,)](out, pm, pl, pacc, SPLIT=SPLIT, BD=D, H=H, num_warps=4)
    return out


def build_case(T=6, R=1, seed=11):
    torch.manual_seed(seed)
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev) * 0.3
    ori_flat = torch.randn(NB_O * BLOCK, D, dtype=torch.bfloat16, device=dev)
    cmp_flat = torch.randn(NB_C * BLOCK, D, dtype=torch.bfloat16, device=dev)
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
    return dict(q=q, ori_flat=ori_flat, cmp_flat=cmp_flat,
                ori_kv=ori_flat.view(NB_O, BLOCK, 1, D),
                cmp_kv=cmp_flat.view(NB_C, BLOCK, 1, D),
                ori_bt=ori_bt, cmp_bt=cmp_bt, cu_q=cu_q, seq_o=seq_o,
                idx=idx, md=md, resid=resid)


def official(c):
    return OP(c["q"], ori_kv=c["ori_kv"], cmp_kv=c["cmp_kv"],
              cmp_sparse_indices=c["idx"].view(c["q"].shape[0], 1, K),
              ori_block_table=c["ori_bt"], cmp_block_table=c["cmp_bt"],
              cu_seqlens_q=c["cu_q"], seqused_ori_kv=c["seq_o"],
              seqused_cmp_kv=c["seq_o"], cmp_residual_kv=c["resid"],
              sinks=SINKS, metadata=c["md"], softmax_scale=SCALE, cmp_ratio=RATIO,
              ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=W - 1, ori_win_right=0,
              layout_q="TND", layout_kv="PA_BBND", topk_value_mode=1,
              return_softmax_lse=False)[0]


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
        o = official(c)
        try:
            m = decode_csa_v4(c["q"], c["ori_flat"], c["cmp_flat"], c["ori_bt"],
                              c["cmp_bt"], c["cu_q"], c["seq_o"], c["idx"],
                              SINKS, SCALE, R)
            torch.npu.synchronize()
        except Exception as e:
            _e = e
            while _e.__cause__:
                _e = _e.__cause__
            print(f"T={T} V4-ERR: {type(_e).__name__}: {str(_e)[:250]}", flush=True)
            break
        fin = torch.isfinite(o.float())
        d = torch.where(fin, (m.float() - o.float()).abs(), torch.zeros((), device=dev))
        tg_m = bench_graph(lambda: decode_csa_v4(c["q"], c["ori_flat"], c["cmp_flat"],
                                                 c["ori_bt"], c["cmp_bt"], c["cu_q"],
                                                 c["seq_o"], c["idx"], SINKS, SCALE, R))
        tg_o = bench_graph(lambda: official(c))
        print(f"T={T:2d} R={R}: med={d.median().item():.5f} max={d.max().item():.4f} "
              f"big%={(d > 0.01).float().mean().item()*100:.1f} | "
              f"v4={tg_m:6.1f}us official={tg_o:6.1f}us ({(1-tg_m/tg_o)*100:+.0f}%)",
              flush=True)
    print("V4-DONE", flush=True)
