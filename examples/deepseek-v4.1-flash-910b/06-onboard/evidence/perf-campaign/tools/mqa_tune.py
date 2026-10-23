"""Triton MQA 并行策略扫描：split-KV 两阶段 vs 单阶段不同 BK/warps。"""
import torch, torch_npu, time, sys
import triton
import triton.language as tl

@triton.jit
def _mqa_split(q_ptr, kv_ptr, idx_ptr, pm_ptr, pl_ptr, pacc_ptr,
               T, D, SEL, scale, NSPLIT,
               BT: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr):
    h = tl.program_id(0)
    sp = tl.program_id(1)
    chunk = tl.cdiv(SEL, NSPLIT)
    lo = sp * chunk
    hi = tl.minimum(lo + chunk, SEL)
    t_off = tl.arange(0, BT)
    d_off = tl.arange(0, BD)
    tmask = t_off < T
    q = tl.load(q_ptr + h.to(tl.int64) * T * D + t_off[:, None] * D + d_off[None, :],
                mask=tmask[:, None], other=0.0)
    m_i = tl.full([BT], float("-inf"), tl.float32)
    l_i = tl.zeros([BT], tl.float32)
    acc = tl.zeros([BT, BD], tl.float32)
    for kb in range(lo, hi, BK):
        k_off = kb + tl.arange(0, BK)
        kmask = k_off < hi
        idx = tl.load(idx_ptr + k_off, mask=kmask, other=0).to(tl.int64)
        kv = tl.load(kv_ptr + idx[:, None] * D + d_off[None, :],
                     mask=kmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(kv)) * scale
        s = tl.where(kmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        p = tl.where(kmask[None, :], p, 0.0)
        corr = tl.exp(m_i - m_new)
        l_i = l_i * corr + tl.sum(p, axis=1)
        acc = acc * corr[:, None] + tl.dot(p.to(tl.bfloat16), kv).to(tl.float32)
        m_i = m_new
    base = (h.to(tl.int64) * NSPLIT + sp)
    tl.store(pm_ptr + base * BT + t_off, m_i)
    tl.store(pl_ptr + base * BT + t_off, l_i)
    tl.store(pacc_ptr + base * BT * BD + t_off[:, None] * BD + d_off[None, :],
             acc, mask=tmask[:, None])

@triton.jit
def _mqa_merge(pm_ptr, pl_ptr, pacc_ptr, out_ptr, T, D, NSPLIT,
               BT: tl.constexpr, BD: tl.constexpr):
    h = tl.program_id(0)
    t_off = tl.arange(0, BT)
    d_off = tl.arange(0, BD)
    tmask = t_off < T
    m = tl.full([BT], float("-inf"), tl.float32)
    for sp in range(NSPLIT):
        base = h.to(tl.int64) * NSPLIT + sp
        m = tl.maximum(m, tl.load(pm_ptr + base * BT + t_off))
    l = tl.zeros([BT], tl.float32)
    acc = tl.zeros([BT, BD], tl.float32)
    for sp in range(NSPLIT):
        base = h.to(tl.int64) * NSPLIT + sp
        mi = tl.load(pm_ptr + base * BT + t_off)
        li = tl.load(pl_ptr + base * BT + t_off)
        w = tl.exp(mi - m)
        l += li * w
        a = tl.load(pacc_ptr + base * BT * BD + t_off[:, None] * BD + d_off[None, :],
                    mask=tmask[:, None], other=0.0)
        acc += a * w[:, None]
    o = acc / l[:, None]
    tl.store(out_ptr + h.to(tl.int64) * T * D + t_off[:, None] * D + d_off[None, :],
             o.to(tl.bfloat16), mask=tmask[:, None])

def sparse_mqa_split(q, kv_pool, flat, scale, NSPLIT, BT=16, BK=32):
    H, T, D = q.shape
    SEL = flat.shape[0]
    out = torch.empty_like(q)
    pm = torch.empty(H * NSPLIT * BT, dtype=torch.float32, device=q.device)
    pl = torch.empty(H * NSPLIT * BT, dtype=torch.float32, device=q.device)
    pacc = torch.empty(H * NSPLIT * BT * D, dtype=torch.float32, device=q.device)
    _mqa_split[(H, NSPLIT)](q, kv_pool, flat, pm, pl, pacc, T, D, SEL, scale,
                            NSPLIT=NSPLIT, BT=BT, BK=BK, BD=D, num_warps=4)
    _mqa_merge[(H,)](pm, pl, pacc, out, T, D, NSPLIT, BT=BT, BD=D, num_warps=4)
    return out

if __name__ == "__main__":
    dev = "npu:0"; torch.manual_seed(0)
    T, H, D, SEL, BLOCK, NB = 6, 8, 512, 640, 128, 512
    SCALE = D ** -0.5
    q = torch.randn(H, T, D, dtype=torch.bfloat16, device=dev)
    kv_pool = torch.randn(NB * BLOCK, D, dtype=torch.bfloat16, device=dev)
    flat = torch.randint(0, NB * BLOCK, (SEL,), dtype=torch.int64, device=dev)
    kv = kv_pool.index_select(0, flat)
    s_ref = torch.matmul(q.float(), kv.float().transpose(0, 1)) * SCALE
    ref = torch.matmul(s_ref.softmax(-1).to(torch.bfloat16), kv)

    for ns in (2, 4, 5):
        try:
            got = sparse_mqa_split(q, kv_pool, flat, SCALE, ns); torch.npu.synchronize()
        except Exception as e:
            print(f"NSPLIT={ns}: ERR {str(e)[:120]}"); continue
        diff = (got.float() - ref.float()).abs().median().item()
        for _ in range(30): sparse_mqa_split(q, kv_pool, flat, SCALE, ns)
        torch.npu.synchronize()
        N = 300; t0 = time.perf_counter()
        for _ in range(N): sparse_mqa_split(q, kv_pool, flat, SCALE, ns)
        torch.npu.synchronize()
        dt = (time.perf_counter() - t0) / N * 1e6
        print(f"NSPLIT={ns}: {dt:6.1f} us/次  median|Δ|={diff:.6f}   (官方170 / PyTorch 144.6)")
