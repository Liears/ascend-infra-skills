"""Triton-Ascend 单内核稀疏 MQA（flash-decoding 风格 online softmax）。
对比：官方 aclnnSparseFlashMla 170us / PyTorch 4-kernel 144.6us。"""
import torch, torch_npu, time, sys
import triton
import triton.language as tl

@triton.jit
def _sparse_mqa_kernel(
    q_ptr, kv_ptr, idx_ptr, out_ptr,
    T, D, SEL, scale,
    BT: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr,
):
    h = tl.program_id(0)
    t_off = tl.arange(0, BT)
    d_off = tl.arange(0, BD)
    tmask = t_off < T

    # q 块 [BT, BD]
    q = tl.load(q_ptr + h.to(tl.int64) * T * D + t_off[:, None] * D + d_off[None, :],
                mask=tmask[:, None], other=0.0)

    m_i = tl.full([BT], float("-inf"), tl.float32)
    l_i = tl.zeros([BT], tl.float32)
    acc = tl.zeros([BT, BD], tl.float32)

    for kb in range(0, SEL, BK):
        k_off = kb + tl.arange(0, BK)
        kmask = k_off < SEL
        idx = tl.load(idx_ptr + k_off, mask=kmask, other=0).to(tl.int64)
        kv = tl.load(kv_ptr + idx[:, None] * D + d_off[None, :],
                     mask=kmask[:, None], other=0.0)          # [BK, BD] bf16
        s = tl.dot(q, tl.trans(kv)) * scale                    # [BT, BK] fp32
        s = tl.where(kmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        p = tl.where(kmask[None, :], p, 0.0)
        corr = tl.exp(m_i - m_new)
        l_i = l_i * corr + tl.sum(p, axis=1)
        acc = acc * corr[:, None] + tl.dot(p.to(tl.bfloat16), kv).to(tl.float32)
        m_i = m_new

    o = acc / l_i[:, None]
    tl.store(out_ptr + h.to(tl.int64) * T * D + t_off[:, None] * D + d_off[None, :],
             o.to(tl.bfloat16), mask=tmask[:, None])


def sparse_mqa(q, kv_pool, flat, scale, BT=16, BK=64):
    H, T, D = q.shape
    SEL = flat.shape[0]
    out = torch.empty_like(q)
    _sparse_mqa_kernel[(H,)](q, kv_pool, flat, out, T, D, SEL, scale,
                             BT=BT, BK=BK, BD=D, num_warps=4)
    return out


if __name__ == "__main__":
    dev = "npu:0"
    torch.manual_seed(0)
    T, H, D, SEL, BLOCK, NB = 6, 8, 512, 640, 128, 512
    SCALE = D ** -0.5
    q = torch.randn(H, T, D, dtype=torch.bfloat16, device=dev)
    kv_pool = torch.randn(NB * BLOCK, D, dtype=torch.bfloat16, device=dev)
    flat = torch.randint(0, NB * BLOCK, (SEL,), dtype=torch.int64, device=dev)

    # 数值参考
    kv = kv_pool.index_select(0, flat)
    s_ref = torch.matmul(q.float(), kv.float().transpose(0, 1)) * SCALE
    ref = torch.matmul(s_ref.softmax(-1).to(torch.bfloat16), kv)

    try:
        got = sparse_mqa(q, kv_pool, flat, SCALE)
        torch.npu.synchronize()
    except Exception as e:
        print("ERR:", str(e)[:400]); sys.exit(1)
    diff = (got.float() - ref.float()).abs()
    rel = (diff / (ref.float().abs() + 1e-3)).median().item()
    print(f"数值: median|Δ|={diff.median().item():.5f}  median rel={rel:.5f}  max|Δ|={diff.max().item():.4f}")

    for _ in range(30): sparse_mqa(q, kv_pool, flat, SCALE)
    torch.npu.synchronize()
    N = 300; t0 = time.perf_counter()
    for _ in range(N): sparse_mqa(q, kv_pool, flat, SCALE)
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) / N * 1e6
    print(f"Triton 单内核: {dt:.1f} us/次   (官方 170 / PyTorch组合 144.6)")
