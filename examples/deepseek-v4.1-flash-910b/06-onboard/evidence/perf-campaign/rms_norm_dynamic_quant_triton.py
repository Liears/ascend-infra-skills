#!/usr/bin/env python3
"""Triton-Ascend fused RMSNorm + per-token dynamic int8 quant (chunked).

UB-friendly: the row is processed in BLOCK-sized chunks (Ascend UB is
256 KB; BLOCK=1024 x fp32 x few buffers fits with margin). Pass 1
accumulates sum(x^2) and max|x*gamma|; pass 2 writes the int8 row.

    m      = max|x * gamma|                     (one pass, with sumsq)
    q      = round_half_away(x * gamma * 127 / m)   (rms-independent!)
    scale  = m / (127 * rms),  rms = sqrt(mean(x^2) + eps)

Reference parity ±1 int8 step (rounding order differs, same as CANN's own
fused kernel). Run inside the serving container.
"""
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _rms_norm_dq_kernel(
    x_ptr, gamma_ptr, q_ptr, scale_ptr,
    H, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    base = x_ptr + row.to(tl.int64) * H

    sumsq = 0.0
    m = 1e-12
    for start in range(0, H, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < H
        x = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(gamma_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        t = x * g
        sumsq += tl.sum(x * x, axis=0)
        m = tl.maximum(m, tl.max(tl.abs(t), axis=0))

    rms = tl.sqrt(sumsq / H + EPS)
    scale = m / (127.0 * rms)
    tl.store(scale_ptr + row, scale)

    inv = 127.0 / m
    for start in range(0, H, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < H
        x = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(gamma_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        q = x * g * inv
        q = tl.where(q >= 0, tl.floor(q + 0.5), -tl.floor(-q + 0.5))
        q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
        tl.store(q_ptr + row.to(tl.int64) * H + offs, q, mask=mask)


def rms_norm_dynamic_quant(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-20):
    B, H = x.shape
    q = torch.empty((B, H), dtype=torch.int8, device=x.device)
    s = torch.empty((B,), dtype=torch.float32, device=x.device)
    _rms_norm_dq_kernel[(B,)](x, gamma, q, s, H, EPS=eps, BLOCK=1024, num_warps=4)
    return q, s


def _reference(x, gamma, eps):
    y = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * gamma
    return torch_npu.npu_dynamic_quant(y.to(torch.bfloat16))


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "npu:0"
    B, H, EPS = 6, 5120, 1e-20
    x = torch.randn(B, H, dtype=torch.bfloat16, device=dev) * 3
    gamma = (torch.randn(H, dtype=torch.bfloat16, device=dev) * 0.1 + 1.0)

    q_ref, s_ref = _reference(x, gamma, EPS)
    q_new, s_new = rms_norm_dynamic_quant(x, gamma, EPS)
    d = (q_new.float() - q_ref.float()).abs()
    print(f"int8  : maxdiff {d.max().item()}, exact-match "
          f"{(q_new == q_ref).float().mean().item()*100:.1f}%")
    ds = (s_new - s_ref).abs() / s_ref
    print(f"scale : max rel diff {ds.max().item():.2e}")

    import time

    def unfused():
        y = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS) * gamma
        return torch_npu.npu_dynamic_quant(y.to(torch.bfloat16))

    def bench(fn, n=2000):
        for _ in range(300):
            fn()
        torch.npu.synchronize()
        t0 = time.time()
        for _ in range(n):
            fn()
        torch.npu.synchronize()
        return (time.time() - t0) / n * 1e6

    u, f = bench(unfused), bench(lambda: rms_norm_dynamic_quant(x, gamma, EPS))
    print(f"unfused (RmsNorm+aclnnDynamicQuant): {u:.1f} us")
    print(f"triton fused (chunked):              {f:.1f} us  ({u/f:.2f}x)")
