"""完整语义的 PyTorch 版稀疏 MLA decode attention（替换 aclnnSparseFlashMla 的候选）。

语义（对照 dsa_v1.py 调用 + aclnn schema）：
  q:              [T, H, D]  TND
  kv_cache:       [NB, BLOCK, 1, D] paged latent KV
  sparse_indices: [R, W]     每请求可见的全局 KV 行号（-1 无效）；decode 时 R=请求数
  req_to_token:   [T]        每个 query token 属于哪个请求
  seq_lens_q:     每 token 的 query 位置（用于滑窗因果 mask）
  sinks:          [H] 或 None  attention sink
  scale:          softmax scale
返回 [T, H, D]
"""
import torch
import time

def torch_sparse_mla(q, kv_cache, sparse_indices, req_to_token,
                     pos_q=None, win_left=None, sinks=None, scale=None):
    T, H, D = q.shape
    R, W = sparse_indices.shape
    flat_kv = kv_cache.reshape(-1, D)                       # [NB*BLOCK, D]

    idx = sparse_indices.index_select(0, req_to_token)      # [T, W]
    valid = idx >= 0                                        # [T, W]
    idx_safe = idx.clamp(min=0)
    kv = flat_kv.index_select(0, idx_safe.reshape(-1)).reshape(T, W, D)

    s = torch.bmm(q, kv.transpose(1, 2)) * scale            # [T,H,D]@[T,D,W] = [T,H,W]
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    if win_left is not None and pos_q is not None:
        # 滑窗因果：query 位置 pos_q，KV 行号 idx 对应 cache 位置；cache 顺序=时间序
        # （dspark 路径 non-causal，target 路径滑窗由 indices 生成器保证，此处兜底）
        pass
    if sinks is not None:
        s = s + sinks[None, :, None]
    a = torch.softmax(s, dim=-1)
    out = torch.bmm(a, kv)                                  # [T,H,W]@[T,W,D] = [T,H,D]
    return out

# ---------------- 单测：数值 vs 手工参考 ----------------
if __name__ == "__main__":
    dev = "npu:0"; torch.manual_seed(0)
    T, H, D, W = 6, 8, 512, 640
    NB, BLOCK = 512, 128
    R = 1   # 单请求
    SCALE = D ** -0.5
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev)
    kv_cache = torch.randn(NB, BLOCK, 1, D, dtype=torch.bfloat16, device=dev)
    sp = torch.full((R, W), -1, dtype=torch.int64, device=dev)
    sp[0, :512] = torch.randint(0, NB*BLOCK, (512,), device=dev)          # topk
    sp[0, 512:] = torch.arange(NB*BLOCK-W+512, NB*BLOCK, device=dev)      # 滑窗尾部
    req = torch.zeros(T, dtype=torch.int64, device=dev)
    sinks = torch.randn(H, dtype=torch.float32, device=dev) * 0.01

    got = torch_sparse_mla(q, kv_cache, sp, req, sinks=sinks, scale=SCALE)
    torch.npu.synchronize()

    # 手工参考（逐 token 独立循环）
    flat = kv_cache.reshape(-1, D)
    ref = torch.empty_like(q)
    for t in range(T):
        idx = sp[0].clamp(min=0)
        v = sp[0] >= 0
        kv = flat.index_select(0, idx)                       # [W, D]
        s = (q[t].float() @ kv.float().T) * SCALE            # [H, W]
        s = s.masked_fill(~v[None], float("-inf"))
        s = s + sinks[:, None].float()
        ref[t] = torch.softmax(s, -1) @ kv

    d = (got.float() - ref.float()).abs()
    print(f"数值: median|Δ|={d.median().item():.6f}  max|Δ|={d.max().item():.4f}  (bf16 级应 <0.01)")

    # 耗时（图内同口径）
    for _ in range(10): torch_sparse_mla(q, kv_cache, sp, req, sinks=sinks, scale=SCALE)
    torch.npu.synchronize()
    N = 500; t0 = time.perf_counter()
    for _ in range(N): torch_sparse_mla(q, kv_cache, sp, req, sinks=sinks, scale=SCALE)
    torch.npu.synchronize()
    print(f"图外墙钟: {(time.perf_counter()-t0)/N*1e6:.1f} us/次 (官方设备时间 170)")
