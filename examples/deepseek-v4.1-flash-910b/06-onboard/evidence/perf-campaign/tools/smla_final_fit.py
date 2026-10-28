"""终局拟合: L=8676 引擎真实尺度 + 垃圾现场(bt尾 stale/0, block0-1 NaN) + 随机跨界索引。"""
import torch, torch_npu
import cann_ops_transformer
from cann_ops_transformer.ops import sparse_flash_mla as _smla_mod  # noqa
OP = torch.ops.cann_ops_transformer.sparse_flash_mla
MD = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata

dev = "npu:0"
torch.manual_seed(11)
H, D, BLOCK = 8, 512, 128
RATIO, L, T, K, W = 2, 8676, 6, 512, 128
NB_C = 900
BTW = 256

q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev) * 0.3
ori_kv = torch.randn(80, BLOCK, 1, D, dtype=torch.bfloat16, device=dev)
pool = torch.randn(NB_C, BLOCK, 1, D, dtype=torch.bfloat16, device=dev)
# 引擎实测垃圾=有限(official 无NaN): 不注 NaN, block0-1 留随机旧数据
ori_bt = torch.zeros(1, BTW, dtype=torch.int32, device=dev); ori_bt[0, :80] = torch.arange(80)
bt2 = torch.zeros(1, BTW, dtype=torch.int32, device=dev)
bt2[0, :34] = torch.arange(820, 854)             # 请求真实块
for c in range(34, 46): bt2[0, c] = 818 + (c - 34)   # stale 尾(引擎 BTDUMP 实测形态)
cu_q = torch.tensor([0, T], dtype=torch.int32, device=dev)
seq_o = torch.tensor([L], dtype=torch.int32, device=dev)
seq_c = torch.tensor([L], dtype=torch.int32, device=dev)  # 引擎: 原始单位
resid = torch.tensor([0], dtype=torch.int32, device=dev)
scale = D ** -0.5
sinks = (torch.randn(H, device=dev) * 0.1).float()
pos_q = (L - T) + torch.arange(T)

md = MD(H, 1, D, cu_seqlens_q=cu_q, seqused_ori_kv=seq_o, seqused_cmp_kv=seq_c,
        cmp_residual_kv=resid, batch_size=1, max_seqlen_q=T,
        max_seqlen_ori_kv=L, max_seqlen_cmp_kv=L,
        ori_topk=0, cmp_topk=K, cmp_ratio=RATIO, ori_mask_mode=4, cmp_mask_mode=3,
        ori_win_left=W - 1, ori_win_right=0, layout_q="TND", layout_kv="PA_BBND",
        has_ori_kv=True, has_cmp_kv=True)

def official(idx):
    o, _ = OP(q, ori_kv=ori_kv, cmp_kv=pool,
              cmp_sparse_indices=idx.reshape(T, 1, K).to(torch.int32),
              ori_block_table=ori_bt, cmp_block_table=bt2,
              cu_seqlens_q=cu_q, seqused_ori_kv=seq_o, seqused_cmp_kv=seq_c,
              cmp_residual_kv=resid, sinks=sinks, metadata=md,
              softmax_scale=scale, cmp_ratio=RATIO, ori_mask_mode=4, cmp_mask_mode=3,
              ori_win_left=W - 1, ori_win_right=0, layout_q="TND", layout_kv="PA_BBND",
              topk_value_mode=1, return_softmax_lse=False)
    return o

def variant(idx, vis, causal, zero_bad_v=True):
    toks = torch.arange(T, device=dev)
    pos = (L - T) + toks
    oflat = ori_kv.reshape(-1, D)
    cols = pos[:, None] - (W - 1) + torch.arange(W, device=dev)[None, :]
    oval = cols >= 0
    cc = cols.clamp(min=0)
    oblk = torch.div(cc, BLOCK, rounding_mode="floor")
    oslot = torch.gather(ori_bt.long()[0][None].expand(T, -1), 1, oblk) * BLOCK + (cc - oblk * BLOCK)
    kv_o = oflat.index_select(0, oslot.reshape(-1)).reshape(T, W, D)
    s_o = torch.bmm(q, kv_o.transpose(1, 2)).float() * scale
    csafe = idx.clamp(min=0)
    cv = idx >= 0
    if vis == "value_causal":        # 值界: idx < (pos+1)//ratio
        cv = cv & (csafe < ((pos + 1) // RATIO)[:, None])
    elif vis == "value_causal_p":    # 值界: idx <= pos
        cv = cv & (csafe <= pos[:, None])
    # else "all"
    if causal == "none":
        pass
    cblk = torch.div(csafe, BLOCK, rounding_mode="floor")
    cbt_v = torch.gather(bt2.long()[0][None].expand(T, -1), 1, cblk)
    cv = cv & (cbt_v >= 0)
    cslot = (cbt_v * BLOCK + (csafe - cblk * BLOCK)).clamp(min=0, max=pool.reshape(-1, D).shape[0] - 1)
    kv_c = pool.reshape(-1, D).index_select(0, cslot.reshape(-1)).reshape(T, K, D)
    s_c = torch.bmm(q, kv_c.transpose(1, 2)).float() * scale
    finite = torch.isfinite(s_c)                # [T,H,K]
    finite_v = finite.all(dim=1)                # [T,K] V 是否有限(逐列)
    cv_v = cv & finite_v                        # [T,K] 列级有效
    if zero_bad_v:
        kv_c = torch.where(cv_v[:, :, None], kv_c, torch.zeros_like(kv_c))
    s = torch.cat([s_o, torch.nan_to_num(s_c, nan=-1e30)], 2)
    val = torch.cat([oval[:, None, :].expand(T, H, W), cv_v[:, None, :].expand(T, H, K)], 2)
    s = torch.where(val, s, torch.full_like(s, float("-inf")))
    s = s + sinks[None, :, None]
    a = torch.softmax(s, -1)
    return torch.bmm(a, torch.cat([kv_o.float(), kv_c.float()], 1)).to(torch.bfloat16)

def cmpd(m, o):
    d = (m.float() - o.float()).abs()
    return (f"med={torch.nan_to_num(d).median().item():.5f} max={torch.nan_to_num(d).max().item():.4f} "
            f"big%={(torch.nan_to_num(d, nan=9.) > 0.01).float().mean().item()*100:.1f} "
            f"nanm%={torch.isnan(m).float().mean().item()*100:.0f}")

idx_r = torch.randint(0, L, (T, K), device=dev)
o_r = official(idx_r)
# 降序(引擎 topk 形态): 每行 p-511..p
idx_d = torch.stack([(pos_q[t] - torch.arange(K, device=dev)).clamp(min=0) for t in range(T)])
o_d = official(idx_d)
print(f"official(desc) nan%={torch.isnan(o_d).float().mean().item()*100:.1f}", flush=True)
for vis in ("all", "value_causal_p", "value_causal"):
    print(f"D {vis:16s}: {cmpd(variant(idx_d, vis, 'none', True), o_d)}", flush=True)
print(f"official nan%={torch.isnan(o_r).float().mean().item()*100:.1f}", flush=True)
for vis in ("all", "value_causal", "value_causal_p"):
    for zb in (True, False):
        print(f"R {vis:16s} zb={zb}: {cmpd(variant(idx_r, vis, 'none', zb), o_r)}", flush=True)
print("DONE", flush=True)
