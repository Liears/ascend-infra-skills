"""离线逆向 npu_sparse_flash_mla (decode 场景): 官方算子 vs 我的组合, 假设秒级切换。
复刻引擎 decode 的喂法: 索引超出 Lc (原始 token 空间), cseq 可选原始/压缩单位。
"""
import os, sys, itertools
import torch, torch_npu

import cann_ops_transformer
from cann_ops_transformer.ops import sparse_flash_mla as _smla_mod  # 触发注册
OP = torch.ops.cann_ops_transformer.sparse_flash_mla
MD = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata
print("ops loaded OK", flush=True)

dev = "npu:0"
torch.manual_seed(7)
H, D, BLOCK = 8, 512, 128
L = 512            # 原始序列长
RATIO = 2
Lc = L // RATIO    # 256 压缩行
T = 6              # decode q_len (dspark verify)
K = 512
W = 128            # ori 窗口 [p-127, p]

NB_O, NB_C = L // BLOCK, Lc // BLOCK        # 4, 2
q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev) * 0.3
ori_kv = torch.randn(NB_O, BLOCK, 1, D, dtype=torch.bfloat16, device=dev)
cmp_kv = torch.randn(NB_C, BLOCK, 1, D, dtype=torch.bfloat16, device=dev)
ori_bt = torch.arange(NB_O, dtype=torch.int32, device=dev)[None]
cmp_bt = torch.arange(NB_C, dtype=torch.int32, device=dev)[None]
cu_q = torch.tensor([0, T], dtype=torch.int32, device=dev)
seq_o = torch.tensor([L], dtype=torch.int32, device=dev)
resid = torch.tensor([0], dtype=torch.int32, device=dev)
scale = D ** -0.5
sinks = torch.zeros(H, dtype=torch.float32, device=dev)  # 先零; 稍后试 None


def official(idx, seq_c, sink=None):
    md = MD(H, 1, D,
            cu_seqlens_q=cu_q,
            seqused_ori_kv=seq_o,
            seqused_cmp_kv=seq_c,
            cmp_residual_kv=resid,
            batch_size=1,
            max_seqlen_q=T,
            max_seqlen_ori_kv=L,
            max_seqlen_cmp_kv=int(seq_c.max().item()),
            ori_topk=0, cmp_topk=K, cmp_ratio=RATIO,
            ori_mask_mode=4, cmp_mask_mode=3,
            ori_win_left=W - 1, ori_win_right=0,
            layout_q="TND", layout_kv="PA_BBND",
            has_ori_kv=True, has_cmp_kv=True)
    out, _ = OP(q.reshape(T, H, D),
                ori_kv=ori_kv, cmp_kv=cmp_kv,
                cmp_sparse_indices=idx.reshape(T, 1, K).to(torch.int32),
                ori_block_table=ori_bt, cmp_block_table=cmp_bt,
                cu_seqlens_q=cu_q, seqused_ori_kv=seq_o, seqused_cmp_kv=seq_c,
                cmp_residual_kv=resid, sinks=sink,
                metadata=md, softmax_scale=scale, cmp_ratio=RATIO,
                ori_mask_mode=4, cmp_mask_mode=3,
                ori_win_left=W - 1, ori_win_right=0,
                layout_q="TND", layout_kv="PA_BBND",
                topk_value_mode=1, return_softmax_lse=False)
    return out


def mine(idx, seq_c, causal="cover_start", cspace="direct", rowcap=True):
    """组合实现. idx: [T,K] long. causal: cover_start|exact|none; cspace: direct|div2"""
    toks = torch.arange(T, device=dev)
    pos = (L - T) + toks                       # 绝对位置 (R=1)
    if causal == "full":
        pos = torch.full_like(pos, L - 1)
    oflat = ori_kv.reshape(-1, D)
    cols = pos[:, None] - (W - 1) + torch.arange(W, device=dev)[None, :]
    oval = cols >= 0
    cc = cols.clamp(min=0)
    oblk = torch.div(cc, BLOCK, rounding_mode="floor")
    ooff = cc - oblk * BLOCK
    oslot = torch.gather(ori_bt.long()[0][None].expand(T, -1), 1, oblk) * BLOCK + ooff
    kv_o = oflat.index_select(0, oslot.reshape(-1)).reshape(T, W, D)
    s_o = torch.bmm(q, kv_o.transpose(1, 2)) * scale

    cflat = cmp_kv.reshape(-1, D)
    csafe = idx.clamp(min=0)
    cvalid = idx >= 0
    if cspace == "div2":
        c = torch.div(csafe, RATIO, rounding_mode="floor")
        cov0 = csafe * RATIO
    else:
        c = csafe
        cov0 = csafe * RATIO
    cvalid = cvalid & (cov0 <= pos[:, None]) if causal in ("cover_start", "full") else cvalid
    if causal == "exact":
        cvalid = cvalid & (cov0 + RATIO - 1 <= pos[:, None])
    cblk = torch.div(c, BLOCK, rounding_mode="floor").clamp(max=cmp_bt.shape[1] - 1)
    coff = c - cblk * BLOCK
    cslot = torch.gather(cmp_bt.long()[0][None].expand(T, -1), 1, cblk) * BLOCK + coff
    if rowcap:
        cvalid = cvalid & (c < Lc)
    cslot = cslot.clamp(min=0, max=cflat.shape[0] - 1)
    kv_c = cflat.index_select(0, cslot.reshape(-1)).reshape(T, K, D)
    s_c = torch.bmm(q, kv_c.transpose(1, 2)) * scale

    s = torch.cat([s_o, s_c], 2)
    val = torch.cat([oval[:, None, :], cvalid[:, None, :]], 2)
    s = s.masked_fill(~val, -1e4)
    if sinks is not None:
        s = s + sinks[None, :, None].to(s.dtype)
    a = torch.softmax(s, -1)
    return torch.bmm(a, torch.cat([kv_o, kv_c], 1))


def mine_dedup(idx, seq_c, mode="amax", causal="cover_start"):
    """scatter 聚合版: 重复 cmp 行合并成单列, score 取 amax/mean/sum."""
    toks = torch.arange(T, device=dev)
    pos = (L - T) + toks
    if causal == "full":
        pos = torch.full_like(pos, L - 1)
    oflat = ori_kv.reshape(-1, D)
    cols = pos[:, None] - (W - 1) + torch.arange(W, device=dev)[None, :]
    oval = cols >= 0
    cc = cols.clamp(min=0)
    oblk = torch.div(cc, BLOCK, rounding_mode="floor")
    ooff = cc - oblk * BLOCK
    oslot = torch.gather(ori_bt.long()[0][None].expand(T, -1), 1, oblk) * BLOCK + ooff
    kv_o = oflat.index_select(0, oslot.reshape(-1)).reshape(T, W, D)
    s_o = torch.bmm(q, kv_o.transpose(1, 2)) * scale

    cvalid = idx >= 0
    csafe = idx.clamp(min=0)
    cov0 = csafe * RATIO
    cvalid = cvalid & (cov0 <= pos[:, None]) & (csafe < Lc)
    kv_all = cmp_kv.reshape(-1, D)
    cslot = csafe  # 行号即 slot(块表恒等)
    kv_c = kv_all.index_select(0, cslot.reshape(-1)).reshape(T, K, D)
    s_c = torch.bmm(q, kv_c.transpose(1, 2)) * scale  # [T,H,K]
    # scatter 到稠密行轴 [T, Lc]
    dense = torch.full((T, H, Lc), float("-inf"), device=dev, dtype=torch.float32)
    flatidx = csafe[:, None].expand(T, H, K).reshape(-1)
    vals = s_c.float().reshape(-1)
    wmask = cvalid[:, None, :].expand(T, H, K).reshape(-1)
    flatidx = flatidx * wmask + (-1) * (~wmask)          # 无效→-1
    vals = vals * wmask + float("-inf") * (~wmask)
    flatidx_c = flatidx.clamp(min=0)
    if mode == "amax":
        dense.scatter_reduce_(2, flatidx_c.reshape(T, H, K),
                              vals.reshape(T, H, K), reduce="amax", include_self=True)
    elif mode == "mean":
        ssum = torch.zeros((T, H, Lc), device=dev, dtype=torch.float32)
        scnt = torch.zeros((T, Lc), device=dev, dtype=torch.float32)
        ssum.scatter_add_(2, flatidx_c.reshape(T, H, K), torch.where(wmask.reshape(T,H,K), vals.reshape(T,H,K), torch.zeros_like(vals.reshape(T,H,K))))
        scnt.scatter_add_(1, csafe.clamp(max=Lc-1), cvalid.float())
        dense = ssum / scnt.clamp(min=1)[:, None, :]
        dense = torch.where(scnt[:, None, :] > 0, dense, torch.full_like(dense, float("-inf")))
    row_valid = (dense > float("-inf"))
    s = torch.cat([s_o.float(), dense], 2)
    val = torch.cat([oval[:, None, :].expand(T, H, W),
                     row_valid.expand(T, H, Lc) if row_valid.dim() == 3 else row_valid[:, None, :].expand(T, H, Lc)], 2)
    s = s.masked_fill(~val, float("-inf"))
    if sinks is not None:
        s = s + sinks[None, :, None]
    a = torch.softmax(s, -1)
    kv_cmp_dense = kv_all  # [Lc, D]; 稠密 V
    kv_cat = torch.cat([kv_o.float(), kv_cmp_dense.float()[None].expand(T, Lc, D)], 1)
    return torch.bmm(a, kv_cat).to(torch.bfloat16)


def cmpd(a, b):
    d = (a.float() - b.float()).abs()
    return f"med={d.median().item():.5f} max={d.max().item():.4f} big%={(d > 0.01).float().mean().item()*100:.1f}"


# ---------- 实验 1: 契约内 (压缩空间索引) — 校准组合实现 ----------
idx_c = torch.randint(0, Lc, (T, K), device=dev)          # [0,256) 压缩行
seq_c_comp = torch.tensor([Lc], dtype=torch.int32, device=dev)

# ---------- 实验 0: 隔离 ori 窗 / 单行 cmp ----------
idx_none = torch.full((T, K), -1, device=dev, dtype=torch.long)
o0a = official(idx_none, seq_c_comp, sinks)
m0a = mine(idx_none, seq_c_comp, "cover_start", "direct")
print("E0a all-(-1) idx (ori+sinks only):", cmpd(m0a, o0a), flush=True)
d0 = (m0a.float() - o0a.float()).abs()
print("    per-tok max:", [round(v, 4) for v in d0.amax(dim=(1, 2)).tolist()], flush=True)
idx_zero = torch.zeros((T, K), device=dev, dtype=torch.long)
o0b = official(idx_zero, seq_c_comp, sinks)
print("E0b all-0 idx (+cmp row0):", cmpd(mine(idx_zero, seq_c_comp, "cover_start", "direct"), o0b), flush=True)

o = official(idx_c, seq_c_comp, sinks)
print("E1 contract/compressed idx:", flush=True)
print("  cover_start:", cmpd(mine(idx_c, seq_c_comp, "cover_start", "direct"), o), flush=True)
print("  exact      :", cmpd(mine(idx_c, seq_c_comp, "exact", "direct"), o), flush=True)
print("  full       :", cmpd(mine(idx_c, seq_c_comp, "full", "direct"), o), flush=True)

# ---------- 实验 0c/0d: 条目数阶梯 ----------
idx_1 = torch.full((T, K), -1, device=dev, dtype=torch.long); idx_1[:, 0] = 5
o0c = official(idx_1, seq_c_comp, sinks)
print("E0c single entry row5:", cmpd(mine(idx_1, seq_c_comp, "cover_start", "direct"), o0c), flush=True)
idx_2 = torch.full((T, K), -1, device=dev, dtype=torch.long); idx_2[:, 0] = 5; idx_2[:, 1] = 5
o0d = official(idx_2, seq_c_comp, sinks)
print("E0d two entries row5 :", cmpd(mine(idx_2, seq_c_comp, "cover_start", "direct"), o0d), flush=True)

# 有效倍数拟合: E0b(512x row0) 官方输出 vs 我的不同倍数 m
import torch.nn.functional as Fn
def mine_mult(idx, m, row=0):
    """cmp 条目有效倍数 m: softmax 里 cmp 总权重 = m * e^{s0}"""
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
    kv0 = cmp_kv.reshape(-1, D)[row].float()
    s0 = torch.einsum("thd,d->th", q.float(), kv0) * scale   # [T,H]
    logw_o = s_o
    logw_c = (s0 + torch.log(torch.tensor(float(m), device=dev)))[:, :, None]  # [T,H,1]
    m_o = logw_o.max(dim=2, keepdim=True).values
    mx = torch.maximum(m_o, logw_c.max(dim=2, keepdim=True).values)            # [T,H,1]
    Zo = (logw_o - mx).exp().sum(2)                                            # [T,H]
    Zc = (logw_c - mx).exp().squeeze(2)                                        # [T,H]
    p_c = Zc / (Zo + Zc)
    w_o = (logw_o - mx).exp() / (Zo + Zc)[:, :, None]     # [T,H,W]
    out = p_c[..., None] * kv0[None, None, :] + torch.einsum("thw,twd->thd", w_o, kv_o.float())
    return out.to(torch.bfloat16)

# E0e: 256x row0 + 256x row5 ; E0g: 300x row0 + 212 invalid
idx_e = torch.full((T, K), -1, device=dev, dtype=torch.long)
idx_e[:, :256] = 0; idx_e[:, 256:] = 5
o0e = official(idx_e, seq_c_comp, sinks)
print("E0e half0-half5 vs naive:", cmpd(mine(idx_e, seq_c_comp, "cover_start", "direct"), o0e), flush=True)
idx_g = torch.full((T, K), -1, device=dev, dtype=torch.long)
idx_g[:, :300] = 0
o0g = official(idx_g, seq_c_comp, sinks)
print("E0g 300x row0:")
for m in (150, 256, 300):
    print(f"   mult={m}:", cmpd(mine_mult(idx_g, m), o0g), flush=True)

# E0i: 仅有的有效条目放在位置 300/301 (前缀 ~254 之外)
idx_i = torch.full((T, K), -1, device=dev, dtype=torch.long)
idx_i[:, 300] = 3; idx_i[:, 301] = 3
o0i = official(idx_i, seq_c_comp, sinks)
print("E0i entries@300 (beyond prefix):", flush=True)
print("   vs naive m=2 :", cmpd(mine(idx_i, seq_c_comp, "cover_start", "direct"), o0i), flush=True)
print("   vs none(m=0) :", cmpd(mine(idx_none, seq_c_comp, "cover_start", "direct"), o0i), flush=True)
# 前缀版组合: 只用前 ceil((pos+1)/ratio) 个条目
def mine_prefix(idx, seq_c):
    out = mine(idx, seq_c, "cover_start", "direct")
    # 手动实现: 屏蔽 entry 位次 >= ceil((pos+1)/ratio)
    toks = torch.arange(T, device=dev)
    pos = (L - T) + toks
    nvis = ((pos + 1) + RATIO - 1) // RATIO
    kvis = torch.arange(K, device=dev)[None, :] < nvis[:, None]
    idx2 = torch.where(kvis, idx, torch.full_like(idx, -1))
    return mine(idx2, seq_c, "cover_start", "direct")
print("E1 prefix variant:", cmpd(mine_prefix(idx_c, seq_c_comp), o), flush=True)
print("E0b prefix variant:", cmpd(mine_prefix(idx_zero, seq_c_comp), o0b), flush=True)

# 边界变体扫描: nvis = floor((p+1)/r) | ceil((p+1)/r) | p//r+1 | (p+1)//r
def mine_prefix2(idx, seq_c, bound):
    toks = torch.arange(T, device=dev)
    pos = (L - T) + toks
    if bound == "floor":
        nvis = (pos + 1) // RATIO
    elif bound == "ceil":
        nvis = ((pos + 1) + RATIO - 1) // RATIO
    elif bound == "p1":
        nvis = pos // RATIO + 1
    kvis = torch.arange(K, device=dev)[None, :] < nvis[:, None]
    idx2 = torch.where(kvis, idx, torch.full_like(idx, -1))
    return mine(idx2, seq_c, "cover_start", "direct")
for b in ("floor", "ceil", "p1"):
    print(f"  E1 bound={b}:", cmpd(mine_prefix2(idx_c, seq_c_comp, b), o), flush=True)
    print(f"  E0b bound={b}:", cmpd(mine_prefix2(idx_zero, seq_c_comp, b), o0b), flush=True)

for m in (1, 2, 4, 16, 64, 128, 256, 512):
    print(f"  E0b mult={m:4d}:", cmpd(mine_mult(idx_zero, m), o0b), flush=True)

# ---------- 实验 4: 去重聚合变体 ----------
print("E4 dedup variants:", flush=True)
print("  E0b amax:", cmpd(mine_dedup(idx_zero, seq_c_comp, "amax"), o0b), flush=True)
print("  E0b mean:", cmpd(mine_dedup(idx_zero, seq_c_comp, "mean"), o0b), flush=True)
print("  E1  amax:", cmpd(mine_dedup(idx_c, seq_c_comp, "amax"), o), flush=True)
print("  E1  mean:", cmpd(mine_dedup(idx_c, seq_c_comp, "mean"), o), flush=True)

# ---------- 实验 2: 引擎 decode 喂法 (原始空间索引 0..511, cseq=512 原始单位) ----------
idx_o = torch.randint(0, L, (T, K), device=dev)           # [0,512) 原始位置
seq_c_orig = torch.tensor([L], dtype=torch.int32, device=dev)
cmp_bt_wide = torch.zeros(1, 68, dtype=torch.int32, device=dev)
cmp_bt_wide[0, :NB_C] = torch.arange(NB_C, dtype=torch.int32, device=dev)
cmp_bt = cmp_bt_wide  # E2/E3 起用宽表(零块=引擎的垃圾块0现场)
o2 = official(idx_o, seq_c_orig, sinks)
print("E2 engine-decode style (idx<L, cseq=L):", flush=True)
print("  direct/cover_start/rowcap :", cmpd(mine(idx_o, seq_c_orig, "cover_start", "direct"), o2), flush=True)
print("  direct/cover_start/norowcap:", cmpd(mine(idx_o, seq_c_orig, "cover_start", "direct", rowcap=False), o2), flush=True)
print("  direct/exact/rowcap       :", cmpd(mine(idx_o, seq_c_orig, "exact", "direct"), o2), flush=True)
print("  div2/cover_start/rowcap   :", cmpd(mine(idx_o, seq_c_orig, "cover_start", "div2"), o2), flush=True)
print("  direct/none/rowcap        :", cmpd(mine(idx_o, seq_c_orig, "none", "direct"), o2), flush=True)

# ---------- 实验 3: 索引原始空间 + cseq 压缩单位 ----------
o3 = official(idx_o, seq_c_comp, sinks)
print("E3 idx orig-space + cseq compressed:", flush=True)
print("  direct/cover_start/rowcap :", cmpd(mine(idx_o, seq_c_comp, "cover_start", "direct"), o3), flush=True)
print("  div2/cover_start          :", cmpd(mine(idx_o, seq_c_comp, "cover_start", "div2"), o3), flush=True)
print("DONE", flush=True)
