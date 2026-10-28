"""捕获执行 vs eager: 我的组合在 NPU graph 捕获下结果是否改变。"""
import torch, torch_npu
import cann_ops_transformer
from cann_ops_transformer.ops import sparse_flash_mla as _smla_mod  # noqa

dev = "npu:0"
torch.manual_seed(3)
H, D, BLOCK = 8, 512, 128
RATIO, L, T, K, W = 2, 8676, 6, 512, 128
NB_C, BTW = 900, 256

q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev) * 0.3
pool = torch.randn(NB_C, BLOCK, 1, D, dtype=torch.bfloat16, device=dev)
ori_kv = torch.randn(80, BLOCK, 1, D, dtype=torch.bfloat16, device=dev)
ori_bt = torch.zeros(1, BTW, dtype=torch.int32, device=dev); ori_bt[0, :80] = torch.arange(80)
bt2 = torch.zeros(1, BTW, dtype=torch.int32, device=dev)
bt2[0, :34] = torch.arange(820, 854)
for c in range(34, 46): bt2[0, c] = 818 + (c - 34)
cu_q = torch.tensor([0, T], dtype=torch.int32, device=dev)
seq_o = torch.tensor([L], dtype=torch.int32, device=dev)
seq_c = torch.tensor([L], dtype=torch.int32, device=dev)
scale = D ** -0.5
sinks = (torch.randn(H, device=dev) * 0.1).float()
idx = torch.randint(0, L, (T, 1, K), device=dev, dtype=torch.int32)  # 图输入 buffer


def my_comp(q, ori_kv, pool, ori_bt, bt2, cu_q, seq_o, seq_c, idx, sinks, scale):
    """v18 组合(引擎版语义: 全条目 + cbt>=0 + isfinite + V清零)。"""
    T, H, D = q.shape
    K = idx.shape[-1]
    toks = torch.arange(T, device=dev)
    pos = (seq_o[0] - T) + toks
    oflat = ori_kv.reshape(-1, D)
    cols = pos[:, None] - (W - 1) + torch.arange(W, device=dev)[None, :]
    oval = cols >= 0
    cc = cols.clamp(min=0)
    oblk = torch.div(cc, BLOCK, rounding_mode="floor")
    oslot = torch.gather(ori_bt.long()[0][None].expand(T, -1), 1, oblk) * BLOCK + (cc - oblk * BLOCK)
    kv_o = oflat.index_select(0, oslot.reshape(-1)).reshape(T, W, D)
    s_o = torch.bmm(q, kv_o.transpose(1, 2)) * scale

    cidx = idx.reshape(T, K).long()
    cvalid = cidx >= 0
    csafe = cidx.clamp(min=0)
    cblk = torch.div(csafe, BLOCK, rounding_mode="floor").clamp(max=bt2.shape[1] - 1)
    cbt_v = torch.gather(bt2.long()[0][None].expand(T, -1), 1, cblk)
    cvalid = cvalid & (cbt_v >= 0)
    cslot = (cbt_v * BLOCK + (csafe - cblk * BLOCK)).clamp(min=0, max=pool.reshape(-1, D).shape[0] - 1)
    kv_c = pool.reshape(-1, D).index_select(0, cslot.reshape(-1)).reshape(T, K, D)
    s_c = torch.bmm(q, kv_c.transpose(1, 2)) * scale
    fin = torch.isfinite(s_c).all(dim=1)
    cvalid = cvalid & fin
    kv_c = kv_c.masked_fill(~cvalid[:, :, None], 0)
    s = torch.cat([s_o, torch.nan_to_num(s_c, nan=-1e30).to(s_o.dtype)], dim=2)
    val = torch.cat([oval[:, None, :], cvalid[:, None, :]], dim=2)
    s = s.masked_fill(~val, -1e4)
    s = s + sinks[None, :, None].to(s.dtype)
    a = torch.softmax(s, dim=-1)
    return torch.bmm(a, torch.cat([kv_o, kv_c], dim=1))


# eager 参考
ref = my_comp(q, ori_kv, pool, ori_bt, bt2, cu_q, seq_o, seq_c, idx, sinks, scale).clone()

# NPU graph 捕获
g = torch.npu.NPUGraph()
s = torch.npu.Stream()
s.wait_stream(torch.npu.current_stream())
with torch.npu.stream(s):
    for _ in range(2):
        my_comp(q, ori_kv, pool, ori_bt, bt2, cu_q, seq_o, seq_c, idx, sinks, scale)
torch.npu.current_stream().wait_stream(s)
out_g = None
with torch.npu.graph(g):
    out_g = my_comp(q, ori_kv, pool, ori_bt, bt2, cu_q, seq_o, seq_c, idx, sinks, scale)
g.replay()
torch.npu.synchronize()
d = (out_g.float() - ref.float()).abs()
print(f"capture-vs-eager: med={d.median().item():.6f} max={d.max().item():.5f} "
      f"big%={(d > 0.01).float().mean().item()*100:.1f}", flush=True)
# 刷新 idx 后 replay(数据通过 buffer 传递)
idx.copy_(torch.randint(0, L, (T, 1, K), device=dev, dtype=torch.int32))
ref2 = my_comp(q, ori_kv, pool, ori_bt, bt2, cu_q, seq_o, seq_c, idx, sinks, scale).clone()
g.replay()
torch.npu.synchronize()
d2 = (out_g.float() - ref2.float()).abs()
print(f"capture-vs-eager(new idx): med={d2.median().item():.6f} max={d2.max().item():.5f} "
      f"big%={(d2 > 0.01).float().mean().item()*100:.1f}", flush=True)
print("DONE", flush=True)
