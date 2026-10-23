"""PyTorch 参考实现：decode 稀疏 MQA attention（SparseFlashMla 的语义子集）。
shape 与真实调用一致：T=6 query, H=8 head(TP8), D=512, KV=640 选中(512 topk+128 滑窗)，
paged KV block=128, HKV=1（所有 head 共享同一份 KV —— MQA 形态）。
目的：测组合实现的每次调用开销，对比 aclnnSparseFlashMla 实测 170us。
"""
import torch, torch_npu, time, sys

dev = "npu:0"
torch.manual_seed(0)
T, H, D = 6, 8, 512
TOPK, WIN = 512, 128
SEL = TOPK + WIN
BLOCK, NB = 128, 512
SCALE = D ** -0.5

q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev)
kv_pool = torch.randn(NB, BLOCK, D, dtype=torch.bfloat16, device=dev)   # paged latent KV
# 选中索引：topk 512 + 滑窗 128，指向 (block, offset)
idx_blk = torch.randint(0, NB, (SEL,), dtype=torch.int64, device=dev)
idx_off = torch.randint(0, BLOCK, (SEL,), dtype=torch.int64, device=dev)
flat = idx_blk * BLOCK + idx_off     # 展平的全局 KV 行号

def ref_mqa():
    kv = kv_pool.reshape(-1, D).index_select(0, flat)          # [SEL, D] gather
    s = torch.einsum('thd,sd->hts', q.float(), kv.float()) * SCALE   # scores [H,T,SEL]
    a = s.softmax(-1).to(torch.bfloat16)
    return torch.einsum('hts,sd->thd', a, kv)                   # [T,H,D]

try:
    ref_mqa(); torch.npu.synchronize()
except Exception as e:
    print("ERR:", str(e)[:250]); sys.exit(1)
for _ in range(30): ref_mqa()
torch.npu.synchronize()
N = 300
t0 = time.perf_counter()
for _ in range(N): ref_mqa()
torch.npu.synchronize()
dt = (time.perf_counter() - t0) / N * 1e6
print(f"PyTorch 组合实现: {dt:.1f} us/次   (aclnnSparseFlashMla 引擎实测: 170 us)")
print(f"  组成: index_select + einsum(scores) + softmax + cast + einsum(out)")
