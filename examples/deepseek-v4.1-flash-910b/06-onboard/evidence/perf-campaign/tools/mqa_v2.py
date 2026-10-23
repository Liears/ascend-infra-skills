"""精简 MQA 参考：4 kernel（gather + bcast-matmul + softmax + matmul），全程 bf16。"""
import torch, torch_npu, time
dev = "npu:0"
torch.manual_seed(0)
T, H, D, SEL, BLOCK, NB = 6, 8, 512, 640, 128, 512
SCALE = D ** -0.5
q = torch.randn(H, T, D, dtype=torch.bfloat16, device=dev)        # [H,T,D]
kv_pool = torch.randn(NB * BLOCK, D, dtype=torch.bfloat16, device=dev)
flat = torch.randint(0, NB * BLOCK, (SEL,), dtype=torch.int64, device=dev)

def v2():
    kv = kv_pool.index_select(0, flat)               # [SEL, D]
    s = torch.matmul(q, kv.transpose(0, 1)) * SCALE  # [H,T,SEL]  (kv 广播, 无复制)
    a = torch.softmax(s, dim=-1)                     # bf16 softmax
    return torch.matmul(a, kv)                       # [H,T,SEL]@[SEL,D] = [H,T,D]

v2(); torch.npu.synchronize()
for _ in range(30): v2()
torch.npu.synchronize()
N = 300; t0 = time.perf_counter()
for _ in range(N): v2()
torch.npu.synchronize()
dt = (time.perf_counter() - t0) / N * 1e6
print(f"精简 4-kernel 版: {dt:.1f} us/次  (官方 170, 朴素组合 493)")
