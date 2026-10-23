"""把 PyTorch 4-kernel MQA 放进 NPU graph，测纯设备执行时间（与引擎内 170us 同口径）。"""
import torch, torch_npu, time
dev = "npu:0"; torch.manual_seed(0)
T, H, D, SEL, BLOCK, NB = 6, 8, 512, 640, 128, 512
SCALE = D ** -0.5
q = torch.randn(H, T, D, dtype=torch.bfloat16, device=dev)
kv_pool = torch.randn(NB * BLOCK, D, dtype=torch.bfloat16, device=dev)
flat = torch.randint(0, NB * BLOCK, (SEL,), dtype=torch.int64, device=dev)

def v2(q, flat):
    kv = kv_pool.index_select(0, flat)
    s = torch.matmul(q, kv.transpose(0, 1)) * SCALE
    a = torch.softmax(s, dim=-1)
    return torch.matmul(a, kv)

# 预热
for _ in range(5): v2(q, flat)
torch.npu.synchronize()

# NPU graph 捕获
try:
    g = torch_npu.npu.NPUGraph()
    s_opt = torch_npu.npu.Stream()
    s_opt.wait_stream(torch_npu.npu.current_stream())
    with torch_npu.npu.stream(s_opt):
        with torch_npu.npu.graph(g):
            out_static = v2(q, flat)
    torch_npu.npu.current_stream().wait_stream(s_opt)
    print("graph 捕获 OK")
except Exception as e:
    print("graph ERR:", str(e)[:250]); raise SystemExit(1)

for _ in range(10): g.replay()
torch.npu.synchronize()
N = 1000; t0 = time.perf_counter()
for _ in range(N): g.replay()
torch.npu.synchronize()
dt = (time.perf_counter() - t0) / N * 1e6
print(f"NPU-graph replay 的 4-kernel MQA: {dt:.1f} us/次  (引擎内 aclnnSparseFlashMla 设备时间 170us)")

# 对照：图外墙钟
t0 = time.perf_counter()
for _ in range(500): v2(q, flat)
torch.npu.synchronize()
print(f"图外墙钟对照: {(time.perf_counter()-t0)/500*1e6:.1f} us/次")
