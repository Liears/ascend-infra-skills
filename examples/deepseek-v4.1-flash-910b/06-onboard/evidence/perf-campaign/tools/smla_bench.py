"""最小 sparse_flash_mla 微基准：decode 形状 batch=1, seqlen=8192, topk=512, 8 heads(TP8)。"""
import torch, torch_npu, time, sys
import cann_ops_transformer as cot
ns = torch.ops.cann_ops_transformer

dev = "npu:0"
B, HQ, HKV, D = 1, 8, 1, 512
SEQ, TOPK, BLOCK = 8192, 512, 128
NB = SEQ // BLOCK
torch.manual_seed(0)

q = torch.randn(B, HQ, D, dtype=torch.bfloat16, device=dev)
ori_kv = torch.randn(NB, BLOCK, HKV, D, dtype=torch.bfloat16, device=dev)
bt = torch.arange(NB, dtype=torch.int32, device=dev).unsqueeze(0)
cu_q = torch.tensor([0, B], dtype=torch.int32, device=dev)
seqused = torch.tensor([SEQ], dtype=torch.int32, device=dev)

md = ns.sparse_flash_mla_metadata(
    num_heads_q=HQ, num_heads_kv=HKV, head_dim=D,
    cu_seqlens_q=cu_q, seqused_q=torch.tensor([1], dtype=torch.int32, device=dev),
    seqused_ori_kv=seqused, batch_size=B,
    max_seqlen_q=1, max_seqlen_ori_kv=SEQ,
    ori_topk=TOPK, cmp_ratio=1,
    ori_mask_mode=4, ori_win_left=127, ori_win_right=0,
    layout_q="TND", layout_kv="PA_BBND",
)
def run():
    return ns.sparse_flash_mla(
        q.reshape(1, HQ, D), ori_kv=ori_kv, ori_block_table=bt,
        cu_seqlens_q=cu_q, seqused_ori_kv=seqused,
        metadata=md, softmax_scale=D**-0.5,
        ori_mask_mode=4, ori_win_left=127, ori_win_right=0,
        layout_q="TND", layout_kv="PA_BBND",
    )
try:
    out = run(); torch.npu.synchronize()
except Exception as e:
    print("ERR:", str(e)[:300]); sys.exit(1)
for _ in range(20): run()
torch.npu.synchronize()
t0 = time.perf_counter()
N = 200
for _ in range(N): run()
torch.npu.synchronize()
dt = (time.perf_counter() - t0) / N * 1e6
print(f"OK: {dt:.1f} us/call  (CANN {torch.version.npu if hasattr(torch.version,'npu') else '?'} / {torch_npu.__version__})")
