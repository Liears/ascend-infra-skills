"""910B3 屋顶线：Fast 稀疏 decode 核（方法抄自 FlashMLA 官方深潜文档，换 910B3 参数）。

硬件参数来源（权威）：CANN 9.1.0 data/platform_config/Ascend910B3.ini
  ai_core_cnt=20 (cube), vector_core_cnt=40, cube_freq=1800MHz,
  l0_a/b=64KB, l0_c=128KB, l1=512KB, ub=192KB, l2=192MB。
吞吐推定：16x16x16 fractal MMA/cycle/core = 4096 FMA/cycle = 8192 FLOP/cycle
  -> 单核 14.75 TFLOPS，20 核 295 TFLOPS（与 910B3 公称 ~280-313T 相符）。
访存口径：HBM ~400GB/s 聚合（910B3 公称），L2 聚合估 ~2TB/s（无官方数，敏感性分析）。
"""
import itertools

FREQ = 1.8e9
CORES = 20
PER_CORE_TFLOPS = 8192 * FREQ / 1e12          # 14.75
PEAK_TFLOPS = PER_CORE_TFLOPS * CORES          # 295
HBM_GBPS = 400.0
L2_GBPS = 2000.0                              # 估计值，敏感性分析用

# Fast 核工作负载（每 q token 一个核对）
H, COLS, D = 16, 640, 512                     # 16 head，640 列（W=128 窗口 + K=512 topk），D=512
FLOP_PER_TOKEN = 2 * H * COLS * D * 2          # MM1 + MM2
KV_BYTES_PER_TOKEN = COLS * D * 2             # bf16 KV 读取
RESCALE_OVH = 1.35                            # 经验余量：NZ 转换/softmax/rescale/发射气泡

print(f"peak = {PEAK_TFLOPS:.0f} TFLOPS aggregate, {PER_CORE_TFLOPS:.2f} TFLOPS/core\n")

for T in (6, 12, 18, 24, 32):
    cores_used = min(T, CORES)
    # 计算：每 token 独占一核对，溢出部分串行
    tokens_per_core = -(-T // cores_used)
    t_cube = tokens_per_core * FLOP_PER_TOKEN / (PER_CORE_TFLOPS * 1e12) * 1e6 * RESCALE_OVH
    # 访存：聚合读 T*640KB，L2 命中率 h
    kv_mb = T * KV_BYTES_PER_TOKEN / 1e6
    for l2_hit in (0.0, 0.8):
        bw = l2_hit * L2_GBPS + (1 - l2_hit) * HBM_GBPS
        t_mem = T * KV_BYTES_PER_TOKEN / (bw * 1e9) * 1e6
        print(f"T={T:2d} ({cores_used:2d} 核, {tokens_per_core} tok/核): "
              f"cube={t_cube:5.1f}us  mem(L2hit={l2_hit:.0%})={t_mem:5.1f}us  "
              f"bound={'mem' if t_mem > t_cube else 'cube'}  kv={kv_mb:.1f}MB")

print("""
官方对照（µbench NPUGraph 内）: T=6/12/18/24 -> 42.4/44.6/46.5/61.3us
SEQ 2048->65536 恒 41us => 官方核的时间几乎全是固定机器开销，不是吞吐。
=> Fast 核对手不是算力或带宽，是 41us 里的机器；预算 10-15us 时 T=6 全 L2 命中约 5-8us。
""")
