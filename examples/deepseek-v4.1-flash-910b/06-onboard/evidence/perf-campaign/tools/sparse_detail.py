import csv, sys
from collections import Counter
def sf(v):
    try: return float(v)
    except Exception: return 0.0
P=sys.argv[1]
shapes=Counter(); blocks=Counter(); n=0
tot=0.0; aic=0.0; scal=0.0; fixp=0.0; mte2=0.0; mac=0.0; mte1=0.0
with open(P,newline="") as fh:
    for r in csv.DictReader(fh):
        if "SparseFlashMla" not in r["Name"]: continue
        du=sf(r["Duration(us)"])
        n+=1; tot+=du
        aic+=sf(r.get("aicore_time(us)")); scal+=sf(r.get("aic_scalar_time(us)"))
        fixp+=sf(r.get("aic_fixpipe_time(us)")); mte2+=sf(r.get("aic_mte2_time(us)"))
        mac+=sf(r.get("aic_mac_time(us)")); mte1+=sf(r.get("aic_mte1_time(us)"))
        shapes[r.get("Input Shapes","")[:70]]+=1
        blocks[r.get("Block Num","")]+=1
print(f"SparseFlashMla n={n} total={tot/1000:.1f} ms  per-op median...")
print(f"  aicore={aic/1000:.1f} MAC={mac/1000:.1f} MTE1={mte1/1000:.1f} MTE2={mte2/1000:.1f} FIXPIPE={fixp/1000:.1f} SCALAR={scal/1000:.1f}")
print(f"  MAC 占 aicore 的 {100*mac/aic if aic else 0:.1f}%")
print(f"\nBlock Num 分布: {dict(blocks.most_common(5))}")
print(f"\nInput Shapes top5:")
for s,c in shapes.most_common(5): print(f"  n={c:5d}  {s}")
