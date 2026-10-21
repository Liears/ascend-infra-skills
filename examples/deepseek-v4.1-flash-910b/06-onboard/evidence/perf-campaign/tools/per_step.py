import csv, sys
from collections import defaultdict
def sf(v):
    try: return float(v)
    except Exception: return 0.0
P=sys.argv[1]
rows=[]
with open(P,newline="") as fh:
    for row in csv.DictReader(fh):
        st=sf(row["Start Time(us)"]); du=sf(row["Duration(us)"])
        rows.append(dict(name=row["Name"],core=row.get("Accelerator Core",""),st=st,dur=du,
            aic=sf(row.get("aicore_time(us)")),mac=sf(row.get("aic_mac_time(us)")),
            mte1=sf(row.get("aic_mte1_time(us)")),mte2=sf(row.get("aic_mte2_time(us)")),
            fixp=sf(row.get("aic_fixpipe_time(us)")),scal=sf(row.get("aic_scalar_time(us)")),
            aiv=sf(row.get("aiv_time(us)")),vec=sf(row.get("aiv_vec_time(us)")),
            aivmte2=sf(row.get("aiv_mte2_time(us)")),aivmte3=sf(row.get("aiv_mte3_time(us)")),
            aivscal=sf(row.get("aiv_scalar_time(us)"))))
anchor=[x["st"] for x in rows if "HcPre" in x["name"]]
anchor.sort()
# HcPre fires 40x per step (40 layers) -> step boundaries every 40 anchors
per_step_anchors=40
steps=len(anchor)//per_step_anchors
lo,hi=anchor[per_step_anchors], anchor[-per_step_anchors]
sel=[x for x in rows if lo<=x["st"]<hi]
nstep=max(1,(len([a for a in anchor if lo<=a<hi]))//per_step_anchors)
print(f"steps={nstep}  window={(hi-lo)/1e6:.2f}s  profile step={(hi-lo)/1000/nstep:.2f} ms")
g=defaultdict(lambda:[0,0.0,0.0,0.0,0.0,0.0])
for x in sel:
    k=x["name"][:46]; a=g[k]
    a[0]+=1; a[1]+=x["dur"]; a[2]+=x["aic"]; a[3]+=x["aiv"]; a[4]+=x["mte2"]; a[5]+=x["aivmte2"]
print(f"\n{'op':<48}{'次/步':>7}{'ms/步':>8}{'AIC':>8}{'AIV':>8}{'MTE2':>8}")
taic=taiv=tmte2=0
for k,(n,d,a,v,m,av) in sorted(g.items(),key=lambda kv:-kv[1][1])[:20]:
    print(f"{k:<48}{n/nstep:>7.1f}{d/1000/nstep:>8.2f}{a/1000/nstep:>8.2f}{v/1000/nstep:>8.2f}{m/1000/nstep:>8.2f}")
    taic+=a; taiv+=v; tmte2+=m
print(f"\n总和(前20): AIC {taic/1000/nstep:.2f} ms/步, AIV {taiv/1000/nstep:.2f} ms/步")
allA=sum(x["aic"] for x in sel)/1000/nstep; allV=sum(x["aiv"] for x in sel)/1000/nstep
allM=sum(x["mte2"] for x in sel)/1000/nstep; allAM=sum(x["aivmte2"] for x in sel)/1000/nstep
print(f"全量: AIC busy {allA:.2f} ms/步 | AIV busy {allV:.2f} ms/步 | AIC-MTE2 {allM:.2f} | AIV-MTE2 {allAM:.2f}")
print(f"→ AIC+AIV = {allA+allV:.2f} ms/步 vs profile 步长 {(hi-lo)/1000/nstep:.2f} ms")
