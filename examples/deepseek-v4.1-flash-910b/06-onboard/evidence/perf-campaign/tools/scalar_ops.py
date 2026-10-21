import csv, sys
from collections import defaultdict
def sf(v):
    try: return float(v)
    except Exception: return 0.0
P=sys.argv[1]
rows=[]
with open(P,newline="") as fh:
    for r in csv.DictReader(fh):
        st=sf(r["Start Time(us)"]); du=sf(r["Duration(us)"])
        if du<=0: continue
        aic=sf(r.get("aicore_time(us)"))
        rows.append(dict(n=r["Name"],c=r.get("Accelerator Core","") or "",st=st,du=du,aic=aic,
            scal=sf(r.get("aic_scalar_time(us)")),fixp=sf(r.get("aic_fixpipe_time(us)")),
            mte2=sf(r.get("aic_mte2_time(us)")),mac=sf(r.get("aic_mac_time(us)"))))
anchor=sorted(r["st"] for r in rows if "HcPre" in r["n"]);PS=40
lo,hi=anchor[PS],anchor[-PS]
sel=[r for r in rows if lo<=r["st"]<hi]
nstep=max(1,len([a for a in anchor if lo<=a<hi])//PS)
step=(hi-lo)/1000/nstep
g=defaultdict(lambda:[0,0.0,0.0,0.0,0.0,0.0,0.0])
for x in sel:
    k=x["n"][:44]; a=g[k]
    a[0]+=1;a[1]+=x["du"];a[2]+=x["aic"];a[3]+=x["scal"];a[4]+=x["fixp"];a[5]+=x["mte2"];a[6]+=x["mac"]
print(f"step={step:.2f}ms  nstep={nstep}")
print(f"\n{'op':<46}{'次/步':>7}{'SCALAR':>8}{'FIXPIPE':>8}{'MTE2':>8}{'MAC':>8}  (ms/步)")
ts=tf=0
for k,(n,d,a,s,f,m,mc) in sorted(g.items(),key=lambda kv:-kv[1][3])[:15]:
    print(f"{k:<46}{n/nstep:>7.1f}{s/1000/nstep:>8.2f}{f/1000/nstep:>8.2f}{m/1000/nstep:>8.2f}{mc/1000/nstep:>8.2f}")
    ts+=s; tf+=f
print(f"\nTop15 SCALAR {ts/1000/nstep:.2f} / FIXPIPE {tf/1000/nstep:.2f} ms/步")
A=sum(x["aic"] for x in sel)/1000/nstep
S=sum(x["scal"] for x in sel)/1000/nstep
F=sum(x["fixp"] for x in sel)/1000/nstep
M=sum(x["mac"] for x in sel)/1000/nstep
E=sum(x["mte2"] for x in sel)/1000/nstep
print(f"全量: AIC {A:.2f} | SCALAR {S:.2f} | FIXPIPE {F:.2f} | MAC {M:.2f} | MTE2 {E:.2f}")
print(f"SCALAR+FIXPIPE 合计 {S+F:.2f} ms = 步长的 {100*(S+F)/step:.1f}%")
