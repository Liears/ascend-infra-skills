import csv, sys
def sf(v):
    try: return float(v)
    except Exception: return 0.0
P=sys.argv[1]
rows=[]
with open(P,newline="") as fh:
    for row in csv.DictReader(fh):
        st=sf(row["Start Time(us)"]); du=sf(row["Duration(us)"])
        if du<=0: continue
        rows.append(dict(n=row["Name"],c=row.get("Accelerator Core","") or "",st=st,
            aic=sf(row.get("aicore_time(us)")),mac=sf(row.get("aic_mac_time(us)")),
            mte1=sf(row.get("aic_mte1_time(us)")),mte2=sf(row.get("aic_mte2_time(us)")),
            fixp=sf(row.get("aic_fixpipe_time(us)")),scal=sf(row.get("aic_scalar_time(us)")),
            aiv=sf(row.get("aiv_time(us)"))))
anchor=sorted(r["st"] for r in rows if "HcPre" in r["n"]); PS=40
lo,hi=anchor[PS],anchor[-PS]
sel=[r for r in rows if lo<=r["st"]<hi]
nstep=max(1,len([a for a in anchor if lo<=a<hi])//PS)
sums={k:sum(r[k] for r in sel)/1000/nstep for k in ("aic","mac","mte1","mte2","fixp","scal","aiv")}
step=(hi-lo)/1000/nstep
print(f"step={step:.2f} ms  steps={nstep}\n")
print("AIC core-time 分解（累计, ms/步）:")
for k,lab in (("mac","MAC 乘加"),("mte2","MTE2 HBM→L1"),("mte1","MTE1 L1→L0"),("fixp","FIXPIPE"),("scal","SCALAR")):
    v=sums[k]; print(f"  {lab:<16}{v:7.2f}  {100*v/sums['aic'] if sums['aic'] else 0:5.1f}% of aicore_time")
print(f"  {'aicore_time 总':<16}{sums['aic']:7.2f}")
# 判定: cube 是否算力受限 => mac 时间 / aicore 时间
r = sums['mac']/sums['aic'] if sums['aic'] else 0
print(f"\nMAC 占比 {100*r:.1f}% => " + ("算力受限（cube 真在算）" if r>0.5 else "**搬运受限（cube 大部分时间在等数据）**"))
