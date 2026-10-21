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
        if du<=0: continue
        rows.append((st, st+du, row["Name"], row.get("Accelerator Core","") or ""))
anchor=sorted(r[0] for r in rows if "HcPre" in r[2])
PS=40
nstep=len(anchor)//PS
lo,hi=anchor[PS], anchor[-PS]
sel=[r for r in rows if lo<=r[0]<hi]
nstep=max(1,len([a for a in anchor if lo<=a<hi])//PS)
span=(hi-lo)/1000.0
print(f"steps={nstep} span={span:.1f}ms step={span/nstep:.2f}ms")

def union(intervals):
    if not intervals: return 0.0
    iv=sorted(intervals); tot=0.0; cs,ce=iv[0]
    for s,e in iv[1:]:
        if s<=ce: ce=max(ce,e)
        else: tot+=ce-cs; cs,ce=s,e
    tot+=ce-cs
    return tot

AIC_CORES={"AI_CORE","MIX_AIC"}
AIV_CORES={"AI_VECTOR_CORE","MIX_AIV"}
comm=[(s,e) for s,e,n,c in sel if "COMMUNICATION" in c.upper()]
# communication ops may be tagged by name
comm=[(s,e) for s,e,n,c in sel if ("hcom" in n.lower() or "AivKernel" in n or "COMMUNICATION" in c.upper())]
aic=[(s,e) for s,e,n,c in sel if c in AIC_CORES]
aiv=[(s,e) for s,e,n,c in sel if c in AIV_CORES]
nA=sum(e-s for s,e in aic)/1000/nstep
nV=sum(e-s for s,e in aiv)/1000/nstep
uA=union(aic)/1000/nstep; uV=union(aiv)/1000/nstep; uC=union(comm)/1000/nstep
step=span/nstep
print(f"\n自身时长/步: AIC {nA:.2f}  AIV {nV:.2f}  COMM {uC:.2f}")
print(f"并集(去重叠)/步: AIC {uA:.2f} ({100*uA/step:.0f}%)  AIV {uV:.2f} ({100*uV/step:.0f}%)")

def inter(A,B):
    A=sorted(A); B=sorted(B); i=j=0; tot=0.0
    while i<len(A) and j<len(B):
        s=max(A[i][0],B[j][0]); e=min(A[i][1],B[j][1])
        if e>s: tot+=e-s
        if A[i][1]<B[j][1]: i+=1
        else: j+=1
    return tot
print(f"\n重叠: AIC∩AIV {inter(aic,aiv)/1000/nstep:.2f}  AIC∩COMM {inter(aic,comm)/1000/nstep:.2f}  AIV∩COMM {inter(aiv,comm)/1000/nstep:.2f} ms/步")
busy=union(aic+aiv)/1000/nstep
print(f"\nAIC∪AIV busy {busy:.2f} ms/步 -> 空闲 {step-busy:.2f} ms/步 ({100*(step-busy)/step:.0f}%)")
print(f"完美流水上界 = max(AIC,AIV) = {max(uA,uV):.2f} ms -> {step/max(uA,uV):.2f}x")
