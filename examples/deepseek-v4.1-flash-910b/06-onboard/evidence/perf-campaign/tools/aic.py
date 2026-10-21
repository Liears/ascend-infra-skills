#!/usr/bin/env python3
"""真实 AICore 利用率：每卡一个 AI Core 芯片 = (npu i, chip 0)。
npu-smi watch 不会自己退出，必须 kill；超时后仍可能无输出，故用 Popen + 定时杀。"""
import subprocess, sys, time, statistics, threading

def one(npu, dur):
    p = subprocess.Popen(["npu-smi","info","watch","-i",str(npu),"-c","0","-d","1","-s","a"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    time.sleep(dur)
    p.kill()
    try:
        out, _ = p.communicate(timeout=5)
    except Exception:
        out = ""
    vals=[]
    for line in (out or "").splitlines()[1:]:
        f=line.split()
        if len(f)>=3:
            try: vals.append(float(f[2]))
            except Exception: pass
    return vals

if __name__=="__main__":
    dur=int(sys.argv[1]) if len(sys.argv)>1 else 5
    res={}
    ths=[threading.Thread(target=lambda n=n: res.__setitem__(n, one(n,dur))) for n in range(8)]
    [t.start() for t in ths]; [t.join() for t in ths]
    allv=[]
    for n in range(8):
        v=res.get(n) or []
        if v:
            print(f"  npu{n}: mean={statistics.mean(v):5.1f}% max={max(v):5.0f}% n={len(v)}")
            allv+=v
    print(f"\n== all: mean={statistics.mean(allv):5.1f}% median={statistics.median(allv):5.1f}% max={max(allv):5.0f}% n={len(allv)}" if allv else "no samples")
