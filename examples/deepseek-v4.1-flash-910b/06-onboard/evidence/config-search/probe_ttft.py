#!/usr/bin/env python3
"""固定负载 TTFT 探针：同一 prompt 文件，流式请求，测首 token 时间。"""
import json, sys, time, urllib.request

base, prompt_file, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3
body = json.load(open(prompt_file))
ttfts = []
for i in range(n):
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/completions",
        data=json.dumps({
            "model": "deepseek-v41", "prompt": body["prompt"],
            "max_tokens": 32, "temperature": 0, "stream": True,
        }).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; total = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.startswith(b"data:") and b'"text"' in line:
                if first is None: first = time.time() - t0
                total += 1
            if total > 40: break
    ttfts.append(first)
    print(f"  run{i+1}: TTFT {first:.3f}s")
ttfts.sort()
print(f"RESULT ttft_median={ttfts[len(ttfts)//2]:.3f} n={n} all={[round(t,3) for t in ttfts]}")
