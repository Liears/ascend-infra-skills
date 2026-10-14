# Open-source implementations: DeepSeek-V4.1-Flash on 910B

Queries issued (11): DeepSeek-V4.1-Flash 910B | DeepSeek-V4.1-Flash 910B3 | DeepSeek-V4.1-Flash 910b | DeepSeek-V4.1-Flash A2 | DeepSeek-V4.1-Flash Atlas 800 A2 | DeepSeek-V4.1-Flash ascend910b | deepseek-v4.1-flash 910B | DeepSeek-V4_1-Flash 910B | DeepSeek-V4-1-Flash 910B | DeepSeekV41Flash 910B | DeepSeek-V4.1-Flash ascend

3 candidate(s), rated by evidence in metadata:

## [verified] chiro2001/deepseek-v4.1-flash-ascend910B  (score 6)
- url: https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B
- source: github | lang: Python | stars: 24 | license: Apache-2.0
- pushed: 2026-10-13 | created: 2026-09-17
- desc: DeepSeek-V4.1-Flash W4A8 optimized vLLM deployment for Ascend 910B (A2) / 910C (A3): patches, one-command serve, self-check and acceptance
- evidence signals: acceptance/perf doc, quantization material

  -> `verify by reading the patch set and the acceptance doc`

## [claimed] qiuhanzhang616-arch/deepseek-v4.1-flash-w4a8-modelarts  (score 1)
- url: https://github.com/qiuhanzhang616-arch/deepseek-v4.1-flash-w4a8-modelarts
- source: github | lang: Shell | stars: 0 | license: NOASSERTION
- pushed: 2026-10-08 | created: 2026-10-08
- desc: W4A8 deployment guides and source scripts for DeepSeek-V4.1-Flash on Huawei Cloud ModelArts Standard / Ascend A2; historical C5 and DSpark5 profiles clearly separated
- evidence signals: quantization material

  -> `metadata shows no implementation evidence; treat as a lead only`

## [claimed] qiuhanzhang616-arch/deepseek-v4.1-flash-modelarts  (score 0)
- url: https://github.com/qiuhanzhang616-arch/deepseek-v4.1-flash-modelarts
- source: github | lang: Shell | stars: 0 | license: NOASSERTION
- pushed: 2026-10-05 | created: 2026-09-21
- desc: Reusable Huawei Cloud ModelArts deployment guide for DeepSeek-V4.1-Flash on 4x8 Ascend A2 (DP4/TP8/EP32, 1M context).

  -> `metadata shows no implementation evidence; treat as a lead only`

---
Rating is metadata-only triage. `verified` here means the repo advertises
implementation evidence (patches/acceptance/repro), NOT that it was run.
Audit the top candidate before citing it:

    git clone --depth 1 <url> /tmp/cand && \
        python3 scripts/audit_repo.py /tmp/cand
