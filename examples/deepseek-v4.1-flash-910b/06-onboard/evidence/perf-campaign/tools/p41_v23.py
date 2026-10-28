import pathlib
P = pathlib.Path("/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py")
src = P.read_text()
assert "_torch_smla41" not in src, "already patched"

helper = '''
# ==== [V41-T41 v23] 纯替换模式: 无双算(排除 metadata 单次消费嫌疑), 捕获期
# Python 分支选定 decode 语义(无数据依赖 where, 防静态折叠)。离线已证:
# decode=全条目/行号直用; prefill eager=v9 压缩语义。
# /tmp/v41_smla41 = "1": decode(T<=64) 用 torch 版。
_T41_STATE = {"mode": "", "mtime": -1.0, "checked": 0.0, "ng": 0}


def _t41_mode():
    import os as _o
    import time as _t
    now = _t.monotonic()
    c = _T41_STATE
    if now - c["checked"] < 0.25:
        return c["mode"]
    c["checked"] = now
    try:
        st = _o.stat("/tmp/v41_smla41")
        if st.st_mtime != c["mtime"]:
            c["mtime"] = st.st_mtime
            with open("/tmp/v41_smla41") as fh:
                c["mode"] = fh.read().strip()
    except OSError:
        c.update({"mode": "", "mtime": -1.0})
    return c["mode"]


def _torch_smla41(q, ori_kv, cmp_kv, ori_bt, cmp_bt, cu_q, seq_lens,
                  cmp_seq_lens, cmp_idx, sinks, scale, win_left, ratio,
                  decode_semantics, block_size=128):
    T, H, D = q.shape
    R = seq_lens.shape[0]
    dev = q.device

    toks = torch.arange(T, device=dev)
    cu_start = cu_q[:R].long()
    cu_end = cu_q[1:R + 1].long()
    req = (((toks[None, :] >= cu_start[:, None]) &
            (toks[None, :] < cu_end[:, None])).to(torch.int64).sum(0) - 1).clamp(0, R - 1)
    base = seq_lens.long() - (cu_end - cu_start)
    pos = torch.gather(base, 0, req) + (toks - torch.gather(cu_start, 0, req))

    oflat = ori_kv.reshape(-1, D)
    o_rows = oflat.shape[0]
    Wd = win_left + 1
    cols = pos[:, None] - win_left + torch.arange(Wd, device=dev)[None, :]
    ovalid = cols >= 0
    cc = cols.clamp(min=0)
    oblk = torch.div(cc, block_size, rounding_mode="floor").clamp(max=ori_bt.shape[1] - 1)
    ooff = cc - oblk * block_size
    obt_v = torch.gather(torch.index_select(ori_bt.long(), 0, req), 1, oblk)
    ovalid = ovalid & (obt_v >= 0)
    oslot = (obt_v * block_size + ooff).clamp(min=0, max=o_rows - 1)
    kv_o = oflat.index_select(0, oslot.reshape(-1)).reshape(T, Wd, D)
    s_o = torch.bmm(q, kv_o.transpose(1, 2)) * scale

    cflat = cmp_kv.reshape(-1, D)
    c_rows = cflat.shape[0]
    K = cmp_idx.shape[-1]
    cidx = cmp_idx.reshape(T, K).long()
    cvalid = cidx >= 0
    csafe = cidx.clamp(min=0)
    if not decode_semantics:
        # prefill(eager): v9 语义 — 压缩行号 + 因果 c*ratio<=pos + 行上界
        cseq_row = torch.index_select(cmp_seq_lens.long(), 0, req)
        cvalid = cvalid & (csafe * ratio <= pos[:, None]) & (csafe < cseq_row[:, None])
    # decode(捕获): 全条目, 无因果 — 离线终局拟合 med=0.00000
    cblk = torch.div(csafe, block_size, rounding_mode="floor").clamp(max=cmp_bt.shape[1] - 1)
    coff = csafe - cblk * block_size
    cbt_v = torch.gather(torch.index_select(cmp_bt.long(), 0, req), 1, cblk)
    cvalid = cvalid & (cbt_v >= 0)
    cslot = (cbt_v * block_size + coff).clamp(min=0, max=c_rows - 1)
    kv_c = cflat.index_select(0, cslot.reshape(-1)).reshape(T, K, D)
    s_c = torch.bmm(q, kv_c.transpose(1, 2)) * scale
    fin = torch.isfinite(s_c)
    cvalid = cvalid & fin.all(dim=1)
    kv_c = kv_c.masked_fill(~cvalid[:, :, None], 0)
    s_c = torch.where(fin, s_c, torch.full_like(s_c, -1e4))

    s = torch.cat([s_o, s_c], dim=2)
    val = torch.cat([ovalid[:, None, :], cvalid[:, None, :]], dim=2)
    kv_cat = torch.cat([kv_o, kv_c], dim=1)
    s = s.masked_fill(~val, -1e4)
    if sinks is not None:
        s = s + sinks[None, :, None].to(s.dtype)
    a = torch.softmax(s, dim=-1)
    return torch.bmm(a, kv_cat)
# ==== [/V41-T41] ====
'''
anchor = "def pad_sparse_indices("
assert anchor in src
src = src.replace(anchor, helper + anchor, 1)

call = """        output, _ = torch.ops._C_ascend.npu_sparse_flash_mla(
            q,"""
assert call in src
new = '''        _t41m = _t41_mode()
        if _t41m == "1" and q.shape[0] <= 64 and cmp_indices is not None:
            try:
                _dec = torch.npu.is_current_stream_capturing()
                return _torch_smla41(
                    q, attn.dsa_attn.swa_cache_layer.kv_cache[0], source_cache,
                    ori_block_table, cmp_block_table, query_start_loc, seq_lens,
                    cmp_seq_lens, cmp_indices, attn.attn_sink, attn.softmax_scale,
                    attn.window_size - 1, ratio, _dec,
                )
            except Exception as _e41:
                if _T41_STATE.get("err", 0) < 3:
                    _T41_STATE["err"] = _T41_STATE.get("err", 0) + 1
                    import traceback
                    print("[T41-ERR]", repr(_e41)[:250], flush=True)
                    traceback.print_exc()
        output, _ = torch.ops._C_ascend.npu_sparse_flash_mla(
            q,'''
src = src.replace(call, new, 1)
P.write_text(src)
print("v23 pure-replacement installed; markers:", src.count("_torch_smla41"))
