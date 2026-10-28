import pathlib
P = pathlib.Path("/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py")
src = P.read_text()
assert "_torch_smla41" not in src, "already patched"

helper = '''
# ==== [V41-T41 v17] 终局语义(引擎尺度离线拟合 med=0.00006 big=0%):
# decode(cseq=原始单位): 行号=索引直用, 全部条目(idx>=0), 无因果过滤
#   (cseq 把官方 visible 机制撑满); prefill(cseq=压缩单位): v9 语义
#   (c=idx, c*ratio<=pos, c<cseq)。检测: cseq*ratio > sl+ratio ⇒ 原始单位。
# /tmp/v41_smla41 = "1mv": decode(T<=64) 返回修复版; 图内双算 official 累计 diff。
_T41_STATE = {"mode": "", "mtime": -1.0, "checked": 0.0, "ng": 0, "last_print": 0.0}
_T41_ACC = None


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


def _t41_acc_new(dev):
    z = lambda: torch.zeros((), device=dev, dtype=torch.float32)
    return {"maxr": z(), "sumr": z(), "bigr": z(), "nelem": z(), "nan": z(),
            "idx_max": z(), "cseq_max": z(),
            "lctr": torch.zeros((), device=dev, dtype=torch.int64)}


def _t41_acc(dev):
    global _T41_ACC
    if _T41_ACC is None or _T41_ACC["maxr"].device != dev:
        _T41_ACC = _t41_acc_new(dev)
    return _T41_ACC


def _torch_smla41(q, ori_kv, cmp_kv, ori_bt, cmp_bt, cu_q, seq_lens,
                  cmp_seq_lens, cmp_idx, sinks, scale, win_left, ratio, block_size=128):
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
    cseq_row = torch.index_select(cmp_seq_lens.long(), 0, req)      # [T]
    # 调用级空间检测: cseq*ratio > sl+ratio ⇒ cseq/索引为原始单位(decode 图路径)
    _cmax = cmp_seq_lens.long().max()
    _slmax = seq_lens.long().max()
    is_orig = _cmax * ratio > _slmax + ratio
    # decode(原始空间): 行号=索引直用 + 条目位置前缀 (pos+1)//ratio (离线拟合实证)
    # prefill(压缩空间): 行号=索引即压缩位置, 全部条目 (v9 引擎内对拍逐位一致)
    c = csafe                                           # 行号=索引直用(两空间同式)
    # 压缩空间(prefill): v9 语义; 原始空间(decode): 仅 idx>=0 (无因果/无前缀)
    cov0 = csafe * ratio
    cvalid_c = cvalid & (cov0 <= pos[:, None]) & (c < cseq_row[:, None])
    cvalid = torch.where(is_orig, cvalid, cvalid_c)
    cblk = torch.div(c, block_size, rounding_mode="floor").clamp(max=cmp_bt.shape[1] - 1)
    coff = c - cblk * block_size
    cbt_v = torch.gather(torch.index_select(cmp_bt.long(), 0, req), 1, cblk)
    cvalid = cvalid & (cbt_v >= 0)
    cslot = (cbt_v * block_size + coff).clamp(min=0, max=c_rows - 1)
    kv_c = cflat.index_select(0, cslot.reshape(-1)).reshape(T, K, D)
    s_c = torch.bmm(q, kv_c.transpose(1, 2)) * scale
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
        _t41_cap = torch.npu.is_current_stream_capturing()
        if _t41m == "1mv" and q.shape[0] <= 64 and cmp_indices is not None:
            try:
                _t41a = _t41_acc(q.device)
                _mine = _torch_smla41(
                    q, attn.dsa_attn.swa_cache_layer.kv_cache[0], source_cache,
                    ori_block_table, cmp_block_table, query_start_loc, seq_lens,
                    cmp_seq_lens, cmp_indices, attn.attn_sink, attn.softmax_scale,
                    attn.window_size - 1, ratio,
                )
                _off, _ = torch.ops._C_ascend.npu_sparse_flash_mla(
                    q,
                    ori_kv=attn.dsa_attn.swa_cache_layer.kv_cache[0],
                    cmp_kv=source_cache,
                    cmp_sparse_indices=cmp_indices,
                    ori_block_table=ori_block_table,
                    cmp_block_table=cmp_block_table,
                    cu_seqlens_q=query_start_loc,
                    seqused_ori_kv=seq_lens,
                    seqused_cmp_kv=cmp_seq_lens,
                    cmp_residual_kv=cmp_residual,
                    sinks=attn.attn_sink,
                    metadata=op_metadata,
                    softmax_scale=attn.softmax_scale,
                    cmp_ratio=ratio,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=attn.window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_BBND",
                    topk_value_mode=1,
                    return_softmax_lse=False,
                )
                _d = (_mine.float() - _off.float()).abs()
                _t41a["maxr"] = torch.maximum(_t41a["maxr"], _d.max())
                _t41a["sumr"] = _t41a["sumr"] + _d.sum()
                _t41a["bigr"] = _t41a["bigr"] + (_d > 0.01).float().sum()
                _t41a["nelem"] = _t41a["nelem"] + _d.numel()
                _t41a["nan"] = _t41a["nan"] + torch.isnan(_mine).float().sum()
                _t41a["nano"] = _t41a.get("nano", torch.zeros((), device=q.device, dtype=torch.float32)) + torch.isnan(_off).float().sum()
                _cidx = cmp_indices.reshape(q.shape[0], -1)
                _t41a["idx_max"] = torch.maximum(_t41a["idx_max"], _cidx.float().max())
                _t41a["cseq_max"] = torch.maximum(_t41a["cseq_max"], cmp_seq_lens.float().max())
                _t41a["lctr"] = _t41a["lctr"] + 1
                if (not _t41_cap) and _T41_STATE["ng"] < 3:
                    _T41_STATE["ng"] += 1
                    print(f"[T41-V22D] T={q.shape[0]} maxr={_d.max().item():.6f} "
                          f"idx_max={_cidx.float().max().item():.0f}", flush=True)
                return _mine
            except Exception as _e41:
                if _T41_STATE.get("err", 0) < 3:
                    _T41_STATE["err"] = _T41_STATE.get("err", 0) + 1
                    import traceback
                    print("[T41-ERR]", repr(_e41)[:250], flush=True)
                    traceback.print_exc()
        elif _t41m == "1mv" and q.shape[0] > 64 and not _t41_cap:
            import time as _t41t
            _now = _t41t.monotonic()
            if _T41_ACC is not None and _now - _T41_STATE["last_print"] > 2.0:
                _T41_STATE["last_print"] = _now
                try:
                    _a = _T41_ACC
                    if not _T41_STATE.get("dumped", False) and cmp_block_table is not None:
                        _T41_STATE["dumped"] = True
                        try:
                            _r = cmp_block_table[0]
                            _tail = _r[34:]
                            print(f"[T41-BTDUMP] cols={_r.shape[0]} "
                                  f"head34_max={int(_r[:34].max())} "
                                  f"tail_min={int(_tail.min())} tail_max={int(_tail.max())} "
                                  f"tail_neg={int((_tail < 0).sum())} "
                                  f"tail_zero={int((_tail == 0).sum())} "
                                  f"tail_uniq={int(_tail.unique().numel())} "
                                  f"tail_head12={_tail[:12].tolist()}", flush=True)
                        except Exception as _e41d:
                            print("[T41-BTDERR]", repr(_e41d)[:150], flush=True)
                    print(f"[T41-V22] lctr={int(_a['lctr'].item())} "
                          f"maxr={_a['maxr'].item():.6f} bigr={int(_a['bigr'].item())} "
                          f"nelem={int(_a['nelem'].item())} nan={int(_a['nan'].item())} nano={int(_a['nano'].item())} "
                          f"idx_max={_a['idx_max'].item():.0f} "
                          f"cseq_max={_a['cseq_max'].item():.0f}", flush=True)
                    for _k in _a:
                        if _k != "lctr":
                            _a[_k].zero_()
                except Exception as _e41:
                    print("[T41-RDERR]", repr(_e41)[:200], flush=True)
        output, _ = torch.ops._C_ascend.npu_sparse_flash_mla(
            q,'''
src = src.replace(call, new, 1)
P.write_text(src)
print("v22 1mv installed; markers:", src.count("_torch_smla41"))
