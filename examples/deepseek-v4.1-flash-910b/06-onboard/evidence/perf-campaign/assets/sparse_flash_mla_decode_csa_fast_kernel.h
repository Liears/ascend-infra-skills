/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_mla_decode_csa_fast_kernel.h
 * \brief Fast decode-only CSA kernel for DeepSeek V4.1 DSA on ascend910b (arch22).
 *
 * 架构（P12；旗标组语义经隔离实验梯确证，见 P12 实录）：
 *  - MIX_AIC_1_2：每个 AIC 配 2 个 AIV，三者构成 mode2 跨核旗标的"组"。
 *    双 AIV 都是真工人（官方同构）：AIV(sub) 负责 head 行 [8sub, 8sub+8) 与
 *    每列块的列 [64sub, 64sub+64)。AIV→AIC 旗标由两个 AIV 各 set 一次
 *    （fire=两者到齐=数据就绪）；AIC→AIV 旗标 AIC set 一次、双 AIV 各 wait。
 *    任何成员提前退出或单边 set 都会死锁/数据错（实证）。
 *  - 静态分派：token t 轮转 numAic 个核对；每 token 固定 640 列
 *    （W=128 窗口 + K=512 topk）= 5 个 BC=128 列块。
 *  - 分块 online softmax（统一公式：rowMax 初值 -FLT_MAX，免 chunk0 分支）。
 *  - AIC 高层 matmul::Matmul：MM1 = Q[16,512]xKV_c^T → wsMm1[16,BC] f32；
 *    MM2 = P[16,BC]xKV_c → wsMm2[16,512] f32。GM workspace 双缓冲中转。
 *  - 无效列：不搬 KV、行清零防 0xNaN、softmax 以 -FLT_MAX 屏蔽。
 */

#ifndef SPARSE_FLASH_MLA_DECODE_CSA_FAST_KERNEL_H
#define SPARSE_FLASH_MLA_DECODE_CSA_FAST_KERNEL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "sparse_flash_mla_decode_common_arch22.h"

using namespace AscendC;

namespace SMLADKernel {

constexpr uint32_t SMLAD_FAST_D = 512;                  // headDim
constexpr uint32_t SMLAD_FAST_H = 16;                   // q heads per token
constexpr uint32_t SMLAD_FAST_HH = 8;                   // 每 AIV 的 head 行数
constexpr uint32_t SMLAD_FAST_W = 128;                  // ori window cols（= 块大小，gate 保证）
constexpr uint32_t SMLAD_FAST_K = 512;                  // cmp topk cols
constexpr uint32_t SMLAD_FAST_COLS = 640;
constexpr uint32_t SMLAD_FAST_BC = 128;                 // 列块
constexpr uint32_t SMLAD_FAST_BH = 64;                  // 每 AIV 的列半宽
constexpr uint32_t SMLAD_FAST_NCHUNK = 5;
constexpr uint32_t SMLAD_FAST_ROW_E = SMLAD_FAST_H * SMLAD_FAST_BC;    // 2048
constexpr uint32_t SMLAD_FAST_ROWH_E = SMLAD_FAST_HH * SMLAD_FAST_BC;  // 1024
constexpr uint32_t SMLAD_FAST_ACC_E = SMLAD_FAST_H * SMLAD_FAST_D;     // 8192
constexpr uint32_t SMLAD_FAST_ACCH_E = SMLAD_FAST_HH * SMLAD_FAST_D;   // 4096
constexpr uint32_t SMLAD_FAST_KV_E = SMLAD_FAST_BC * SMLAD_FAST_D;     // 65536
constexpr uint32_t SMLAD_FAST_KVH_E = SMLAD_FAST_BH * SMLAD_FAST_D;    // 32768
constexpr float SMLAD_FAST_NEG_MAX = -3.402823466e+38F;               // -FLT_MAX

// 跨核旗标（tag 2..9；与本核内官方路径互斥不共存）
constexpr uint32_t SMLAD_FAST_F_KV_READY = 2;           // + slot(0/1)：双 AIV 已写 wsKv 各自半区
constexpr uint32_t SMLAD_FAST_F_MM1_DONE = 4;           // + slot：AIC 已写 wsMm1
constexpr uint32_t SMLAD_FAST_F_P_READY = 6;            // + slot：双 AIV 已写 wsP 各自半区
constexpr uint32_t SMLAD_FAST_F_MM2_DONE = 8;           // + slot：AIC 已写 wsMm2

template <typename SMLADT>
class SparseFlashMlaDecodeCsaFast {
public:
    using Q_T = typename SMLADT::queryType;
    using KV_T = typename SMLADT::kvType;
    using OUT_T = typename SMLADT::outputType;
    using T = float;

    __aicore__ inline SparseFlashMlaDecodeCsaFast() {}

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
                                __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                                __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedQ,
                                __gm__ uint8_t *seqUsedKV, __gm__ uint8_t *seqUsedCmpKV,
                                __gm__ uint8_t *cmpResidualKV, __gm__ uint8_t *oriTopkLength,
                                __gm__ uint8_t *cmpTopkLength, __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                const SparseFlashMlaDecodeTilingData *__restrict tiling, __gm__ uint8_t *gmTiling,
                                TPipe *tPipe);

    __aicore__ inline void Process();

private:
    using ATypeNd = matmul::MatmulType<TPosition::GM, CubeFormat::ND, Q_T, false>;
    using BTypeMm1 = matmul::MatmulType<TPosition::GM, CubeFormat::ND, KV_T, true>;    // KV^T
    using BTypeMm2 = matmul::MatmulType<TPosition::GM, CubeFormat::ND, KV_T, false>;
    using CTypeF32 = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    static constexpr MatmulConfig smladFastMmCfg{false, false, true, 0, 0, 0, false, false, false, false, false, 0,
                                                 0, 0, 0, 0, 0, 0, true};
    using Mm1T = matmul::MatmulImpl<ATypeNd, BTypeMm1, CTypeF32, CTypeF32, smladFastMmCfg>;
    using Mm2T = matmul::MatmulImpl<ATypeNd, BTypeMm2, CTypeF32, CTypeF32, smladFastMmCfg>;
    Mm1T mm1;
    Mm2T mm2;

    __aicore__ inline void RunCube(uint32_t token);
    __aicore__ inline void RunVector(uint32_t token);
    __aicore__ inline uint32_t ReqOfToken(uint32_t token);
    __aicore__ inline void GatherHalf(uint32_t token, uint32_t chunk);
    __aicore__ inline void SoftmaxHalf(uint32_t chunk);
    __aicore__ inline void Finalize(uint32_t token);
    __aicore__ inline float ScalarExp(float x);

    const SparseFlashMlaDecodeTilingData *__restrict tilingData = nullptr;
    TPipe *pipe = nullptr;
    uint32_t aiCoreIdx = 0;
    uint32_t subBlk = 0;         // 0/1：本 AIV 的子块号（行/列半区）
    uint32_t numAic = 1;
    uint32_t bSize = 1;
    float smScale = 1.0f;
    int64_t oriKvStride0 = 0;
    int64_t cmpKvStride0 = 0;
    uint32_t oriMaxBlk = 0;
    uint32_t cmpMaxBlk = 0;
    bool hasSinks = false;
    // 本核事件（FetchEventID 托管）
    event_t evM2V;   // MTE2→V（GM 读回→向量用）
    event_t evV3;    // V→MTE3（向量产出→拷出）
    event_t ev23;    // MTE2→MTE3（gather 入→拷出）
    event_t ev32;    // MTE3→MTE2（kv 拷出→mm2 读入复用）
    event_t evV2;    // V→MTE2（acc 用完→下块 gather）

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> oriKvGm;
    GlobalTensor<KV_T> cmpKvGm;
    GlobalTensor<int32_t> obtGm;
    GlobalTensor<int32_t> cbtGm;
    GlobalTensor<int32_t> topkGm;
    GlobalTensor<int32_t> cuQGm;
    GlobalTensor<int32_t> seqOriGm;
    GlobalTensor<int32_t> seqCmpGm;
    GlobalTensor<T> sinksGm;
    GlobalTensor<OUT_T> outGm;

    GlobalTensor<KV_T> wsKvGm;   // [T][2][BC*D]
    GlobalTensor<T> wsMm1Gm;     // [T][2][H*BC]
    GlobalTensor<KV_T> wsPGm;    // [T][2][H*BC]
    GlobalTensor<T> wsMm2Gm;     // [T][2][H*D]

    // AIV UB
    LocalTensor<KV_T> kvUb;      // 64KB：本子块 64 行 gather 中转 / mm2 读回复用
    LocalTensor<T> mm1Ub;        // 4KB+尾：mm1 读回 + softmax 工作区
    LocalTensor<KV_T> pUb;       // 2KB：P bf16（本子块 8 行）
    LocalTensor<T> accUb;        // 16KB（本子块 8 行）
    LocalTensor<T> maskUb;       // 64 f32：0 / -FLT_MAX
    LocalTensor<T> mask2dUb;     // 4KB：[8,128] 列屏蔽
    LocalTensor<T> shiftUb;      // 4KB：[8,128] 行移位（独立，防跨块污染——实证）
    LocalTensor<T> tmpUb;        // 64B：ScalarExp 通道
    LocalTensor<T> rowMaxUb;     // 8（mm1 尾部）
    LocalTensor<T> rowSumUb;
    LocalTensor<T> sinkUb;
};

// ============================================================ Init
template <typename SMLADT>
__aicore__ inline void SparseFlashMlaDecodeCsaFast<SMLADT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *oriSparseIndices,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
    __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKV, __gm__ uint8_t *seqUsedCmpKV,
    __gm__ uint8_t *cmpResidualKV, __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength,
    __gm__ uint8_t *sinks, __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, const SparseFlashMlaDecodeTilingData *__restrict tiling, __gm__ uint8_t *gmTiling,
    TPipe *tPipe)
{
    tilingData = tiling;
    pipe = tPipe;
    if ASCEND_IS_AIV {
        const uint32_t idx = GetBlockIdx();
        subBlk = idx & 1U;
        aiCoreIdx = idx / 2;
    } else {
        subBlk = 0;
        aiCoreIdx = GetBlockIdx();
    }
    numAic = tiling->fastParams.numAic;
    bSize = tiling->baseParams.batchSize;
    smScale = tiling->baseParams.softmaxScale;
    oriKvStride0 = tiling->baseParams.oriKvStride0;
    cmpKvStride0 = tiling->cmpParams.cmpKvStride0;
    oriMaxBlk = tiling->baseParams.oriMaxBlockNumPerBatch;
    cmpMaxBlk = tiling->cmpParams.cmpMaxBlockNumPerBatch;
    hasSinks = (sinks != nullptr);

    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    oriKvGm.SetGlobalBuffer((__gm__ KV_T *)oriKV);
    cmpKvGm.SetGlobalBuffer((__gm__ KV_T *)cmpKV);
    obtGm.SetGlobalBuffer((__gm__ int32_t *)oriBlockTable);
    cbtGm.SetGlobalBuffer((__gm__ int32_t *)cmpBlockTable);
    topkGm.SetGlobalBuffer((__gm__ int32_t *)cmpSparseIndices);
    cuQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ, bSize + 1);
    seqOriGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedKV, bSize);
    seqCmpGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedCmpKV, bSize);
    if (hasSinks) {
        sinksGm.SetGlobalBuffer((__gm__ T *)sinks, SMLAD_FAST_H);
    }
    outGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);

    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    const uint32_t totalQ = tiling->fastParams.totalQTokens;
    constexpr uint64_t kvSlotB = (uint64_t)SMLAD_FAST_KV_E * sizeof(KV_T);   // 128KB
    constexpr uint64_t mm1SlotB = (uint64_t)SMLAD_FAST_ROW_E * sizeof(T);    // 8KB
    constexpr uint64_t pSlotB = (uint64_t)SMLAD_FAST_ROW_E * sizeof(KV_T);   // 4KB
    constexpr uint64_t mm2SlotB = (uint64_t)SMLAD_FAST_ACC_E * sizeof(T);    // 32KB
    const uint64_t kvBase = 0;
    const uint64_t mm1Base = (uint64_t)totalQ * 2 * kvSlotB;
    const uint64_t pBase = mm1Base + (uint64_t)totalQ * 2 * mm1SlotB;
    const uint64_t mm2Base = pBase + (uint64_t)totalQ * 2 * pSlotB;
    wsKvGm.SetGlobalBuffer((__gm__ KV_T *)(user + kvBase), (uint64_t)totalQ * 2 * SMLAD_FAST_KV_E);
    wsMm1Gm.SetGlobalBuffer((__gm__ T *)(user + mm1Base), (uint64_t)totalQ * 2 * SMLAD_FAST_ROW_E);
    wsPGm.SetGlobalBuffer((__gm__ KV_T *)(user + pBase), (uint64_t)totalQ * 2 * SMLAD_FAST_ROW_E);
    wsMm2Gm.SetGlobalBuffer((__gm__ T *)(user + mm2Base), (uint64_t)totalQ * 2 * SMLAD_FAST_ACC_E);

    if ASCEND_IS_AIC {
        mm1.Init(&tiling->fastParams.mm1Tiling, tPipe);
        mm2.Init(&tiling->fastParams.mm2Tiling, tPipe);
    }
}

template <typename SMLADT>
__aicore__ inline uint32_t SparseFlashMlaDecodeCsaFast<SMLADT>::ReqOfToken(uint32_t token)
{
    for (uint32_t b = 0; b < bSize; b++) {
        int32_t lo = cuQGm.GetValue(b);
        int32_t hi = cuQGm.GetValue(b + 1);
        if ((int32_t)token >= lo && (int32_t)token < hi) {
            return b;
        }
    }
    return bSize - 1;
}

// ============================================================ AIC
template <typename SMLADT>
__aicore__ inline void SparseFlashMlaDecodeCsaFast<SMLADT>::RunCube(uint32_t token)
{
    constexpr uint64_t kvSlotE = SMLAD_FAST_KV_E;
    constexpr uint64_t rowSlotE = SMLAD_FAST_ROW_E;
    constexpr uint64_t accSlotE = SMLAD_FAST_ACC_E;
    const uint64_t tokenKv = (uint64_t)token * 2 * kvSlotE;
    const uint64_t tokenRow = (uint64_t)token * 2 * rowSlotE;
    const uint64_t tokenAcc = (uint64_t)token * 2 * accSlotE;
    const uint64_t qOff = (uint64_t)token * SMLAD_FAST_ACC_E;

    for (uint32_t c = 0; c < SMLAD_FAST_NCHUNK; c++) {
        const uint32_t slot = c & 1U;
        CrossCoreWaitFlag<ConstInfo::SMLAD_SYNC_MODE2, PIPE_MTE3>(SMLAD_FAST_F_KV_READY + slot);
        mm1.SetOrgShape(SMLAD_FAST_H, SMLAD_FAST_BC, SMLAD_FAST_D);
        mm1.SetSingleShape(SMLAD_FAST_H, SMLAD_FAST_BC, SMLAD_FAST_D);
        mm1.SetTensorA(queryGm[qOff], false);
        mm1.SetTensorB(wsKvGm[tokenKv + slot * kvSlotE], true);
        mm1.template IterateAll<false>(wsMm1Gm[tokenRow + slot * rowSlotE], 0);
        CrossCoreSetFlag<ConstInfo::SMLAD_SYNC_MODE2, PIPE_FIX>(SMLAD_FAST_F_MM1_DONE + slot);

        CrossCoreWaitFlag<ConstInfo::SMLAD_SYNC_MODE2, PIPE_MTE3>(SMLAD_FAST_F_P_READY + slot);
        mm2.SetOrgShape(SMLAD_FAST_H, SMLAD_FAST_D, SMLAD_FAST_BC);
        mm2.SetSingleShape(SMLAD_FAST_H, SMLAD_FAST_D, SMLAD_FAST_BC);
        mm2.SetTensorA(wsPGm[tokenRow + slot * rowSlotE], false);
        mm2.SetTensorB(wsKvGm[tokenKv + slot * kvSlotE], false);
        mm2.template IterateAll<false>(wsMm2Gm[tokenAcc + slot * accSlotE], 0);
        CrossCoreSetFlag<ConstInfo::SMLAD_SYNC_MODE2, PIPE_FIX>(SMLAD_FAST_F_MM2_DONE + slot);
    }
}

// 标量 exp（设备侧无 expf）：借 8 元素向量 Exp 通道
template <typename SMLADT>
__aicore__ inline float SparseFlashMlaDecodeCsaFast<SMLADT>::ScalarExp(float x)
{
    tmpUb.SetValue(0, x);
    Exp(tmpUb, tmpUb, 8);
    return tmpUb.GetValue(0);
}

// ============================================================ AIV: gather（本子块 64 列）
template <typename SMLADT>
__aicore__ inline void SparseFlashMlaDecodeCsaFast<SMLADT>::GatherHalf(uint32_t token, uint32_t chunk)
{
    const uint32_t b = ReqOfToken(token);
    const int32_t cuLo = cuQGm.GetValue(b);
    const int32_t actS1 = cuQGm.GetValue(b + 1) - cuLo;
    const int32_t actS2 = seqOriGm.GetValue(b);
    const int32_t actCmpS2 = seqCmpGm.GetValue(b);
    const int32_t tokenInReq = (int32_t)token - cuLo;
    const uint64_t kvSlotE = SMLAD_FAST_KV_E;
    const uint32_t slot = chunk & 1U;
    const uint32_t colBase = chunk * SMLAD_FAST_BC + subBlk * SMLAD_FAST_BH;

    DataCopyExtParams cp;
    cp.blockCount = 1;
    cp.blockLen = SMLAD_FAST_D * sizeof(KV_T);
    cp.srcStride = 0;
    cp.dstStride = 0;
    DataCopyPadExtParams<KV_T> pad{false, 0, 0, 0};

    uint32_t nInvalid = 0;
    for (uint32_t j = 0; j < SMLAD_FAST_BH; j++) {
        const uint32_t col = colBase + j;
        int64_t elemOff = -1;
        if (col < SMLAD_FAST_W) {
            const int32_t pos = actS2 - actS1 + tokenInReq;
            const int32_t o = pos - (int32_t)(SMLAD_FAST_W - 1) + (int32_t)col;
            if (o >= 0 && o <= pos) {
                const int32_t blk = obtGm.GetValue((uint64_t)b * oriMaxBlk + (uint64_t)(o / (int32_t)SMLAD_FAST_W));
                if (blk >= 0) {
                    elemOff = (int64_t)blk * oriKvStride0 + (int64_t)(o % (int32_t)SMLAD_FAST_W) * SMLAD_FAST_D;
                }
            }
            if (elemOff >= 0) {
                DataCopyPad(kvUb[(uint64_t)j * SMLAD_FAST_D], oriKvGm[elemOff], cp, pad);
            }
        } else {
            const int32_t idx = topkGm.GetValue((uint64_t)token * SMLAD_FAST_K + (col - SMLAD_FAST_W));
            if (idx >= 0 && idx < actCmpS2) {
                const int32_t blk = cbtGm.GetValue((uint64_t)b * cmpMaxBlk + (uint64_t)(idx / (int32_t)SMLAD_FAST_W));
                if (blk >= 0) {
                    elemOff = (int64_t)blk * cmpKvStride0 + (int64_t)(idx % (int32_t)SMLAD_FAST_W) * SMLAD_FAST_D;
                }
            }
            if (elemOff >= 0) {
                DataCopyPad(kvUb[(uint64_t)j * SMLAD_FAST_D], cmpKvGm[elemOff], cp, pad);
            }
        }
        if (elemOff < 0) {
            maskUb.SetValue(j, SMLAD_FAST_NEG_MAX);
            nInvalid++;
        }
    }
    if (nInvalid > 0) {
        for (uint32_t j = 0; j < SMLAD_FAST_BH; j++) {
            if (maskUb.GetValue(j) < 0) {
                Duplicate(kvUb[(uint64_t)j * SMLAD_FAST_D], (KV_T)0, SMLAD_FAST_D);
            }
        }
        SetFlag<HardEvent::V_MTE3>(evV3);
        WaitFlag<HardEvent::V_MTE3>(evV3);
    }
    // 无条件重建本子块列半区的屏蔽（防上一块陈旧 -inf 残留）
    for (uint32_t h = 0; h < SMLAD_FAST_HH; h++) {
        Adds(mask2dUb[h * SMLAD_FAST_BC + subBlk * SMLAD_FAST_BH], maskUb, (T)0, SMLAD_FAST_BH);
    }
    SetFlag<HardEvent::MTE2_MTE3>(ev23);
    WaitFlag<HardEvent::MTE2_MTE3>(ev23);
    const uint64_t dstOff = (uint64_t)token * 2 * kvSlotE + slot * kvSlotE +
                            (uint64_t)subBlk * SMLAD_FAST_KVH_E;
    DataCopy(wsKvGm[dstOff], kvUb, SMLAD_FAST_KVH_E);
    CrossCoreSetFlag<ConstInfo::SMLAD_SYNC_MODE2, PIPE_MTE3>(SMLAD_FAST_F_KV_READY + slot);
}

// ============================================================ AIV: 分块 online softmax（本子块 8 行）
template <typename SMLADT>
__aicore__ inline void SparseFlashMlaDecodeCsaFast<SMLADT>::SoftmaxHalf(uint32_t chunk)
{
    constexpr uint32_t nRow = SMLAD_FAST_HH;   // 8
    constexpr uint32_t nCol = SMLAD_FAST_BC;   // 128（列内偏移由 wsMm1 半区读入体现）
    LocalTensor<T> mm1 = mm1Ub;

    Muls(mm1, mm1, smScale, SMLAD_FAST_ROWH_E);
    Add(mm1, mm1, mask2dUb, SMLAD_FAST_ROWH_E);
    T newMax[nRow];
    for (uint32_t h = 0; h < nRow; h++) {
        LocalTensor<T> row = mm1[h * nCol];
        Max(tmpUb, row, row[nCol / 2], nCol / 2);
        Max(tmpUb, tmpUb, tmpUb[nCol / 4], nCol / 4);
        Max(tmpUb, tmpUb, tmpUb[nCol / 8], nCol / 8);
        T m = tmpUb.GetValue(0);
        for (uint32_t e = 1; e < 8; e++) {
            T v = tmpUb.GetValue(e);
            if (v > m) {
                m = v;
            }
        }
        newMax[h] = m;
    }
    T corr[nRow];
    for (uint32_t h = 0; h < nRow; h++) {
        const T oldMax = rowMaxUb.GetValue(h);
        const T oldSum = rowSumUb.GetValue(h);
        const T m = Max(newMax[h], oldMax);
        corr[h] = ScalarExp(oldMax - m);
        rowMaxUb.SetValue(h, m);
        rowSumUb.SetValue(h, oldSum * corr[h]);
        Duplicate(shiftUb[h * nCol], -m, nCol);
        Muls(accUb[h * SMLAD_FAST_D], accUb[h * SMLAD_FAST_D], corr[h], SMLAD_FAST_D);
    }
    Add(mm1, mm1, shiftUb, SMLAD_FAST_ROWH_E);
    Exp(mm1, mm1, SMLAD_FAST_ROWH_E);
    for (uint32_t h = 0; h < nRow; h++) {
        LocalTensor<T> row = mm1[h * nCol];
        Add(tmpUb, row, row[nCol / 2], nCol / 2);
        Add(tmpUb, tmpUb, tmpUb[nCol / 4], nCol / 4);
        Add(tmpUb, tmpUb, tmpUb[nCol / 8], nCol / 8);
        T s = (T)0;
        for (uint32_t e = 0; e < 8; e++) {
            s = s + tmpUb.GetValue(e);
        }
        rowSumUb.SetValue(h, rowSumUb.GetValue(h) + s);
    }
    Cast(pUb, mm1, RoundMode::CAST_RINT, SMLAD_FAST_ROWH_E);
}

// ============================================================ AIV: 收尾（本子块 8 行）
template <typename SMLADT>
__aicore__ inline void SparseFlashMlaDecodeCsaFast<SMLADT>::Finalize(uint32_t token)
{
    for (uint32_t h = 0; h < SMLAD_FAST_HH; h++) {
        const T sink = sinkUb.GetValue(h);
        const T m = rowMaxUb.GetValue(h);
        const T den = rowSumUb.GetValue(h) + ScalarExp(sink - m);
        const T inv = (den > (T)0) ? ((T)1 / den) : (T)0;
        Muls(accUb[h * SMLAD_FAST_D], accUb[h * SMLAD_FAST_D], inv, SMLAD_FAST_D);
    }
    LocalTensor<OUT_T> outUb = kvUb.template ReinterpretCast<OUT_T>();
    SetFlag<HardEvent::V_MTE2>(evV2);
    WaitFlag<HardEvent::V_MTE2>(evV2);
    Cast(outUb, accUb, RoundMode::CAST_RINT, SMLAD_FAST_ACCH_E);
    SetFlag<HardEvent::V_MTE3>(evV3);
    WaitFlag<HardEvent::V_MTE3>(evV3);
    DataCopy(outGm[(uint64_t)token * SMLAD_FAST_ACC_E + (uint64_t)subBlk * SMLAD_FAST_ACCH_E], outUb,
             SMLAD_FAST_ACCH_E);
}

// ============================================================ AIV 主流程（双工人同构）
template <typename SMLADT>
__aicore__ inline void SparseFlashMlaDecodeCsaFast<SMLADT>::RunVector(uint32_t token)
{
    // UB 预算 ≈ 64K(kv) + 5K(mm1+尾) + 2K(p) + 16K(acc) + 0.3K(mask) + 4K(mask2d) + 64B(tmp)
    evM2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    evV3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    ev23 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    ev32 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    evV2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    TBuf<> kvBuf;
    TBuf<> mm1Buf;
    TBuf<> pBuf;
    TBuf<> accBuf;
    TBuf<> maskBuf;
    TBuf<> mask2dBuf;
    TBuf<> shiftBuf;
    TBuf<> tmpBuf;
    pipe->InitBuffer(kvBuf, SMLAD_FAST_KVH_E * sizeof(KV_T));
    pipe->InitBuffer(mm1Buf, (SMLAD_FAST_ROWH_E + 64) * sizeof(T));
    pipe->InitBuffer(pBuf, SMLAD_FAST_ROWH_E * sizeof(KV_T));
    pipe->InitBuffer(accBuf, SMLAD_FAST_ACCH_E * sizeof(T));
    pipe->InitBuffer(maskBuf, SMLAD_FAST_BH * sizeof(T) + 32);
    pipe->InitBuffer(mask2dBuf, SMLAD_FAST_ROWH_E * sizeof(T));
    pipe->InitBuffer(shiftBuf, SMLAD_FAST_ROWH_E * sizeof(T));
    pipe->InitBuffer(tmpBuf, 64);
    kvUb = kvBuf.Get<KV_T>();
    mm1Ub = mm1Buf.Get<T>();
    pUb = pBuf.Get<KV_T>();
    accUb = accBuf.Get<T>();
    maskUb = maskBuf.Get<T>();
    mask2dUb = mask2dBuf.Get<T>();
    shiftUb = shiftBuf.Get<T>();
    tmpUb = tmpBuf.Get<T>();
    rowMaxUb = mm1Ub[SMLAD_FAST_ROWH_E];
    rowSumUb = mm1Ub[SMLAD_FAST_ROWH_E + 16];
    sinkUb = mm1Ub[SMLAD_FAST_ROWH_E + 32];

    Duplicate(maskUb, (T)0, SMLAD_FAST_BH);
    Duplicate(mask2dUb, (T)0, SMLAD_FAST_ROWH_E);
    Duplicate(accUb, (T)0, SMLAD_FAST_ACCH_E);
    for (uint32_t h = 0; h < SMLAD_FAST_HH; h++) {
        rowMaxUb.SetValue(h, SMLAD_FAST_NEG_MAX);
        rowSumUb.SetValue(h, (T)0);
    }
    if (hasSinks) {
        DataCopy(sinkUb, sinksGm[(uint64_t)subBlk * SMLAD_FAST_HH], SMLAD_FAST_HH);
        SetFlag<HardEvent::MTE2_V>(evM2V);
        WaitFlag<HardEvent::MTE2_V>(evM2V);
    } else {
        Duplicate(sinkUb, (T)0, SMLAD_FAST_HH);
    }
    SetFlag<HardEvent::V_MTE2>(evV2);
    WaitFlag<HardEvent::V_MTE2>(evV2);

    constexpr uint64_t rowSlotE = SMLAD_FAST_ROW_E;
    constexpr uint64_t accSlotE = SMLAD_FAST_ACC_E;
    const uint64_t tokenRow = (uint64_t)token * 2 * rowSlotE;
    const uint64_t tokenAcc = (uint64_t)token * 2 * accSlotE;
    LocalTensor<T> mm2Ub = kvUb.template ReinterpretCast<T>();

    for (uint32_t c = 0; c < SMLAD_FAST_NCHUNK; c++) {
        const uint32_t slot = c & 1U;
        GatherHalf(token, c);
        CrossCoreWaitFlag<ConstInfo::SMLAD_SYNC_MODE2, PIPE_FIX>(SMLAD_FAST_F_MM1_DONE + slot);
        DataCopy(mm1Ub, wsMm1Gm[tokenRow + slot * rowSlotE + (uint64_t)subBlk * SMLAD_FAST_ROWH_E],
                 SMLAD_FAST_ROWH_E);
        SetFlag<HardEvent::MTE2_V>(evM2V);
        WaitFlag<HardEvent::MTE2_V>(evM2V);
        SoftmaxHalf(c);
        SetFlag<HardEvent::V_MTE3>(evV3);
        WaitFlag<HardEvent::V_MTE3>(evV3);
        DataCopy(wsPGm[tokenRow + slot * rowSlotE + (uint64_t)subBlk * SMLAD_FAST_ROWH_E], pUb,
                 SMLAD_FAST_ROWH_E);
        CrossCoreSetFlag<ConstInfo::SMLAD_SYNC_MODE2, PIPE_MTE3>(SMLAD_FAST_F_P_READY + slot);
        CrossCoreWaitFlag<ConstInfo::SMLAD_SYNC_MODE2, PIPE_FIX>(SMLAD_FAST_F_MM2_DONE + slot);
        SetFlag<HardEvent::MTE3_MTE2>(ev32);
        WaitFlag<HardEvent::MTE3_MTE2>(ev32);
        DataCopy(mm2Ub, wsMm2Gm[tokenAcc + slot * accSlotE + (uint64_t)subBlk * SMLAD_FAST_ACCH_E],
                 SMLAD_FAST_ACCH_E);
        SetFlag<HardEvent::MTE2_V>(evM2V);
        WaitFlag<HardEvent::MTE2_V>(evM2V);
        Add(accUb, accUb, mm2Ub, SMLAD_FAST_ACCH_E);
        SetFlag<HardEvent::V_MTE2>(evV2);
        WaitFlag<HardEvent::V_MTE2>(evV2);
        Duplicate(maskUb, (T)0, SMLAD_FAST_BH);
    }
    Finalize(token);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(evM2V);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(evV3);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(ev23);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(ev32);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(evV2);
}

// ============================================================ Process
template <typename SMLADT>
__aicore__ inline void SparseFlashMlaDecodeCsaFast<SMLADT>::Process()
{
    const uint32_t totalQ = tilingData->fastParams.totalQTokens;
    const uint32_t step = (numAic == 0U) ? 1U : numAic;
    if ASCEND_IS_AIV {
        for (uint32_t t = aiCoreIdx; t < totalQ; t += step) {
            RunVector(t);
        }
    } else {
        for (uint32_t t = aiCoreIdx; t < totalQ; t += step) {
            RunCube(t);
        }
    }
}

} // namespace SMLADKernel
#endif // SPARSE_FLASH_MLA_DECODE_CSA_FAST_KERNEL_H
