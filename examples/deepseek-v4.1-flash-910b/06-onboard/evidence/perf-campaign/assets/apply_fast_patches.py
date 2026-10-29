#!/usr/bin/env python3
"""P12 M3: 给容器内 sparse_flash_mla_decode 打 Fast 核三补丁（tiling.h / tiling.cpp / 入口 cpp）。
每个替换都断言恰好命中一次；失败即退出非零。幂等：已打过则跳过。"""
import pathlib, sys

BASE = pathlib.Path('/vllm-workspace/vllm-ascend/csrc/attention/sparse_flash_mla_decode')
OK = []

def patch(path, old, new, tag):
    p = pathlib.Path(path)
    s = p.read_text()
    if new.strip() and new in s:
        OK.append(f'{tag}: already applied')
        return
    n = s.count(old)
    assert n == 1, f'{tag}: anchor count={n} (expect 1)'
    p.write_text(s.replace(old, new))
    OK.append(f'{tag}: applied')

# ---------------- P1: tiling.h ----------------
h = BASE / 'op_host/sparse_flash_mla_decode_tiling.h'
patch(h,
"""BEGIN_TILING_DATA_DEF(SparseFlashMlaDecodeTilingData)
TILING_DATA_FIELD_DEF_STRUCT(SparseFlashMlaDecodeSwaParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(SparseFlashMlaDecodeCmpParams, cmpParams);
END_TILING_DATA_DEF""",
"""BEGIN_TILING_DATA_DEF(SparseFlashMlaDecodeFastParams)
TILING_DATA_FIELD_DEF(uint32_t, fastFlag)
TILING_DATA_FIELD_DEF(uint32_t, totalQTokens)
TILING_DATA_FIELD_DEF(uint32_t, numAic)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1Tiling)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2Tiling)
END_TILING_DATA_DEF

BEGIN_TILING_DATA_DEF(SparseFlashMlaDecodeTilingData)
TILING_DATA_FIELD_DEF_STRUCT(SparseFlashMlaDecodeSwaParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(SparseFlashMlaDecodeCmpParams, cmpParams);
TILING_DATA_FIELD_DEF_STRUCT(SparseFlashMlaDecodeFastParams, fastParams);
END_TILING_DATA_DEF""",
'P1 tiling.h struct')

# ---------------- P2: tiling.cpp gate ----------------
cpp = BASE / 'op_host/sparse_flash_mla_decode_tiling.cpp'
GATE = """    // ---------------- Fast decode path (P12): 静态分派融合核 gate ----------------
    uint32_t fastFlag = 0U;
    do {
        if (tilingInfo->npuArch != NpuArch::DAV_2201) { break; }
        if (tilingInfo->perfMode != SMLADTemplateMode::CSA_TEMPLATE_MODE) { break; }
        if (tilingInfo->qType != ge::DT_BF16 || tilingInfo->oriKvType != ge::DT_BF16 ||
            tilingInfo->cmpKvType != ge::DT_BF16 || tilingInfo->outputType != ge::DT_BF16) { break; }
        if (tilingInfo->qLayout != SMLADLayout::TND || tilingInfo->kvLayout != SMLADLayout::PA_BBND ||
            tilingInfo->outLayout != SMLADLayout::TND) { break; }
        if (tilingInfo->qTSize == 0U || tilingInfo->qTSize > 32U || tilingInfo->s1Size > 32U) { break; }
        if (tilingInfo->gSize != 16U || tilingInfo->qHeadDim != 512U || tilingInfo->n2Size != 1U) { break; }
        if (tilingInfo->oriWinLeft != 127 || tilingInfo->oriWinRight != 0) { break; }
        if (tilingInfo->cmpRatio != 2 || tilingInfo->sparseBlockCount != 512) { break; }
        if (tilingInfo->oriBlockSize != 128 || tilingInfo->cmpBlockSize != 128 || tilingInfo->blockSize != 128) {
            break;
        }
        if (tilingInfo->returnSoftmaxLse || tilingInfo->actualLenDimsQ == 0U || tilingInfo->actualLenDimsKV == 0U ||
            tilingInfo->actualLenDimsCmpKV == 0U) { break; }
        if (tilingInfo->oriMaskMode != 4U || tilingInfo->cmpMaskMode != 3U) { break; }
        matmul_tiling::PlatformInfo mmPlat;
        mmPlat.socVersion = ascendcPlatform.GetSocVersion();
        uint64_t l1Size = 0;
        uint64_t l0aSize = 0;
        uint64_t l0bSize = 0;
        uint64_t l0cSize = 0;
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        mmPlat.l1Size = l1Size;
        mmPlat.l0CSize = l0cSize;
        mmPlat.ubSize = ubSize;
        mmPlat.l0ASize = l0aSize;
        mmPlat.l0BSize = l0bSize;
        auto mmDtypeBf16 = static_cast<matmul_tiling::DataType>(ge::DT_BF16);
        auto mmDtypeFp32 = static_cast<matmul_tiling::DataType>(ge::DT_FLOAT);
        matmul_tiling::MultiCoreMatmulTiling mm1T(mmPlat);
        mm1T.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDtypeBf16, false);
        mm1T.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDtypeBf16, true);
        mm1T.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDtypeFp32);
        mm1T.SetOrgShape(16, 128, 512);
        mm1T.SetShape(16, 128, 512);
        mm1T.SetBufferSpace(l1Size, l0cSize, ubSize);
        matmul_tiling::MultiCoreMatmulTiling mm2T(mmPlat);
        mm2T.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDtypeBf16, false);
        mm2T.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDtypeBf16, false);
        mm2T.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDtypeFp32);
        mm2T.SetOrgShape(16, 512, 128);
        mm2T.SetShape(16, 512, 128);
        mm2T.SetBufferSpace(l1Size, l0cSize, ubSize);
        if (mm1T.GetTiling(tilingData_.fastParams.mm1Tiling) == -1) { break; }
        if (mm2T.GetTiling(tilingData_.fastParams.mm2Tiling) == -1) { break; }
        constexpr uint64_t fastWsPerToken = 2ULL * (128ULL * 1024 + 8ULL * 1024 + 4ULL * 1024 + 32ULL * 1024);
        const uint64_t fastWsNeed =
            static_cast<uint64_t>(ascendcPlatform.GetLibApiWorkSpaceSize()) + tilingInfo->qTSize * fastWsPerToken;
        if (workSpaces[0] < fastWsNeed) {
            workSpaces[0] = fastWsNeed;
        }
        fastFlag = 1U;
        OP_LOGI(tilingInfo->opName, "SMLAD fast decode path enabled: T=%u.", tilingInfo->qTSize);
    } while (0);
    tilingData_.fastParams.set_fastFlag(fastFlag);
    tilingData_.fastParams.set_totalQTokens(fastFlag ? tilingInfo->qTSize : 0U);
    tilingData_.fastParams.set_numAic(aicNum);
    usedCoreNum_ = aicNum;"""
patch(cpp, "    usedCoreNum_ = aicNum;", GATE, 'P2 tiling.cpp gate')

# ---------------- P3: 入口 cpp 宏 ----------------
entry = BASE / 'op_kernel/sparse_flash_mla_decode.cpp'
patch(entry,
"""#include "arch22/sparse_flash_mla_decode_csa_kernel.h"
#include "arch22/sparse_flash_mla_decode_swa_kernel.h\"""",
"""#include "arch22/sparse_flash_mla_decode_csa_kernel.h"
#include "arch22/sparse_flash_mla_decode_csa_fast_kernel.h"
#include "arch22/sparse_flash_mla_decode_swa_kernel.h\"""",
'P3a entry include')

OLD_MACRO = """#else
#define SMLAD_OP_IMPL(templateClass, tilingdataClass, ...) \\
    do { \\
        templateClass<SMLADType<__VA_ARGS__>> op; \\
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling); \\
        const tilingdataClass *__restrict tiling_data = &tiling_data_in; \\
        op.Init(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqlensQ, \\
                cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedQ, seqUsedOriKV, seqUsedCmpKV, cmpResidualKV, oriTopkLength, \\
                cmpTopkLength, sinks, metadata, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe); \\
        op.Process(); \\
    } while (0)
#endif"""
NEW_MACRO = """#else
#define SMLAD_OP_IMPL(templateClass, tilingdataClass, ...) \\
    do { \\
        using SMLADFastDT = SMLADType<__VA_ARGS__>; \\
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling); \\
        const tilingdataClass *__restrict tiling_data = &tiling_data_in; \\
        if (tiling_data->fastParams.fastFlag != 0U && \\
            IsSameType<typename SMLADFastDT::queryType, bfloat16_t>::value && \\
            IsSameType<typename SMLADFastDT::kvType, bfloat16_t>::value && \\
            SMLADFastDT::templateMode == CSA_TEMPLATE) { \\
            SMLADKernel::SparseFlashMlaDecodeCsaFast<SMLADFastDT> fastOp; \\
            fastOp.Init(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable, \\
                        cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedQ, seqUsedOriKV, seqUsedCmpKV, \\
                        cmpResidualKV, oriTopkLength, cmpTopkLength, sinks, metadata, attentionOut, softmaxLse, \\
                        user, tiling_data, tiling, &tPipe); \\
            fastOp.Process(); \
        } else { \\
            templateClass<SMLADType<__VA_ARGS__>> op; \\
            op.Init(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable, \\
                    cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedQ, seqUsedOriKV, seqUsedCmpKV, \\
                    cmpResidualKV, oriTopkLength, cmpTopkLength, sinks, metadata, attentionOut, softmaxLse, \\
                    user, tiling_data, tiling, &tPipe); \\
            op.Process(); \\
        } \\
    } while (0)
#endif"""
patch(entry, OLD_MACRO, NEW_MACRO, 'P3b entry macro')

for line in OK:
    print(line)
print('ALL-PATCHES-OK')
