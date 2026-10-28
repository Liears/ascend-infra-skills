/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file sparse_flash_mla.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"
#include "aclnnop/aclnn_sparse_flash_mla_decode.h"

namespace op_api {
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_4 = 4;

constexpr int64_t SMLA_METADATA_SIZE = 1024;

const c10::optional<at::Tensor> smla_get_valid_tensor(const c10::optional<at::Tensor> &tensor_opt, at::Device device)
{
    return tensor_opt.has_value() ? tensor_opt : torch::empty({0}, torch::dtype(torch::kInt32).device(device));
};

at::Tensor sparse_flash_mla_metadata(
    int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv, const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_ori_kv,
    const c10::optional<at::Tensor> &seqused_cmp_kv, const c10::optional<at::Tensor> &cmp_residual_kv,
    const c10::optional<at::Tensor> &ori_topk_length, const c10::optional<at::Tensor> &cmp_topk_length,
    int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_ori_kv, int64_t max_seqlen_cmp_kv, int64_t ori_topk,
    int64_t cmp_topk, int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_win_left,
    int64_t ori_win_right, c10::string_view layout_q, c10::string_view layout_kv, bool has_ori_kv, bool has_cmp_kv)
{
    at::Device output_device = at::Device(std::string("npu"));
    if (cu_seqlens_q.has_value()) {
        output_device = cu_seqlens_q.value().device();
    } else if (cu_seqlens_ori_kv.has_value()) {
        output_device = cu_seqlens_ori_kv.value().device();
    } else if (cu_seqlens_cmp_kv.has_value()) {
        output_device = cu_seqlens_cmp_kv.value().device();
    } else if (seqused_q.has_value()) {
        output_device = seqused_q.value().device();
    } else if (seqused_ori_kv.has_value()) {
        output_device = seqused_ori_kv.value().device();
    } else if (seqused_cmp_kv.has_value()) {
        output_device = seqused_cmp_kv.value().device();
    } else if (cmp_residual_kv.has_value()) {
        output_device = cmp_residual_kv.value().device();
    } else if (ori_topk_length.has_value()) {
        output_device = ori_topk_length.value().device();
    } else if (cmp_topk_length.has_value()) {
        output_device = cmp_topk_length.value().device();
    }

    at::Tensor output = torch::empty({SMLA_METADATA_SIZE}, torch::dtype(torch::kInt32).device(output_device));
    auto cu_seqlens_q_val = smla_get_valid_tensor(cu_seqlens_q, output_device);
    auto cu_seqlens_ori_kv_val = smla_get_valid_tensor(cu_seqlens_ori_kv, output_device);
    auto cu_seqlens_cmp_kv_val = smla_get_valid_tensor(cu_seqlens_cmp_kv, output_device);
    auto seqused_q_val = smla_get_valid_tensor(seqused_q, output_device);
    auto seqused_ori_kv_val = smla_get_valid_tensor(seqused_ori_kv, output_device);
    auto seqused_cmp_kv_val = smla_get_valid_tensor(seqused_cmp_kv, output_device);
    auto cmp_residual_kv_val = smla_get_valid_tensor(cmp_residual_kv, output_device);
    auto ori_topk_length_val = smla_get_valid_tensor(ori_topk_length, output_device);
    auto cmp_topk_length_val = smla_get_valid_tensor(cmp_topk_length, output_device);

    // convert str
    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    ACLNN_CMD(aclnnSparseFlashMlaMetadata, cu_seqlens_q_val, cu_seqlens_ori_kv_val, cu_seqlens_cmp_kv_val,
              seqused_q_val, seqused_ori_kv_val, seqused_cmp_kv_val, cmp_residual_kv_val, ori_topk_length_val,
              cmp_topk_length_val, num_heads_q, num_heads_kv, head_dim, batch_size, max_seqlen_q, max_seqlen_ori_kv,
              max_seqlen_cmp_kv, ori_topk, cmp_topk, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left,
              ori_win_right, layout_q_ptr, layout_kv_ptr, has_ori_kv, has_cmp_kv, output);
    return output;
}

void CheckQueryShape(const at::Tensor &q, const std::string &layout_q_str)
{
    TORCH_CHECK(layout_q_str == "BSND" || layout_q_str == "TND",
                "The layout of query only support BSND and TND, but got ", layout_q_str);
    TORCH_CHECK(q.numel() > 0, "Tensor query is empty.");
    for (int64_t i = 0; i < q.dim(); i++) {
        TORCH_CHECK(q.size(i) > 0, "All values within query's shape should be greater "
            "than 0, but shape[", i, "] is ", q.size(i));
    }
    if (layout_q_str == "BSND") {
        TORCH_CHECK(q.dim() == DIM_4,
                    "When the layout of query is BSND, the query dimension must be 4, but got ", q.dim());
    } else {
        TORCH_CHECK(q.dim() == DIM_3,
                    "When the layout of query is TND, the query dimension must be 3, but got ", q.dim());
    }
}

int64_t GetKvHeadNum(const c10::optional<at::Tensor> &ori_kv, const c10::optional<at::Tensor> &cmp_kv,
                     const std::string &layout_kv_str)
{
    TORCH_CHECK(ori_kv.has_value() || cmp_kv.has_value(),
                "ori_kv or cmp_kv is required when return_softmax_lse is true.");
    const at::Tensor &kv = ori_kv.has_value() ? ori_kv.value() : cmp_kv.value();
    if (layout_kv_str == "TND") {
        return kv.size(DIM_1);
    }
    return kv.size(DIM_2);
}

std::tuple<at::Tensor, at::Tensor> MakeSparseFlashMlaOutputs(const at::Tensor &q,
                                                             const c10::optional<at::Tensor> &ori_kv,
                                                             const c10::optional<at::Tensor> &cmp_kv,
                                                             const std::string &layout_q_str,
                                                             const std::string &layout_kv_str,
                                                             bool return_softmax_lse)
{
    CheckQueryShape(q, layout_q_str);
    at::Tensor atten_out = at::empty_like(q);
    at::Tensor softmax_lse;
    if (!return_softmax_lse) {
        softmax_lse = at::empty({}, q.options().dtype(torch::kFloat32));
        return {atten_out, softmax_lse};
    }

    int64_t kv_head_num = GetKvHeadNum(ori_kv, cmp_kv, layout_kv_str);
    if (layout_q_str == "BSND") {
        softmax_lse = at::empty({q.size(DIM_0), kv_head_num, q.size(DIM_1), q.size(DIM_2) / kv_head_num},
                                q.options().dtype(torch::kFloat32));
    } else {
        softmax_lse = at::empty({kv_head_num, q.size(DIM_0), q.size(DIM_1) / kv_head_num},
                                q.options().dtype(torch::kFloat32));
    }
    return {atten_out, softmax_lse};
}

std::tuple<at::Tensor, at::Tensor> sparse_flash_mla_decode(
    const at::Tensor &q,
    const c10::optional<at::Tensor> &ori_kv, const c10::optional<at::Tensor> &cmp_kv,
    const c10::optional<at::Tensor> &ori_sparse_indices, const c10::optional<at::Tensor> &cmp_sparse_indices,
    const c10::optional<at::Tensor> &ori_block_table, const c10::optional<at::Tensor> &cmp_block_table,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_ori_kv,
    const c10::optional<at::Tensor> &cu_seqlens_cmp_kv, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_ori_kv, const c10::optional<at::Tensor> &seqused_cmp_kv,
    const c10::optional<at::Tensor> &cmp_residual_kv,
    const c10::optional<at::Tensor> &ori_topk_length, const c10::optional<at::Tensor> &cmp_topk_length,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &metadata,
    double softmax_scale, int64_t cmp_ratio,
    int64_t ori_mask_mode, int64_t cmp_mask_mode,
    int64_t ori_win_left, int64_t ori_win_right,
    c10::string_view layout_q, c10::string_view layout_kv,
    int64_t topk_value_mode, bool return_softmax_lse)
{
    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    // convert str
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    // construct the atten_out tensor
    std::tuple<at::Tensor, at::Tensor> sparse_flash_mla_atten_out = op_api::MakeSparseFlashMlaOutputs(
        q, ori_kv, cmp_kv, layout_q_str, layout_kv_str, return_softmax_lse);
    at::Tensor atten_out = std::get<0>(sparse_flash_mla_atten_out);
    at::Tensor softmax_lse = std::get<1>(sparse_flash_mla_atten_out);

    ACLNN_CMD(aclnnSparseFlashMlaDecode, q,
              ori_kv, cmp_kv,
              ori_sparse_indices, cmp_sparse_indices,
              ori_block_table, cmp_block_table,
              cu_seqlens_q, cu_seqlens_ori_kv,
              cu_seqlens_cmp_kv, seqused_q,
              seqused_ori_kv, seqused_cmp_kv,
              cmp_residual_kv,
              ori_topk_length, cmp_topk_length,
              sinks, metadata,
              softmax_scale, cmp_ratio,
              ori_mask_mode, cmp_mask_mode,
              ori_win_left, ori_win_right,
              layout_q_ptr, layout_kv_ptr,
              topk_value_mode, return_softmax_lse,
              atten_out, softmax_lse);
    return std::tuple<at::Tensor, at::Tensor>(atten_out, softmax_lse);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparse_flash_mla_metadata", &sparse_flash_mla_metadata, "sparse_flash_mla_metadata");
    m.def("sparse_flash_mla_decode", &sparse_flash_mla_decode, "sparse_flash_mla_decode");
}
} // namespace op_api