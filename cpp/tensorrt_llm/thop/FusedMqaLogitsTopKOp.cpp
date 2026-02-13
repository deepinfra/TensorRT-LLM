/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include "tensorrt_llm/kernels/FusedMqaLogitsTopK.h"

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

void fused_mqa_logits_topk_prefill(th::Tensor const& q, th::Tensor const& k, th::Tensor const& k_scales,
    th::Tensor const& weights, th::Tensor const& cu_seqlen_ks, th::Tensor const& cu_seqlen_ke,
    th::Tensor const& indices, int64_t index_topk)
{
    // Validate inputs.
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && k_scales.is_cuda() && weights.is_cuda() && cu_seqlen_ks.is_cuda()
            && cu_seqlen_ke.is_cuda() && indices.is_cuda(),
        "All tensors must be CUDA tensors");
    TORCH_CHECK(q.get_device() == k.get_device() && q.get_device() == indices.get_device(),
        "All tensors must be on the same device");

    // q: [num_tokens, num_heads, head_dim] (FP8)
    TORCH_CHECK(q.dim() == 3, "q must be a 3D tensor [num_tokens, num_heads, head_dim]");
    // k: [num_kv_tokens, head_dim] (FP8)
    TORCH_CHECK(k.dim() == 2, "k must be a 2D tensor [num_kv_tokens, head_dim]");
    // k_scales: [num_kv_tokens] or [num_kv_tokens, 1]
    TORCH_CHECK(k_scales.dim() == 1 || k_scales.dim() == 2, "k_scales must be 1D or 2D");
    // weights: [num_tokens, num_heads]
    TORCH_CHECK(weights.dim() == 2, "weights must be a 2D tensor [num_tokens, num_heads]");
    // cu_seqlen_ks/ke: [num_tokens]
    TORCH_CHECK(cu_seqlen_ks.dim() == 1 && cu_seqlen_ke.dim() == 1, "cu_seqlen_ks/ke must be 1D tensors");
    // indices: [num_tokens, topk]
    TORCH_CHECK(indices.dim() == 2, "indices must be a 2D tensor [num_tokens, topk]");

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(k_scales.is_contiguous(), "k_scales must be contiguous");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(cu_seqlen_ks.is_contiguous(), "cu_seqlen_ks must be contiguous");
    TORCH_CHECK(cu_seqlen_ke.is_contiguous(), "cu_seqlen_ke must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");

    int32_t numTokens = static_cast<int32_t>(q.size(0));
    int32_t numHeads = static_cast<int32_t>(q.size(1));
    int32_t headDim = static_cast<int32_t>(q.size(2));
    int32_t numKvTokens = static_cast<int32_t>(k.size(0));

    TORCH_CHECK(indices.size(0) == numTokens, "indices first dim must match q.size(0)");
    TORCH_CHECK(indices.size(1) >= index_topk, "indices second dim must be >= index_topk");

    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    tk::invokeFusedMqaLogitsTopKPrefill(reinterpret_cast<__nv_fp8_e4m3 const*>(q.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3 const*>(k.data_ptr()), k_scales.data_ptr<float>(), weights.data_ptr<float>(),
        cu_seqlen_ks.data_ptr<int32_t>(), cu_seqlen_ke.data_ptr<int32_t>(), indices.data_ptr<int32_t>(), numTokens,
        numKvTokens, numHeads, headDim, static_cast<int32_t>(index_topk), stream);
}

void fused_mqa_logits_topk_decode(th::Tensor const& q, th::Tensor const& k_cache, th::Tensor const& weights,
    th::Tensor const& context_lens, th::Tensor const& block_table, th::Tensor const& indices, int64_t next_n,
    int64_t index_topk, int64_t tokens_per_block)
{
    // Validate inputs.
    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && weights.is_cuda() && context_lens.is_cuda()
            && block_table.is_cuda() && indices.is_cuda(),
        "All tensors must be CUDA tensors");
    TORCH_CHECK(q.get_device() == k_cache.get_device() && q.get_device() == indices.get_device(),
        "All tensors must be on the same device");

    // q: [num_rows, num_heads, head_dim] (FP8) — num_rows = batch * next_n
    TORCH_CHECK(q.dim() == 3, "q must be a 3D tensor [num_rows, num_heads, head_dim]");
    // weights: [num_rows, num_heads]
    TORCH_CHECK(weights.dim() == 2, "weights must be a 2D tensor [num_rows, num_heads]");
    // context_lens: [batch_size]
    TORCH_CHECK(context_lens.dim() == 1, "context_lens must be a 1D tensor");
    // block_table: [batch_size, max_blocks]
    TORCH_CHECK(block_table.dim() == 2, "block_table must be a 2D tensor [batch_size, max_blocks]");
    // indices: [num_rows, topk]
    TORCH_CHECK(indices.dim() == 2, "indices must be a 2D tensor [num_rows, topk]");

    int32_t numRows = static_cast<int32_t>(q.size(0));
    int32_t numHeads = static_cast<int32_t>(q.size(1));
    int32_t headDim = static_cast<int32_t>(q.size(2));
    int32_t batchSize = static_cast<int32_t>(context_lens.size(0));
    int32_t maxBlocks = static_cast<int32_t>(block_table.size(1));

    TORCH_CHECK(numRows == batchSize * next_n, "q.size(0) must equal batch_size * next_n");
    TORCH_CHECK(indices.size(0) == numRows, "indices first dim must match q.size(0)");
    TORCH_CHECK(indices.size(1) >= index_topk, "indices second dim must be >= index_topk");

    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    tk::invokeFusedMqaLogitsTopKDecode(reinterpret_cast<__nv_fp8_e4m3 const*>(q.data_ptr()), k_cache.data_ptr(),
        weights.data_ptr<float>(), context_lens.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(),
        indices.data_ptr<int32_t>(), numRows, batchSize, static_cast<int32_t>(next_n), numHeads, headDim,
        static_cast<int32_t>(index_topk), static_cast<int32_t>(tokens_per_block), maxBlocks, stream);
}

} // end namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_mqa_logits_topk_prefill(Tensor q, Tensor k, Tensor k_scales, Tensor weights, "
        "Tensor cu_seqlen_ks, Tensor cu_seqlen_ke, Tensor indices, int index_topk=2048) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_mqa_logits_topk_prefill", &tensorrt_llm::torch_ext::fused_mqa_logits_topk_prefill);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_mqa_logits_topk_decode(Tensor q, Tensor k_cache, Tensor weights, "
        "Tensor context_lens, Tensor block_table, Tensor indices, int next_n, "
        "int index_topk=2048, int tokens_per_block=64) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_mqa_logits_topk_decode", &tensorrt_llm::torch_ext::fused_mqa_logits_topk_decode);
}
