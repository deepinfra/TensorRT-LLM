/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include <cuda_fp8.h>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

/// Fused FP8 MQA logits computation + streaming top-K for DSA indexer (prefill path).
///
/// Fuses the fp8_mqa_logits GEMM (Q @ K^T with ReLU + weighted head reduction) with
/// a streaming top-K selection in shared memory, eliminating the [seq_len, seq_len_kv]
/// intermediate logits tensor written to global memory.
///
/// @param q           FP8 query tensor, shape [num_tokens, num_heads, head_dim]
/// @param k           FP8 key tensor (contiguous), shape [num_kv_tokens, head_dim]
/// @param kScales     Per-token key scales, shape [num_kv_tokens]
/// @param weights     Per-head weights for head reduction, shape [num_tokens, num_heads]
/// @param cuSeqlenKs  Per-token attention window start indices, shape [num_tokens]
/// @param cuSeqlenKe  Per-token attention window end indices, shape [num_tokens]
/// @param outIndices  Output top-K indices, shape [num_tokens, topK], padded with -1
/// @param numTokens   Number of query tokens
/// @param numKvTokens Number of KV tokens in contiguous key buffer
/// @param numHeads    Number of indexer heads (32 or 64)
/// @param headDim     Head dimension (64 or 128)
/// @param topK        Number of top indices to select per query (e.g. 2048)
/// @param stream      CUDA stream
void invokeFusedMqaLogitsTopKPrefill(__nv_fp8_e4m3 const* q, __nv_fp8_e4m3 const* k, float const* kScales,
    float const* weights, int32_t const* cuSeqlenKs, int32_t const* cuSeqlenKe, int32_t* outIndices, int32_t numTokens,
    int32_t numKvTokens, int32_t numHeads, int32_t headDim, int32_t topK, cudaStream_t stream);

/// Fused FP8 MQA logits + streaming top-K for DSA indexer (decode path, paged KV cache).
///
/// @param q              FP8 query tensor, shape [numRows, num_heads, head_dim]
/// @param kCache         Paged KV cache buffer (raw bytes)
/// @param weights        Per-head weights, shape [numRows, num_heads]
/// @param contextLens    Context lengths per request, shape [batchSize]
/// @param blockTable     Block table for paged cache, shape [batchSize, maxBlocks]
/// @param outIndices     Output top-K indices, shape [numRows, topK]
/// @param numRows        Total number of query rows (batchSize * nextN)
/// @param batchSize      Number of requests
/// @param nextN          Next-N tokens per request
/// @param numHeads       Number of indexer heads
/// @param headDim        Head dimension
/// @param topK           Number of top indices to select
/// @param tokensPerBlock Tokens per KV cache block
/// @param maxBlocks      Max number of blocks in block table
/// @param stream         CUDA stream
void invokeFusedMqaLogitsTopKDecode(__nv_fp8_e4m3 const* q, void const* kCache, float const* weights,
    int32_t const* contextLens, int32_t const* blockTable, int32_t* outIndices, int32_t numRows, int32_t batchSize,
    int32_t nextN, int32_t numHeads, int32_t headDim, int32_t topK, int32_t tokensPerBlock, int32_t maxBlocks,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
