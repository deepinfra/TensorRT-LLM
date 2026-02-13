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

#include "fusedMqaLogitsTopK.cuh"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <algorithm>

using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

// Common tile sizes.
constexpr uint32_t kBlockKV = 256;
constexpr uint32_t kNumThreads = 512;

// Compute SMEM requirement for the fused kernel.
// Layout: QueryState[BLOCK_Q] + candScores[BLOCK_Q * candidateCap] + candIndices[BLOCK_Q * candidateCap]
template <uint32_t BLOCK_Q, uint32_t kCandidateCap>
constexpr size_t smemSize()
{
    // QueryState is 8 bytes each (float threshold + int count), aligned to 128.
    size_t stateBytes = (BLOCK_Q * 8 + 127) & ~static_cast<size_t>(127);
    size_t candBytes = BLOCK_Q * kCandidateCap * (sizeof(float) + sizeof(int32_t));
    return stateBytes + candBytes;
}

// Helper to pick grid size: use all SMs but don't over-subscribe.
int getGridSize(int numBlocks, int smemBytes)
{
    int numSMs = 0;
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId);

    // Aim for 1 block per SM, but cap to the actual number of blocks.
    return std::min(numBlocks, numSMs);
}

} // namespace

// ============================================================================
// Prefill launcher
// ============================================================================

void invokeFusedMqaLogitsTopKPrefill(__nv_fp8_e4m3 const* q, __nv_fp8_e4m3 const* k, float const* kScales,
    float const* weights, int32_t const* cuSeqlenKs, int32_t const* cuSeqlenKe, int32_t* outIndices, int32_t numTokens,
    int32_t numKvTokens, int32_t numHeads, int32_t headDim, int32_t topK, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(topK > 0 && topK <= 4096, "topK must be in [1, 4096]");
    TLLM_CHECK_WITH_INFO(numTokens > 0, "numTokens must be positive");

    // Select kernel instantiation based on (numHeads, headDim).
    // BLOCK_Q = 128 / numHeads following DeepGEMM convention (block_qh=128).
    // kCandidateCap = topK + kBlockKV to give overflow margin.

    // We instantiate for the two primary configs:
    //   (numHeads=32, headDim=64)  -> BLOCK_Q=4
    //   (numHeads=32, headDim=128) -> BLOCK_Q=4
    //   (numHeads=64, headDim=64)  -> BLOCK_Q=2
    //   (numHeads=64, headDim=128) -> BLOCK_Q=2

    auto launchKernel = [&](auto kernelFunc, int blockQ, size_t smem)
    {
        int numQBlocks = (numTokens + blockQ - 1) / blockQ;
        int gridSize = getGridSize(numQBlocks, static_cast<int>(smem));

        // Set max dynamic SMEM if needed.
        cudaFuncSetAttribute(kernelFunc, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));

        kernelFunc<<<gridSize, kNumThreads, smem, stream>>>(
            q, k, kScales, weights, cuSeqlenKs, cuSeqlenKe, outIndices, numTokens, numKvTokens, topK);
    };

    // topK=2048 is the expected value; candidateCap = topK + BLOCK_KV = 2304.
    // For other topK values we use the same buffer size (capped at 2304).
    constexpr uint32_t kTopK = 2048;
    constexpr uint32_t kCandidateCap = kTopK + kBlockKV; // 2304

    if (numHeads == 32 && headDim == 64)
    {
        constexpr uint32_t BLOCK_Q = 4; // 128 / 32
        constexpr size_t smem = smemSize<BLOCK_Q, kCandidateCap>();
        launchKernel(
            fused_mqa_topk::fusedMqaLogitsTopKPrefillKernel<32, 64, BLOCK_Q, kBlockKV, kTopK, kCandidateCap>, BLOCK_Q,
            smem);
    }
    else if (numHeads == 32 && headDim == 128)
    {
        constexpr uint32_t BLOCK_Q = 4;
        constexpr size_t smem = smemSize<BLOCK_Q, kCandidateCap>();
        launchKernel(
            fused_mqa_topk::fusedMqaLogitsTopKPrefillKernel<32, 128, BLOCK_Q, kBlockKV, kTopK, kCandidateCap>, BLOCK_Q,
            smem);
    }
    else if (numHeads == 64 && headDim == 64)
    {
        constexpr uint32_t BLOCK_Q = 2; // 128 / 64
        constexpr size_t smem = smemSize<BLOCK_Q, kCandidateCap>();
        launchKernel(
            fused_mqa_topk::fusedMqaLogitsTopKPrefillKernel<64, 64, BLOCK_Q, kBlockKV, kTopK, kCandidateCap>, BLOCK_Q,
            smem);
    }
    else if (numHeads == 64 && headDim == 128)
    {
        constexpr uint32_t BLOCK_Q = 2;
        constexpr size_t smem = smemSize<BLOCK_Q, kCandidateCap>();
        launchKernel(
            fused_mqa_topk::fusedMqaLogitsTopKPrefillKernel<64, 128, BLOCK_Q, kBlockKV, kTopK, kCandidateCap>, BLOCK_Q,
            smem);
    }
    else
    {
        TLLM_THROW("Unsupported (numHeads=%d, headDim=%d) for fused MQA logits top-K kernel", numHeads, headDim);
    }

    sync_check_cuda_error(stream);
}

// ============================================================================
// Decode launcher
// ============================================================================

void invokeFusedMqaLogitsTopKDecode(__nv_fp8_e4m3 const* q, void const* kCache, float const* weights,
    int32_t const* contextLens, int32_t const* blockTable, int32_t* outIndices, int32_t numRows, int32_t batchSize,
    int32_t nextN, int32_t numHeads, int32_t headDim, int32_t topK, int32_t tokensPerBlock, int32_t maxBlocks,
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(topK > 0 && topK <= 4096, "topK must be in [1, 4096]");
    TLLM_CHECK_WITH_INFO(numRows > 0, "numRows must be positive");

    constexpr uint32_t kTopK = 2048;
    constexpr uint32_t kCandidateCap = kTopK + kBlockKV;
    // In decode mode, BLOCK_Q=1 (one query per CTA).
    constexpr uint32_t BLOCK_Q = 1;
    constexpr size_t smem = smemSize<BLOCK_Q, kCandidateCap>();

    // headDimPlusScale: FP8 bytes + 4 bytes for float32 scale.
    int32_t headDimPlusScale = headDim + 4;

    int gridSize = getGridSize(numRows, static_cast<int>(smem));

    auto launchKernel = [&](auto kernelFunc)
    {
        cudaFuncSetAttribute(kernelFunc, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        kernelFunc<<<gridSize, kNumThreads, smem, stream>>>(q, kCache, weights, contextLens, blockTable, outIndices,
            numRows, batchSize, nextN, topK, tokensPerBlock, maxBlocks, headDimPlusScale);
    };

    if (numHeads == 32 && headDim == 64)
    {
        launchKernel(
            fused_mqa_topk::fusedMqaLogitsTopKDecodeKernel<32, 64, BLOCK_Q, kBlockKV, kTopK, kCandidateCap>);
    }
    else if (numHeads == 32 && headDim == 128)
    {
        launchKernel(
            fused_mqa_topk::fusedMqaLogitsTopKDecodeKernel<32, 128, BLOCK_Q, kBlockKV, kTopK, kCandidateCap>);
    }
    else if (numHeads == 64 && headDim == 64)
    {
        launchKernel(
            fused_mqa_topk::fusedMqaLogitsTopKDecodeKernel<64, 64, BLOCK_Q, kBlockKV, kTopK, kCandidateCap>);
    }
    else if (numHeads == 64 && headDim == 128)
    {
        launchKernel(
            fused_mqa_topk::fusedMqaLogitsTopKDecodeKernel<64, 128, BLOCK_Q, kBlockKV, kTopK, kCandidateCap>);
    }
    else
    {
        TLLM_THROW("Unsupported (numHeads=%d, headDim=%d) for fused MQA logits top-K decode kernel", numHeads, headDim);
    }

    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
