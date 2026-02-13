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

#include <cfloat>
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{
namespace fused_mqa_topk
{

// ============================================================================
// Streaming Top-K data structure operating in shared memory.
//
// Each query maintains a candidate buffer in SMEM. Incoming (score, index)
// pairs are appended when they exceed a running threshold. When the buffer
// overflows, a cooperative bitonic sort compacts it down to exactly kTopK
// elements, updating the threshold.
// ============================================================================

/// Swap two elements if the first should come after the second, given the
/// requested sort direction.
__device__ __forceinline__ void bitonicCompareSwap(
    float* scores, int32_t* indices, int i, int j, bool ascending)
{
    if ((scores[i] < scores[j]) == ascending)
    {
        float tmpS = scores[i];
        scores[i] = scores[j];
        scores[j] = tmpS;
        int32_t tmpI = indices[i];
        indices[i] = indices[j];
        indices[j] = tmpI;
    }
}

/// Cooperative bitonic sort on contiguous SMEM arrays.
/// All `numThreads` threads participate; each handles a strided subset of
/// the comparison network. The array length must be a power of two or zero-
/// padded to one.
///
/// Sorts in DESCENDING order (largest scores first).
__device__ void bitonicSortSmem(float* scores, int32_t* indices, int n, int numThreads, int tid)
{
    // Pad to next power of two conceptually — we only sort `n` elements but
    // the bitonic network assumes power-of-two length. Scores beyond `n` are
    // initialised to -FLT_MAX by the caller so they sort to the end.
    int nPow2 = 1;
    while (nPow2 < n)
    {
        nPow2 <<= 1;
    }

    for (int size = 2; size <= nPow2; size <<= 1)
    {
        for (int stride = size >> 1; stride > 0; stride >>= 1)
        {
            __syncthreads();
            for (int idx = tid; idx < nPow2 / 2; idx += numThreads)
            {
                int pos = 2 * idx - (idx & (stride - 1));
                // Determine sub-sequence sort direction for bitonic merge.
                bool ascending = ((pos & size) != 0);
                int partner = pos + stride;
                if (partner < n)
                {
                    // descending overall, so flip: ascending==false means
                    // "first half descending"
                    bitonicCompareSwap(scores, indices, pos, partner, ascending);
                }
            }
        }
    }
    __syncthreads();
}

// ============================================================================
// Fused MQA Logits + Top-K kernel for prefill (contiguous K cache).
//
// This kernel computes Q @ K^T with FP8 inputs, applies per-KV scaling,
// ReLU activation, weighted head reduction, and streams the results through
// a SMEM top-K accumulator, avoiding the massive intermediate logits tensor.
//
// The GEMM is computed in software using vectorised FP8->FP32 dot products
// (no TMA / UMMA — this is the initial portable implementation targeting
// SM90+). A future SM100-optimised variant can replace the dot-product
// inner loop with UMMA instructions.
//
// Template parameters:
//   kNumHeads  — number of indexer heads (e.g. 32)
//   kHeadDim   — per-head dimension (e.g. 64 or 128)
//   BLOCK_Q    — number of query tokens processed per CTA
//   BLOCK_KV   — number of KV positions processed per tile
//   kTopK      — number of top indices to keep per query (e.g. 2048)
//   kCandidateCap — capacity of the SMEM candidate buffer per query
//                   (must be >= kTopK + BLOCK_KV)
// ============================================================================

template <uint32_t kNumHeads, uint32_t kHeadDim, uint32_t BLOCK_Q, uint32_t BLOCK_KV, uint32_t kTopK,
    uint32_t kCandidateCap>
__global__ void __launch_bounds__(512) fusedMqaLogitsTopKPrefillKernel(__nv_fp8_e4m3 const* __restrict__ q,
    __nv_fp8_e4m3 const* __restrict__ k, float const* __restrict__ kScales, float const* __restrict__ weights,
    int32_t const* __restrict__ cuSeqlenKs, int32_t const* __restrict__ cuSeqlenKe, int32_t* __restrict__ outIndices,
    int32_t numTokens, int32_t numKvTokens, int32_t topK)
{
    // ---- shared memory layout ----
    // Per query: threshold (float), candidate count (int), candidate scores
    // and indices.
    extern __shared__ char smemRaw[];

    // Per-query bookkeeping in smem.
    struct QueryState
    {
        float threshold;
        int candidateCount;
    };

    auto* queryStates = reinterpret_cast<QueryState*>(smemRaw);
    // Candidate buffers start after QueryState array, aligned.
    float* candScoreBase
        = reinterpret_cast<float*>(smemRaw + ((BLOCK_Q * sizeof(QueryState) + 127) & ~127));
    int32_t* candIndexBase = reinterpret_cast<int32_t*>(candScoreBase + BLOCK_Q * kCandidateCap);

    int const tid = threadIdx.x;
    int const numThreads = blockDim.x;

    // Persistent grid-stride loop over Q-block tiles.
    int const numQBlocks = (numTokens + BLOCK_Q - 1) / BLOCK_Q;
    for (int qBlockIdx = blockIdx.x; qBlockIdx < numQBlocks; qBlockIdx += gridDim.x)
    {
        int const qStart = qBlockIdx * BLOCK_Q;
        int const qEnd = min(qStart + static_cast<int>(BLOCK_Q), numTokens);
        int const numQInBlock = qEnd - qStart;

        // ---- Initialise per-query top-K state ----
        for (int i = tid; i < numQInBlock; i += numThreads)
        {
            queryStates[i].threshold = -FLT_MAX;
            queryStates[i].candidateCount = 0;
        }
        // Pad candidate scores for sorting (bitonic sort needs power-of-2).
        int const totalCandSlots = BLOCK_Q * kCandidateCap;
        for (int i = tid; i < totalCandSlots; i += numThreads)
        {
            candScoreBase[i] = -FLT_MAX;
            candIndexBase[i] = -1;
        }
        __syncthreads();

        // ---- Iterate over KV blocks ----
        int const numKvBlocks = (numKvTokens + BLOCK_KV - 1) / BLOCK_KV;
        for (int kvBlockIdx = 0; kvBlockIdx < numKvBlocks; ++kvBlockIdx)
        {
            int const kvStart = kvBlockIdx * BLOCK_KV;
            int const kvEnd = min(kvStart + static_cast<int>(BLOCK_KV), numKvTokens);

            // Each thread processes a subset of (q_local, kv_local) pairs.
            int const totalPairs = numQInBlock * (kvEnd - kvStart);
            for (int pairIdx = tid; pairIdx < totalPairs; pairIdx += numThreads)
            {
                int const qLocal = pairIdx / (kvEnd - kvStart);
                int const kvLocal = pairIdx % (kvEnd - kvStart);
                int const qGlobal = qStart + qLocal;
                int const kvGlobal = kvStart + kvLocal;

                // Causal masking: skip if kv position outside [ks, ke).
                int const ks = cuSeqlenKs[qGlobal];
                int const ke = cuSeqlenKe[qGlobal];
                if (kvGlobal < ks || kvGlobal >= ke)
                {
                    continue;
                }

                // Compute dot product across all heads, apply kv_scale, ReLU,
                // weight, and reduce.
                float reducedScore = 0.0f;
                float const kvScale = kScales[kvGlobal];

                for (uint32_t h = 0; h < kNumHeads; ++h)
                {
                    float dot = 0.0f;
                    // q layout: [numTokens, numHeads, headDim]
                    __nv_fp8_e4m3 const* qPtr = q + (static_cast<int64_t>(qGlobal) * kNumHeads + h) * kHeadDim;
                    // k layout: [numKvTokens, headDim]
                    __nv_fp8_e4m3 const* kPtr = k + static_cast<int64_t>(kvGlobal) * kHeadDim;

                    for (uint32_t d = 0; d < kHeadDim; ++d)
                    {
                        dot += static_cast<float>(qPtr[d]) * static_cast<float>(kPtr[d]);
                    }
                    dot *= kvScale;
                    // ReLU
                    dot = fmaxf(dot, 0.0f);
                    // Weighted head reduction
                    float w = weights[static_cast<int64_t>(qGlobal) * kNumHeads + h];
                    reducedScore += dot * w;
                }

                // ---- Streaming top-K insertion ----
                if (reducedScore > queryStates[qLocal].threshold)
                {
                    // Warp-level ballot for conflict-free insertion.
                    unsigned int activeMask = __activemask();
                    // We need per-query ballot, but since different threads may
                    // target different qLocal, use a simple atomic for now.
                    int pos = atomicAdd(&queryStates[qLocal].candidateCount, 1);
                    if (pos < static_cast<int>(kCandidateCap))
                    {
                        candScoreBase[qLocal * kCandidateCap + pos] = reducedScore;
                        candIndexBase[qLocal * kCandidateCap + pos] = kvGlobal - ks;
                    }
                }
            } // end pair loop

            __syncthreads();

            // ---- Check for overflow and compact ----
            for (int qLocal = 0; qLocal < numQInBlock; ++qLocal)
            {
                if (queryStates[qLocal].candidateCount >= static_cast<int>(kCandidateCap))
                {
                    // Cooperative bitonic sort, keep top-kTopK.
                    float* scores = candScoreBase + qLocal * kCandidateCap;
                    int32_t* indices = candIndexBase + qLocal * kCandidateCap;
                    int count = min(queryStates[qLocal].candidateCount, static_cast<int>(kCandidateCap));

                    bitonicSortSmem(scores, indices, count, numThreads, tid);

                    // Update threshold to the kTopK-th element (0-indexed).
                    if (tid == 0)
                    {
                        int threshIdx = min(static_cast<int>(kTopK) - 1, count - 1);
                        queryStates[qLocal].threshold = scores[threshIdx];
                        queryStates[qLocal].candidateCount = min(static_cast<int>(kTopK), count);
                    }
                    __syncthreads();

                    // Clear slots beyond the kept top-K.
                    int kept = queryStates[qLocal].candidateCount;
                    for (int i = kept + tid; i < static_cast<int>(kCandidateCap); i += numThreads)
                    {
                        scores[i] = -FLT_MAX;
                        indices[i] = -1;
                    }
                    __syncthreads();
                }
            }
        } // end KV block loop

        // ---- Final selection: sort remaining candidates per query ----
        for (int qLocal = 0; qLocal < numQInBlock; ++qLocal)
        {
            float* scores = candScoreBase + qLocal * kCandidateCap;
            int32_t* indices = candIndexBase + qLocal * kCandidateCap;
            int count = min(queryStates[qLocal].candidateCount, static_cast<int>(kCandidateCap));

            if (count > 0)
            {
                bitonicSortSmem(scores, indices, count, numThreads, tid);
            }

            // Write top-K indices to global memory.
            int const qGlobal = qStart + qLocal;
            int32_t* outRow = outIndices + static_cast<int64_t>(qGlobal) * topK;
            int numToWrite = min(static_cast<int>(kTopK), count);
            for (int i = tid; i < topK; i += numThreads)
            {
                if (i < numToWrite)
                {
                    outRow[i] = indices[i];
                }
                else
                {
                    outRow[i] = -1;
                }
            }
            __syncthreads();
        }
    } // end Q-block loop
}

// ============================================================================
// Fused MQA Logits + Top-K kernel for decode (paged KV cache).
//
// Similar to the prefill kernel but reads K from a paged cache using
// block_table indirection.
// ============================================================================

template <uint32_t kNumHeads, uint32_t kHeadDim, uint32_t BLOCK_Q, uint32_t BLOCK_KV, uint32_t kTopK,
    uint32_t kCandidateCap>
__global__ void __launch_bounds__(512) fusedMqaLogitsTopKDecodeKernel(__nv_fp8_e4m3 const* __restrict__ q,
    void const* __restrict__ kCache, float const* __restrict__ weights, int32_t const* __restrict__ contextLens,
    int32_t const* __restrict__ blockTable, int32_t* __restrict__ outIndices, int32_t numRows, int32_t batchSize,
    int32_t nextN, int32_t topK, int32_t tokensPerBlock, int32_t maxBlocks, int32_t headDimPlusScale)
{
    extern __shared__ char smemRaw[];

    struct QueryState
    {
        float threshold;
        int candidateCount;
    };

    auto* queryStates = reinterpret_cast<QueryState*>(smemRaw);
    float* candScoreBase
        = reinterpret_cast<float*>(smemRaw + ((BLOCK_Q * sizeof(QueryState) + 127) & ~127));
    int32_t* candIndexBase = reinterpret_cast<int32_t*>(candScoreBase + BLOCK_Q * kCandidateCap);

    int const tid = threadIdx.x;
    int const numThreads = blockDim.x;

    // Each CTA handles one row (one query token in decode).
    for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x)
    {
        int const requestIdx = rowIdx / nextN;
        int const nextNOffset = rowIdx % nextN;
        int const contextLen = contextLens[requestIdx];
        // In decode mode, the row attends to positions [0, contextLen - nextN + nextNOffset].
        int const rowEnd = contextLen - nextN + nextNOffset + 1;

        // Initialise
        if (tid == 0)
        {
            queryStates[0].threshold = -FLT_MAX;
            queryStates[0].candidateCount = 0;
        }
        int const totalCandSlots = kCandidateCap;
        for (int i = tid; i < totalCandSlots; i += numThreads)
        {
            candScoreBase[i] = -FLT_MAX;
            candIndexBase[i] = -1;
        }
        __syncthreads();

        // Iterate over KV positions in blocks.
        int const numKvBlocks = (rowEnd + BLOCK_KV - 1) / BLOCK_KV;
        for (int kvBlockIdx = 0; kvBlockIdx < numKvBlocks; ++kvBlockIdx)
        {
            int const kvStart = kvBlockIdx * BLOCK_KV;
            int const kvEnd = min(kvStart + static_cast<int>(BLOCK_KV), rowEnd);

            for (int kvPos = kvStart + tid; kvPos < kvEnd; kvPos += numThreads)
            {
                // Look up the paged cache for this KV position.
                int const pageIdx = kvPos / tokensPerBlock;
                int const tokenInPage = kvPos % tokensPerBlock;
                int const blockOffset = blockTable[static_cast<int64_t>(requestIdx) * maxBlocks + pageIdx];

                // kCache layout: [num_blocks, tokensPerBlock, 1, headDimPlusScale] bytes
                // FP8 data is the first headDim bytes, scale is the next 4 bytes (float32).
                auto const* pageBase = reinterpret_cast<uint8_t const*>(kCache)
                    + static_cast<int64_t>(blockOffset) * tokensPerBlock * headDimPlusScale
                    + static_cast<int64_t>(tokenInPage) * headDimPlusScale;
                auto const* kPtr = reinterpret_cast<__nv_fp8_e4m3 const*>(pageBase);
                float kvScale;
                // Scale is stored as float32 after headDim FP8 bytes.
                memcpy(&kvScale, pageBase + kHeadDim, sizeof(float));

                // Compute score: Q @ K^T with ReLU + weighted head reduction.
                float reducedScore = 0.0f;
                for (uint32_t h = 0; h < kNumHeads; ++h)
                {
                    float dot = 0.0f;
                    __nv_fp8_e4m3 const* qPtr
                        = q + (static_cast<int64_t>(rowIdx) * kNumHeads + h) * kHeadDim;
                    for (uint32_t d = 0; d < kHeadDim; ++d)
                    {
                        dot += static_cast<float>(qPtr[d]) * static_cast<float>(kPtr[d]);
                    }
                    dot *= kvScale;
                    dot = fmaxf(dot, 0.0f);
                    float w = weights[static_cast<int64_t>(rowIdx) * kNumHeads + h];
                    reducedScore += dot * w;
                }

                // Streaming top-K insertion.
                if (reducedScore > queryStates[0].threshold)
                {
                    int pos = atomicAdd(&queryStates[0].candidateCount, 1);
                    if (pos < static_cast<int>(kCandidateCap))
                    {
                        candScoreBase[pos] = reducedScore;
                        candIndexBase[pos] = kvPos;
                    }
                }
            }

            __syncthreads();

            // Check for overflow and compact.
            if (queryStates[0].candidateCount >= static_cast<int>(kCandidateCap))
            {
                int count = min(queryStates[0].candidateCount, static_cast<int>(kCandidateCap));
                bitonicSortSmem(candScoreBase, candIndexBase, count, numThreads, tid);

                if (tid == 0)
                {
                    int threshIdx = min(static_cast<int>(kTopK) - 1, count - 1);
                    queryStates[0].threshold = candScoreBase[threshIdx];
                    queryStates[0].candidateCount = min(static_cast<int>(kTopK), count);
                }
                __syncthreads();

                int kept = queryStates[0].candidateCount;
                for (int i = kept + tid; i < static_cast<int>(kCandidateCap); i += numThreads)
                {
                    candScoreBase[i] = -FLT_MAX;
                    candIndexBase[i] = -1;
                }
                __syncthreads();
            }
        } // end KV block loop

        // Final selection.
        {
            int count = min(queryStates[0].candidateCount, static_cast<int>(kCandidateCap));
            if (count > 0)
            {
                bitonicSortSmem(candScoreBase, candIndexBase, count, numThreads, tid);
            }
            int32_t* outRow = outIndices + static_cast<int64_t>(rowIdx) * topK;
            int numToWrite = min(static_cast<int>(kTopK), count);
            for (int i = tid; i < topK; i += numThreads)
            {
                outRow[i] = (i < numToWrite) ? candIndexBase[i] : -1;
            }
            __syncthreads();
        }
    } // end row loop
}

} // namespace fused_mqa_topk
} // namespace kernels
} // namespace tensorrt_llm
