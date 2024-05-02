/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <tensorrt_llm/common/tensor.h>

#include <optional>
#include <vector>

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::layers
{

class DecodingSetupParams
{
public:
    std::optional<std::vector<float>> temperature;              // [1] or [batch_size] on cpu
    std::optional<std::vector<runtime::SizeType32>> min_length; // [1] or [batch_size] on cpu
    std::optional<std::vector<float>> repetition_penalty;       // [1] or [batch_size] on cpu
    std::optional<std::vector<float>> presence_penalty;         // [1] or [batch_size] on cpu
    std::optional<std::vector<float>> frequency_penalty;        // [1] or [batch_size] on cpu
};

class DecodingParams
{
public:
    DecodingParams(runtime::SizeType32 step, runtime::SizeType32 ite, tc::Tensor logits, tc::Tensor end_ids, tc::Tensor min_p)
        : step{step}
        , ite{ite}
        , logits{std::move(logits)}
        , end_ids{std::move(end_ids)}
        , min_p{std::move(min_p)}
    {
    }

    // mandatory parameters
    runtime::SizeType32 step;
    runtime::SizeType32 ite;
    tc::Tensor logits;                     // [local_batch_size, beam_width, vocab_size_padded]
    tc::Tensor end_ids;                    // [local_batch_size]
    tc::Tensor min_p;                      // [local_batch_size]
    std::optional<tc::Tensor> batch_slots; // [local_batch_size], on pinned memory
    std::optional<tc::Tensor> finished;    // [batch_size * beam_width]
};

class DecodingOutputParams
{
public:
    explicit DecodingOutputParams(tc::Tensor outputIds)
        : output_ids{std::move(outputIds)}
    {
    }

    // mandatory parameters
    tc::Tensor output_ids; // [max_seq_len, batch_size]

    // optional parameters
    std::optional<tc::Tensor> finished;        // [batch_size * beam_width], optional
    std::optional<tc::Tensor> sequence_length; // [batch_size * beam_width], optional
    std::optional<tc::Tensor> cum_log_probs;   // [batch_size * beam_width], necessary in beam search
    std::optional<tc::Tensor>
        output_log_probs;                 // [request_ouptut_length, batch_size * beam_width], must be float*, optional
    std::optional<tc::Tensor> parent_ids; // [max_seq_len, batch_size * beam_width], necessary in beam search

    tc::Tensor output_ids_ptr;            // [batch_size] int* (2-d array), each int* has [beam_width, max_seq_len]

    // Medusa params
    std::optional<tc::Tensor> nextDraftTokens;       // [batch_size, max_draft_tokens_per_step]
    std::optional<tc::Tensor> acceptedLengths;       // [batch_size]
    std::optional<tc::Tensor> acceptedLengthsCumSum; // [batch_size + 1]
    std::optional<tc::Tensor> medusaPathsOffsets;    // [batch_size * max_medusa_heads]
};

} // namespace tensorrt_llm::layers
