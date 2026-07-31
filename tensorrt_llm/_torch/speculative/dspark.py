# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# DSpark worker / metadata mirror the DFlash plumbing (capture target-layer
# hidden states, accept the previous block with standard verification, draft a
# new block in one backbone forward), adapted to DSpark's draft model which
# produces the whole block (and its confidence-truncated length) inside a single
# ``DSparkDraftModel.forward`` rather than via mask-token cross-attention.

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID
from .interface import SpecMetadata, SpecWorkerBase

if TYPE_CHECKING:
    from ...llmapi.llm_args import DSparkDecodingConfig


@dataclass
class DSparkSpecMetadata(SpecMetadata):
    """Metadata for DSpark speculative decoding.

    Captures hidden states from the target model's ``layers_to_capture`` during
    the target forward pass. DSpark captures the *mean over the multi-head
    (mHC) residual streams* at each captured layer (handled by the target-side
    capture hook), concatenated across layers, and feeds them to the draft
    model's ``main_proj`` + ``main_norm`` (inside ``DSparkDraftModel.forward``)
    as the captured-context attention input (``main_x``).

    Mirrors :class:`DFlashSpecMetadata`; the only DSpark-specific detail is that
    the per-layer captured width is the model hidden size (post hc-mean), so the
    buffer is ``[max_num_tokens, hidden_size * num_capture_layers]``.
    """

    batch_indices_cuda: Optional[torch.Tensor] = None

    # Hidden state capture fields
    layers_to_capture: Optional[List[int]] = None
    hidden_size: int = 0
    max_num_tokens: int = 0
    dtype: torch.dtype = torch.bfloat16
    captured_hidden_states: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.batch_indices_cuda = torch.empty(
            [self.max_num_requests],
            dtype=torch.int,
            device="cuda",
        )

        self.is_spec_dec_tree = False
        self.is_spec_dec_dynamic_tree = False

        # Set up hidden state capture buffer
        if self.layers_to_capture is not None and len(self.layers_to_capture) > 0:
            self.layers_to_capture = sorted(list(self.layers_to_capture))
            self.num_capture_layers = len(self.layers_to_capture)
            # O(1) lookups for is_layer_capture() and maybe_capture_hidden_states()
            self._capture_layer_set = frozenset(self.layers_to_capture)
            self._layer_to_idx = {lid: i for i, lid in enumerate(self.layers_to_capture)}
            self.captured_hidden_states = torch.empty(
                (self.max_num_tokens, self.hidden_size * self.num_capture_layers),
                dtype=self.dtype,
                device="cuda",
            )
            logger.info(
                f"DSpark: capturing hidden states from layers {self.layers_to_capture}, "
                f"buffer shape {self.captured_hidden_states.shape}"
            )
        else:
            self.num_capture_layers = 0
            self._capture_layer_set = frozenset()
            self._layer_to_idx = {}

    def prepare(self):
        assert self.request_ids is not None
        num_seqs = len(self.request_ids)
        batch_indices = torch.arange(
            num_seqs, dtype=torch.int, device="cpu", pin_memory=prefer_pinned()
        )
        self.batch_indices_cuda[:num_seqs].copy_(batch_indices, non_blocking=True)

        # CUDA-graph-safe path: maintain the request->slot mapping on the host
        # (outside the captured region) and mirror it into ``_batch_to_slot`` so the
        # captured gen forward can index the rolling windows by tensor. Mirrors
        # ``DFlashSpecMetadata.prepare`` (dflash.py:96-113).
        worker = getattr(self, "_dspark_worker", None)
        if worker is not None and worker._win_inited:
            current = set(self.request_ids)
            for rid in list(worker._req_to_slot.keys()):
                if rid not in current:
                    slot = worker._req_to_slot.pop(rid)
                    worker._ctx_len[slot] = 0
                    worker._kv_windows[slot].zero_()
                    worker._free_slots.append(slot)
            # Assign a persistent rolling-window slot to every real generation
            # request that never ran a context/seed forward on this worker. In
            # disaggregated serving the prompt is prefilled (and the window
            # seeded) on the *context* server, so ``_seed_context_windows`` never
            # runs on the generation server and ``_req_to_slot`` stays empty;
            # without this, all concurrent gen requests fall through to the shared
            # scratch row below and corrupt each other's draft window at batch
            # size > 1 (GitHub #16767). Context-prefix entries are left to
            # ``_seed_context_windows``; the ADP-idle (id 0) and CUDA-graph
            # padding dummies are kept on the scratch row.
            num_contexts = max(0, len(self.request_ids) - self.num_generations)
            for rid in self.request_ids[num_contexts:]:
                if (
                    rid != ATTENTION_DP_DUMMY_REQUEST_ID
                    and rid < worker._graph_dummy_id_floor
                    and rid not in worker._req_to_slot
                ):
                    worker._assign_slot(rid, reset=False)
            # Unknown request IDs (synthetic warmup / CUDA-graph padding, ADP idle
            # requests, or disagg seed forwards without a real id) map to the
            # dedicated throwaway scratch row so they cannot overwrite a live
            # request's rolling window (they previously aliased to slot 0).
            scratch = worker._scratch_slot
            mapping = torch.tensor(
                [worker._req_to_slot.get(rid, scratch) for rid in self.request_ids],
                dtype=torch.long,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            worker._batch_to_slot[:num_seqs].copy_(mapping, non_blocking=True)

    def is_layer_capture(self, layer_id: int) -> bool:
        return layer_id in self._capture_layer_set

    def maybe_capture_hidden_states(
        self, layer_id: int, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> None:
        """Capture hidden states from a target model layer into the buffer.

        DeepSeek-V4 keeps the multi-head (mHC) residual stream flattened as
        ``[num_tokens, hc_mult * hidden]``; DSpark captures the *mean over the hc
        streams* (reference ``h.mean(dim=2)`` with ``h`` shaped
        ``[*, hc_mult, hidden]``). We reduce here so the V4 decoder layer's
        existing capture call is unchanged. A ``[num_tokens, hidden]`` input
        (already reduced / non-mHC) is stored as-is.
        """
        if self.captured_hidden_states is None:
            return
        i = self._layer_to_idx.get(layer_id)
        if i is not None:
            num_tokens = hidden_states.shape[0]
            to_save = hidden_states + residual if residual is not None else hidden_states
            # mHC residual -> mean over the hc_mult streams.
            if to_save.shape[-1] != self.hidden_size:
                hc_mult = to_save.shape[-1] // self.hidden_size
                to_save = to_save.reshape(num_tokens, hc_mult, self.hidden_size).mean(dim=1)
            self.captured_hidden_states[
                :num_tokens, i * self.hidden_size : (i + 1) * self.hidden_size
            ].copy_(to_save, non_blocking=True)

    def get_hidden_states(self, num_tokens: int) -> Optional[torch.Tensor]:
        """Get captured hidden states (all layers concatenated)."""
        if self.captured_hidden_states is None:
            return None
        return self.captured_hidden_states[
            :num_tokens, : self.hidden_size * self.num_capture_layers
        ]


class DSparkWorker(SpecWorkerBase):
    """Worker for DSpark speculative decoding.

    DSpark drafts a whole block of ``block_size`` tokens in one backbone forward
    (``DSparkDraftModel.forward``): it projects the captured target-layer hidden
    states (``main_proj`` + ``main_norm``) into the draft's captured-context
    attention, runs the ``num_stages`` DSpark blocks over a rolling captured
    window, refines the per-position logits with the Markov head, and predicts a
    per-position acceptance confidence used to truncate the proposed prefix.

    Unlike DFlash, the draft does NOT use the paged KV cache or mask-token
    cross-attention: its attention K/V come from the worker-owned rolling window
    of projected captured context (one ``main_kv`` per decode step, per stage).
    Acceptance of the previous block goes through the unified
    :meth:`SpecWorkerBase.sample_and_accept_draft_tokens` (strict target-verify,
    or rejection sampling for a non-greedy batch), so greedy parity with no-spec
    is preserved regardless of draft quality.

    The rolling window is kept consistent across the whole decode: it is seeded
    from the prompt's captured context at prefill and back-filled with the
    intermediate accepted tokens of a multi-accept step (both via
    ``DSparkDraftModel.write_context_windows``), in addition to the per-step bonus
    write done by the generation path. These affect draft acceptance rate only,
    not correctness, which the standard target verify guarantees.

    Reference: DeepSeek DeepSpec (https://github.com/deepseek-ai/DeepSpec).
    """

    def __init__(
        self,
        spec_config: "DSparkDecodingConfig",
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(use_separate_draft_kv_cache)
        self.spec_config = spec_config
        self.mapping = mapping

        # Per-slot rolling captured-context KV windows, built lazily on the
        # first forward (fixed-size for slot-indexed reads/writes).
        self._win_inited = False
        self._kv_windows: Optional[torch.Tensor] = None  # [max_batch, num_stages, win, hd]
        self._ctx_len: Optional[torch.Tensor] = None  # [max_batch] abs decode position
        self._win = 0

        # Slot management. ``_req_to_slot`` (python dict) + ``_free_slots`` are the
        # source of truth, updated in prepare()/forward(); ``_batch_to_slot`` is the
        # CUDA mirror (request-order -> slot) read by the CUDA-graph-safe batched
        # gen path (set on the host in prepare(), so the captured forward indexes
        # the rolling windows through a tensor instead of a python dict lookup).
        self._req_to_slot = {}  # request_id -> slot index
        self._free_slots = deque()  # available slot indices
        self._batch_to_slot: Optional[torch.Tensor] = None  # [max_batch] long, cuda
        # Index of the throwaway "scratch" window row that absorbs padded /
        # unknown request IDs (set in ``_lazy_init`` to ``max_batch``); it is
        # never handed out through ``_free_slots``.
        self._scratch_slot = 0

        # Worker-owned CUDA graphs for the block draft, replayed when the outer
        # engine runs eagerly (mixed ctx+gen iterations are never captured by the
        # engine's gen-only graphs, which otherwise leaves the draft's many small
        # kernels launch-bound). Keyed by padded gen-batch bucket.
        self._pad_slot: Optional[int] = None  # spare kv_windows row for padded rows
        self._draft_graphs = {}  # bucket -> (torch.cuda.CUDAGraph, out logits)
        self._draft_graph_pool = None
        self._draft_graph_warmed = {}  # bucket -> bool (one eager pass pre-capture)
        self._draft_graph_disabled = False
        self._draft_graph_buckets: List[int] = []

        # The generation draft path is the batched, host-sync-free
        # ``_draft_gen_block_batched`` + ``DSparkDraftModel.forward_batched`` +
        # ``dspark_attention_forward_batched``: it is correct in eager mode AND safe
        # to capture into the target's CUDA graph (DSpark is a one-engine drafter —
        # its worker forward runs inside that graph, so the draft path MUST be
        # capture-safe whenever ``cuda_graph_config`` is set).

        logger.info(
            f"DSparkWorker initialized with "
            f"use_separate_draft_kv_cache={use_separate_draft_kv_cache}"
        )

    @property
    def max_draft_len(self) -> int:
        return self.spec_config.max_draft_len

    def _lazy_init(self, draft_model, spec_metadata) -> None:
        block_size = int(draft_model.block_size)
        if block_size != self.max_draft_len:
            raise ValueError(
                "DSpark draft model block_size must equal worker max_draft_len; "
                f"got block_size={block_size} and max_draft_len={self.max_draft_len}"
            )

        if self._win_inited:
            return
        max_batch = spec_metadata.max_num_requests
        num_stages = draft_model.num_stages
        self._win = int(draft_model._attn_params["window_size"])
        head_dim = int(draft_model._attn_params["head_dim"])

        # Real requests occupy slots ``[0, max_batch)``; one extra "scratch" row
        # at index ``max_batch`` absorbs padded / unknown request IDs (CUDA-graph
        # padding, ADP idle requests, or disagg seed forwards that arrive without
        # a real request id) so they can never overwrite a live request's rolling
        # window. Previously such IDs aliased to slot 0 and corrupted whichever
        # real request occupied it. The scratch row is never handed out through
        # ``_free_slots`` and its contents are throwaway. The draft-graph
        # buckets reuse the same row as their pad slot (``_pad_slot``): padded
        # bucket rows only ever produce throwaway window writes, exactly the
        # guarantee the scratch row provides.
        self._scratch_slot = max_batch
        self._pad_slot = max_batch
        num_rows = max_batch + 1

        # CUDA-graph padding requests carry ids in
        # ``[CUDA_GRAPH_DUMMY_REQUEST_ID - runtime_draft_len, CUDA_GRAPH_DUMMY_REQUEST_ID]``,
        # while real request ids start at ``max_batch_size`` and grow, so a simple
        # floor cleanly separates them. Together with ``ATTENTION_DP_DUMMY_REQUEST_ID``
        # (0) these dummies must route to the scratch row (see ``prepare()``) and
        # never consume a real slot. Imported lazily to break the
        # dspark -> cuda_graph_runner -> speculative.utils -> dspark import cycle.
        from ..pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID

        self._graph_dummy_id_floor = CUDA_GRAPH_DUMMY_REQUEST_ID - self.max_draft_len

        self._kv_windows = torch.zeros(
            (num_rows, num_stages, self._win, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self._ctx_len = torch.zeros(num_rows, dtype=torch.long, device="cuda")
        self._batch_to_slot = torch.zeros(max_batch, dtype=torch.long, device="cuda")
        self._free_slots = deque(range(max_batch))
        self._req_to_slot = {}

        # Staging buffers for the draft-graph path: the graphed core reads its
        # per-request inputs from these fixed addresses.
        self._g_slots = torch.zeros(max_batch, dtype=torch.long, device="cuda")
        self._g_nacc = torch.ones(max_batch, dtype=torch.long, device="cuda")
        self._g_bonus = torch.zeros(max_batch, dtype=torch.long, device="cuda")
        self._g_base = torch.zeros(max_batch, dtype=torch.long, device="cuda")
        # Same bucket policy as the engine's decode graphs ([1, 2, 4] + multiples
        # of 8, CudaGraphConfig._generate_cuda_graph_batch_sizes), so the draft
        # pads num_gens exactly like the target pads its captured batch sizes.
        self._draft_graph_buckets = sorted(
            {
                b
                for b in [1, 2, 4] + list(range(8, max_batch + 1, 8)) + [max_batch]
                if b <= max_batch
            }
        )
        self._win_inited = True
        logger.info(
            f"DSpark: allocated rolling KV windows "
            f"[{num_rows}, {num_stages}, {self._win}, {head_dim}] "
            f"({max_batch} request slots + 1 scratch/pad row), "
            f"draft-graph buckets {self._draft_graph_buckets}"
        )

    def _assign_slot(self, req_id: int, reset: bool) -> int:
        """Get (or refresh) the slot for a request; reset clears its window."""
        if reset and req_id in self._req_to_slot:
            old = self._req_to_slot.pop(req_id)
            self._ctx_len[old] = 0
            self._kv_windows[old].zero_()
            self._free_slots.append(old)
        if req_id not in self._req_to_slot:
            if not self._free_slots:
                raise RuntimeError(
                    "DSpark has no free rolling-window slots for request "
                    f"{req_id}; increase max_num_requests"
                )
            slot = self._free_slots.popleft()
            self._req_to_slot[req_id] = slot
            self._ctx_len[slot] = 0
            self._kv_windows[slot].zero_()
        return self._req_to_slot[req_id]

    def _seed_context_windows(
        self,
        draft_model,
        spec_metadata: "DSparkSpecMetadata",
        attn_metadata,
        total_target_tokens: int,
    ) -> None:
        """Seed context chunks using their absolute positions.

        A request can arrive in multiple prefill chunks. Only its first chunk
        starts at position zero and resets the persistent rolling window;
        continuation chunks append to the same request slot. Chunk geometry
        comes from host-side metadata (seq lens + cached-token counts), so the
        seeding adds no device sync, and all requests' windows are written in
        one batched call.
        """
        captured = spec_metadata.get_hidden_states(total_target_tokens)
        num_cached = attn_metadata.kv_cache_params.num_cached_tokens_per_seq
        win = self._win

        slots, ends, rows = [], [], []
        max_keep = 0
        context_offset = 0
        for i in range(attn_metadata.num_contexts):
            chunk_len = int(attn_metadata._seq_lens[i])
            if chunk_len == 0:
                continue
            # The chunk's first absolute position equals the tokens already in
            # the KV cache (previous chunks and/or reused blocks).
            first_position = int(num_cached[i])
            slot = self._assign_slot(
                spec_metadata.request_ids[i], reset=first_position == 0
            )
            slots.append(slot)
            ends.append(first_position + chunk_len)
            if captured is not None:
                keep = min(win, chunk_len)
                # A prompt token at absolute position p is stored in frame p+1,
                # matching the generation path's start_pos convention.
                rows.append(
                    (
                        context_offset + chunk_len - keep,
                        first_position + chunk_len - keep + 1,
                        keep,
                    )
                )
                max_keep = max(max_keep, keep)
            context_offset += chunk_len

        if not slots:
            return
        slots_cuda = torch.tensor(slots, dtype=torch.long, device="cuda")
        self._ctx_len[slots_cuda] = torch.tensor(ends, dtype=torch.long, device="cuda")
        if captured is None or max_keep == 0:
            return

        j = torch.arange(max_keep)
        starts = torch.tensor([r[0] for r in rows], dtype=torch.long)
        pos0 = torch.tensor([r[1] for r in rows], dtype=torch.long)
        keeps = torch.tensor([r[2] for r in rows], dtype=torch.long)
        # Keep positions contiguous past each row's valid range so every row
        # targets distinct window frames: masked entries are read-modify-write
        # no-ops and must not collide with the row's real writes.
        idx = (starts.unsqueeze(1) + j).clamp_(max=captured.shape[0] - 1).cuda()
        pos = (pos0.unsqueeze(1) + j).cuda()
        valid = (j.unsqueeze(0) < keeps.unsqueeze(1)).cuda()
        draft_model.write_context_windows_batched(
            captured[idx], pos, slots_cuda, valid, self._kv_windows
        )

    def _draft_gen_block_batched(
        self,
        draft_model,
        spec_metadata: "DSparkSpecMetadata",
        attn_metadata,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        total_target_tokens: int,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """CUDA-graph-safe batched gen draft (all gen requests in one forward).

        Free of host syncs and data-dependent shapes: per-request quantities
        (``nacc``, the bonus, ``main_hidden``, ``start_pos``, the multi-accept
        back-fill) are gathered as tensors, slots come from the host-built
        ``_batch_to_slot`` mirror, and the backbone runs once via
        ``DSparkDraftModel.forward_batched``. Returns the per-position corrected
        block logits ``[num_gens, K, vocab]`` (or ``None`` when there is nothing to
        draft); the worker feeds them to ``SpecWorkerBase.sample_draft_tokens``.
        Confidence truncation stays disabled — the full block is proposed.
        """
        num_gens = batch_size - num_contexts
        K = self.max_draft_len
        Kp1 = K + 1
        device = accepted_tokens.device

        if num_gens == 0:
            return None
        captured = spec_metadata.get_hidden_states(total_target_tokens)
        if captured is None:
            return None

        # gen-only graph batches have num_ctx_tokens == 0; mixed eager batches put
        # the gen tokens after the context tokens.
        gen_start = attn_metadata.num_ctx_tokens
        slots = self._batch_to_slot[num_contexts:batch_size]  # [G]
        nacc = num_accepted_tokens[num_contexts:batch_size].long()  # [G]
        gidx = nacc - 1  # [G] index of the bonus within each verified prefix

        # Bonus token = last accepted token of the verified prefix.
        bonus = (
            accepted_tokens[num_contexts:batch_size].gather(1, gidx.unsqueeze(1)).squeeze(1).long()
        )  # [G]

        # Captured target hidden at the bonus position within each request's Kp1
        # processed tokens.
        arange_g = torch.arange(num_gens, device=device)
        base = gen_start + arange_g * Kp1  # [G]

        if self._can_use_draft_graph(spec_metadata, all_rank_num_tokens):
            block_logits = self._run_draft_block_graphed(
                draft_model, spec_metadata, slots, nacc, bonus, base, num_gens
            )
            if block_logits is not None:
                return block_logits
        return self._draft_block_core(
            draft_model, captured, slots, nacc, bonus, base, all_rank_num_tokens
        )

    def _draft_block_core(
        self,
        draft_model,
        captured: torch.Tensor,
        slots: torch.Tensor,
        nacc: torch.Tensor,
        bonus: torch.Tensor,
        base: torch.Tensor,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Window back-fill + one-shot block draft, on fixed-size [G] inputs.

        Pure tensor ops: correct eagerly and under CUDA graph capture (either the
        engine's gen-only decode graph, or the worker-owned draft graph, whose
        capture passes the staging buffers as ``slots``/``nacc``/``bonus``/``base``
        and the full-length capture buffer as ``captured``).
        """
        K = self.max_draft_len
        device = nacc.device
        gidx = nacc - 1  # [G] index of the bonus within each verified prefix
        main_hidden = captured[base + gidx]  # [G, ncap*hidden]

        # Fixed-size ([G, K]) masked back-fill of the intermediate accepted tokens
        # (everything but the bonus) into the rolling window — same frames as the
        # eager path (old+1 .. old+nacc-1), with j >= nacc-1 masked out.
        old = self._ctx_len[slots]  # [G] pre-increment decode position
        j = torch.arange(K, device=device)  # [K]
        interim_valid = j.unsqueeze(0) < (nacc.unsqueeze(1) - 1)  # [G, K]
        interim_pos = old.unsqueeze(1) + 1 + j.unsqueeze(0)  # [G, K]
        interim_base = (base.unsqueeze(1) + j.unsqueeze(0)).clamp(
            min=0, max=captured.shape[0] - 1
        )  # [G, K] (clamped; invalid entries are masked out anyway)
        interim_hidden = captured[interim_base]  # [G, K, ncap*hidden]
        draft_model.write_context_windows_batched(
            interim_hidden, interim_pos, slots, interim_valid, self._kv_windows
        )

        # Advance the decode position by the accepted count; start_pos (= post-
        # increment ctx_len) matches the eager path's frame value.
        start_pos = old + nacc  # [G]
        self._ctx_len[slots] = start_pos

        # Surface the per-position corrected block logits ([num_gens, K, vocab])
        # and let SpecWorkerBase.sample_draft_tokens do the (greedy or rejection)
        # sampling + TP gather + draft_probs scatter, rather than argmaxing here.
        _toks, _num_proposed, block_logits = draft_model.forward_batched(
            main_hidden,
            bonus,
            start_pos,
            kv_windows=self._kv_windows,
            slots=slots,
            temperature=0.0,
            confidence_threshold=0.0,
            return_logits=True,
            all_rank_num_tokens=all_rank_num_tokens,
        )
        return block_logits

    def _can_use_draft_graph(self, spec_metadata, all_rank_num_tokens) -> bool:
        # Replay the worker-owned draft graph only when the engine itself runs
        # eagerly: never inside the engine's own warmup/capture passes
        # (is_cuda_graph / an actively capturing stream), and never under ADP
        # lockstep, where every rank must invoke the draft MoE with the globally
        # agreed token counts (per-rank bucket padding would desync the
        # FUSED_COMM phase-flip barrier).
        return (
            not self._draft_graph_disabled
            and all_rank_num_tokens is None
            and not getattr(spec_metadata, "is_cuda_graph", False)
            and not torch.cuda.is_current_stream_capturing()
        )

    def _run_draft_block_graphed(
        self,
        draft_model,
        spec_metadata: "DSparkSpecMetadata",
        slots: torch.Tensor,
        nacc: torch.Tensor,
        bonus: torch.Tensor,
        base: torch.Tensor,
        num_gens: int,
    ) -> Optional[torch.Tensor]:
        """Replay the block draft as its own CUDA graph (capture on first need).

        The gen batch is padded up to a bucket size; padded rows draft from the
        spare ``_pad_slot`` window with ``nacc=1``/``bonus=0``, so their (garbage)
        logits rows are computed harmlessly and sliced away. Each bucket runs
        eagerly once (kernel warmup) before being captured. Returns None to fall
        back to the eager core (oversized batch or a failed capture).
        """
        bucket = next((b for b in self._draft_graph_buckets if b >= num_gens), None)
        if bucket is None:
            return None

        self._g_slots[:num_gens].copy_(slots)
        self._g_slots[num_gens:bucket].fill_(self._pad_slot)
        self._g_nacc[:num_gens].copy_(nacc)
        self._g_nacc[num_gens:bucket].fill_(1)
        self._g_bonus[:num_gens].copy_(bonus)
        self._g_bonus[num_gens:bucket].fill_(0)
        self._g_base[:num_gens].copy_(base)
        self._g_base[num_gens:bucket].fill_(0)
        # The core advances ctx_len for every row it draws, including the padded
        # rows; pin the pad slot back to 0 so its RoPE position cannot creep past
        # the frequency table over long-running serving.
        self._ctx_len[self._pad_slot] = 0

        entry = self._draft_graphs.get(bucket)
        if entry is None:
            # The graphed core must index the full-length capture buffer: the
            # captured gathers keep this address, and per-iteration absolute
            # indices in _g_base may exceed any single iteration's token count.
            captured_full = spec_metadata.get_hidden_states(spec_metadata.max_num_tokens)
            core_args = (
                draft_model,
                captured_full,
                self._g_slots[:bucket],
                self._g_nacc[:bucket],
                self._g_bonus[:bucket],
                self._g_base[:bucket],
            )
            if not self._draft_graph_warmed.get(bucket, False):
                self._draft_graph_warmed[bucket] = True
                return self._draft_block_core(*core_args)[:num_gens]
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, pool=self._draft_graph_pool):
                    out = self._draft_block_core(*core_args)
            except Exception as e:
                # A failed mid-serving capture is not recoverable per-bucket;
                # permanently fall back to the eager core instead of crashing.
                logger.warning(
                    f"DSpark: draft-block CUDA graph capture failed for bucket "
                    f"{bucket}; falling back to eager draft. Error: {e}"
                )
                self._draft_graph_disabled = True
                return None
            if self._draft_graph_pool is None:
                self._draft_graph_pool = graph.pool()
            entry = (graph, out)
            self._draft_graphs[bucket] = entry

        graph, out = entry
        graph.replay()
        return out[:num_gens]

    def _forward_impl(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
        resource_manager=None,
    ):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        raw_logits = logits
        K = self.max_draft_len

        self._lazy_init(draft_model, spec_metadata)
        # Backref so DSparkSpecMetadata.prepare() can maintain the host slot map
        # and mirror it into _batch_to_slot for the CUDA-graph-safe gen path.
        spec_metadata._dspark_worker = self
        self._execute_guided_decoder_if_present(logits)

        # Target-verify acceptance via the unified SpecWorkerBase entry: it
        # reshapes the stored draft tokens (default (num_gens, runtime_draft_len)
        # hook), then routes to strict or rejection sampling. Greedy parity with
        # the previous hand-rolled path is preserved (rejection only engages for a
        # non-greedy batch with valid draft_probs).
        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata
        )

        total_target_tokens = input_ids.shape[0]

        # CUDA-graph warmup guard: the warmup forwards (is_cuda_graph set, stream
        # NOT yet capturing) run synthetic gen batches that would otherwise advance
        # the persistent rolling-window state. Snapshot and restore it so warmup is
        # side-effect-free. (During the capture pass itself the stream IS capturing,
        # so we skip the save/restore and let the ops be recorded; real requests
        # reset their slot's window+ctx_len at prefill, wiping any capture-time
        # mutation.)
        is_warmup = (
            getattr(spec_metadata, "is_cuda_graph", False)
            and not torch.cuda.is_current_stream_capturing()
        )
        if is_warmup:
            saved_ctx_len = self._ctx_len.clone()
            saved_windows = self._kv_windows.clone()

        # Assign / reset window slots for context (prefill) requests and seed each
        # request's rolling KV window from its prompt's captured context, so the
        # first generation step drafts against real context instead of an all-zero
        # window (acceptance-rate only; verified decoding keeps output correct).
        if num_contexts > 0:
            self._seed_context_windows(
                draft_model,
                spec_metadata,
                attn_metadata,
                total_target_tokens,
            )

        # FUSED_COMM MoE backends (DeepGEMM MegaMoE) synchronize EP ranks with an
        # in-kernel phase-flip NVLink barrier that flips on every kernel call, so
        # every rank must invoke the draft MoE the same number of times and with
        # the same globally-gathered per-rank token list, or the barrier desyncs
        # (hang / "unspecified launch failure"). The draft runs over generation
        # requests only, each expanded to ``block`` positions, so the per-rank
        # draft-MoE token count is ``num_gens * block``. ``all_rank_num_gens`` is
        # gathered at metadata-prep time (model_engine, outside any CUDA-graph
        # capture region); it is None for non-ADP / single-rank runs, where the
        # local ``[num_tokens]`` fallback in ``_forward_stage`` is correct.
        block = int(draft_model.block_size)
        all_rank_num_gens = getattr(spec_metadata, "all_rank_num_gens", None)
        # A rank with zero local gen requests still has to cross the draft MoE's
        # cross-rank barrier, but DeepseekV4MoE's router / shared-expert dense
        # GEMMs reject a 0-row input (cuBLAS CUBLAS_STATUS_INVALID_VALUE), so such
        # a rank runs a single 1-row dummy through the MoE (like ADP padding).
        # Encode that as ``1`` in the globally-shared per-rank token list so every
        # rank agrees on the FUSED_COMM chunk count and per-rank slice.
        all_rank_draft_tokens = (
            [max(1, int(g) * block) for g in all_rank_num_gens]
            if all_rank_num_gens is not None
            else None
        )
        global_has_gen = (
            max(all_rank_num_gens) > 0 if all_rank_num_gens is not None else num_gens > 0
        )

        if num_gens > 0:
            # The batched gen-block draft returns the per-position corrected block
            # logits [num_gens, K, vocab] and is CUDA-graph-safe.
            gen_logits = self._draft_gen_block_batched(
                draft_model,
                spec_metadata,
                attn_metadata,
                accepted_tokens,
                num_accepted_tokens,
                num_contexts,
                batch_size,
                total_target_tokens,
                all_rank_num_tokens=all_rank_draft_tokens,
            )
            if gen_logits is not None:
                # SpecWorkerBase samples the draft tokens. With a guided decoder
                # configured the drafts stay UNCONSTRAINED: the target-verify
                # bitmask chain stops at the first grammar-invalid draft token,
                # which then cannot match the masked target argmax and is
                # rejected; the matcher rollback runs inside
                # CapturableGuidedDecoder.fetch_batch (unconstrained_draft).
                gen_draft_tokens = self.sample_draft_tokens(
                    gen_logits, spec_metadata, batch_size, num_contexts=num_contexts
                )
                # The context one-hot must match the width the gen scatter just
                # published to draft_probs, NOT gen_logits.shape[-1].
                gen_vocab = spec_metadata.draft_probs_last_dim
            else:
                gen_draft_tokens = torch.zeros((num_gens, K), dtype=torch.int32, device="cuda")
                gen_vocab = None
        else:
            # No local generation requests: if any peer EP rank has some, we must
            # still cross the draft MoE's cross-rank barrier the same number of
            # times (zero-token) so a FUSED_COMM phase-flip barrier stays lockstep.
            if global_has_gen:
                draft_model.run_moe_lockstep_noop(all_rank_draft_tokens, accepted_tokens.device)
            gen_draft_tokens = torch.empty((0, K), dtype=torch.int32, device="cuda")
            gen_vocab = None

        # Context requests are not drafted by the block worker (zero placeholder
        # token); fill their draft-prob slot rows with a legal one-hot so they are
        # a valid distribution when they become gen requests next iteration.
        self.write_context_onehot_draft_probs(spec_metadata, num_contexts, num_gens, K, gen_vocab)

        if num_contexts > 0:
            ctx_draft_tokens = torch.zeros((num_contexts, K), dtype=torch.int32, device="cuda")
            next_draft_tokens = torch.cat([ctx_draft_tokens, gen_draft_tokens], dim=0)
        else:
            next_draft_tokens = gen_draft_tokens

        next_new_tokens = self._prepare_next_new_tokens(
            accepted_tokens,
            next_draft_tokens,
            spec_metadata.batch_indices_cuda,
            batch_size,
            num_accepted_tokens,
        )

        if is_warmup:
            self._ctx_len.copy_(saved_ctx_len)
            self._kv_windows.copy_(saved_windows)

        return {
            "logits": raw_logits,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }
