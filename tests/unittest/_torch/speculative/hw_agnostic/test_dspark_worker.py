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
"""GPU unit tests for the DSpark worker and metadata plumbing.

Covers the framework-side logic that does NOT need the full draft model:
``DSparkSpecMetadata`` hidden-state capture (incl. the mHC hc-mean reduction)
and ``DSparkWorker`` slot / rolling-KV-window management. The end-to-end block
draft and acceptance path is covered by the DSpark test in
``integration/defs/accuracy/test_llm_api_pytorch.py``.
"""

import types

import pytest
import torch

from tensorrt_llm._torch.speculative.dspark import DSparkSpecMetadata, DSparkWorker
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DSpark metadata/worker allocate CUDA buffers"
)

HIDDEN = 128
NCAP = 3
HC_MULT = 4


def _make_metadata(max_num_requests=8, max_num_tokens=64, layers=(58, 59, 60)):
    return DSparkSpecMetadata(
        max_draft_len=5,
        max_total_draft_tokens=5,
        spec_dec_mode=SpeculativeDecodingMode.DSPARK,
        max_num_requests=max_num_requests,
        layers_to_capture=list(layers),
        hidden_size=HIDDEN,
        max_num_tokens=max_num_tokens,
        dtype=torch.bfloat16,
    )


def test_metadata_buffer_and_layer_lookup():
    meta = _make_metadata()
    assert meta.num_capture_layers == NCAP
    assert meta.captured_hidden_states.shape == (64, HIDDEN * NCAP)
    # sorted, O(1) membership
    assert meta.is_layer_capture(58) and meta.is_layer_capture(60)
    assert not meta.is_layer_capture(0) and not meta.is_layer_capture(61)


def test_metadata_capture_plain_hidden():
    """A [num_tokens, hidden] capture is stored at the layer's slice as-is."""
    meta = _make_metadata()
    hs = torch.randn(4, HIDDEN, device="cuda", dtype=torch.bfloat16)
    meta.maybe_capture_hidden_states(59, hs)  # layer 59 -> capture index 1
    got = meta.get_hidden_states(4)
    assert torch.equal(got[:, HIDDEN : 2 * HIDDEN], hs)


def test_metadata_capture_hc_mean_reduction():
    """A flattened mHC residual [N, hc_mult*hidden] is reduced by mean over hc."""
    meta = _make_metadata()
    mhc = torch.randn(4, HC_MULT * HIDDEN, device="cuda", dtype=torch.bfloat16)
    meta.maybe_capture_hidden_states(58, mhc)  # layer 58 -> capture index 0
    expected = mhc.reshape(4, HC_MULT, HIDDEN).mean(dim=1)
    got = meta.get_hidden_states(4)
    assert torch.equal(got[:, 0:HIDDEN], expected)


def test_metadata_no_capture_for_unlisted_layer():
    meta = _make_metadata()
    meta.captured_hidden_states.zero_()
    meta.maybe_capture_hidden_states(10, torch.randn(4, HIDDEN, device="cuda"))
    assert torch.count_nonzero(meta.get_hidden_states(4)) == 0


def test_metadata_prepare_batch_indices():
    meta = _make_metadata()
    meta.request_ids = [7, 3, 5]
    meta.prepare()
    assert meta.batch_indices_cuda[:3].tolist() == [0, 1, 2]


def _make_worker():
    cfg = types.SimpleNamespace(
        max_draft_len=5,
        spec_dec_mode=SpeculativeDecodingMode.DSPARK,
        confidence_threshold=0.5,
    )
    from tensorrt_llm.mapping import Mapping

    return DSparkWorker(cfg, Mapping())


def _fake_draft_model(num_stages=3, window_size=128, head_dim=64):
    return types.SimpleNamespace(
        num_stages=num_stages,
        block_size=5,
        _attn_params={"window_size": window_size, "head_dim": head_dim},
    )


def test_worker_lazy_init_window_buffers():
    worker = _make_worker()
    dm = _fake_draft_model(num_stages=3, window_size=128, head_dim=64)
    meta = _make_metadata(max_num_requests=8)
    worker._lazy_init(dm, meta)
    # One spare row (the draft-graph pad slot) beyond the 8 request slots.
    assert worker._kv_windows.shape == (9, 3, 128, 64)
    assert worker._ctx_len.shape == (9,)
    assert worker._pad_slot == 8
    assert list(worker._free_slots) == list(range(8))
    assert worker._batch_to_slot is not None
    assert worker._batch_to_slot.shape == (8,)
    assert worker._batch_to_slot.device.type == "cuda"
    # idempotent
    buf_id = id(worker._kv_windows)
    worker._lazy_init(dm, meta)
    assert id(worker._kv_windows) == buf_id


def test_worker_rejects_mismatched_block_size():
    worker = _make_worker()
    draft_model = _fake_draft_model()
    draft_model.block_size = 4

    with pytest.raises(ValueError, match="block_size must equal worker max_draft_len"):
        worker._lazy_init(draft_model, _make_metadata())


def test_worker_slot_assignment_and_reset():
    worker = _make_worker()
    worker._lazy_init(_fake_draft_model(), _make_metadata(max_num_requests=4))

    s0 = worker._assign_slot(100, reset=False)
    s1 = worker._assign_slot(101, reset=False)
    assert s0 != s1
    # same request id -> same slot (no reset)
    assert worker._assign_slot(100, reset=False) == s0

    # mark a position, then reset -> slot freed + window/pos cleared
    worker._ctx_len[s0] = 42
    worker._kv_windows[s0].fill_(1.0)
    s0b = worker._assign_slot(100, reset=True)
    assert int(worker._ctx_len[s0b]) == 0
    assert float(worker._kv_windows[s0b].abs().sum()) == 0.0


def test_worker_slot_exhaustion_preserves_live_request():
    worker = _make_worker()
    worker._lazy_init(_fake_draft_model(), _make_metadata(max_num_requests=1))

    slot = worker._assign_slot(100, reset=False)
    worker._ctx_len[slot] = 42
    worker._kv_windows[slot].fill_(1.0)

    with pytest.raises(RuntimeError, match="no free rolling-window slots"):
        worker._assign_slot(101, reset=False)

    assert worker._req_to_slot == {100: slot}
    assert int(worker._ctx_len[slot]) == 42
    assert torch.all(worker._kv_windows[slot] == 1.0)


def test_seed_context_windows_preserves_state_across_prefill_chunks():
    class DraftModel:
        num_stages = 1
        block_size = 5
        _attn_params = {"window_size": 8, "head_dim": 4}

        def __init__(self):
            self.written_positions = []

        def write_context_windows_batched(self, hidden, positions, slots, mask, kv_windows):
            # Record only the real (unmasked) writes, mirroring the writer's
            # contract that masked entries are no-ops.
            self.written_positions.append(positions[mask].clone())
            kv_windows[slots] += 1

    def _ctx_chunk(chunk_len, cached):
        return types.SimpleNamespace(
            num_contexts=1,
            _seq_lens=[chunk_len],
            kv_cache_params=types.SimpleNamespace(num_cached_tokens_per_seq=[cached]),
        )

    worker = _make_worker()
    draft_model = DraftModel()
    metadata = types.SimpleNamespace(
        max_num_requests=1,
        request_ids=[100],
        get_hidden_states=lambda _num_tokens: torch.zeros(
            3, HIDDEN * NCAP, device="cuda", dtype=torch.bfloat16
        ),
    )
    worker._lazy_init(draft_model, metadata)

    worker._seed_context_windows(draft_model, metadata, _ctx_chunk(3, cached=0), 3)
    slot = worker._req_to_slot[100]
    assert int(worker._ctx_len[slot]) == 3

    metadata.get_hidden_states = lambda _num_tokens: torch.zeros(
        2, HIDDEN * NCAP, device="cuda", dtype=torch.bfloat16
    )
    worker._seed_context_windows(draft_model, metadata, _ctx_chunk(2, cached=3), 2)

    assert int(worker._ctx_len[slot]) == 5
    assert [positions.tolist() for positions in draft_model.written_positions] == [
        [1, 2, 3],
        [4, 5],
    ]
    assert torch.all(worker._kv_windows[slot] == 2.0)


def test_prepare_builds_batch_to_slot_on_batched_path():
    """prepare() mirrors the host slot map into _batch_to_slot (default batched path)."""
    worker = _make_worker()
    meta = _make_metadata(max_num_requests=4)
    worker._lazy_init(_fake_draft_model(), meta)  # batched is the default
    meta._dspark_worker = worker

    # Assign slots for two requests (as the prefill path would).
    sa = worker._assign_slot(100, reset=True)
    sb = worker._assign_slot(101, reset=True)

    meta.request_ids = [101, 100]
    meta.prepare()
    # Mirror reflects request-order -> slot.
    assert worker._batch_to_slot[:2].tolist() == [sb, sa]


def test_prepare_frees_stale_slots_on_batched_path():
    """A request that drops out of the batch returns its slot to the free pool."""
    worker = _make_worker()
    meta = _make_metadata(max_num_requests=4)
    worker._lazy_init(_fake_draft_model(), meta)
    meta._dspark_worker = worker

    sa = worker._assign_slot(100, reset=True)
    worker._assign_slot(101, reset=True)
    worker._ctx_len[sa] = 17

    # Only request 101 survives; 100's slot must be freed + cleared.
    meta.request_ids = [101]
    meta.prepare()
    assert 100 not in worker._req_to_slot
    assert sa in worker._free_slots
    assert int(worker._ctx_len[sa]) == 0


def test_forward_mixed_batch_routes_through_base_entries(monkeypatch):
    """Mixed (context + gen) batch: ``forward`` must route acceptance and
    production through the unified ``SpecWorkerBase`` entries, one-hot-fill the
    context requests' draft-prob rows, and assemble
    ``next_draft_tokens = [ctx zeros ; gen argmax]``.

    Spies replace the base sampling entries and the heavy sub-calls (context
    seeding, per-request draft backbone) so this exercises the worker's
    context/gen orchestration — the exact surface the #15775 refactor changed —
    without a real draft model or MPI.
    """
    worker = _make_worker()
    worker.guided_decoder = None
    dm = _fake_draft_model(num_stages=3, window_size=128, head_dim=64)

    K = worker.max_draft_len
    vocab = 16
    num_contexts, num_gens = 2, 3
    batch_size = num_contexts + num_gens

    meta = _make_metadata(max_num_requests=8)
    meta.request_ids = [10, 11, 20, 21, 22]  # 2 context + 3 gen
    meta.prepare()

    attn_metadata = types.SimpleNamespace(
        num_seqs=batch_size,
        num_contexts=num_contexts,
        num_ctx_tokens=0,
        num_tokens=batch_size,
    )

    # Acceptance: return a fixed verified prefix (one accepted token per request).
    accepted = torch.arange(batch_size * (K + 1), dtype=torch.int32, device="cuda").reshape(
        batch_size, K + 1
    )
    num_accepted = torch.ones(batch_size, dtype=torch.int32, device="cuda")
    accept_calls = {}

    def fake_accept(logits, am, sm):
        accept_calls["args"] = (am, sm)
        return accepted, num_accepted

    monkeypatch.setattr(worker, "sample_and_accept_draft_tokens", fake_accept)
    # Context-window seeding is covered by its own test; stub it out here.
    monkeypatch.setattr(worker, "_seed_context_windows", lambda *a, **k: None)

    # The gen-block helper now returns the corrected block logits [num_gens,K,vocab].
    gen_logits = torch.randn(num_gens, K, vocab, device="cuda")
    monkeypatch.setattr(worker, "_draft_gen_block_batched", lambda *a, **k: gen_logits)

    sdt_calls = {}
    # The gen scatter publishes the FULL (post-TP-gather) vocab width, which is
    # wider than the sharded gen_logits width (`vocab`). The worker must pass this
    # published width (draft_probs_last_dim) to write_context_onehot_draft_probs,
    # NOT gen_logits.shape[-1].
    FULL_VOCAB = 97

    def fake_sample_draft_tokens(gl, sm, bs, *, num_contexts):
        sdt_calls["logits"] = gl
        sdt_calls["batch_size"] = bs
        sdt_calls["num_contexts"] = num_contexts
        sm.draft_probs_last_dim = FULL_VOCAB  # simulate the full-vocab scatter
        return gl.argmax(dim=-1).to(torch.int32)

    monkeypatch.setattr(worker, "sample_draft_tokens", fake_sample_draft_tokens)

    onehot_calls = {}
    monkeypatch.setattr(
        worker,
        "write_context_onehot_draft_probs",
        lambda sm, nc, ng, k, gv: onehot_calls.update(nc=nc, ng=ng, k=k, gv=gv),
    )

    input_ids = torch.zeros(batch_size, dtype=torch.long, device="cuda")
    position_ids = torch.zeros(batch_size, dtype=torch.long, device="cuda")
    hidden = torch.zeros(batch_size, HIDDEN, device="cuda", dtype=torch.bfloat16)
    logits = torch.zeros(batch_size, vocab, device="cuda")

    out = worker.forward(input_ids, position_ids, hidden, logits, attn_metadata, meta, dm)

    # Acceptance went through the unified entry with the right metadata objects.
    assert accept_calls["args"] == (attn_metadata, meta)
    # Production fed the [num_gens, K, vocab] block logits to the base sampler,
    # with num_contexts so it slices the gen segment.
    assert sdt_calls["num_contexts"] == num_contexts
    assert sdt_calls["logits"].shape == (num_gens, K, vocab)
    # Context rows one-hot-filled with the *scatter* width (draft_probs_last_dim,
    # FULL_VOCAB), not the sharded gen_logits width (vocab).
    assert onehot_calls == {"nc": num_contexts, "ng": num_gens, "k": K, "gv": FULL_VOCAB}

    # next_draft_tokens = [context zeros ; gen argmax]; gen subset is not polluted
    # by the context rows.
    nd = out["next_draft_tokens"]
    assert nd.shape == (batch_size, K)
    assert torch.all(nd[:num_contexts] == 0)
    assert torch.equal(nd[num_contexts:], gen_logits.argmax(dim=-1).to(torch.int32))
    # Verified tokens are surfaced unchanged.
    assert torch.equal(out["new_tokens"], accepted)
    assert torch.equal(out["new_tokens_lens"], num_accepted)


def _graph_capable_fake_draft_model(num_stages=3, window_size=16, head_dim=8, vocab=32):
    """Fake draft model whose batched hooks are pure tensor ops (capturable).

    ``forward_batched`` mixes every input (captured hidden, bonus, start_pos and
    the rolling-window content) into the logits so state divergence between the
    eager and graphed paths cannot go unnoticed; ``write_context_windows_batched``
    does the same masked frame write as the real model (torch.where blend, no
    data-dependent shapes).
    """
    dm = types.SimpleNamespace(
        num_stages=num_stages,
        block_size=5,
        _attn_params={"window_size": window_size, "head_dim": head_dim},
    )

    def write_context_windows_batched(hidden, pos, slots, valid, kv_windows):
        win = kv_windows.shape[2]
        frame = pos % win  # [G, K]
        g = slots.view(-1, 1).expand_as(frame)  # [G, K]
        val = hidden.float().sum(dim=-1)  # [G, K]
        cur = kv_windows[g, 0, frame, 0].float()
        kv_windows[g, 0, frame, 0] = torch.where(valid, val, cur).to(kv_windows.dtype)

    def forward_batched(main_hidden, bonus, start_pos, *, kv_windows, slots, **kwargs):
        num_gens = main_hidden.shape[0]
        wsum = kv_windows[slots].float().sum(dim=(1, 2, 3))  # [G]
        row = main_hidden.float().sum(dim=-1) + bonus.float() + start_pos.float() + wsum
        offsets = torch.arange(dm.block_size * vocab, device=row.device, dtype=torch.float32)
        logits = row.view(num_gens, 1, 1) + offsets.view(1, dm.block_size, vocab)
        return None, None, logits

    dm.write_context_windows_batched = write_context_windows_batched
    dm.forward_batched = forward_batched
    return dm


def test_draft_gen_block_graphed_matches_eager():
    """Warm, capture and replay of the worker-owned draft graph reproduce the
    eager core exactly (logits and window/position state), incl. bucket padding."""
    worker = _make_worker()
    dm = _graph_capable_fake_draft_model()
    meta = _make_metadata(max_num_requests=8, max_num_tokens=64)
    worker._lazy_init(dm, meta)

    num_contexts, num_gens = 1, 3  # pads up to bucket 4
    batch_size = num_contexts + num_gens
    K, Kp1 = 5, 6
    gen_start = 7  # fake context token count
    total_target_tokens = gen_start + num_gens * Kp1

    torch.manual_seed(0)
    meta.captured_hidden_states.normal_()
    for i, rid in enumerate([11, 22, 33]):
        assert worker._assign_slot(rid, reset=True) == i
    worker._batch_to_slot[num_contexts:batch_size] = torch.arange(3, device="cuda")
    worker._kv_windows.normal_()
    worker._ctx_len[:3] = torch.tensor([4, 9, 2], device="cuda")

    accepted = torch.randint(0, 31, (batch_size, Kp1), device="cuda")
    num_accepted = torch.tensor([1, 2, 6, 3], device="cuda", dtype=torch.int32)
    attn_metadata = types.SimpleNamespace(num_ctx_tokens=gen_start)

    win0, ctx0 = worker._kv_windows.clone(), worker._ctx_len.clone()

    def run_block():
        return worker._draft_gen_block_batched(
            dm, meta, attn_metadata, accepted, num_accepted,
            num_contexts, batch_size, total_target_tokens,
        )

    def reset_state():
        worker._kv_windows.copy_(win0)
        worker._ctx_len.copy_(ctx0)

    # Eager reference (graph path disabled).
    worker._draft_graph_disabled = True
    ref = run_block().clone()
    win_ref, ctx_ref = worker._kv_windows.clone(), worker._ctx_len.clone()

    worker._draft_graph_disabled = False
    for phase in ("warm", "capture", "replay"):
        reset_state()
        got = run_block()
        assert torch.equal(got, ref), phase
        assert torch.equal(worker._kv_windows[:3], win_ref[:3]), phase
        assert torch.equal(worker._ctx_len[:3], ctx_ref[:3]), phase
    assert 4 in worker._draft_graphs  # bucket 4 was captured
