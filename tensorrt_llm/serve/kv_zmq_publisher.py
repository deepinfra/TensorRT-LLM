# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Read-only ZMQ tee for KV-cache events.

``KvZmqPublisher`` translates TensorRT-LLM's native KV-cache events into the
vLLM ZMQ wire format and broadcasts them on a ``zmq.PUB`` socket, so that an
external consumer (e.g. an NVIDIA Dynamo RadixTree indexer) can subscribe and
reconstruct the cache state -- including a precomputed per-block ``tokens_hash``
(content hash) that the SSE ``/kv_cache_events`` path does not expose.

Threading model: the engine's asyncio drain loop
(``OpenAIServer.kv_event_processor``) only calls ``enqueue()``, a non-blocking
hand-off onto a bounded ``queue.Queue``. A single dedicated consumer thread
owns everything else -- it pulls raw events off the queue, translates/filters
them, publishes on the PUB socket, appends to the replay buffer, and services
replay requests on the ROUTER socket. Because exactly one thread touches the
sockets and the buffer, no lock is needed (this mirrors vLLM's
``ZmqEventPublisher._publisher_thread``). The ZMQ-side work (msgpack encode +
send) is thus moved off the asyncio loop and never stalls the SSE path.

Because this sits next to the production routing path, every hand-off is
bounded and lossy by design: the inbound queue is bounded (overflow increments
``queue_dropped`` -- unrecoverable, the consumer fell behind), and the PUB
socket is bounded (``SNDHWM``) with non-blocking sends (overflow increments
``dropped`` -- recoverable via replay).

The translation logic mirrors the reference ``ZmqKvEventPublisher`` /
``_handle_kv_event`` in ``dynamo/components/src/dynamo/trtllm/publisher.py``
(partial-block tracking, attention-DP rank passthrough, window-size filtering,
event-id gap detection) so the emitted stream is byte-compatible with what a
Dynamo indexer expects.
"""

import queue
import threading
import time
from collections import deque, namedtuple
from typing import Any, Optional

import zmq

from tensorrt_llm.logger import logger

# Sentinel pushed onto the event queue by shutdown() to wake the consumer
# thread immediately instead of waiting out its get() timeout.
_SHUTDOWN = object()

# msgpack encoder for the vLLM wire payload. Prefer msgspec (what the
# vLLM/Dynamo reference publisher uses, and what requirements.txt pins); fall
# back to the standalone msgpack package, which the TRT-LLM runtime image
# already ships. Both emit identical msgpack bytes for the plain
# [float, [dict], int] batch we send, so the consumer cannot tell them apart.
try:
    import msgspec

    def _encode_msgpack(obj):
        return msgspec.msgpack.encode(obj)
except ImportError:
    import msgpack

    def _encode_msgpack(obj):
        return msgpack.packb(obj, use_bin_type=True)


# XXH3-64 hashing for per-block token hashes. The router (Dynamo standalone
# indexer) no longer hashes token_ids itself -- the engine now sends one
# precomputed ``tokens_hash`` per block. This value MUST be byte-identical to
# dynamo's ``compute_block_hash_for_seq`` or the indexer's queries miss every
# block. Verified by tokens_hash_parity_test.py against the dynamo binding.
try:
    import xxhash

    def _xxh3_64(data: bytes, seed: int = 0) -> int:
        return xxhash.xxh3_64_intdigest(data, seed)
except ImportError:

    def _xxh3_64(data: bytes, seed: int = 0) -> int:
        raise RuntimeError(
            "xxhash is required to compute KV-cache tokens_hash for the ZMQ "
            "tee; install it (pip install xxhash) in the engine image")


# Seed for XXH3, must match dynamo_kv_router::protocols::XXH3_SEED.
XXH3_SEED = 1337


def _compute_block_tokens_hash(token_ids: list[int],
                               lora_name: Optional[str] = None,
                               mm_hashes: Optional[list[int]] = None) -> int:
    """Local per-block token hash, matching dynamo's compute_block_hash_for_seq.

    Equivalent to ``compute_block_hash_for_seq(tokens, kv_block_size, ...)[i]``
    for a single full (non-eagle) block: XXH3-64 over the block's token ids as
    little-endian u32s, seeded by ``XXH3_SEED`` (mixed with the LoRA name when
    present), with sorted multimodal ``mm_hash`` values (u64 LE) appended.
    """
    if lora_name:
        seed = (XXH3_SEED + _xxh3_64(lora_name.encode())) & 0xFFFFFFFFFFFFFFFF
    else:
        seed = XXH3_SEED
    buf = b"".join((t & 0xFFFFFFFF).to_bytes(4, "little") for t in token_ids)
    if mm_hashes:
        for h in sorted(mm_hashes):
            buf += (h & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little")
    return _xxh3_64(buf, seed)


# One snapshot node per published full block, keyed by signed block_hash in
# ``kv_snapshot``. Carries just enough to re-emit a BlockStored event and to
# topologically order the rebuild (parents before children). See
# dynamo standalone_indexer/docs.md.
_SnapNode = namedtuple("_SnapNode", ["tokens_hash", "parent_hash", "depth"])


def _to_signed_i64(value: Optional[int]) -> Optional[int]:
    """Convert a Python int to signed 64-bit range by two's complement."""
    if value is None:
        return None

    if value >= 2**63:
        return value - 2**64
    if value < -(2**63):
        return ((value + 2**63) % 2**64) - 2**63
    return value


class KvZmqPublisher:
    """Pure-Python ZMQ PUBLISHER tee for TensorRT-LLM KV-cache events.

    Event format (vLLM-compatible): ``[timestamp, [events], data_parallel_rank]``.
    Wire format: 3-frame multipart ``[topic, 8-byte big-endian sequence, payload]``
    where ``payload`` is the msgpack-serialized batch.

    Replay: when ``replay_endpoint`` is set, a ROUTER socket is bound and a ring
    buffer of the last ``buffer_steps`` ``(seq, payload)`` pairs is kept. Both
    live on the same consumer thread that does the publishing, which polls the
    ROUTER between queue items (``poll(0)``). A subscriber that detects a
    sequence gap (PUB/SUB is lossy) sends the missing start sequence as an
    8-byte big-endian integer; the publisher streams back every buffered batch
    from that sequence onward, then an end-of-sequence sentinel (``seq = -1``,
    empty payload). This is byte-compatible with vLLM's ``ZmqEventPublisher``
    replay protocol, so an existing Dynamo / standalone-indexer replay client
    works unchanged.

    Args:
        zmq_endpoint: endpoint to bind the PUB socket to, e.g. ``tcp://*:5557``.
        kv_block_size: KV-cache block size in tokens (``kv_cache_config.tokens_per_block``).
        replay_endpoint: endpoint to bind the replay ROUTER socket to, e.g.
            ``tcp://*:5558``. When ``None`` the replay path is disabled (no
            buffer, no ROUTER socket); the consumer thread still runs to drain
            the queue and publish.
        buffer_steps: number of past batches kept in the replay ring buffer.
        queue_maxsize: bound on the inbound event queue. When the consumer
            thread cannot keep up the queue fills and the oldest-arriving events
            are dropped (``queue_dropped``); these are unrecoverable.
        sndhwm: send high-water mark (max queued messages before drops kick in).
        topic: ZMQ topic to publish on (empty string == all topics, vLLM default).
    """

    # Emit a single aggregate warning every time the dropped-event count crosses
    # a multiple of this, so a missing/slow subscriber does not spam the log.
    _DROP_LOG_INTERVAL = 10000

    # End-of-replay sentinel: signed -1 as an 8-byte big-endian int, matching
    # vLLM's ZmqEventPublisher.END_SEQ so existing replay clients recognize it.
    END_SEQ = (-1).to_bytes(8, "big", signed=True)

    # Snapshot-follows sentinel for the replay protocol: when the requested
    # start_seq predates the ring buffer, the publisher replies with this header
    # (payload = snapshot_version as an 8-byte big-endian int), then an
    # AllBlocksCleared batch and the depth-ordered BlockStored rebuild. See
    # dynamo standalone_indexer/docs.md.
    SNAPSHOT_SEQ = (-2).to_bytes(8, "big", signed=True)

    # Number of BlockStored events packed into one snapshot batch. Packing keeps
    # the snapshot's frame count far below the replay socket's send HWM; one
    # event per frame would truncate a large tree (the failure this prevents).
    _SNAPSHOT_BATCH_BLOCKS = 1000

    def __init__(self,
                 zmq_endpoint: str,
                 kv_block_size: int,
                 replay_endpoint: Optional[str] = None,
                 buffer_steps: int = 10000,
                 queue_maxsize: int = 100000,
                 sndhwm: int = 100000,
                 topic: str = "") -> None:
        self.zmq_endpoint = zmq_endpoint
        self.kv_block_size = kv_block_size
        self.topic = topic
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PUB)
        # SAFETY: bound the outbound buffer so a missing/slow subscriber cannot
        # make the engine's drain loop block or grow memory without limit. Must
        # be set before bind() to take effect.
        self.socket.setsockopt(zmq.SNDHWM, sndhwm)
        self.socket.bind(zmq_endpoint)  # PUB binds (broadcast); subscribers connect.
        self.sequence = 0
        self.dropped = 0
        self.queue_dropped = 0

        # --- threading / replay state -------------------------------------- #
        # Inbound hand-off from the asyncio drain loop. queue.Queue is the
        # thread-safe primitive for the OS-thread consumer (vs asyncio.Queue,
        # which only serves coroutines). This is the ONLY object shared across
        # threads: the drain loop put_nowait()s, the consumer get()s.
        self._event_queue: "queue.Queue" = queue.Queue(maxsize=queue_maxsize)
        # Ring buffer of recently published (seq, payload) pairs. Touched ONLY
        # by the consumer thread (append on publish, read on replay), so it
        # needs no lock -- the single-thread invariant replaces it.
        self.replay_endpoint = replay_endpoint
        self._buffer: deque = deque(maxlen=buffer_steps)
        # Per-worker reconstructed tree, maintained on the consumer thread only
        # (no lock -- same single-thread invariant as _buffer). Populated only
        # when replay is enabled; used to serve a full snapshot when a
        # subscriber's requested start_seq has already been evicted from _buffer.
        self.kv_snapshot: dict[int, _SnapNode] = {}
        # Seq of the most recently published batch (== snapshot_version); -1
        # until the first publish. Stamped in _publish_event.
        self._last_published_seq = -1
        self._running = True
        self._replay_socket: Optional[zmq.Socket] = None
        if replay_endpoint is not None:
            # ROUTER lets one socket serve many DEALER/REQ clients and reply
            # request -> many batches. Owned by the consumer thread, same as
            # the PUB socket, so the two are never used concurrently.
            self._replay_socket = self.ctx.socket(zmq.ROUTER)
            # A replay streams the WHOLE ring buffer in one tight loop. ROUTER's
            # mute action is to DROP silently (not block), so a send HWM below
            # buffer_steps truncates large replays -- the subscriber gets a hole
            # in the middle, every later block fails ParentBlockNotFound, and its
            # tree collapses. Keep the HWM >= buffer_steps (queued count is capped
            # by the ring anyway) so a full replay always fits. Must be set before
            # bind() to take effect.
            replay_sndhwm = max(sndhwm, buffer_steps)
            self._replay_socket.setsockopt(zmq.SNDHWM, replay_sndhwm)
            self._replay_socket.bind(replay_endpoint)
            logger.info(
                f"KvZmqPublisher replay enabled on {replay_endpoint} "
                f"(buffer_steps={buffer_steps}, sndhwm={replay_sndhwm})")

        # --- translation / filtering state (mirrors the reference publisher) ---
        # Block hashes for partial blocks (fewer than kv_block_size tokens). They
        # are never stored by the router, so their removal events are suppressed.
        self.partial_block_hashes: set[int] = set()
        # Window-size tracking: the engine emits "created" events at startup, one
        # per attention window. We keep only events from the max-window (global
        # attention) layer to avoid duplicate hashes, matching router behavior.
        self.max_window_size: Optional[int] = None
        self.processing_initial_created_events = True
        # Engine event-id gap detection (best effort, for diagnostics only).
        self._last_engine_event_id: Optional[int] = None

        logger.info(
            f"KvZmqPublisher bound to {zmq_endpoint} (topic='{topic}', "
            f"kv_block_size={kv_block_size}, sndhwm={sndhwm})")

        # Single consumer thread owns the PUB socket, the ROUTER socket, the
        # ring buffer, and the translation state. Started last, so all the
        # state it touches in handle() is fully initialized before it can run.
        # It always runs (even with replay disabled) because it is what drains
        # the queue and publishes.
        self._consumer_thread = threading.Thread(
            target=self.zmq_loop, daemon=True, name="kv-zmq-consumer")
        self._consumer_thread.start()

    # ------------------------------------------------------------------ #
    # Producer side: non-blocking hand-off from the asyncio drain loop    #
    # ------------------------------------------------------------------ #
    def enqueue(self, event: dict) -> None:
        """Hand a raw native event to the consumer thread (never blocks).

        Called from ``OpenAIServer.kv_event_processor`` for every event,
        BEFORE the SSE path's KVHash conversion (which drops ``token_ids``).
        The event dict is shared read-only with that path -- neither side may
        mutate it. If the consumer thread has fallen behind and the queue is
        full, the event is dropped and counted; such drops are unrecoverable
        (unlike PUB-send drops, which replay can fix).
        """
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            self.queue_dropped += 1
            if self.queue_dropped % self._DROP_LOG_INTERVAL == 0:
                logger.warning(
                    f"KvZmqPublisher queue full: dropped {self.queue_dropped} "
                    f"events before publish (consumer thread fell behind)")

    # ------------------------------------------------------------------ #
    # Transport: native -> vLLM wire format (copied from reference)       #
    # ------------------------------------------------------------------ #
    def publish_stored(self,
                       tokens_hashes: list[int],
                       block_hashes: list[int],
                       parent_hash: Optional[int] = None,
                       block_mm_infos: Optional[list] = None,
                       attention_dp_rank: int = 0,
                       lora_name: Optional[str] = None) -> None:
        """Publish a BlockStored event in vLLM format.

        ``tokens_hashes`` carries one precomputed local block hash per block, in
        the same order as ``block_hashes``, replacing the old ``token_ids``
        array -- the router no longer hashes tokens.
        """
        block_hashes_signed = [_to_signed_i64(h) for h in block_hashes]
        tokens_hashes_signed = [_to_signed_i64(h) for h in tokens_hashes]
        parent_hash_signed = (_to_signed_i64(parent_hash)
                              if parent_hash is not None else None)

        event: dict[str, Any] = {
            "type": "BlockStored",
            "block_hashes": block_hashes_signed,
            "parent_block_hash": parent_hash_signed,
            "tokens_hashes": tokens_hashes_signed,
            "block_size": self.kv_block_size,
        }
        if lora_name is not None:
            event["lora_name"] = lora_name
        if block_mm_infos is not None:
            event["block_mm_infos"] = block_mm_infos

        # Record into the snapshot before publishing so kv_snapshot and
        # _last_published_seq (stamped in _publish_event) stay consistent.
        if self.replay_endpoint is not None:
            self._snapshot_record_stored(block_hashes_signed,
                                         tokens_hashes_signed,
                                         parent_hash_signed)

        self._publish_event(event, attention_dp_rank)

    def publish_removed(self,
                       block_hashes: list[int],
                       attention_dp_rank: int = 0) -> None:
        """Publish a BlockRemoved event in vLLM format."""
        block_hashes_signed = [_to_signed_i64(h) for h in block_hashes]
        event = {
            "type": "BlockRemoved",
            "block_hashes": block_hashes_signed,
        }
        if self.replay_endpoint is not None:
            for bh in block_hashes_signed:
                self.kv_snapshot.pop(bh, None)
        self._publish_event(event, attention_dp_rank)

    def publish_all_cleared(self) -> None:
        """Publish an AllBlocksCleared event in vLLM format."""
        if self.replay_endpoint is not None:
            self.kv_snapshot.clear()
        self._publish_event({"type": "AllBlocksCleared"})

    def _snapshot_record_stored(self, block_hashes_signed: list,
                                tokens_hashes_signed: list,
                                parent_hash_signed: Optional[int]) -> None:
        """Fold a published BlockStored into ``kv_snapshot``.

        Blocks within one event chain: block[0]'s parent is the event parent,
        block[i]'s parent is block[i-1]. ``depth`` is ``parent.depth + 1`` (0 for
        roots, whose ``parent_hash`` is None, or for a block whose parent is not
        yet known -- such orphans are skipped at emit time). Runs on the consumer
        thread only, so ``kv_snapshot`` needs no lock.
        """
        snapshot = self.kv_snapshot
        parent = parent_hash_signed
        for block_hash, tokens_hash in zip(block_hashes_signed,
                                           tokens_hashes_signed):
            if parent is None:
                depth = 0
            else:
                parent_node = snapshot.get(parent)
                depth = parent_node.depth + 1 if parent_node is not None else 0
            snapshot[block_hash] = _SnapNode(tokens_hash=tokens_hash,
                                             parent_hash=parent,
                                             depth=depth)
            parent = block_hash

    def _publish_event(self, event: dict, attention_dp_rank: int = 0) -> None:
        """Serialize and broadcast a single event (non-blocking)."""
        # vLLM batch format: [timestamp, [events], data_parallel_rank].
        timestamp = time.time()
        batch = [timestamp, [event], attention_dp_rank]
        payload = _encode_msgpack(batch)

        # The sequence counter is advanced unconditionally (even on drop) so a
        # subscriber sees an honest gap rather than silently-missing data.
        seq = self.sequence
        sequence_bytes = seq.to_bytes(8, byteorder="big")
        self.sequence += 1
        # Track the latest published seq; this is the snapshot_version reported
        # to a far-behind subscriber (see _send_snapshot).
        self._last_published_seq = seq

        # Buffer the batch for replay BEFORE attempting the live send, and do
        # so unconditionally -- even if the send below drops. A drop means a
        # subscriber fell behind (HWM full); the dropped batches are exactly
        # the ones it will ask to replay once it catches up. Buffering only on
        # success would leave a hole in the buffer at precisely the sequence
        # numbers replay needs. The maxlen ring evicts the oldest as it fills;
        # gaps older than that window are not recoverable from us. No lock:
        # this runs on the consumer thread, the only thread touching _buffer.
        if self._replay_socket is not None:
            self._buffer.append((seq, payload))

        try:
            self.socket.send_multipart(
                [self.topic.encode(), sequence_bytes, payload],
                flags=zmq.NOBLOCK)
        except zmq.Again:
            # No subscriber / buffer full: drop the LIVE send. Expected and
            # harmless for a read-only tee -- the batch is still in the replay
            # buffer above. Log only at coarse intervals to avoid spam.
            self.dropped += 1
            if self.dropped % self._DROP_LOG_INTERVAL == 0:
                logger.warning(
                    f"KvZmqPublisher dropped {self.dropped} events "
                    f"(no subscriber or HWM reached on {self.zmq_endpoint})")

    # ------------------------------------------------------------------ #
    # Translation + filtering (adapted from reference _handle_kv_event)   #
    # ------------------------------------------------------------------ #
    def handle(self, event: dict) -> None:
        """Translate one native KV-cache event and publish it to ZMQ.
        ``event`` is the raw native dict as produced by
        ``KVCacheEventSerializer`` (the same shape ``kv_event_processor``
        consumes), so ``token_ids`` are still present.
        """
        # Drop events that are not from the global (max-window) attention layer.
        if self.should_drop_event(event):
            return

        event_id = event["event_id"]
        if self._last_engine_event_id is not None:
            expected_id = self._last_engine_event_id + 1
            if event_id != expected_id:
                logger.warning(
                    f"Non-consecutive engine event_id: expected {expected_id}, "
                    f"got {event_id}")
        self._last_engine_event_id = event_id

        data = event["data"]
        event_type = data["type"]

        if event_type == "stored":
            self.processing_initial_created_events = False
            parent_hash = _to_signed_i64(data["parent_hash"])
            # lora_name applies to the whole sequence and feeds the hash seed,
            # so read it before hashing each block below.
            lora_name = data.get("lora_name")
            tokens_hashes: list[int] = []
            block_hashes: list[int] = []
            block_mm_infos: list[Optional[dict]] = []
            kv_block_size = self.kv_block_size
            partial_block_hashes = self.partial_block_hashes
            for block in data["blocks"]:
                block_tokens = block["tokens"]
                token_num_in_block = len(block_tokens)
                if token_num_in_block > kv_block_size:
                    logger.error(
                        f"Block contains {token_num_in_block} tokens, which is "
                        f"greater than kv_block_size {kv_block_size}")
                    return
                block_hash = _to_signed_i64(block["block_hash"])
                if block_hash is None:
                    logger.warning(
                        f"Skipping block with None hash containing "
                        f"{token_num_in_block} tokens")
                    continue
                # Partial block: not stored by the router. Record its hash so we
                # can suppress its later removal event, then stop (a partial
                # block is always the last block in a stored event).
                if token_num_in_block < kv_block_size:
                    partial_block_hashes.add(block_hash)
                    break

                # Multimodal object hashes for this block, if any. They are both
                # folded into the token hash (so blocks with identical tokens but
                # different mm objects hash differently) and passed through as
                # block_mm_infos (the indexer keeps them as mm_extra_info).
                mm_keys = block.get("mm_keys")
                mm_hashes = None
                if mm_keys:
                    mm_hashes = [
                        int(mk["hash"][:16], 16) for mk in mm_keys
                        if mk.get("type") == "mm_key" and mk.get("hash")
                    ]
                    if mm_hashes:
                        block_mm_infos.append({
                            "mm_objects": [{
                                "mm_hash": h,
                                "offsets": []
                            } for h in mm_hashes]
                        })
                    else:
                        mm_hashes = None
                        block_mm_infos.append(None)
                else:
                    block_mm_infos.append(None)

                # Hash the block's tokens here (the router no longer does it) and
                # discard the token ids -- only the hash rides the wire.
                token_ids_in_block = [int(t["token_id"]) for t in block_tokens]
                tokens_hashes.append(
                    _compute_block_tokens_hash(token_ids_in_block, lora_name,
                                               mm_hashes))
                block_hashes.append(block_hash)

            attention_dp_rank = event.get("attention_dp_rank", 0)
            self.publish_stored(tokens_hashes, block_hashes, parent_hash,
                                block_mm_infos, attention_dp_rank, lora_name)

        elif event_type == "removed":
            self.processing_initial_created_events = False
            removed_block_hashes: list[int] = []
            for block_hash in data["block_hashes"]:
                block_hash = _to_signed_i64(block_hash)
                if block_hash is None:
                    continue
                if block_hash in self.partial_block_hashes:
                    # Partial blocks were never published as stored; skip their
                    # removal too.
                    self.partial_block_hashes.remove(block_hash)
                    continue
                removed_block_hashes.append(block_hash)

            attention_dp_rank = event.get("attention_dp_rank", 0)
            self.publish_removed(removed_block_hashes, attention_dp_rank)

        elif event_type == "created" and self.processing_initial_created_events:
            self.update_max_window_size(event)

    def update_max_window_size(self, event: dict) -> None:
        """Track the largest attention window seen in the initial events."""
        if "window_size" in event:
            window_size = event["window_size"]
            if self.max_window_size is None or window_size > self.max_window_size:
                self.max_window_size = window_size
                logger.debug(
                    f"kv events max_window_size updated to {self.max_window_size}")

    def should_drop_event(self, event: dict) -> bool:
        """Keep only events from the global (max-window) attention layer.

        Two cases:
        1. No ``window_size`` in the event (older engines): keep everything.
        2. ``window_size`` present: keep all events until the initial "created"
           events have been processed (so ``max_window_size`` is known), then
           accept only events whose ``window_size`` equals the max.
        """
        if "window_size" not in event or self.processing_initial_created_events:
            return False
        return event["window_size"] != self.max_window_size

    # ------------------------------------------------------------------ #
    # Consumer thread: drain queue -> publish, and service replay         #
    # ------------------------------------------------------------------ #
    def zmq_loop(self) -> None:
        """Single thread that publishes queued events and answers replay.

        Mirrors vLLM's ``ZmqEventPublisher._publisher_thread``: each iteration
        first services a pending replay request (non-critical, ``poll(0)`` so
        it never blocks the publish path), then pulls one event off the queue
        and publishes it. Because this is the only thread touching the sockets
        and the ring buffer, no lock is required. Keeps running until shutdown
        AND the queue has been drained, so in-flight events are not lost.
        """
        while self._running or not self._event_queue.empty():
            if self._replay_socket is not None and self._replay_socket.poll(0):
                try:
                    self._service_replay()
                except Exception as e:
                    logger.warning(f"KvZmqPublisher replay error: {e}")

            try:
                event = self._event_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if event is _SHUTDOWN:
                break
            try:
                # handle() translates/filters and calls _publish_event, which
                # appends to the buffer and sends on the PUB socket.
                self.handle(event)
            except Exception as e:
                logger.warning(f"KvZmqPublisher handle error (event dropped): {e}")
                continue

    def _service_replay(self) -> None:
        """Answer one replay request: incremental batches or a full snapshot.

        Request frame (from a DEALER/REQ client, identity prepended by ROUTER):
        ``[client_id, b"", start_seq_bytes]``.

        If ``start_seq`` is still in the ring buffer, reply incrementally: one
        multipart per batch ``[client_id, b"", seq_bytes, payload]`` for every
        ``seq >= start_seq``. If ``start_seq`` predates the buffer (evicted), the
        requested data is gone, so reply with a full snapshot instead (see
        ``_send_snapshot``) -- otherwise the subscriber would get a hole and its
        tree would collapse. Either way, finish with a sentinel
        ``[client_id, b"", END_SEQ, b""]``. Runs on the consumer thread, so it
        reads ``_buffer``/``kv_snapshot`` directly -- no other thread mutates them.
        """
        assert self._replay_socket is not None
        frame = self._replay_socket.recv_multipart()
        if len(frame) != 3:
            logger.warning(f"KvZmqPublisher invalid replay request: {frame}")
            return
        client_id, _, start_seq_bytes = frame
        start_seq = int.from_bytes(start_seq_bytes, "big")

        if not self._buffer:
            # Nothing buffered yet: nothing to replay and no snapshot to build.
            self._replay_socket.send_multipart(
                [client_id, b"", self.END_SEQ, b""])
            return

        if start_seq < self._buffer[0][0]:
            # Requested data has already been evicted from the ring buffer ->
            # serve a full snapshot (clear + depth-ordered rebuild) instead of a
            # hole that would collapse the subscriber's tree.
            self._send_snapshot(client_id)
        else:
            for seq, payload in self._buffer:
                if seq >= start_seq:
                    self._replay_socket.send_multipart(
                        [client_id, b"", seq.to_bytes(8, "big"), payload])
        # End-of-sequence marker so the client knows replay is complete.
        self._replay_socket.send_multipart(
            [client_id, b"", self.END_SEQ, b""])

    def _send_snapshot(self, client_id: bytes) -> None:
        """Stream a full snapshot of the current tree to one replay client.

        Wire (see dynamo standalone_indexer/docs.md):
          [id, b"", SNAPSHOT_SEQ, S]              header; S = snapshot_version
          [id, b"", S, batch([AllBlocksCleared])] clear the worker first
          [id, b"", S, batch([BlockStored, ...])] depth-ordered, packed rebuild
          ...                                     (END_SEQ is sent by the caller)

        Runs on the consumer thread, so kv_snapshot is not mutated concurrently
        and S == _last_published_seq is consistent with its contents.
        """
        S = self._last_published_seq
        s_bytes = S.to_bytes(8, "big")

        # Header: tells the subscriber a snapshot follows and which watermark to
        # adopt once it ends.
        self._replay_socket.send_multipart(
            [client_id, b"", self.SNAPSHOT_SEQ, s_bytes])

        # Clear the worker's stale state before the rebuild. A plain vLLM batch
        # so the existing apply path handles it (and fans out to lower tiers).
        clear_batch = [time.time(), [{"type": "AllBlocksCleared"}], 0]
        self._replay_socket.send_multipart(
            [client_id, b"", s_bytes, _encode_msgpack(clear_batch)])

        # Depth-ordered rebuild: a parent has strictly smaller depth than its
        # child, so sorting by depth guarantees parents are emitted first. Skip
        # orphans (a non-root block whose parent was evicted) -- emitting one
        # would be rejected (ParentBlockNotFound) and strand its subtree; they
        # are re-learned from the live stream.
        present: set = set()
        skipped = 0
        emitted = 0
        batch_events: list = []

        def _flush() -> None:
            if not batch_events:
                return
            batch = [time.time(), list(batch_events), 0]
            self._replay_socket.send_multipart(
                [client_id, b"", s_bytes, _encode_msgpack(batch)])
            batch_events.clear()

        for block_hash, node in sorted(self.kv_snapshot.items(),
                                       key=lambda item: item[1].depth):
            if node.parent_hash is not None and node.parent_hash not in present:
                skipped += 1
                continue
            present.add(block_hash)
            batch_events.append({
                "type": "BlockStored",
                "block_hashes": [block_hash],
                "parent_block_hash": node.parent_hash,
                "tokens_hashes": [node.tokens_hash],
                "block_size": self.kv_block_size,
            })
            emitted += 1
            if len(batch_events) >= self._SNAPSHOT_BATCH_BLOCKS:
                _flush()
        _flush()

        if skipped:
            logger.warning(
                f"KvZmqPublisher snapshot skipped {skipped} orphan block(s) "
                f"(parent evicted) while serving version {S}")
        logger.info(
            f"KvZmqPublisher served snapshot version={S} ({emitted} blocks)")

    def shutdown(self) -> None:
        """Stop the consumer thread, close sockets, and terminate the context."""
        # Signal the consumer to exit and wake it immediately with a sentinel
        # (so it doesn't wait out the get() timeout), then wait for it to stop
        # touching the sockets before we close them.
        self._running = False
        try:
            self._event_queue.put_nowait(_SHUTDOWN)
        except queue.Full:
            pass
        if self._consumer_thread is not None:
            self._consumer_thread.join(timeout=2.0)
        try:
            if self.socket is not None:
                self.socket.close()
            if self._replay_socket is not None:
                self._replay_socket.close(linger=0)
            if self.ctx is not None:
                self.ctx.term()
        finally:
            logger.info(
                f"KvZmqPublisher shut down ({self.dropped} events dropped total)")
