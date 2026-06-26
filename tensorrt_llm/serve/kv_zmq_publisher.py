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
reconstruct the cache state -- including the ``token_ids`` that the SSE
``/kv_cache_events`` path drops.

Threading model: the engine's asyncio drain loop
(``OpenAIServer.kv_event_processor``) only calls ``enqueue()``, a non-blocking
hand-off onto a bounded ``queue.Queue``. A single dedicated consumer thread
owns everything else -- it pulls raw events off the queue, translates/filters
them, publishes on the PUB socket, and (when enabled) feeds each event into the
in-process ``LocalKvIndexer``. Because exactly one thread touches the PUB socket
and the translation state, no lock is needed (this mirrors vLLM's
``ZmqEventPublisher._publisher_thread``). The ZMQ-side work (msgpack encode +
send) is thus moved off the asyncio loop and never stalls the SSE path.

Recovery: instead of a fixed-window ZMQ ROUTER replay deque, each event is also
fed into a Dynamo ``LocalKvIndexer`` (radix tree + replay ring buffer +
snapshot), enabled via ``enable_local_indexer``. A consumer that misses events
queries the worker out-of-band (HTTP, served elsewhere) and gets either the
buffered events or a full tree snapshot. The indexer is fed ``event_id == seq``
(the same ZMQ batch sequence), so the consumer keeps detecting gaps on ``seq``.

Because this sits next to the production routing path, every hand-off is
bounded and lossy by design: the inbound queue is bounded (overflow increments
``queue_dropped`` -- unrecoverable, the consumer fell behind), and the PUB
socket is bounded (``SNDHWM``) with non-blocking sends (overflow increments
``dropped`` -- recoverable from the local indexer).

The translation logic mirrors the reference ``ZmqKvEventPublisher`` /
``_handle_kv_event`` in ``dynamo/components/src/dynamo/trtllm/publisher.py``
(partial-block tracking, attention-DP rank passthrough, window-size filtering,
event-id gap detection) so the emitted stream is byte-compatible with what a
Dynamo indexer expects.
"""

import queue
import threading
import time
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

    Recovery: when ``enable_local_indexer`` is set, every published event is also
    fed into an in-process Dynamo ``LocalKvIndexer`` (radix tree + replay ring
    buffer of ``buffer_steps`` events + snapshot), keyed by ``event_id == seq``.
    A consumer that detects a sequence gap recovers out-of-band by querying the
    worker (HTTP, served by the engine) and receives either the buffered events
    or a full tree snapshot. This replaces the old fixed-window ZMQ ROUTER replay.

    Args:
        zmq_endpoint: endpoint to bind the PUB socket to, e.g. ``tcp://*:5557``.
        kv_block_size: KV-cache block size in tokens (``kv_cache_config.tokens_per_block``).
        enable_local_indexer: build the in-process ``LocalKvIndexer`` used for
            recovery. When ``False`` the consumer thread still drains the queue
            and publishes, but no recovery state is kept.
        worker_id: stable id stamped on indexed events (informational for
            recovery; the consumer applies events under the worker it queried).
        buffer_steps: capacity of the local indexer's replay ring buffer (events).
        queue_maxsize: bound on the inbound event queue. When the consumer
            thread cannot keep up the queue fills and the oldest-arriving events
            are dropped (``queue_dropped``); these are unrecoverable.
        sndhwm: send high-water mark (max queued messages before drops kick in).
        topic: ZMQ topic to publish on (empty string == all topics, vLLM default).
    """

    # Emit a single aggregate warning every time the dropped-event count crosses
    # a multiple of this, so a missing/slow subscriber does not spam the log.
    _DROP_LOG_INTERVAL = 10000

    def __init__(self,
                 zmq_endpoint: str,
                 kv_block_size: int,
                 enable_local_indexer: bool = False,
                 worker_id: int = 0,
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
        self._running = True

        # In-process recovery state: a Dynamo LocalKvIndexer (radix tree +
        # replay ring buffer + snapshot), fed every published event keyed by
        # event_id == seq. Stays None when disabled. Fed only by the consumer
        # thread; queried by the (out-of-band) recovery handler. The indexer is
        # internally synchronized, so no lock is needed here.
        self._indexer = None
        if enable_local_indexer:
            # Imported lazily so the tee works without the compiled wrapper
            # unless recovery is actually enabled.
            from kv_local_indexer import LocalIndexer
            self._indexer = LocalIndexer(worker_id, kv_block_size, buffer_steps)
            logger.info(
                f"KvZmqPublisher local indexer enabled "
                f"(worker_id={worker_id}, buffer_steps={buffer_steps})")

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

        # Single consumer thread owns the PUB socket and the translation state,
        # and feeds the local indexer. Started last, so all the state it touches
        # in handle() is fully initialized before it can run. It always runs
        # (even without the local indexer) because it drains the queue and
        # publishes.
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
                       token_ids: list[int],
                       num_block_tokens: list[int],
                       block_hashes: list[int],
                       parent_hash: Optional[int] = None,
                       block_mm_infos: Optional[list] = None,
                       attention_dp_rank: int = 0,
                       lora_name: Optional[str] = None) -> None:
        """Publish a BlockStored event in vLLM format."""
        block_hashes_signed = [_to_signed_i64(h) for h in block_hashes]
        parent_hash_signed = (_to_signed_i64(parent_hash)
                              if parent_hash is not None else None)

        event: dict[str, Any] = {
            "type": "BlockStored",
            "block_hashes": block_hashes_signed,
            "parent_block_hash": parent_hash_signed,
            "token_ids": token_ids,
            "block_size": self.kv_block_size,
        }
        if lora_name is not None:
            event["lora_name"] = lora_name
        if block_mm_infos is not None:
            event["block_mm_infos"] = block_mm_infos

        seq = self._publish_event(event, attention_dp_rank)
        self._feed_stored(seq, token_ids, block_hashes_signed,
                          parent_hash_signed, attention_dp_rank, lora_name)

    def publish_removed(self,
                       block_hashes: list[int],
                       attention_dp_rank: int = 0) -> None:
        """Publish a BlockRemoved event in vLLM format."""
        block_hashes_signed = [_to_signed_i64(h) for h in block_hashes]
        event = {
            "type": "BlockRemoved",
            "block_hashes": block_hashes_signed,
        }
        seq = self._publish_event(event, attention_dp_rank)
        self._feed_removed(seq, block_hashes_signed, attention_dp_rank)

    def publish_all_cleared(self) -> None:
        """Publish an AllBlocksCleared event in vLLM format."""
        seq = self._publish_event({"type": "AllBlocksCleared"})
        self._feed_cleared(seq)

    def _publish_event(self, event: dict, attention_dp_rank: int = 0) -> int:
        """Serialize and broadcast a single event; return its sequence number.

        The returned ``seq`` is used by the caller as the local indexer's
        ``event_id``, so the live ZMQ stream and the indexer share one id space.
        """
        # vLLM batch format: [timestamp, [events], data_parallel_rank].
        timestamp = time.time()
        batch = [timestamp, [event], attention_dp_rank]
        payload = _encode_msgpack(batch)

        # The sequence counter is advanced unconditionally (even on drop) so a
        # subscriber sees an honest gap rather than silently-missing data.
        seq = self.sequence
        sequence_bytes = seq.to_bytes(8, byteorder="big")
        self.sequence += 1

        try:
            self.socket.send_multipart(
                [self.topic.encode(), sequence_bytes, payload],
                flags=zmq.NOBLOCK)
        except zmq.Again:
            # No subscriber / buffer full: drop the LIVE send. Expected and
            # harmless for a read-only tee -- the event is still recoverable
            # from the local indexer. Log only at coarse intervals to avoid spam.
            self.dropped += 1
            if self.dropped % self._DROP_LOG_INTERVAL == 0:
                logger.warning(
                    f"KvZmqPublisher dropped {self.dropped} events "
                    f"(no subscriber or HWM reached on {self.zmq_endpoint})")

        return seq

    # ------------------------------------------------------------------ #
    # Feed the in-process local indexer (recovery state)                  #
    # ------------------------------------------------------------------ #
    # These run on the consumer thread in seq order, so the indexer's replay
    # buffer stays ordered and consecutive (event_id == seq). An indexing
    # failure is logged and swallowed so it never breaks the live publish path.
    def _feed_stored(self, seq, token_ids, block_hashes, parent_hash,
                     attention_dp_rank, lora_name) -> None:
        if self._indexer is None:
            return
        try:
            self._indexer.apply_stored(seq, token_ids, block_hashes,
                                       parent_hash, attention_dp_rank, lora_name)
        except Exception as e:
            logger.warning(
                f"KvZmqPublisher local indexer apply_stored failed: {e}")

    def _feed_removed(self, seq, block_hashes, attention_dp_rank) -> None:
        if self._indexer is None:
            return
        try:
            self._indexer.apply_removed(seq, block_hashes, attention_dp_rank)
        except Exception as e:
            logger.warning(
                f"KvZmqPublisher local indexer apply_removed failed: {e}")

    def _feed_cleared(self, seq) -> None:
        if self._indexer is None:
            return
        try:
            self._indexer.apply_cleared(seq)
        except Exception as e:
            logger.warning(
                f"KvZmqPublisher local indexer apply_cleared failed: {e}")

    # ------------------------------------------------------------------ #
    # Recovery query (served over HTTP by the engine)                     #
    # ------------------------------------------------------------------ #
    def get_recovery_json(self,
                          start: Optional[int] = None,
                          end: Optional[int] = None) -> Optional[str]:
        """Return recovery events for ``[start, end]`` as a JSON string.

        ``start`` is the first ``event_id`` (== ZMQ ``seq``) the caller is
        missing; ``None`` requests a full snapshot. The result is an
        externally-tagged ``WorkerKvQueryResponse`` (``Events`` / ``TreeDump`` /
        ``TooNew`` / ``InvalidRange`` / ``Error``). Returns ``None`` when the
        local indexer is disabled.

        Thread-safe (the indexer is internally synchronized), but may be heavy
        (a full tree dump), so callers should run it off the asyncio event loop.
        """
        if self._indexer is None:
            return None
        return self._indexer.get_events_json(start, end)

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
            token_ids: list[int] = []
            num_block_tokens: list[int] = []
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
                num_block_tokens.append(token_num_in_block)
                block_hashes.append(block_hash)
                token_ids.extend(int(t["token_id"]) for t in block_tokens)

                mm_keys = block.get("mm_keys")
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
                        block_mm_infos.append(None)
                else:
                    block_mm_infos.append(None)

            lora_name = data.get("lora_name")
            attention_dp_rank = event.get("attention_dp_rank", 0)
            self.publish_stored(token_ids, num_block_tokens, block_hashes,
                                parent_hash, block_mm_infos, attention_dp_rank,
                                lora_name)

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
    # Consumer thread: drain queue -> translate -> publish + feed indexer  #
    # ------------------------------------------------------------------ #
    def zmq_loop(self) -> None:
        """Single thread that translates queued events and publishes them.

        Mirrors vLLM's ``ZmqEventPublisher._publisher_thread``: pull one event
        off the queue, translate/filter it, publish on the PUB socket, and feed
        the local indexer. Because this is the only thread touching the PUB
        socket and the translation state, no lock is required. Keeps running
        until shutdown AND the queue has been drained, so in-flight events are
        not lost.
        """
        while self._running or not self._event_queue.empty():
            try:
                event = self._event_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if event is _SHUTDOWN:
                break
            try:
                # handle() translates/filters and calls publish_*, which sends
                # on the PUB socket and feeds the local indexer.
                self.handle(event)
            except Exception as e:
                logger.warning(f"KvZmqPublisher handle error (event dropped): {e}")
                continue

    def shutdown(self) -> None:
        """Stop the consumer thread, close the socket, and terminate the context."""
        # Signal the consumer to exit and wake it immediately with a sentinel
        # (so it doesn't wait out the get() timeout), then wait for it to stop
        # touching the socket before we close it.
        self._running = False
        try:
            self._event_queue.put_nowait(_SHUTDOWN)
        except queue.Full:
            pass
        if self._consumer_thread is not None:
            self._consumer_thread.join(timeout=2.0)
        if self._indexer is not None:
            try:
                self._indexer.shutdown()
            except Exception as e:
                logger.warning(
                    f"KvZmqPublisher local indexer shutdown failed: {e}")
        try:
            if self.socket is not None:
                self.socket.close()
            if self.ctx is not None:
                self.ctx.term()
        finally:
            logger.info(
                f"KvZmqPublisher shut down ({self.dropped} events dropped total)")
