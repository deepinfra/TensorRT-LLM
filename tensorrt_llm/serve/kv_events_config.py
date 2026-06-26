# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serving-layer config for the read-only KV-cache event ZMQ tee.

Field names and defaults mirror vLLM's ``vllm.config.KVEventsConfig`` so the
operator-facing knobs line up across engines (and the wire format already
matches). A single ``--kv_events_config`` JSON object replaces the older pair of
``--kv_events_zmq_endpoint`` / ``--kv_events_replay_endpoint`` flags.

This module is dependency-light (stdlib only) on purpose: it is imported at CLI
parse time, before the heavier ``KvZmqPublisher`` (which pulls in ``zmq``).
"""

import dataclasses
import json
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class KVEventsConfig:
    """Configuration for KV-cache event publishing over ZMQ.

    Mirrors vLLM's ``KVEventsConfig``:

    - ``enable_kv_cache_events``: master switch; the tee is off unless this is
      true (matches vLLM, where presence of an endpoint alone does nothing).
    - ``publisher``: ``"zmq"`` to publish, ``"null"`` to disable. Set
      automatically from ``enable_kv_cache_events`` when left at the default.
    - ``endpoint``: PUB address. ``tcp://*:5557`` to bind (broadcast).
    - ``replay_endpoint``: DEPRECATED. The old ZMQ ROUTER replay path has been
      removed; recovery now goes through the in-process local KV indexer
      (``enable_local_indexer``) over HTTP. Kept only so existing config JSON
      that still sets it does not fail to parse; the value is ignored.
    - ``enable_local_indexer``: build an in-process Dynamo ``LocalKvIndexer``
      (radix tree + replay ring buffer + snapshot) that the standalone global
      indexer can query to recover missed events or pull a full snapshot.
    - ``buffer_steps``: number of recent events kept for replay (the local
      indexer's ring-buffer capacity).
    - ``hwm``: ZeroMQ high-water-mark for the PUB socket.
    - ``max_queue_size``: max events buffered in memory before drops.
    - ``topic``: ZMQ topic prefix on every published frame.
    """

    enable_kv_cache_events: bool = False
    publisher: Optional[Literal["null", "zmq"]] = None
    endpoint: str = "tcp://*:5557"
    replay_endpoint: Optional[str] = None  # deprecated, ignored (see docstring)
    enable_local_indexer: bool = False
    buffer_steps: int = 10000
    hwm: int = 100000
    max_queue_size: int = 100000
    topic: str = ""

    def __post_init__(self):
        # Mirror vLLM exactly: an unset (None) publisher follows the master
        # switch; an explicitly-set publisher (incl. "null") is left untouched.
        if self.publisher is None:
            self.publisher = "zmq" if self.enable_kv_cache_events else "null"

    @classmethod
    def from_cli(cls, value: Optional[str]) -> Optional["KVEventsConfig"]:
        """Parse the ``--kv_events_config`` JSON string into a config.

        Returns ``None`` when no value is given (tee disabled). Raises
        ``ValueError`` on malformed JSON or unknown keys, so a typo fails fast
        at startup instead of silently disabling the tee.
        """
        if not value:
            return None
        try:
            data = json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"--kv_events_config is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ValueError("--kv_events_config must be a JSON object")
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"--kv_events_config has unknown keys {sorted(unknown)}; "
                f"valid keys are {sorted(known)}")
        return cls(**data)
