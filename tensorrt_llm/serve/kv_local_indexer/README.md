# kv-local-indexer

A thin PyO3 wrapper around Dynamo's `LocalKvIndexer`, used by the TRT-LLM
KV-event worker (`tensorrt_llm/serve/kv_zmq_publisher.py`).

It gives each worker an in-process radix tree + ring buffer of recent KV events,
so the standalone global indexer can recover missed events (`Events`) or pull a
full snapshot (`TreeDump`) directly from the worker. This replaces the old
fixed-window ZMQ replay deque.

## What it exposes

`LocalIndexer(worker_id, kv_block_size, max_buffer_size=10000)` with:

- `apply_stored(seq, token_ids, block_hashes, parent_hash=None, dp_rank=0, lora_name=None, is_eagle=None)`
- `apply_removed(seq, block_hashes, dp_rank=0)`
- `apply_cleared(seq, dp_rank=0)`
- `get_events_json(start=None, end=None) -> str`  (the recovery query)
- `shutdown()`

`seq` is used directly as the event's `event_id` (one event per ZMQ batch).

## Build

Requires the Rust toolchain pinned in `rust-toolchain.toml` (1.93.1) and the
Dynamo build prerequisites (libclang; see Dynamo's bindings build notes). The
crate path-depends on `../../../../dynamo/lib/kv-router`; adjust or switch to a
pinned git dep for CI/image builds (see `Cargo.toml`).

```bash
cd tensorrt_llm/serve/kv_local_indexer
maturin develop --release      # build + install into the active venv
# or
maturin build --release        # produce a wheel under target/wheels/
```

Then `from kv_local_indexer import LocalIndexer` works in the engine image.
