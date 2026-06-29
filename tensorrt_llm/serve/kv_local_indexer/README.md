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

## Dependency on Dynamo

This crate depends on `dynamo-kv-router`, pinned in `Cargo.toml` to a **public
upstream git commit** of https://github.com/ai-dynamo/dynamo.git:

    rev = "b2b7868090a22e02a0cf5733eea7d53e8e0f0829"   # ai-dynamo/dynamo main

You do **not** need a local dynamo checkout — cargo fetches that commit during
the build. We build with `default-features = false`, which compiles only the
in-process indexer + protocols. The `standalone-indexer` feature is OFF, so the
deepinfra-fork changes (which live entirely under that feature) are not compiled
here; upstream at this rev is byte-identical to the fork for what this wrapper
needs. To bump it, change `rev`, rebuild (regenerates `Cargo.lock`), commit both.

## Build

Requires the Rust toolchain pinned in `rust-toolchain.toml` (1.93.1) plus
libclang and maturin, **and network access** (to fetch the dynamo git rev).

The normal path is the containerized build, which produces the cp312 wheel that
`Dockerfile.python` installs:

```bash
tensorrt_llm/serve/kv_local_indexer/build_wheel.sh <pytrtllm-builder-tag>
# -> drops kv_local_indexer-*.whl into <repo>/wheels/  (committed to the repo)
```

For local dev in an active venv:

```bash
cd tensorrt_llm/serve/kv_local_indexer
maturin develop --release      # build + install into the active venv
# or
maturin build --release        # produce a wheel under target/wheels/
```

Then `from kv_local_indexer import LocalIndexer` works in the engine image.

## What's committed (so others don't rebuild from scratch)

- **Source** (`src/lib.rs`, `Cargo.toml`, build glue) — always.
- **`Cargo.lock`** — pins every transitive dep for reproducible builds.
- **The prebuilt wheel** under `<repo>/wheels/` — lets images build with no Rust
  toolchain or dynamo fetch. ⚠️ It is a cp312 / x86_64 / glibc binary tied to the
  base image's ABI, and it can go stale: **rebuild and re-commit it whenever you
  change `src/lib.rs` or the dynamo `rev`.**
