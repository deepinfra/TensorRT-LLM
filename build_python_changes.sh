#!/usr/bin/env bash
# Build a TRT-LLM engine image with local Python changes patched in.
#
# Usage: build_python_changes.sh <output-tag> <pytrtllm-builder-tag>
#   e.g. build_python_changes.sh v1.3.0rc16-myfeature v1.3.0rc16-deepmain
#
# The KV local indexer is NOT built here -- there is no Rust in this repo. It is a
# prebuilt wheel pulled from the deepinfra/kv-local-indexer GitHub release. Bump
# KV_VER to pick up a newer indexer. Needs `gh auth login` once (private repo;
# an SSH key alone cannot fetch release assets).
set -euo pipefail

KV_VER=v0.1.0

git diff pytrtllm-builder/"$2" -- tensorrt_llm > python.patch

# Pull the pinned, prebuilt kv_local_indexer wheel into ./wheels (Dockerfile.python
# installs it). Clear stale wheels first so exactly KV_VER is installed. Tolerant:
# if the download fails (not logged in / offline) the image still builds, just
# without KV recovery -- the warning makes that visible.
mkdir -p wheels && rm -f wheels/*.whl
gh release download "$KV_VER" --repo deepinfra/kv-local-indexer \
    --pattern '*.whl' --dir wheels --clobber \
  || echo "WARN: could not fetch kv_local_indexer ${KV_VER}; building WITHOUT it"

docker build -t localhost:30500/pytrtllm:"$1" --build-arg url=localhost:30500/pytrtllm-builder:"$2" . -f Dockerfile.python
