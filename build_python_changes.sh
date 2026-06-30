#!/usr/bin/env bash
# Build a TRT-LLM engine image with local Python changes patched in.
#
# Usage: build_python_changes.sh <output-tag> <pytrtllm-builder-tag>
#   e.g. build_python_changes.sh v1.3.0rc16-myfeature v1.3.0rc16-deepmain
#
# The KV local indexer is a prebuilt PUBLIC wheel that Dockerfile.python installs
# straight from its GitHub release -- no Rust, no auth, nothing fetched here. To
# change the indexer version, edit KV_WHEEL_URL in Dockerfile.python.
set -euo pipefail

git diff pytrtllm-builder/"$2" -- tensorrt_llm > python.patch

docker build -t localhost:30500/pytrtllm:"$1" --build-arg url=localhost:30500/pytrtllm-builder:"$2" . -f Dockerfile.python
