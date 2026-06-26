#!/usr/bin/env bash
# Build the kv_local_indexer wheel INSIDE the TRT-LLM base image, so the
# Python/glibc ABI matches the engine image it will be installed into.
#
# The resulting wheel lands in <TensorRT-LLM>/wheels/, which Dockerfile.python
# COPYs and pip-installs.
#
# Usage:
#   build_wheel.sh <pytrtllm-builder-tag>
#
# Requirements:
#   - The dynamo checkout must sit next to the TensorRT-LLM checkout (the crate
#     path-deps ../../../../dynamo/lib/kv-router). Both must be under one parent.
#   - The base-image container needs the Rust 1.93.1 toolchain + libclang +
#     maturin. This script installs them if missing; in an air-gapped setup,
#     pre-bake them into the base image or build on a host that already has the
#     Dynamo build toolchain (then double-check it targets python3.12).
set -euo pipefail

BASE_TAG="${1:?usage: build_wheel.sh <pytrtllm-builder-tag>}"
BASE_IMAGE="localhost:30500/pytrtllm-builder:${BASE_TAG}"

CRATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRTLLM_ROOT="$(cd "${CRATE_DIR}/../../.." && pwd)"   # .../TensorRT-LLM
PARENT="$(cd "${TRTLLM_ROOT}/.." && pwd)"            # must also contain dynamo/
OUT_DIR="${TRTLLM_ROOT}/wheels"
mkdir -p "${OUT_DIR}"

if [[ ! -d "${PARENT}/dynamo/lib/kv-router" ]]; then
  echo "ERROR: ${PARENT}/dynamo/lib/kv-router not found." >&2
  echo "       The crate path-deps it; check out dynamo next to TensorRT-LLM." >&2
  exit 1
fi

echo "Building kv_local_indexer wheel in ${BASE_IMAGE} ..."
# Mount only the two repos to fresh /work/* paths (not the whole home dir to
# /src), and cd inside the command rather than using -w: bind-mounting over an
# existing dir + auto-created workdir trips runc on some docker versions
# ("mkdir ...: file exists"). The relative path-dep resolves because both repos
# land as siblings under /work.
docker run --rm \
  -v "${TRTLLM_ROOT}":/work/TensorRT-LLM \
  -v "${PARENT}/dynamo":/work/dynamo \
  "${BASE_IMAGE}" bash -lc '
    set -euo pipefail
    cd /work/TensorRT-LLM/tensorrt_llm/serve/kv_local_indexer
    export RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
    export PATH=/opt/cargo/bin:${PATH}
    # Keep build artifacts inside the container, not in the mounted source tree.
    export CARGO_TARGET_DIR=/tmp/kv_target
    if ! command -v cargo >/dev/null 2>&1; then
      curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain 1.93.1
    fi
    command -v clang >/dev/null 2>&1 || \
      (apt-get update && apt-get install -y --no-install-recommends clang libclang-dev)
    pip install -q -U maturin
    # --compatibility linux: this wheel is installed into the same image it is
    # built from, so skip the manylinux audit (it links system libs).
    maturin build --release -i python3.12 --compatibility linux \
      --out /work/TensorRT-LLM/wheels
  '

echo "Done. Wheels in ${OUT_DIR}:"
ls -1 "${OUT_DIR}"/*.whl
