git diff 415509c9a2f432db4eb2e4d5f5efbe5e03bfcb37 -- tensorrt_llm > python.patch

docker build -t localhost:30500/pytrtllm:"$1" --build-arg url=localhost:30500/pytrtllm-builder:"$2" . -f Dockerfile.python
