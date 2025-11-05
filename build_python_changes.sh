git diff 3c3e59e3c91787d46f544ff3f2fe4df06c3e721a -- tensorrt_llm > python.patch

docker build -t localhost:30500/pytrtllm:"$1" --build-arg url=localhost:30500/pytrtllm-builder:"$2" . -f Dockerfile.python
