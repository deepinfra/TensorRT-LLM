git diff 3c8a304724b4771b45ee77b4db6f2f21e8dff6cd -- tensorrt_llm > python.patch

docker build -t localhost:30500/pytrtllm:"$1" --build-arg url=localhost:30500/pytrtllm-builder:"$2" . -f Dockerfile.python
