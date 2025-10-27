git diff 024abfdb3921cc64a881d87b544bfa99f6c7a45c -- tensorrt_llm > python.patch

docker build -t localhost:30500/pytrtllm:"$1" --build-arg url=localhost:30500/pytrtllm-builder:"$2" . -f Dockerfile.python
