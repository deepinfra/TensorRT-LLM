git diff 64513797f2055d4c58a12f0328bb7ea72a104758 -- tensorrt_llm > python.patch

docker build -t localhost:30500/pytrtllm:"$1" --build-arg url=localhost:30500/pytrtllm-builder:"$2" . -f Dockerfile.python
