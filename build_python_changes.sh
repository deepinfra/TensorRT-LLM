git diff a6cbc65b740bd51a7414a756b5a42577c6ca6ed3 -- tensorrt_llm > python.patch

docker build -t localhost:30500/pytrtllm:"$1" --build-arg url=localhost:30500/pytrtllm-builder:"$2" . -f Dockerfile.python
