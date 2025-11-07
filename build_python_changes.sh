git diff 9ec01e6afb800cbedc0dac42cf77b23d9bda36d9 -- tensorrt_llm > python.patch

docker build -t localhost:30500/pytrtllm:"$1" --build-arg url=localhost:30500/pytrtllm-builder:"$2" . -f Dockerfile.python
