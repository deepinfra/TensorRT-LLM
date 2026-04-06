git diff 46d3f74fdd60d2c96bee00e0b939ab1fd2241dc1 -- tensorrt_llm > python.patch

docker build -t localhost:30500/pytrtllm:"$1" --build-arg url=localhost:30500/pytrtllm-builder:"$2" . -f Dockerfile.python
