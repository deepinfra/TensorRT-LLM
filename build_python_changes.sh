git diff 876c1152da69fed212e64a4e9dbfb021048cdafd -- tensorrt_llm > python.patch

docker build -t localhost:30500/pytrtllm:"$1" --build-arg url=localhost:30500/pytrtllm-builder:"$2" . -f Dockerfile.python
