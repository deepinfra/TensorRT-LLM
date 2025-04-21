git diff 943f3ff8f6bf21a3e2de4689f8261af32962c09e -- tensorrt_llm > python.patch

docker build -t localhost:30500/tensorrt-llm:"$1" --build-arg url=localhost:30500/tensorrt-llm-builder:"$2" . -f Dockerfile.python
