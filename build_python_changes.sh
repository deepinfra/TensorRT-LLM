git show > python.patch

docker build -t localhost:30500/tensorrt-llm:"$1" --build-arg url=localhost:30500/tensorrt-llm-builder:"$1" . -f Dockerfile.python
