git diff 20d629e1b1505f41d0f7bcf4edca2c02dc27e9ed -- tensorrt_llm > python.patch

docker build -t localhost:30500/tensorrt-llm:"$1" --build-arg url=localhost:30500/tensorrt-llm-builder:feb25 . -f Dockerfile.python
