git diff 5262f3c1056469133284e4faeee936979d298d9c -- tensorrt_llm > python.patch

docker build -t localhost:30500/tensorrt-llm:"$1" --build-arg url=localhost:30500/tensorrt-llm-builder:"$2" . -f Dockerfile.python
