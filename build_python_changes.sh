git diff 61599950dc062ab2fa83a648bf96870f9ae39cfa -- tensorrt_llm > python.patch

docker build -t localhost:30500/tensorrt-llm:"$1" --build-arg url=localhost:30500/tensorrt-llm-builder:"$2" . -f Dockerfile.python
