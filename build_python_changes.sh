git diff 3058fdb97f3518933c7fab4260f599c1718da45e -- tensorrt_llm > python.patch

docker build -t localhost:30500/tensorrt-llm:"$1" --build-arg url=localhost:30500/tensorrt-llm-builder:"$2" . -f Dockerfile.python
