# docker run -v .:/workspace -t -i --rm --ipc=host mlperf/object_detection
docker run --cpuset-cpus="8-11" --cpus="4" --ipc=host --rm --runtime=nvidia -it -v $(pwd):/workspace mlperf/object_detection
