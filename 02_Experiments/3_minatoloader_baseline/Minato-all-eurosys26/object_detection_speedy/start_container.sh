nvidia-docker run -v .:/workspace -t -i --rm --ipc=host --gpus all -v /raid/data/object_detection/datasets:/datasets mlperf/object_detection
