docker run -v $(pwd):/workspace -v /raid/data/object_detection/datasets/:/datasets -t -i --rm --gpus 8 --ipc=host object_detection:speedy1
