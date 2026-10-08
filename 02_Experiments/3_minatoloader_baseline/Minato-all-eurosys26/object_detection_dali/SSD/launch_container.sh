#!/bin/bash
docker run --gpus all -v /dl-bench/rnouaj/DeepLearningExamples/PyTorch/Detection/SSD/:/workspace -v /raid/data/object_detection/datasets/:/datasets -t -i --rm --gpus 8 --ipc=host ssd_dali:rnouaj
