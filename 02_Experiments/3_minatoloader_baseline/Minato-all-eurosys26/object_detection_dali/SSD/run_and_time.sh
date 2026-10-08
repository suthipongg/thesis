#!/bin/bash



torchrun --nproc_per_node=4 \
       main.py --batch-size 72 \
               --mode benchmark-training \
               --benchmark-warmup 1000 \
               --benchmark-iterations 200 \
               --amp \
               --data /projects/I20240005/coco \
