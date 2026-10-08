#!/bin/bash

# Runs benchmark and reports time to convergence

pushd pytorch

# Single GPU training
time python -m torch.distributed.launch --nproc_per_node=4 tools/train_mlperf.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
       SOLVER.IMS_PER_BATCH 48  TEST.IMS_PER_BATCH 4 SOLVER.MAX_ITER 1440 SOLVER.STEPS "(960, 1280)" SOLVER.BASE_LR 0.0025
       

# Multi GPU training
# python -m torch.distributed.launch --nproc_per_node=3
# solver iter: 7200
# solver steps: 4800,6400
popd
