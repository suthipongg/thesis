import numpy as np 
from maskrcnn_benchmark.config import cfg
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import csv
import os
import sys
print("path now", sys.path)
import sys
sys.path.append("/projects/I20240005/rnouaj/object_detection_speedy/pytorch/")
print("path after", sys.path)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
import argparse
import os
import functools
import logging
import random
import datetime
import time

import torch
sys.path.append("/projects/I20240005/rnouaj/object_detection_speedy/pytorch/")

from maskrcnn_benchmark.config import cfg


def download_weights(cfg):


    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    arguments["save_checkpoints"] = cfg.SAVE_CHECKPOINTS
    
    

    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)
