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
import datetime
import logging
import time
import csv
import torch
import matplotlib.pyplot as plt 
import sys
import torch.distributed as dist
from queue import Empty
from maskrcnn_benchmark.utils.comm import get_world_size, is_main_process
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from torch.cuda.amp import autocast, GradScaler
from maskrcnn_benchmark.config import cfg


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

# Instead of zeroing, set parameter grads to None
# Prevents extraneous copy as we're not accumulating
def set_grads_to_none(model):
    for param in model.parameters():
        param.grad = None



def compute_target_size(boxlist):
  
    

    object_size = sys.getsizeof(boxlist[0])
    print("object_size",object_size)
    total_size = boxlist[0].get_object_size() + object_size
    print("total_size",total_size)

    # Return the total size in bytes (object + tensors)
    return  total_size / (1024 * 1024)  # Convert bytes to MB


def compute_image_size(images):
    """Computes the size of image tensors in MB."""
    if hasattr(images, "tensors"):
        images_tensor = images.tensors
        size = images_tensor.element_size() * images_tensor.numel()
        print(f"Image size: {size / (1024 * 1024):.3f} MB")
        return size / (1024 * 1024)  # Convert bytes to MB
    return 0


def do_train(
    batch_queue,
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    per_iter_start_callback_fn=None,
    per_iter_end_callback_fn=None,
    throughput_file=None
):

    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = 14400
    start_iter = arguments["iteration"]
    print("start_iter",start_iter)
    model.train()
    start_training_time = time.time()
    start_time = time.time()
    data_ready_event = torch.cuda.Event()

    
    # Create CUDA streams
    stream_compute = torch.cuda.Stream()
    stream_data = torch.cuda.Stream()
    
    # Throughput tracking variables
    last_log_time = time.time()
    processed_size = 0
    total_processed = 0
    
    # Prefetch initial batches
    current_batches = None
    next_batches = None
    next_images, next_targets = None, None
    nbatch = 0
    data_ready_events = []


    processed_size = 0
    prefetch_factor = 2  # Number of batches to prefetch
    buffer = []
    stop_training = False  # Flag to handle empty queue cases
    losses_list = []  # Store loss values


    # Prefetch initial batches asynchronously
    with torch.cuda.stream(stream_data):
        for _ in range(1):
            try:
                batch = batch_queue.get(timeout=20)
                images, targets, _ = batch
                images_gpu = images.to(device, non_blocking=True)  # Use pinned memory for faster transfers
                targets_gpu = [t.to(device) for t in targets]   # Ensure targets are also async transferred
                event = torch.cuda.Event()
                event.record(stream_data)
                buffer.append((images_gpu, targets_gpu, event))
            except Empty:
                stop_training = True
                
                # break

    for iteration in range(start_iter, max_iter):




        print("speedyloader - do_train-epoch", iteration)

        # if per_iter_start_callback_fn is not None:
        #     per_iter_start_callback_fn(iteration=iteration)
        # # if stop_training and not buffer:
        #     logger.error("Training stopped due to lack of data.")
        #     break

        # Get next batch from buffer
        if buffer:
            current_images, current_targets, current_event = buffer.pop(0)
        else:
            logger.warning("Buffer underflow at iteration %d", iteration)
            continue  # Skip iteration if no batch available

        # Process current batch on compute stream
        with torch.cuda.stream(stream_compute):
            current_event.wait(stream_compute)  # Ensure data is ready before computation

            
            
            # Forward pass
            loss_dict = model(current_images, current_targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            losses.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses_list.append(losses.item())  # Store loss as a scalar


            # Track processed data size
            processed_size += compute_image_size(current_images) + compute_target_size(current_targets)

        # Prefetch next batches asynchronously
        with torch.cuda.stream(stream_data):
            while len(buffer) < prefetch_factor:
                try:
                    batch = batch_queue.get(timeout=5)  # Reduced timeout to avoid CPU contention
                    images, targets, _ = batch
                    images_gpu = images.to(device, non_blocking=True)
                    targets_gpu = [t.to(device) for t in targets]
                    event = torch.cuda.Event()
                    event.record(stream_data)
                    buffer.append((images_gpu, targets_gpu, event))
                except Empty:
                    stop_training = True  # Mark that no more data is available
                    print("No more data available for prefetching.")
                    break

                # Track processed data size
            # processed_size += compute_image_size(current_images)
            # print("compute_image_size(current_images)",compute_image_size(current_images))
            # print("compute_target_size(current_targets)",compute_target_size(current_targets))
            # processed_size += compute_target_size(current_targets)

        # 4. THROUGHPUT LOGGING
        if time.time() - last_log_time >= 5:
            time_diff = time.time() - last_log_time
            throughput = processed_size / 5 # MB/s
            iter_persec = nbatch / time_diff
            with open("/projects/I20240005/rnouaj/object_detection_speedy1/results_#diff_GPUs/4gpu.csv", 'a', newline='') as f:
                f.write(f"{iteration},{throughput},{time.time() - start_training_time},{time_diff}, {iter_persec}\n")
            processed_size = 0
            last_log_time = time.time()
            nbatch =0

        # 5. LOGGING & CHECKPOINTS
        if iteration % 10 == 0:
            batch_time = time.time() - start_time
            meters.update(time=batch_time)
            logger.info(
                f"Iter: {iteration} | "
                f"Time: {batch_time:.3f}s | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                f"Memory: {torch.cuda.max_memory_allocated()/1024**2:.1f}MB"
            )
            
        if iteration % checkpoint_period == 0:
            checkpointer.save(f"model_{iteration}", **arguments)
        nbatch += 1
          # per-epoch work (testing)
        if per_iter_end_callback_fn is not None:
            # Note: iteration has been incremented previously for
            # human-readable checkpoint names (i.e. 60000 instead of 59999)
            # so need to adjust again here
            early_exit = per_iter_end_callback_fn(iteration=iteration)
            print("early_exit",early_exit)
            if early_exit:
                break

    # Final throughput calculation
    total_time = time.time() - start_training_time
    logger.info(f"Average throughput: {total_processed/total_time/(1024**2):.2f} MB/s")




    if per_iter_end_callback_fn is not None:
        if early_exit:
            return True
        else:
            return False
    else:
        print("No per_iter_end_callback_fn provided, returning False")
        return None

    # torch.save(losses_list, "losses_speedy.pth")
   

    # # Load loss values
    # losses = torch.load("losses_speedy.pth")

    # # Plot
    # plt.figure(figsize=(10, 5))
    # plt.plot(losses, label="Training Loss")
    # plt.xlabel("Iteration")
    # plt.ylabel("Loss")
    # plt.title("Loss Function Fluctuation Over Training")
    # plt.legend()
    # plt.savefig("loss_plot.png")