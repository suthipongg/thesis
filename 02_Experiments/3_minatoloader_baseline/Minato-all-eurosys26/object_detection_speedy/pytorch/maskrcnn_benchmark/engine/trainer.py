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
import torch.distributed as dist
import matplotlib.pyplot as plt
from maskrcnn_benchmark.utils.comm import get_world_size, is_main_process
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

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



def compute_target_size(targets):
    target_size_bytes = 0
    for t in targets:
        if isinstance(t, torch.Tensor):
            target_size_bytes += t.element_size() * t.numel()
        elif isinstance(t, dict):  # If targets contain dictionaries (e.g., COCO format)
            for v in t.values():
                if isinstance(v, torch.Tensor):
                    target_size_bytes += v.element_size() * v.numel()
    return target_size_bytes / (1024 * 1024)  # Convert bytes to MB

def compute_image_size(images):
    images_tensor = images.tensors
    image_size_bytes = images_tensor.element_size() * images_tensor.numel()


    image_size_mb = image_size_bytes / (1024 * 1024)
    return  image_size_mb


throughput_file2 = '/projects/I20240005/rnouaj/object_detection_speedy/prefetch_factor/24pref.csv'
def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    per_iter_start_callback_fn=None,
    per_iter_end_callback_fn=None, throughput_file=None
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    last_time = time.time()
    processed_size = 0
    last_iteration = 0
    losses_list = []  # Store loss values



    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        if iteration == 0:
            with open(throughput_file2, 'w') as f:
               f.write('iteration,throughput(MBs), iter/s, iteration_time\n')
               torch.cuda.synchronize()


        iteration_start_time = time.time()

        #print("speedyloader - do_train-epoch", iteration)

        # if per_iter_start_callback_fn is not None:
        #     per_iter_start_callback_fn(iteration=iteration)

        data_time = time.time() - end
        
        arguments["iteration"] = iteration

        scheduler.step()
    
        test_time = time.time()
        images = images.to(device)
        
        targets = [target.to(device) for target in targets]
        loss_dict = model(images, targets)



        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        losses.backward()

        
        preproc_start = time.time()
        optimizer.step()
        optimizer.zero_grad()
        processed_size += compute_image_size(images) + compute_target_size(targets)
        print("iteration", iteration , "size", processed_size)
        losses_list.append(losses.item())  # Store loss as a scalar
 

        torch.cuda.synchronize()
          
        now = time.time()
        batch_time = now - end
        # batch_time = time.time() - end
        # print("bitchebe-preproctime0: ", now-preproc_start)
        # print("bitchebe-preproctime1-data: ", now-test_time)
        # print("bitchebe-preproctime2-batch: ", batch_time)
        end = time.time()
     
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        iteration_duration1 = time.time() - iteration_start_time

        if time.time() - last_time >= 5:
            # Calculate number of iterations per second over the last 5 seconds
            iterations_per_second = (iteration - last_iteration) / (time.time() - last_time)
            size = compute_image_size(images) + compute_target_size(targets)
            throughput_mbs = processed_size / (time.time() - last_time)  # MB/s
            
            # Log the result

            with open(throughput_file2, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([iteration, throughput_mbs, iterations_per_second, time.time() - start_training_time])
                torch.cuda.synchronize()
            # Update the last recorded time and batch number
            last_time = time.time()
            processed_size = 0
            last_iteration = iteration
            torch.cuda.synchronize()



        if iteration % 10 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0 and arguments["save_checkpoints"]:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter and arguments["save_checkpoints"]:
            checkpointer.save("model_final", **arguments)

        # per-epoch work (testing)
        if per_iter_end_callback_fn is not None:
            # Note: iteration has been incremented previously for
            # human-readable checkpoint names (i.e. 60000 instead of 59999)
            # so need to adjust again here
            early_exit = per_iter_end_callback_fn(iteration=iteration-1)
            if early_exit:
                break
        torch.cuda.synchronize()

        

     


    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )



    torch.save(losses_list, "losses.pth")
   

    # Load loss values
    losses = torch.load("losses.pth")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Testing Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Pytorch")
    plt.legend()
    plt.savefig("loss_Pytorch.png")


    if per_iter_end_callback_fn is not None:
        if early_exit:
            return True
        else:
            return False
    else:
        return None