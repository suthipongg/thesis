# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

from torch.autograd import Variable
import torch
import time
import csv
from apex import amp
import random
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

def get_random_image_size():
    """Generate a random image size with width fixed at 300 and height varying between 100 and 300."""
    width = 300
    height = random.randint(100, 200)  # Random height between 100 and 300
    return (width, height)

def compute_image_size(img, batch_size = 64, channels=3, dtype=torch.float32):
    """
    Compute the size of a randomly sized image tensor in MB.
    
    Args:
        channels (int): Number of channels (e.g., 3 for RGB, 1 for grayscale).
        dtype (torch.dtype): Data type of the image tensor.

    Returns:
        float: Image size in megabytes (MB).
    """
    width, height = get_random_image_size()
    print("img size", img.shape)
    image_tensor = torch.empty((batch_size, channels, height, width), dtype=dtype)  # Create an empty tensor with the size
    image_size_bytes = image_tensor.element_size() * image_tensor.numel()
    return image_size_bytes / (1024 * 1024)  # Convert bytes to MB


def train_loop(model, loss_func, scaler, epoch, optim, train_dataloader, val_dataloader, encoder, iteration, logger, args, mean, std, start_epoch_time):
    start_training_time = time.time()
    print("train_loop")
    print("start_training_time", start_training_time)
    size = 0
    for nbatch, data in enumerate(train_dataloader):
        print("nbatch", nbatch)
        iteration_start_time = time.time()
        img = data[0][0][0]
        bbox = data[0][1][0]
        label = data[0][2][0]
        label = label.type(torch.cuda.LongTensor)
        bbox_offsets = data[0][3][0]
        bbox_offsets = bbox_offsets.cuda()
        img.sub_(mean).div_(std)
        if not args.no_cuda:
            img = img.cuda()
            bbox = bbox.cuda()
            label = label.cuda()
            bbox_offsets = bbox_offsets.cuda()

        N = img.shape[0]
        if bbox_offsets[-1].item() == 0:
            print("No labels in batch")
            continue

        # output is ([N*8732, 4], [N*8732], need [N, 8732, 4], [N, 8732] respectively
        M = bbox.shape[0] // N
        bbox = bbox.view(N, M, 4)
        label = label.view(N, M)

        with torch.cuda.amp.autocast(enabled=args.amp):
            if args.data_layout == 'channels_last':
                img = img.to(memory_format=torch.channels_last)
            ploc, plabel = model(img)

            ploc, plabel = ploc.float(), plabel.float()
            trans_bbox = bbox.transpose(1, 2).contiguous().cuda()
            gloc = Variable(trans_bbox, requires_grad=False)
            glabel = Variable(label, requires_grad=False)

            loss = loss_func(ploc, plabel, gloc, glabel)

        if args.warmup is not None:
            warmup(optim, args.warmup, iteration, args.learning_rate)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()
        # iteration_time1 = time.time() - iteration_start_time
        # print('Iteration time: ', iteration_time1)


     
        if args.local_rank == 0:
            logger.update_iter(epoch, iteration, loss.item())
        iteration += 1

    return iteration

import torch
import time
import csv
from torch.autograd import Variable

def benchmark_train_loop(model, loss_func, scaler, epoch, optim, train_dataloader, val_dataloader, encoder, iteration, logger, args, mean, std, start_epoch_time, step, start_training_time):
    start_time = time.time()
    iteration10 = time.time()
    result = torch.zeros((1,)).cuda()
     # Track time and the number of iterations
    last_time = time.time()
    last_iteration = 0
    size = 0


    # Create two CUDA streams
    data_stream = torch.cuda.Stream()
    train_stream = torch.cuda.Stream()

    for nbatch, data in enumerate(loop(train_dataloader)):
        step +=1
        if nbatch >= args.benchmark_warmup:
            torch.cuda.synchronize()
            start_time = time.time()

        # Load data in a separate stream
        with torch.cuda.stream(data_stream):
            img = data[0][0][0]
            bbox = data[0][1][0]
            label = data[0][2][0].type(torch.cuda.LongTensor)
            bbox_offsets = data[0][3][0].cuda()

            img.sub_(mean).div_(std)
            if not args.no_cuda:
                img = img.cuda(non_blocking=True)
                bbox = bbox.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

        # Ensure data loading is completed before training
        data_stream.synchronize()

        N = img.shape[0]
        if bbox_offsets[-1].item() == 0:
            print("No labels in batch")
            continue

        M = bbox.shape[0] // N
        bbox = bbox.view(N, M, 4)
        label = label.view(N, M)
        start_time_training = time.time() 
        

        # Training step in a separate stream
        with torch.cuda.stream(train_stream):
            with torch.cuda.amp.autocast(enabled=args.amp):
                if args.data_layout == 'channels_last':
                    img = img.to(memory_format=torch.channels_last)

                ploc, plabel = model(img)
                ploc, plabel = ploc.float(), plabel.float()

                trans_bbox = bbox.transpose(1, 2).contiguous().cuda()
                gloc = Variable(trans_bbox, requires_grad=False)
                glabel = Variable(label, requires_grad=False)

                loss = loss_func(ploc, plabel, gloc, glabel)

            if args.warmup is not None:
                warmup(optim, args.warmup, iteration, args.learning_rate)
            



            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
            size += compute_image_size(img) 

        train_stream.synchronize()



        print(f"nbatch: {nbatch}")

        if nbatch == 0:
            with open('training_metrics_A100.csv', 'w') as f:
                f.write('iteration,throughput(MBs), iter/s, iteration_time\n')
            torch.cuda.synchronize()

        iteration_time = time.time() - start_time
        print(f"Iteration time: {iteration_time}, nbatch {nbatch}")
        with open('training_time_dali.csv', 'a') as f:
            f.write([ "iteration_time"])
            torch.cuda.synchronize()

        if nbatch >= args.benchmark_warmup + args.benchmark_iterations:
            break

        torch.cuda.synchronize()
        logger.update(args.batch_size * args.N_gpu, time.time() - start_time)

        # Check if 5 seconds have passed since the last recording
        if time.time() - last_time >= 5:
            # Calculate number of iterations per second over the last 5 seconds
            iterations_per_second = (iteration - last_iteration) / (time.time() - last_time)
            
            throughput_mbs = size / (time.time() - last_time) 
            
            # Log the result

            with open('training_metrics_A100.csv', 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([step, throughput_mbs, iterations_per_second, time.time() - start_training_time])
                torch.cuda.synchronize()
            # Update the last recorded time and batch number
            last_time = time.time()
            last_iteration = iteration
            size = 0
            torch.cuda.synchronize()

 
     


        # Increment iteration

        print(f"Iteration time: {iteration_time}, nbatch {nbatch}")

        if nbatch >= args.benchmark_warmup + args.benchmark_iterations:
            break

        torch.cuda.synchronize()
        logger.update(args.batch_size * args.N_gpu, time.time() - start_time)

        # Increment iteration
        iteration += 1

    result.data[0] = logger.print_result()

    if args.N_gpu > 1:
        torch.distributed.reduce(result, 0)

    if args.local_rank == 0:
        print(f'Training performance = {float(result.data[0])} FPS')

    return iteration  # Return updated iteration count


def loop(dataloader, reset=True):
    while True:
        for data in dataloader:
            yield data
        if reset:
            dataloader.reset()

def benchmark_inference_loop(model, loss_func, scaler, epoch, optim, train_dataloader, val_dataloader, encoder, iteration, logger, args, mean, std):
    assert args.N_gpu == 1, 'Inference benchmark only on 1 gpu'
    model.eval()
    val_datas = loop(val_dataloader, False)

    for i in range(args.benchmark_warmup + args.benchmark_iterations):
        torch.cuda.synchronize()
        start_time = time.time()

        data = next(val_datas)
        img = data[0]
        with torch.no_grad():
            if not args.no_cuda:
                img = img.cuda()
            img.sub_(mean).div_(std)
            with torch.cuda.amp.autocast(enabled=args.amp):
                _ = model(img)

        torch.cuda.synchronize()
        end_time = time.time()


        if i >= args.benchmark_warmup:
            logger.update(args.eval_batch_size, end_time - start_time)

    logger.print_result()

def warmup(optim, warmup_iters, iteration, base_lr):
    if iteration < warmup_iters:
        new_lr = 1. * base_lr / warmup_iters * iteration
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr


def load_checkpoint(model, checkpoint):
    """
    Load model from checkpoint.
    """
    print("loading model checkpoint", checkpoint)
    od = torch.load(checkpoint)

    # remove proceeding 'N.' from checkpoint that comes from DDP wrapper
    saved_model = od["model"]
    model.load_state_dict(saved_model)


def tencent_trick(model):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0},
            {'params': decay}]
