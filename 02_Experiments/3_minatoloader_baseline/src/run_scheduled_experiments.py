import os
import sys
import time
import argparse
import json
import threading
import psutil
import torch
import numpy as np
from unittest.mock import patch
import torch.multiprocessing as mp
from queue import Empty

# Add Minato path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
MINATO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../MinatoLoader/Minato"))
if MINATO_DIR not in sys.path:
    sys.path.insert(0, MINATO_DIR)

from data_loading.data_loader import get_data_split
from torchvision import transforms
from torch.utils.data import DataLoader

from model.unet3d import Unet3D
from model.losses import DiceCELoss, DiceScore

# Import our Scheduled loader
from worker_scheduler import ScheduledAsynchronousLoader

def get_heavy_train_transforms():
    # We must import inside because the classes are loaded dynamically based on system
    from data_loading.pytorch_loader import ElasticDeformation, HeavyGaussianBlur, RandFlip, Cast, RandomBrightnessAugmentation, GaussianNoise
    rand_flip = RandFlip()
    cast = Cast(types=(np.float32, np.uint8))
    heavy_blur = HeavyGaussianBlur(sigma_range=(1, 5))
    elastic_deform = ElasticDeformation(alpha=1000, sigma=30)
    rand_scale = RandomBrightnessAugmentation(factor=0.3, prob=0.1)
    rand_noise = GaussianNoise(mean=0.0, std=0.1, prob=0.1)
    return transforms.Compose([rand_flip, cast, elastic_deform, heavy_blur, rand_scale, rand_noise])

class QueueMonitor(threading.Thread):
    def __init__(self, log_file, dataloader):
        super().__init__()
        self.log_file = log_file
        self.dataloader = dataloader
        self.stop_event = threading.Event()
        self.epoch = 1
        self.start_time = time.time()
        self.daemon = True

    def run(self):
        with open(self.log_file, "w") as f:
            while not self.stop_event.is_set():
                elapsed = (time.time() - self.start_time) * 1000
                ram_gb = psutil.virtual_memory().used / (1024**3)
                cpu_p = psutil.cpu_percent(interval=None)
                
                log_data = {
                    "elapsed_ms": elapsed,
                    "epoch": self.epoch,
                    "cpu_percent": cpu_p,
                    "ram_gb": ram_gb
                }
                
                if hasattr(self.dataloader, "queue") and self.dataloader.queue is not None:
                    log_data["fast_q_size"] = self.dataloader.queue.qsize()
                    log_data["slow_processed_q_size"] = self.dataloader.slow_processed_queue.qsize()
                    log_data["active_workers"] = len(self.dataloader.producers)
                
                f.write(json.dumps(log_data) + "\n")
                f.flush()
                time.sleep(0.2)

class TimeoutScheduler(threading.Thread):
    def __init__(self, timeout_val, time_queue):
        super().__init__()
        self.timeout_val = timeout_val
        self.time_queue = time_queue
        self.stop_event = threading.Event()
        self.daemon = True
        self.times = []
        
    def run(self):
        while not self.stop_event.is_set():
            time.sleep(1.0)
            added = False
            while not self.time_queue.empty():
                try:
                    t = self.time_queue.get_nowait()
                    self.times.append(t)
                    added = True
                except Empty:
                    break
            
            # Keep last 100 times to adapt to changing dynamics
            if len(self.times) > 100:
                self.times = self.times[-100:]
                
            if added and len(self.times) >= 10:
                p75 = float(np.percentile(self.times, 75))
                self.timeout_val.value = p75
                # print(f"[TimeoutScheduler] Updated adaptive timeout to {p75:.2f}s")

def get_adaptive_pyt_train(PytTrain):
    class AdaptivePytTrain(PytTrain):
        def __init__(self, *args, timeout_val, time_queue, adaptive=True, **kwargs):
            super().__init__(*args, **kwargs)
            self.timeout_val = timeout_val
            self.time_queue = time_queue
            self.adaptive = adaptive
            
        def __getitem__(self, idx):
            start = time.time()
            if self.adaptive:
                self.sample_timeout = self.timeout_val.value
            res = super().__getitem__(idx)
            elapsed = time.time() - start
            
            if res is not None and self.adaptive:
                try:
                    self.time_queue.put(elapsed, block=False)
                except:
                    pass
            return res
    return AdaptivePytTrain

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive_timeout", action="store_true", help="Enable C2: Adaptive Timeout")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    batch_log_path = os.path.join(args.out_dir, "batch_logs.jsonl")
    epoch_log_path = os.path.join(args.out_dir, "epoch_result.csv")
    queue_log_path = os.path.join(args.out_dir, "queue_monitor.jsonl")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Dataset setup with Heavy Augmentation Patch
    data_dir = "/home/mew/Desktop/mew/study/Master degree/thesis/00_datasets/kits19_preproc"
    x_train, x_val, y_train, y_val = get_data_split(data_dir, num_shards=1, shard_id=0)
    train_kwargs = {"patch_size": [64, 64, 32], "oversampling": 0.4, "seed": 42}
    
    from data_loading.pytorch_loader import PytTrain, PytVal
    from data_loading.pytorch_loader import ElasticDeformation, HeavyGaussianBlur, RandFlip, Cast, RandomBrightnessAugmentation, GaussianNoise
    
    slow_queue = mp.Queue()
    slow_processed_queue = mp.Queue()
    
    # Timeout stuff
    timeout_val = mp.Value('d', 2.0)
    time_queue = mp.Queue(maxsize=1000)
    AdaptivePytTrainClass = get_adaptive_pyt_train(PytTrain)
    
    with patch('data_loading.pytorch_loader.get_train_transforms', side_effect=get_heavy_train_transforms):
        train_dataset = AdaptivePytTrainClass(x_train, y_train, slow_queue, slow_processed_queue, 
                                              timeout_val=timeout_val, time_queue=time_queue, 
                                              adaptive=args.adaptive_timeout, **train_kwargs)
    
    val_dataset = PytVal(x_val, y_val)

    # 2. DataLoader setup
    # Starts with 2 workers, max 8
    train_loader = ScheduledAsynchronousLoader(train_dataset, device=device, shards=1, sampler=None, 
                                      batch_size=args.batch_size, shuffle=True, pin_memory=True, 
                                      num_workers=2, max_workers=8, rank=0, slow_processed_queue=slow_processed_queue)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True) 

    # 3. Model setup
    model = Unet3D(1, 3, normalization="instancenorm", activation="relu").to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, use_softmax=True, layout="NCDHW", include_background=False).to(device)
    score_fn = DiceScore(to_onehot_y=True, use_argmax=True, layout="NCDHW", include_background=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.8, momentum=0.9, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler()

    # 4. Queue Monitor & Timeout Scheduler
    q_monitor = QueueMonitor(queue_log_path, train_loader)
    q_monitor.start()
    
    t_scheduler = None
    if args.adaptive_timeout:
        t_scheduler = TimeoutScheduler(timeout_val, time_queue)
        t_scheduler.start()

    # CSV Header
    with open(epoch_log_path, "w") as f:
        f.write("Epoch,Phase,Loss,Mean_Dice,Throughput_Img_Sec,GPU_Busy_Ratio,Total_Starvation_Sec,RAM_GB,VRAM_GB,Avg_Timeout\n")

    batch_log_file = open(batch_log_path, "w")

    print(f"Starting Scheduled MinatoLoader (Adaptive Timeout: {args.adaptive_timeout})")
    
    for epoch in range(1, args.epochs + 1):
        q_monitor.epoch = epoch
        model.train()
        
        train_iter = iter(train_loader)
        batch_idx = 0
        total_wait_time = 0.0
        total_compute_time = 0.0
        epoch_start_time = time.time()
        
        t_req = time.time()
        
        while True:
            try:
                batch = next(train_iter)
            except StopIteration:
                break
                
            t_data = time.time()
            wait_ms = (t_data - t_req) * 1000
            total_wait_time += (t_data - t_req)
            
            img, lbl = batch
            img, lbl = img.to(device), lbl.to(device)
            
            t_gpu_start = time.time()
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(img)
                loss = loss_fn(out, lbl)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            t_gpu_end = time.time()
            compute_ms = (t_gpu_end - t_gpu_start) * 1000
            total_compute_time += (t_gpu_end - t_gpu_start)
            
            vram_gb = torch.cuda.memory_allocated() / (1024**3)
            ram_gb = psutil.virtual_memory().used / (1024**3)
            cpu_p = psutil.cpu_percent(interval=None)
            
            log_data = {
                "epoch": epoch,
                "batch_idx": batch_idx,
                "t_dataloader_start": t_req,
                "t_dataloader_end": t_data,
                "t_gpu_start": t_gpu_start,
                "t_gpu_end": t_gpu_end,
                "dataloader_wait_ms": wait_ms,
                "gpu_compute_ms": compute_ms,
                "vram_gb": vram_gb,
                "ram_gb": ram_gb,
                "cpu_percent": cpu_p,
                "batch_loss": loss.item(),
                "timeout_val": timeout_val.value
            }
            if hasattr(train_loader, "queue") and train_loader.queue is not None:
                log_data["fast_q_size"] = train_loader.queue.qsize()
                log_data["slow_q_size"] = train_loader.slow_processed_queue.qsize()
                log_data["active_workers"] = len(train_loader.producers)
                
            batch_log_file.write(json.dumps(log_data) + "\n")
            batch_log_file.flush()
            
            batch_idx += 1
            t_req = time.time()
            
        epoch_time = time.time() - epoch_start_time
        throughput = (batch_idx * args.batch_size) / epoch_time
        gpu_busy = total_compute_time / (total_compute_time + total_wait_time) if (total_compute_time + total_wait_time) > 0 else 0
        
        # Validation
        model.eval()
        val_losses = []
        dice_scores = []
        with torch.no_grad():
            for v_img, v_lbl in val_loader:
                v_img, v_lbl = v_img.to(device), v_lbl.to(device)
                with torch.cuda.amp.autocast():
                    from runtime.inference import sliding_window_inference
                    out, v_lbl = sliding_window_inference(
                        inputs=v_img,
                        labels=v_lbl,
                        roi_shape=[64, 64, 32],
                        model=model,
                        overlap=0.5,
                        mode="constant",
                        padding_val=-2.2
                    )
                    v_loss = loss_fn(out, v_lbl)
                    v_dice = score_fn(out, v_lbl)
                val_losses.append(v_loss.item())
                dice_scores.append(v_dice.cpu().numpy())
                
                del out, v_loss, v_dice, v_img, v_lbl
                torch.cuda.empty_cache()
                
        mean_val_loss = np.mean(val_losses)
        mean_dice = np.mean([d[0] for d in dice_scores])
        
        with open(epoch_log_path, "a") as f:
            f.write(f"{epoch},Val,{mean_val_loss:.4f},{mean_dice:.4f},{throughput:.2f},{gpu_busy:.4f},{total_wait_time:.4f},{ram_gb:.2f},{vram_gb:.2f},{timeout_val.value:.4f}\n")
            
        print(f"Epoch {epoch} | T-put: {throughput:.2f} img/s | GPU Busy: {gpu_busy*100:.1f}% | Wait: {total_wait_time:.1f}s | Dice: {mean_dice:.4f} | Timeout: {timeout_val.value:.2f}s")
        
    q_monitor.stop_event.set()
    if t_scheduler is not None:
        t_scheduler.stop_event.set()
    if hasattr(train_loader, "stop_epoch"):
        train_loader.stop_epoch()
    
    batch_log_file.close()
    print("Done!")

if __name__ == "__main__":
    main()
    os._exit(0)
