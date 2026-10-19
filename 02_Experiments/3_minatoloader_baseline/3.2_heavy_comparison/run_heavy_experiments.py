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
from data_loading.Asynchrnous_dataloader import AsynchronousLoader
import torch.multiprocessing as mp

from model.unet3d import Unet3D
from model.losses import DiceCELoss, DiceScore

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
    def __init__(self, log_file, dataloader, system):
        super().__init__()
        self.log_file = log_file
        self.dataloader = dataloader
        self.system = system
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
                
                if self.system == "minato" and hasattr(self.dataloader, "queue"):
                    log_data["fast_q_size"] = self.dataloader.queue.qsize()
                    log_data["slow_processed_q_size"] = self.dataloader.slow_processed_queue.qsize()
                    # Cannot easily get slow_queue size as it's passed directly to PytTrain in Minato
                
                f.write(json.dumps(log_data) + "\n")
                f.flush()
                time.sleep(0.2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["pytorch", "minato"], required=True)
    parser.add_argument("--workers", type=int, required=True)
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
    
    if args.system == "pytorch":
        PYTORCH_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../MinatoLoader/PyTorch"))
        sys.path.insert(0, PYTORCH_DIR)
        
        # Remove cached modules so we get the PyTorch version
        for k in list(sys.modules.keys()):
            if k.startswith("data_loading."):
                del sys.modules[k]
                
        from data_loading.pytorch_loader import PytTrain, PytVal
        from data_loading.pytorch_loader import ElasticDeformation, HeavyGaussianBlur, RandFlip, Cast, RandomBrightnessAugmentation, GaussianNoise
        sys.path.pop(0) # remove it
        
        with patch('data_loading.pytorch_loader.get_train_transforms', side_effect=get_heavy_train_transforms):
            train_dataset = PytTrain(x_train, y_train, **train_kwargs)
            
    else:
        from data_loading.pytorch_loader import PytTrain, PytVal
        from data_loading.pytorch_loader import ElasticDeformation, HeavyGaussianBlur, RandFlip, Cast, RandomBrightnessAugmentation, GaussianNoise
        slow_queue = mp.Queue()
        slow_processed_queue = mp.Queue()
        with patch('data_loading.pytorch_loader.get_train_transforms', side_effect=get_heavy_train_transforms):
            train_dataset = PytTrain(x_train, y_train, slow_queue, slow_processed_queue, **train_kwargs)
    
    val_dataset = PytVal(x_val, y_val)

    # 2. DataLoader setup
    if args.system == "pytorch":
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.workers, pin_memory=True, drop_last=True, prefetch_factor=2)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        train_loader = AsynchronousLoader(train_dataset, device=device, shards=1, sampler=None, 
                                          batch_size=args.batch_size, shuffle=True, pin_memory=True, 
                                          num_workers=args.workers, rank=0, slow_processed_queue=slow_processed_queue)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True) # Using PyTorch loader for val to avoid Minato issues

    # 3. Model setup
    model = Unet3D(1, 3, normalization="instancenorm", activation="relu").to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, use_softmax=True, layout="NCDHW", include_background=False).to(device)
    score_fn = DiceScore(to_onehot_y=True, use_argmax=True, layout="NCDHW", include_background=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.8, momentum=0.9, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler()

    # 4. Queue Monitor
    q_monitor = QueueMonitor(queue_log_path, train_loader, args.system)
    q_monitor.start()

    # CSV Header
    with open(epoch_log_path, "w") as f:
        f.write("Epoch,Phase,Loss,Mean_Dice,Throughput_Img_Sec,GPU_Busy_Ratio,Total_Starvation_Sec,RAM_GB,VRAM_GB\n")

    batch_log_file = open(batch_log_path, "w")

    print(f"Starting {args.system} with {args.workers} workers, batch_size {args.batch_size}")
    
    for epoch in range(1, args.epochs + 1):
        q_monitor.epoch = epoch
        model.train()
        
        # Iterator setup
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
            
            if batch is None:
                t_req = time.time()
                continue
                
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
                "batch_loss": loss.item()
            }
            if args.system == "minato" and hasattr(train_loader, "queue"):
                log_data["fast_q_size"] = train_loader.queue.qsize()
                log_data["slow_q_size"] = train_loader.slow_processed_queue.qsize()
                
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
        mean_dice = np.mean([d[0] for d in dice_scores]) # class 1 dice
        
        with open(epoch_log_path, "a") as f:
            f.write(f"{epoch},Val,{mean_val_loss:.4f},{mean_dice:.4f},{throughput:.2f},{gpu_busy:.4f},{total_wait_time:.4f},{ram_gb:.2f},{vram_gb:.2f}\n")
            
        print(f"Epoch {epoch} | T-put: {throughput:.2f} img/s | GPU Busy: {gpu_busy*100:.1f}% | Wait: {total_wait_time:.1f}s | Dice: {mean_dice:.4f}")
        
    q_monitor.stop_event.set()
    if hasattr(train_loader, "stop_threads"):
        train_loader.stop_threads()
    
    batch_log_file.close()
    print("Done!")

if __name__ == "__main__":
    main()
    os._exit(0)
