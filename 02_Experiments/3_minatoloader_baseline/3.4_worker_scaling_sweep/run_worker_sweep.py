#!/usr/bin/env python3
"""
Worker Scaling Sweep Benchmark (Workers 1 to 8)
================================================
Investigates the scaling inflection point of PyTorch DataLoader vs MinatoLoader
under Heavy 3D Augmentation on Hybrid CPU (Intel Core i5-12450HX: 4P + 4E cores).
"""

import os
import sys
import time
import argparse
import json
import threading
import psutil
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch
import torch.multiprocessing as mp
from pathlib import Path

# Add paths
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
from model.unet3d import Unet3D
from model.losses import DiceCELoss, DiceScore

def get_heavy_train_transforms():
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
                f.write(json.dumps(log_data) + "\n")
                f.flush()
                time.sleep(0.2)

def run_single_benchmark(system: str, num_workers: int, epochs: int, batch_size: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    batch_log_path = os.path.join(out_dir, "batch_logs.jsonl")
    queue_log_path = os.path.join(out_dir, "queue_monitor.jsonl")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = "/home/mew/Desktop/mew/study/Master degree/thesis/00_datasets/kits19_preproc"
    x_train, x_val, y_train, y_val = get_data_split(data_dir, num_shards=1, shard_id=0)
    train_kwargs = {"patch_size": [64, 64, 32], "oversampling": 0.4, "seed": 42}

    if system == "pytorch":
        PYTORCH_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../MinatoLoader/PyTorch"))
        sys.path.insert(0, PYTORCH_DIR)
        for k in list(sys.modules.keys()):
            if k.startswith("data_loading."):
                del sys.modules[k]
        from data_loading.pytorch_loader import PytTrain
        sys.path.pop(0)

        with patch('data_loading.pytorch_loader.get_train_transforms', side_effect=get_heavy_train_transforms):
            train_dataset = PytTrain(x_train, y_train, **train_kwargs)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True, prefetch_factor=2
        )
    else:
        for k in list(sys.modules.keys()):
            if k.startswith("data_loading."):
                del sys.modules[k]
        from data_loading.pytorch_loader import PytTrain
        slow_queue = mp.Queue()
        slow_processed_queue = mp.Queue()
        with patch('data_loading.pytorch_loader.get_train_transforms', side_effect=get_heavy_train_transforms):
            train_dataset = PytTrain(x_train, y_train, slow_queue, slow_processed_queue, **train_kwargs)
        from data_loading.Asynchrnous_dataloader import data_stop_event, data_resume_event
        data_stop_event.clear()
        data_resume_event.set()
        train_loader = AsynchronousLoader(
            train_dataset, device=device, shards=1, sampler=None,
            batch_size=batch_size, shuffle=True, pin_memory=True,
            num_workers=num_workers, rank=0, slow_processed_queue=slow_processed_queue
        )

    model = Unet3D(1, 3, normalization="instancenorm", activation="relu").to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, use_softmax=True, layout="NCDHW", include_background=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.8, momentum=0.9, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler()

    q_monitor = QueueMonitor(queue_log_path, train_loader, system)
    q_monitor.start()

    epoch_records = []
    batch_log_file = open(batch_log_path, "w")

    print(f"\n=======================================================")
    print(f" Running {system.upper()} with {num_workers} workers ({epochs} epochs)")
    print(f"=======================================================")

    try:
        for epoch in range(1, epochs + 1):
            q_monitor.epoch = epoch
            model.train()
            epoch_start_time = time.time()
            total_compute_time = 0.0
            total_wait_time = 0.0
            batch_idx = 0

            t_req = time.time()
            for batch in train_loader:
                t_data = time.time()
                wait_ms = (t_data - t_req) * 1000
                total_wait_time += (t_data - t_req)

                if isinstance(batch, (tuple, list)):
                    img, lbl = batch
                elif isinstance(batch, dict):
                    img, lbl = batch["image"], batch["label"]
                else:
                    img, lbl = batch
                img = img.to(device, non_blocking=True)
                lbl = lbl.to(device, non_blocking=True)

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
                if system == "minato" and hasattr(train_loader, "queue"):
                    log_data["fast_q_size"] = train_loader.queue.qsize()
                    log_data["slow_q_size"] = train_loader.slow_processed_queue.qsize()

                batch_log_file.write(json.dumps(log_data) + "\n")
                batch_log_file.flush()

                batch_idx += 1
                t_req = time.time()

            epoch_time = time.time() - epoch_start_time
            throughput = (batch_idx * batch_size) / epoch_time if epoch_time > 0 else 0
            gpu_busy = total_compute_time / (total_compute_time + total_wait_time) if (total_compute_time + total_wait_time) > 0 else 0
            avg_wait = (total_wait_time / batch_idx * 1000) if batch_idx > 0 else 0

            print(f" Epoch {epoch}/{epochs}: Throughput = {throughput:6.2f} img/s | GPU Busy = {gpu_busy*100:5.1f}% | Starvation = {total_wait_time:6.2f}s | Avg Wait = {avg_wait:5.1f}ms | RAM = {ram_gb:4.1f}GB")

            epoch_records.append({
                "System": system,
                "Workers": num_workers,
                "Epoch": epoch,
                "Throughput_Img_Sec": round(throughput, 2),
                "GPU_Busy_Ratio": round(gpu_busy, 4),
                "Total_Starvation_Sec": round(total_wait_time, 4),
                "Avg_Wait_Ms": round(avg_wait, 2),
                "RAM_GB": round(ram_gb, 2),
                "VRAM_GB": round(vram_gb, 2),
                "CPU_Percent": round(cpu_p, 1),
                "Total_Batches": batch_idx
            })

    finally:
        q_monitor.stop_event.set()
        batch_log_file.close()
        # Fast non-blocking cleanup
        if hasattr(train_loader, 'queue'):
            try:
                train_loader.queue.cancel_join_thread()
            except Exception:
                pass
        if hasattr(train_loader, 'slow_processed_queue'):
            try:
                train_loader.slow_processed_queue.cancel_join_thread()
            except Exception:
                pass
        if 'slow_queue' in locals() and slow_queue:
            try:
                slow_queue.cancel_join_thread()
            except Exception:
                pass
        if hasattr(train_loader, 'producers'):
            for p in train_loader.producers:
                try:
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=0.2)
                except Exception:
                    pass
        if hasattr(train_loader, '_shutdown_workers'):
            try:
                train_loader._shutdown_workers()
            except Exception:
                pass
        from data_loading.Asynchrnous_dataloader import data_stop_event, data_resume_event
        data_stop_event.clear()
        data_resume_event.set()
        torch.cuda.empty_cache()

    return epoch_records

def main():
    import subprocess
    parser = argparse.ArgumentParser(description="Sweep DataLoader worker count from 1 to 8")
    parser.add_argument("--workers", type=str, default="1,2,3,4,5,6,7,8", help="Comma-separated workers to test")
    parser.add_argument("--systems", type=str, default="pytorch,minato", help="Comma-separated systems (pytorch,minato)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs per test (default: 3)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory for sweep results")
    parser.add_argument("--single_run", action="store_true", help="Run only one single configuration in isolated process")
    parser.add_argument("--system", type=str, choices=["pytorch", "minato"], help="System for single_run")
    parser.add_argument("--worker", type=int, help="Worker count for single_run")
    args = parser.parse_args()

    out_base = Path(args.out_dir)
    runs_dir = out_base / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_base / "worker_sweep_summary.csv"

    # Single isolated run mode
    if args.single_run:
        assert args.system is not None and args.worker is not None, "system and worker must be provided for single_run"
        run_dir = runs_dir / f"{args.system}_w{args.worker}"
        rec = run_single_benchmark(
            system=args.system,
            num_workers=args.worker,
            epochs=args.epochs,
            batch_size=args.batch_size,
            out_dir=str(run_dir)
        )
        # Append to summary CSV
        df_new = pd.DataFrame(rec)
        if summary_csv.exists():
            df_existing = pd.read_csv(summary_csv)
            # Remove any previous records for this system/worker
            df_existing = df_existing[~((df_existing['System'] == args.system) & (df_existing['Workers'] == args.worker))]
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_csv(summary_csv, index=False)
        print(f"✅ Successfully completed and saved {args.system} w={args.worker}")
        sys.stdout.flush()
        os._exit(0)

    # Sweep coordinator mode
    workers_list = [int(w.strip()) for w in args.workers.split(",") if w.strip()]
    systems_list = [s.strip().lower() for s in args.systems.split(",") if s.strip()]

    completed_pairs = set()
    if summary_csv.exists():
        try:
            existing_df = pd.read_csv(summary_csv)
            for _, r in existing_df.iterrows():
                completed_pairs.add((str(r['System']).lower(), int(r['Workers'])))
            print(f"Found existing summary at {summary_csv}")
            print(f"Already completed configurations: {completed_pairs}")
        except Exception:
            pass

    print("\n=======================================================")
    print(" 🚀 STARTING WORKER SCALING SWEEP (ISOLATED PROCESSES)")
    print(f" Systems: {systems_list}")
    print(f" Workers: {workers_list}")
    print(f" Epochs : {args.epochs}")
    print(f" Output : {out_base}")
    print("=======================================================\n")

    for system in systems_list:
        for w in workers_list:
            if (system, w) in completed_pairs:
                print(f"⏩ Skipping {system.upper()} w={w} (already completed in {summary_csv})")
                continue

            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--single_run",
                "--system", system,
                "--worker", str(w),
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--out_dir", str(out_base)
            ]
            print(f"\n>>> Spawning clean isolated process for {system.upper()} w={w} <<<")
            res = subprocess.run(cmd)
            if res.returncode != 0:
                print(f"⚠️ Warning: {system} w={w} finished with return code {res.returncode}")
            time.sleep(1) # Cooldown

    print("\n\n=======================================================")
    print(" 📊 SWEEP BENCHMARK COMPLETE! SUMMARY TABLE:")
    print("=======================================================")
    final_df = pd.DataFrame(all_records)
    # Filter steady-state (e.g. epoch >= 2)
    steady_df = final_df[final_df['Epoch'] >= 2] if args.epochs > 1 else final_df
    agg_table = steady_df.groupby(['System', 'Workers']).agg({
        'Throughput_Img_Sec': 'mean',
        'GPU_Busy_Ratio': 'mean',
        'Total_Starvation_Sec': 'mean',
        'Avg_Wait_Ms': 'mean',
        'RAM_GB': 'max',
        'CPU_Percent': 'mean'
    }).reset_index()
    print(agg_table.to_string(index=False))

if __name__ == "__main__":
    main()
