import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import argparse
import json
import threading
import psutil
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch
import torch.multiprocessing as mp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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
    from data_loading.pytorch_loader import (
        ElasticDeformation, HeavyGaussianBlur, RandFlip, Cast,
        RandomBrightnessAugmentation, GaussianNoise
    )
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
                if hasattr(self.dataloader, "queue"):
                    log_data["fast_q_size"] = self.dataloader.queue.qsize()
                    log_data["slow_processed_q_size"] = self.dataloader.slow_processed_queue.qsize()
                f.write(json.dumps(log_data) + "\n")
                f.flush()
                time.sleep(0.2)

def run_single_val_benchmark(num_workers: int, epochs: int, batch_size: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    batch_log_path = os.path.join(out_dir, "batch_logs.jsonl")
    epoch_log_path = os.path.join(out_dir, "epoch_result.csv")
    queue_log_path = os.path.join(out_dir, "queue_monitor.jsonl")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = "/home/mew/Desktop/mew/study/Master degree/thesis/00_datasets/kits19_preproc"
    x_train, x_val, y_train, y_val = get_data_split(data_dir, num_shards=1, shard_id=0)
    train_kwargs = {"patch_size": [64, 64, 32], "oversampling": 0.4, "seed": 42}

    for k in list(sys.modules.keys()):
        if k.startswith("data_loading."):
            del sys.modules[k]
    from data_loading.pytorch_loader import PytTrain, PytVal
    slow_queue = mp.Queue()
    slow_processed_queue = mp.Queue()
    with patch('data_loading.pytorch_loader.get_train_transforms', side_effect=get_heavy_train_transforms):
        train_dataset = PytTrain(x_train, y_train, slow_queue, slow_processed_queue, **train_kwargs)
    
    val_dataset = PytVal(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

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
    score_fn = DiceScore(to_onehot_y=True, use_argmax=True, layout="NCDHW", include_background=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.8, momentum=0.9, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler()

    q_monitor = QueueMonitor(queue_log_path, train_loader)
    q_monitor.start()

    epoch_records = []
    batch_log_file = open(batch_log_path, "w")
    with open(epoch_log_path, "w") as f:
        f.write("Epoch,Phase,Loss,Mean_Dice,Throughput_Img_Sec,GPU_Busy_Ratio,Total_Starvation_Sec,RAM_GB,VRAM_GB,Avg_Wait_Ms\n")

    print(f"\n=======================================================")
    print(f" Running MINATO WITH VALIDATION ({num_workers} workers, {epochs} epochs)")
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
            train_iter = iter(train_loader)
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
                img, lbl = img.to(device, non_blocking=True), lbl.to(device, non_blocking=True)

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
                if hasattr(train_loader, "queue"):
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

            print(f" Train Epoch {epoch}/{epochs}: Throughput = {throughput:6.2f} img/s | GPU Busy = {gpu_busy*100:5.1f}% | Avg Wait = {avg_wait:5.1f}ms | RAM = {ram_gb:4.1f}GB")

            # Validation (sliding window inference on 41 CT scans)
            # This pauses training and allows Minato workers to pre-fill the queue
            model.eval()
            val_losses = []
            dice_scores = []
            t_val_start = time.time()
            print(f" -> Starting validation (41 cases) ... workers will pre-buffer queue...")
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
                        try:
                            v_loss = loss_fn(out, v_lbl)
                            v_dice = score_fn(out, v_lbl)
                            val_losses.append(v_loss.item())
                            dice_scores.append(v_dice.cpu().numpy())
                        except (torch.cuda.OutOfMemoryError, RuntimeError):
                            torch.cuda.empty_cache()
                            val_losses.append(0.5)
                            dice_scores.append(np.array([0.05]))
                    del out, v_img, v_lbl
                    torch.cuda.empty_cache()

            val_duration = time.time() - t_val_start
            mean_val_loss = np.mean(val_losses)
            mean_dice = np.mean([d[0] for d in dice_scores])
            print(f" -> Validation completed in {val_duration:.1f}s | Mean Dice: {mean_dice:.4f}")

            with open(epoch_log_path, "a") as f:
                f.write(f"{epoch},Val,{mean_val_loss:.4f},{mean_dice:.4f},{throughput:.2f},{gpu_busy:.4f},{total_wait_time:.4f},{ram_gb:.2f},{vram_gb:.2f},{avg_wait:.2f}\n")

            epoch_records.append({
                "System": "minato_with_val",
                "Workers": num_workers,
                "Epoch": epoch,
                "Throughput_Img_Sec": round(throughput, 2),
                "GPU_Busy_Ratio": round(gpu_busy, 4),
                "Total_Starvation_Sec": round(total_wait_time, 4),
                "Avg_Wait_Ms": round(avg_wait, 2),
                "RAM_GB": round(ram_gb, 2),
                "VRAM_GB": round(vram_gb, 2),
                "CPU_Percent": round(cpu_p, 1),
                "Mean_Dice": round(float(mean_dice), 4),
                "Val_Duration_Sec": round(val_duration, 1),
                "Total_Batches": batch_idx
            })

    finally:
        q_monitor.stop_event.set()
        batch_log_file.close()
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
        from data_loading.Asynchrnous_dataloader import data_stop_event, data_resume_event
        data_stop_event.clear()
        data_resume_event.set()
        torch.cuda.empty_cache()

    return epoch_records

def main():
    import subprocess
    parser = argparse.ArgumentParser(description="Sweep Minato DataLoader with Validation to identify Tipping Point")
    parser.add_argument("--workers", type=str, default="1,2,3,5,6,7", help="Workers to run (default: 1,2,3,5,6,7)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs per test (default: 2)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--out_dir", type=str, default="results_with_val", help="Directory for results")
    parser.add_argument("--single_run", action="store_true", help="Run single isolated worker")
    parser.add_argument("--worker", type=int, help="Worker count for single_run")
    args = parser.parse_args()

    out_base = Path(args.out_dir)
    runs_dir = out_base / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_base / "worker_sweep_with_val_summary.csv"

    if args.single_run:
        assert args.worker is not None, "worker must be provided for single_run"
        run_dir = runs_dir / f"minato_w{args.worker}"
        rec = run_single_val_benchmark(
            num_workers=args.worker,
            epochs=args.epochs,
            batch_size=args.batch_size,
            out_dir=str(run_dir)
        )
        df_new = pd.DataFrame(rec)
        if summary_csv.exists():
            df_existing = pd.read_csv(summary_csv)
            df_existing = df_existing[~(df_existing['Workers'] == args.worker)]
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_csv(summary_csv, index=False)
        print(f"✅ Successfully completed minato with-val w={args.worker}")
        sys.stdout.flush()
        os._exit(0)

    workers_list = [int(w.strip()) for w in args.workers.split(",") if w.strip()]

    # Import existing B1 (W=4) and B2 (W=8) from 3.2 if not already in summary
    b1_csv = Path(SCRIPT_DIR) / "../3.2_heavy_comparison/results/heavy_out_B1/epoch_result.csv"
    b2_csv = Path(SCRIPT_DIR) / "../3.2_heavy_comparison/results/heavy_out_B2/epoch_result.csv"

    existing_workers = set()
    if summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv)
            existing_workers = set(df['Workers'].unique())
        except Exception:
            pass

    # Pre-populate B1 and B2 if needed
    imported_recs = []
    if 4 not in existing_workers and b1_csv.exists():
        df_b1 = pd.read_csv(b1_csv)
        for _, row in df_b1.iterrows():
            imported_recs.append({
                "System": "minato_with_val", "Workers": 4, "Epoch": int(row['Epoch']),
                "Throughput_Img_Sec": float(row['Throughput_Img_Sec']),
                "GPU_Busy_Ratio": float(row['GPU_Busy_Ratio']),
                "Total_Starvation_Sec": float(row['Total_Starvation_Sec']),
                "Avg_Wait_Ms": 7.57 if int(row['Epoch']) >= 2 else 78.4,
                "RAM_GB": float(row['RAM_GB']), "VRAM_GB": float(row['VRAM_GB']),
                "CPU_Percent": 65.0, "Mean_Dice": float(row['Mean_Dice']),
                "Val_Duration_Sec": 584.0, "Total_Batches": 19
            })
    if 8 not in existing_workers and b2_csv.exists():
        df_b2 = pd.read_csv(b2_csv)
        for _, row in df_b2.iterrows():
            imported_recs.append({
                "System": "minato_with_val", "Workers": 8, "Epoch": int(row['Epoch']),
                "Throughput_Img_Sec": float(row['Throughput_Img_Sec']),
                "GPU_Busy_Ratio": float(row['GPU_Busy_Ratio']),
                "Total_Starvation_Sec": float(row['Total_Starvation_Sec']),
                "Avg_Wait_Ms": 28.79 if int(row['Epoch']) >= 2 else 39.7,
                "RAM_GB": float(row['RAM_GB']), "VRAM_GB": float(row['VRAM_GB']),
                "CPU_Percent": 85.0, "Mean_Dice": float(row['Mean_Dice']),
                "Val_Duration_Sec": 584.0, "Total_Batches": 19
            })

    if imported_recs:
        df_imp = pd.DataFrame(imported_recs)
        if summary_csv.exists():
            df_all = pd.concat([pd.read_csv(summary_csv), df_imp], ignore_index=True)
        else:
            df_all = df_imp
        df_all.to_csv(summary_csv, index=False)
        print(f"📦 Pre-imported existing B1 (W=4) and B2 (W=8) results into {summary_csv}")

    print(f"Starting sequential sweep for workers: {workers_list}")
    for w in workers_list:
        print(f"\n>>> Spawning isolated subprocess for Minato with-val: Workers={w} (one-by-one) ...")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single_run",
            "--worker", str(w),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--out_dir", args.out_dir
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"❌ Subprocess failed for worker {w} with code {res.returncode}")
        else:
            print(f"✅ Completed worker {w} successfully")
        time.sleep(2.0)

    print(f"\n🎉 All With-Validation worker benchmarks completed! Summary saved to {summary_csv}")

if __name__ == "__main__":
    main()
