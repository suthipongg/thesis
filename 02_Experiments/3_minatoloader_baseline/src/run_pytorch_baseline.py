"""
PyTorch DataLoader Baseline
============================
รัน 3D-UNet + KiTS19 ด้วย PyTorch DataLoader มาตรฐาน
เก็บข้อมูลครบทุก metric ก่อน ไม่ plot ระหว่างรัน

Output:
  results/pytorch/{timestamp}_e{E}_bs{B}_w{W}/
    ├── epoch_result.csv    ← per-epoch summary
    ├── batch_logs.jsonl    ← per-batch raw data
    └── system_info.json    ← environment snapshot

Usage:
    source ../.venv_minato/bin/activate
    python run_pytorch_baseline.py --num_workers 2 --epochs 5

Note: ไม่แก้โค้ดใน MinatoLoader/ เลย — import ตรงจาก path
"""
import os
import sys
import time
import json
import csv
import argparse
import platform
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Setup Paths ──────────────────────────────────────────────
SRC_DIR    = Path(__file__).parent
BASE_DIR   = SRC_DIR.parent
THESIS_DIR = BASE_DIR.parent.parent
MINATO_DIR = BASE_DIR / "MinatoLoader" / "Minato"

# Add MinatoLoader to path so we can import their modules directly
for p in [str(MINATO_DIR), str(BASE_DIR / "MinatoLoader")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Import MinatoLoader's components (no modification) ───────
from data_loading.pytorch_loader import get_train_transforms, RandBalancedCrop
from data_loading.pytorch_loader import PytVal
from model.unet3d import Unet3D
from model.losses import DiceCELoss
from runtime.inference import sliding_window_inference


# ─────────────────────────────────────────────────────────────
# Profiled Dataset Wrapper
# ─────────────────────────────────────────────────────────────
class ProfiledPytTrain(torch.utils.data.Dataset):
    """
    Standard PyTorch Dataset for KiTS19 + 3D-UNet.
    Replicates MinatoLoader's data augmentations exactly, but removes the
    custom multiprocessing queues and background workers found in PytTrain.
    """
    def __init__(self, images, labels, flags):
        self.images = images
        self.labels = labels
        self.train_transforms = get_train_transforms()
        self.patch_size = flags.input_shape
        self.oversampling = flags.oversampling
        self.rand_crop = RandBalancedCrop(self.patch_size, self.oversampling)

        # Preload (mmap)
        self.images_data = [np.load(f, mmap_mode="r") for f in self.images]
        self.labels_data = [np.load(f, mmap_mode="r") for f in self.labels]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        t_disk_start = time.perf_counter()
        
        # Disk IO (mmap read)
        image = self.images_data[idx]
        label = self.labels_data[idx]
        
        t_trans_start = time.perf_counter()

        # Transform & Crop
        data = {"image": image, "label": label}
        data = self.rand_crop(data)
        for transform in self.train_transforms.transforms:
            data = transform(data)
            
        img_tensor = torch.from_numpy(data["image"].copy()).float()
        lbl_tensor = torch.from_numpy(data["label"].copy()).long()

        # result is tuple (image, label) — PytTrain already does transform internally
        # We approximate: disk_time = up to load, trans_time = rest of __getitem__
        # Since PytTrain combines load+transform, we split by timing the whole thing
        # and attribute the last portion to transform (augmentation).
        # For 3D: random_crop + flips are expensive — we capture total __getitem__ time.

        t_trans_end = time.perf_counter()

        wi = torch.utils.data.get_worker_info()
        worker_id = wi.id if wi else 0

        return img_tensor, lbl_tensor, t_disk_start, t_trans_start, t_trans_end, worker_id


class ProfiledPytVal(torch.utils.data.Dataset):
    """Same timing wrapper for validation."""
    def __init__(self, images, labels):
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        t_disk_start = time.perf_counter()
        image = np.load(self.images[idx])
        label = np.load(self.labels[idx])
        t_trans_start = time.perf_counter()
        t_trans_end = time.perf_counter()
        
        img_tensor = torch.from_numpy(image).float()
        lbl_tensor = torch.from_numpy(label).long()
        
        wi = torch.utils.data.get_worker_info()
        worker_id = wi.id if wi else 0

        return img_tensor, lbl_tensor, t_disk_start, t_trans_start, t_trans_end, worker_id


# ─────────────────────────────────────────────────────────────
# Custom collate (handle profiling timestamps)
# ─────────────────────────────────────────────────────────────
def profiled_collate(batch):
    """Collate for ProfiledPytTrain — merges timing arrays."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    images      = torch.stack([b[0] for b in batch])
    labels      = torch.stack([b[1] for b in batch])
    disk_starts = torch.tensor([b[2] for b in batch], dtype=torch.float64)
    trans_starts= torch.tensor([b[3] for b in batch], dtype=torch.float64)
    trans_ends  = torch.tensor([b[4] for b in batch], dtype=torch.float64)
    worker_ids  = torch.tensor([b[5] for b in batch], dtype=torch.int32)

    return images, labels, disk_starts, trans_starts, trans_ends, worker_ids


# ─────────────────────────────────────────────────────────────
# Epoch runner (same structure as exp.py)
# ─────────────────────────────────────────────────────────────
def run_epoch(dataloader, model, loss_fn, optimizer, scaler,
              device, is_train, epoch, amp_enabled):
    if is_train:
        model.train()
    else:
        model.eval()

    process = psutil.Process(os.getpid())

    # Accumulators
    batch_logs        = []
    cumulative_loss   = []
    total_samples     = 0
    total_dice        = 0.0

    # GPU event tracking
    start_events = []
    end_events   = []

    # Timing references
    torch.cuda.synchronize()
    epoch_start_time   = time.perf_counter()
    epoch_start_ts_wall= time.time()
    epoch_start_ev     = torch.cuda.Event(enable_timing=True)
    epoch_start_ev.record()
    end_time = epoch_start_time

    pbar = tqdm(dataloader, desc=f"{'Train' if is_train else 'Val  '} E{epoch}", leave=False)

    for iteration, batch in enumerate(pbar):
        if batch is None:
            continue

        images, labels, disk_starts, trans_starts, trans_ends, worker_ids = batch

        # ── CPU timing ──────────────────────────────────────
        batch_start_ts    = time.time()
        dataloader_wait   = time.perf_counter() - end_time

        disk_durations  = trans_starts - disk_starts        # tensor [B]
        trans_durations = trans_ends   - trans_starts       # tensor [B]

        # ── GPU events ──────────────────────────────────────
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)
        start_ev.record()

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── Forward / Backward ──────────────────────────────
        with autocast(enabled=amp_enabled):
            if is_train:
                output = model(images)
                loss   = loss_fn(output, labels)
            else:
                with torch.no_grad():
                    output, labels = sliding_window_inference(
                        inputs=images,
                        labels=labels,
                        roi_shape=[128, 128, 128],
                        model=model,
                        overlap=0.5,
                        mode="gaussian",
                        padding_val=-2.2
                    )
                    loss = loss_fn(output, labels)

        if is_train:
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()

        end_ev.record()

        start_events.append(start_ev)
        end_events.append(end_ev)

        # ── Resource snapshot ────────────────────────────────
        ram_gb  = process.memory_info().rss / 1e9
        vram_gb = torch.cuda.memory_allocated(device) / 1e9

        # ── Per-batch log entry ──────────────────────────────
        disk_start_rel_ms = (disk_starts.min().item() - epoch_start_ts_wall) * 1000.0
        trans_end_rel_ms  = (trans_ends.max().item()  - epoch_start_ts_wall) * 1000.0

        entry = {
            "epoch":              epoch,
            "batch_id":           iteration,
            "phase":              "train" if is_train else "val",
            # timing (ms)
            "disk_ms":            disk_durations.sum().item()  * 1000.0,
            "disk_mean_ms":       disk_durations.mean().item() * 1000.0,
            "disk_max_ms":        disk_durations.max().item()  * 1000.0,
            "trans_ms":           trans_durations.sum().item() * 1000.0,
            "dataloader_wait_ms": dataloader_wait * 1000.0,
            # gpu timing (filled after sync)
            "gpu_start_rel_ms":   None,
            "gpu_end_rel_ms":     None,
            "gpu_compute_ms":     None,
            "gpu_starvation_ms":  None,
            # worker info
            "worker_id":          int(worker_ids[0].item()),
            "disk_start_rel_ms":  disk_start_rel_ms,
            "trans_end_rel_ms":   trans_end_rel_ms,
            # resource
            "ram_gb":             round(ram_gb,  3),
            "vram_gb":            round(vram_gb, 3),
            "batch_size":         images.size(0),
            "start_ts":           batch_start_ts,
            "timestamp_relative": time.perf_counter() - epoch_start_time,
        }
        batch_logs.append(entry)
        cumulative_loss.append(loss.item())
        total_samples += images.size(0)
        end_time = time.perf_counter()

        # ── Prevent OOM during validation ───────────────────────
        if not is_train:
            del images, labels, output, loss
            torch.cuda.empty_cache()

    # ── End of epoch: sync GPU ────────────────────────────────
    torch.cuda.synchronize()
    epoch_total_time = time.perf_counter() - epoch_start_time

    # ── Backfill GPU timing ───────────────────────────────────
    gpu_compute_list = []
    gpu_idle_list    = []

    for i, (se, ee) in enumerate(zip(start_events, end_events)):
        gpu_start_ms = epoch_start_ev.elapsed_time(se)
        gpu_end_ms   = epoch_start_ev.elapsed_time(ee)
        compute_ms   = gpu_end_ms - gpu_start_ms

        batch_logs[i]["gpu_start_rel_ms"] = gpu_start_ms
        batch_logs[i]["gpu_end_rel_ms"]   = gpu_end_ms
        batch_logs[i]["gpu_compute_ms"]   = compute_ms

        if i == 0:
            idle_ms = gpu_start_ms
        else:
            prev_end = batch_logs[i-1]["gpu_end_rel_ms"]
            idle_ms  = max(0.0, gpu_start_ms - prev_end)

        batch_logs[i]["gpu_starvation_ms"] = idle_ms
        gpu_compute_list.append(compute_ms / 1000.0)
        if i > 0:
            gpu_idle_list.append(idle_ms / 1000.0)

    # ── Compute epoch-level metrics ───────────────────────────
    def safe_mean(lst):
        lst = lst[1:] if len(lst) > 1 else lst
        return float(np.mean(lst)) if lst else 0.0

    total_compute   = sum(gpu_compute_list)
    total_starvation= sum(gpu_idle_list)
    denom = total_compute + total_starvation

    epoch_metrics = {
        "Loss":                      np.mean(cumulative_loss),
        "Mean_Dice":                 0.0,  # placeholder — compute separately if needed
        "Throughput_Img_Sec":        total_samples / epoch_total_time,
        "Avg_Disk_Read":             safe_mean([e["disk_ms"]        / 1000 for e in batch_logs]),
        "Avg_Disk_Read_mean":        safe_mean([e["disk_mean_ms"]   / 1000 for e in batch_logs]),
        "Avg_Disk_Read_max":         safe_mean([e["disk_max_ms"]    / 1000 for e in batch_logs]),
        "Avg_Transform":             safe_mean([e["trans_ms"]       / 1000 for e in batch_logs]),
        "Avg_Wait":                  safe_mean([e["dataloader_wait_ms"] / 1000 for e in batch_logs]),
        "Avg_GPU_Compute":           safe_mean(gpu_compute_list),
        "Avg_GPU_Idle":              safe_mean(gpu_idle_list),
        "Total_GPU_Compute":         total_compute,
        "Total_GPU_Idle":            total_starvation,
        "Avg_RAM_GB":                safe_mean([e["ram_gb"]  for e in batch_logs]),
        "Peak_RAM_GB":               max(e["ram_gb"]  for e in batch_logs),
        "Avg_VRAM_GB":               safe_mean([e["vram_gb"] for e in batch_logs]),
        "Peak_VRAM":                 torch.cuda.max_memory_allocated(device) / 1e9,
        "GPU_Busy_Ratio_All":        total_compute / denom if denom > 0 else 0.0,
        "GPU_Busy_Ratio_NoWarmup":   (sum(gpu_compute_list[1:]) /
                                      (sum(gpu_compute_list[1:]) + total_starvation)
                                      if sum(gpu_compute_list[1:]) + total_starvation > 0 else 0.0),
        "Epoch_Time_Sec":            epoch_total_time,
        "Total_GPU_Starvation_Sec":  total_starvation,
        "epoch_batch_logs":          batch_logs,
    }
    return epoch_metrics


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PyTorch DataLoader Baseline — KiTS19 + 3D-UNet")
    parser.add_argument("--data_dir",    default=str(THESIS_DIR / "00_datasets" / "kits19_preproc"))
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs",      type=int, default=5)
    parser.add_argument("--batch_size",  type=int, default=2)
    parser.add_argument("--amp",         action="store_true", default=True)
    parser.add_argument("--no_amp",      action="store_true", default=False)
    parser.add_argument("--dry_run",     action="store_true", help="2 epochs, 5 batches only")
    parser.add_argument("--eval_cases_txt", default=str(MINATO_DIR / "evaluation_cases.txt"))
    args = parser.parse_args()

    amp_enabled = args.amp and not args.no_amp
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  PyTorch DataLoader Baseline")
    print(f"  Workers={args.num_workers}  BS={args.batch_size}  Epochs={args.epochs}  AMP={amp_enabled}")
    print(f"  Device: {device}")
    print(f"  Data:   {args.data_dir}")
    print(f"{'='*60}\n")

    # ── Output dir ────────────────────────────────────────────
    ts       = int(time.time())
    dry_tag  = "_dry" if args.dry_run else ""
    run_name = f"{ts}_e{args.epochs}_bs{args.batch_size}_w{args.num_workers}{dry_tag}"
    out_dir  = BASE_DIR / "results" / "pytorch" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Output: {out_dir}\n")

    # ── System info snapshot ──────────────────────────────────
    system_info = {
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python_version":  platform.python_version(),
        "torch_version":   torch.__version__,
        "cuda_version":    torch.version.cuda,
        "gpu_name":        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "cpu_model":       platform.processor(),
        "cpu_cores":       psutil.cpu_count(logical=True),
        "ram_total_gb":    round(psutil.virtual_memory().total / 1e9, 2),
        "vram_total_gb":   round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
                           if torch.cuda.is_available() else 0,
        "loader":          "pytorch",
        "num_workers":     args.num_workers,
        "batch_size":      args.batch_size,
        "epochs":          args.epochs,
        "amp":             amp_enabled,
        "dry_run":         args.dry_run,
        "data_dir":        args.data_dir,
    }
    with open(out_dir / "system_info.json", "w") as f:
        json.dump(system_info, f, indent=2)

    # ── Read split ────────────────────────────────────────────
    with open(args.eval_cases_txt) as f:
        val_cases = [l.strip() for l in f if l.strip()]

    all_npy   = list(Path(args.data_dir).glob("*_x.npy"))
    all_ids   = sorted(set(p.name.replace("_x.npy", "").replace("case_", "")
                           for p in all_npy))
    train_ids = [c for c in all_ids if c not in val_cases]
    val_ids   = [c for c in all_ids if c in val_cases]

    print(f"📊 Dataset split: {len(train_ids)} train / {len(val_ids)} val cases")

    # ── Flags object (matches MinatoLoader's flags interface) ──
    class Flags:
        input_shape     = [128, 128, 128]
        val_input_shape = [128, 128, 128]
        oversampling    = 0.4
        overlap         = 0.5
        seed            = -1
        batch_size      = args.batch_size
        num_workers     = args.num_workers
        layout          = "NCDHW"

    flags = Flags()

    x_train = [os.path.join(args.data_dir, f"case_{c}_x.npy") for c in train_ids]
    y_train = [os.path.join(args.data_dir, f"case_{c}_y.npy") for c in train_ids]
    x_val   = [os.path.join(args.data_dir, f"case_{c}_x.npy") for c in val_ids]
    y_val   = [os.path.join(args.data_dir, f"case_{c}_y.npy") for c in val_ids]

    train_dataset = ProfiledPytTrain(x_train, y_train, flags)
    val_dataset   = ProfiledPytVal(x_val, y_val)

    if args.dry_run:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(min(10, len(train_dataset))))
        val_dataset   = Subset(val_dataset,   range(min(8,  len(val_dataset))))
        print("🔵 DRY RUN: using small subset\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=profiled_collate,
        drop_last=True,
    )
    val_loader   = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=profiled_collate,
        drop_last=False,
    )

    # ── Model ─────────────────────────────────────────────────
    model   = Unet3D(in_channels=1, n_class=3, normalization="instancenorm", activation="relu").to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, use_softmax=True, layout=flags.layout, include_background=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.8, momentum=0.9, weight_decay=0.0)
    scaler    = GradScaler()

    # ── CSV header ────────────────────────────────────────────
    csv_path = out_dir / "epoch_result.csv"
    log_path = out_dir / "batch_logs.jsonl"

    CSV_FIELDS = [
        "Epoch", "Phase", "Loss", "Mean_Dice",
        "Throughput_Img_Sec",
        "Disk_Read_Sec", "Disk_Read_Mean_Sec", "Disk_Read_Max_Sec",
        "Transform_Sec", "DataLoader_Wait_Sec",
        "GPU_Compute_Sec", "GPU_Idle_Starvation_Sec",
        "Total_GPU_Compute_Sec", "Total_GPU_Idle_Sec",
        "RAM_GB", "Peak_RAM_GB", "VRAM_GB", "Peak_VRAM",
        "GPU_Busy_Ratio_All", "GPU_Busy_Ratio_NoWarmup",
        "Epoch_Time_Sec", "Total_GPU_Starvation_Sec",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_FIELDS)

    # ── Training loop ─────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        print(f"\n[{time.strftime('%H:%M:%S')}] 🚀 Epoch {epoch}/{args.epochs}")

        # --- Train ---
        torch.cuda.reset_peak_memory_stats(device)
        train_m = run_epoch(
            train_loader, model, loss_fn, optimizer, scaler,
            device, is_train=True, epoch=epoch, amp_enabled=amp_enabled
        )
        print(f"  Train | Loss: {train_m['Loss']:.4f} | "
              f"Throughput: {train_m['Throughput_Img_Sec']:.2f} samp/s | "
              f"GPU Starvation: {train_m['Total_GPU_Starvation_Sec']:.2f}s | "
              f"GPU Busy: {train_m['GPU_Busy_Ratio_NoWarmup']*100:.1f}%")

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch, "Train",
                train_m["Loss"], train_m["Mean_Dice"],
                train_m["Throughput_Img_Sec"],
                train_m["Avg_Disk_Read"],  train_m["Avg_Disk_Read_mean"], train_m["Avg_Disk_Read_max"],
                train_m["Avg_Transform"],  train_m["Avg_Wait"],
                train_m["Avg_GPU_Compute"], train_m["Avg_GPU_Idle"],
                train_m["Total_GPU_Compute"], train_m["Total_GPU_Idle"],
                train_m["Avg_RAM_GB"],  train_m["Peak_RAM_GB"],
                train_m["Avg_VRAM_GB"], train_m["Peak_VRAM"],
                train_m["GPU_Busy_Ratio_All"], train_m["GPU_Busy_Ratio_NoWarmup"],
                train_m["Epoch_Time_Sec"], train_m["Total_GPU_Starvation_Sec"],
            ])

        with open(log_path, "a") as f:
            for entry in train_m["epoch_batch_logs"]:
                f.write(json.dumps(entry) + "\n")

        # --- Validation ---
        torch.cuda.reset_peak_memory_stats(device)
        val_m = run_epoch(
            val_loader, model, loss_fn, optimizer, scaler,
            device, is_train=False, epoch=epoch, amp_enabled=amp_enabled
        )
        print(f"  Val   | Loss: {val_m['Loss']:.4f} | "
              f"Throughput: {val_m['Throughput_Img_Sec']:.2f} samp/s")

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch, "Validation",
                val_m["Loss"], val_m["Mean_Dice"],
                val_m["Throughput_Img_Sec"],
                val_m["Avg_Disk_Read"],  val_m["Avg_Disk_Read_mean"], val_m["Avg_Disk_Read_max"],
                val_m["Avg_Transform"],  val_m["Avg_Wait"],
                val_m["Avg_GPU_Compute"], val_m["Avg_GPU_Idle"],
                val_m["Total_GPU_Compute"], val_m["Total_GPU_Idle"],
                val_m["Avg_RAM_GB"],  val_m["Peak_RAM_GB"],
                val_m["Avg_VRAM_GB"], val_m["Peak_VRAM"],
                val_m["GPU_Busy_Ratio_All"], val_m["GPU_Busy_Ratio_NoWarmup"],
                val_m["Epoch_Time_Sec"], val_m["Total_GPU_Starvation_Sec"],
            ])

        with open(log_path, "a") as f:
            for entry in val_m["epoch_batch_logs"]:
                f.write(json.dumps(entry) + "\n")

    print(f"\n✅ Done! Results in: {out_dir}")
    print(f"   epoch_result.csv  : {csv_path}")
    print(f"   batch_logs.jsonl  : {log_path}")
    print(f"   system_info.json  : {out_dir / 'system_info.json'}")


if __name__ == "__main__":
    main()
