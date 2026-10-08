"""
MinatoLoader Baseline
======================
รัน 3D-UNet + KiTS19 ด้วย MinatoLoader's AsynchronousLoader
เก็บข้อมูลทุก metric รวมถึง queue states และ sample path tracking

Output:
  results/minato/{timestamp}_e{E}_bs{B}_w{W}/
    ├── epoch_result.csv      ← per-epoch summary (compatible กับ plot_results.py)
    ├── batch_logs.jsonl      ← per-batch (common fields + minato-exclusive)
    ├── queue_monitor.jsonl   ← queue sizes polled ทุก 100ms
    ├── sample_trace.jsonl    ← per-sample fast/slow path
    └── system_info.json      ← environment snapshot

Usage:
    source ../.venv_minato/bin/activate
    python run_minato_baseline.py --num_workers 2 --epochs 5

Note: ไม่แก้โค้ดใน MinatoLoader/ เลย
"""
import os
import sys
import time
import json
import csv
import argparse
import platform
import threading
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ── Setup Paths ──────────────────────────────────────────────
SRC_DIR    = Path(__file__).parent
BASE_DIR   = SRC_DIR.parent
THESIS_DIR = BASE_DIR.parent.parent
MINATO_DIR = BASE_DIR / "MinatoLoader" / "Minato"

for p in [str(MINATO_DIR), str(BASE_DIR / "MinatoLoader"), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Import MinatoLoader's components ─────────────────────────
from data_loading.Asynchrnous_dataloader import AsynchronousLoader
from model.unet3d import Unet3D
from model.losses import DiceCELoss
from runtime.inference import sliding_window_inference

# ── Import our instrumentation ────────────────────────────────
from instrumented_dataset import InstrumentedPytTrain, QueueMonitor


# ─────────────────────────────────────────────────────────────
# MinatoLoader Flags (matches their runtime/arguments.py interface)
# ─────────────────────────────────────────────────────────────
class MinatoFlags:
    """Replicates the argparse Namespace that MinatoLoader's code expects."""
    def __init__(self, args):
        self.data_dir       = args.data_dir
        self.raw_dir        = args.data_dir   # not used by loader, but required by flags
        self.log_dir        = "/tmp"
        self.loader         = "minato"
        self.local_rank     = 0
        self.disable_logging= True
        self.epochs         = args.epochs
        self.quality_threshold = 0.908
        self.ga_steps       = 1
        self.warmup_steps   = 4
        self.batch_size     = args.batch_size
        self.layout         = "NCDHW"
        self.input_shape    = [128, 128, 128]
        self.val_input_shape= [128, 128, 128]
        self.seed           = -1
        self.num_workers    = args.num_workers
        self.exec_mode      = "train"
        self.benchmark      = False
        self.amp            = args.amp and not args.no_amp
        self.optimizer      = "sgd"
        self.learning_rate  = 0.8
        self.init_learning_rate = 1e-4
        self.lr_warmup_epochs = 5
        self.lr_decay_epochs  = []
        self.lr_decay_factor  = 1.0
        self.lamb_betas     = [0.9, 0.999]
        self.momentum       = 0.9
        self.weight_decay   = 0.0
        self.evaluate_every = None
        self.start_eval_at  = None
        self.verbose        = True
        self.normalization  = "instancenorm"
        self.activation     = "relu"
        self.oversampling   = 0.4
        self.overlap        = 0.5
        self.include_background = False
        self.cudnn_benchmark    = False
        self.cudnn_deterministic= False


# ─────────────────────────────────────────────────────────────
# Patched AsynchronousLoader — wraps __next__ to measure batch assembly time
# and capture queue state at delivery
# ─────────────────────────────────────────────────────────────
class InstrumentedAsyncLoader:
    """
    Thin wrapper around AsynchronousLoader that adds timing and queue monitoring.
    Delegates everything else to the real loader.
    """
    def __init__(self, async_loader, batch_extra_log_ref):
        """
        async_loader         : the real AsynchronousLoader instance
        batch_extra_log_ref  : list(dict) — we append one dict per batch with extra fields
        """
        self._loader        = async_loader
        self._extra_log     = batch_extra_log_ref

    def __iter__(self):
        self._loader_iter = iter(self._loader)
        return self

    def __next__(self):
        t_start = time.perf_counter()
        batch   = next(self._loader_iter)
        assembly_ms = (time.perf_counter() - t_start) * 1000.0

        # Snapshot queue sizes at the moment batch was delivered
        try:
            fast_q_size = self._loader.queue.qsize()
        except Exception:
            fast_q_size = -1
        try:
            slow_q_size = self._loader.slow_processed_queue.qsize()
        except Exception:
            slow_q_size = -1

        # Count None items in batch (slow path samples that got returned anyway)
        # In MinatoLoader, slow samples are served from slow_processed_queue,
        # so they're not None but they have higher latency
        self._extra_log.append({
            "batch_assembly_ms":      round(assembly_ms, 3),
            "fast_q_size_at_delivery":fast_q_size,
            "slow_q_size_at_delivery":slow_q_size,
        })

        return batch

    def __len__(self):
        return len(self._loader)

    @property
    def sampler(self):
        return getattr(self._loader, "sampler", None)


# ─────────────────────────────────────────────────────────────
# Epoch runner (same structure as run_pytorch_baseline.py)
# ─────────────────────────────────────────────────────────────
def run_epoch(loader, model, loss_fn, optimizer, scaler,
              device, is_train, epoch, amp_enabled,
              batch_extra_log_ref, epoch_ref):

    epoch_ref[0] = epoch

    if is_train:
        model.train()
    else:
        model.eval()

    process = psutil.Process(os.getpid())

    batch_logs      = []
    cumulative_loss = []
    total_samples   = 0
    start_events    = []
    end_events      = []

    torch.cuda.synchronize()
    epoch_start_time    = time.perf_counter()
    epoch_start_ts_wall = time.time()
    epoch_start_ev      = torch.cuda.Event(enable_timing=True)
    epoch_start_ev.record()
    end_time = epoch_start_time

    batch_extra_log_ref.clear()

    pbar = tqdm(loader, desc=f"{'Train' if is_train else 'Val  '} E{epoch}", leave=False)

    for iteration, batch in enumerate(pbar):
        if batch is None:
            continue

        # MinatoLoader's AsynchronousLoader yields (image, label) directly
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            images, labels = batch
        else:
            continue

        batch_start_ts  = time.time()
        dataloader_wait = time.perf_counter() - end_time

        # Extra info from InstrumentedAsyncLoader (queue state at this iteration)
        extra = batch_extra_log_ref[iteration] if iteration < len(batch_extra_log_ref) else {}

        # ── GPU events ──────────────────────────────────────
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)
        start_ev.record()

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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

        ram_gb  = process.memory_info().rss / 1e9
        vram_gb = torch.cuda.memory_allocated(device) / 1e9

        entry = {
            "epoch":                   epoch,
            "batch_id":                iteration,
            "phase":                   "train" if is_train else "val",
            # timing — MinatoLoader doesn't expose per-sample disk/trans time easily
            # so we use dataloader_wait as the proxy for Data Loading Time
            "disk_ms":                 None,   # not directly accessible from outside
            "disk_mean_ms":            None,
            "disk_max_ms":             None,
            "trans_ms":                None,
            "dataloader_wait_ms":      dataloader_wait * 1000.0,
            # gpu timing (filled after sync)
            "gpu_start_rel_ms":        None,
            "gpu_end_rel_ms":          None,
            "gpu_compute_ms":          None,
            "gpu_starvation_ms":       None,
            # worker info
            "worker_id":               0,      # MinatoLoader manages workers internally
            "disk_start_rel_ms":       None,
            "trans_end_rel_ms":        None,
            # resource
            "ram_gb":                  round(ram_gb,  3),
            "vram_gb":                 round(vram_gb, 3),
            "batch_size":              images.size(0),
            "start_ts":                batch_start_ts,
            "timestamp_relative":      time.perf_counter() - epoch_start_time,
            # ── MinatoLoader-exclusive ──────────────────────
            "batch_assembly_ms":       extra.get("batch_assembly_ms"),
            "fast_q_size_at_delivery": extra.get("fast_q_size_at_delivery"),
            "slow_q_size_at_delivery": extra.get("slow_q_size_at_delivery"),
        }
        batch_logs.append(entry)
        cumulative_loss.append(loss.item())
        total_samples += images.size(0)
        end_time = time.perf_counter()

    # ── End of epoch sync ────────────────────────────────────
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

    def safe_mean(lst):
        lst = lst[1:] if len(lst) > 1 else lst
        return float(np.mean(lst)) if lst else 0.0

    total_compute    = sum(gpu_compute_list)
    total_starvation = sum(gpu_idle_list)
    denom = total_compute + total_starvation

    # slow ratio from sample_trace (filled by InstrumentedPytTrain externally)
    epoch_metrics = {
        "Loss":                      np.mean(cumulative_loss) if cumulative_loss else 0.0,
        "Mean_Dice":                 0.0,
        "Throughput_Img_Sec":        total_samples / epoch_total_time,
        "Avg_Disk_Read":             None,
        "Avg_Disk_Read_mean":        None,
        "Avg_Disk_Read_max":         None,
        "Avg_Transform":             None,
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
        "Slow_Sample_Ratio":         None,  # filled from sample_trace.jsonl after run
        "Mean_Batch_Assembly_Ms":    safe_mean([e["batch_assembly_ms"]
                                                for e in batch_logs
                                                if e.get("batch_assembly_ms") is not None]),
        "epoch_batch_logs":          batch_logs,
    }
    return epoch_metrics


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MinatoLoader Baseline — KiTS19 + 3D-UNet")
    parser.add_argument("--data_dir",    default=str(THESIS_DIR / "00_datasets" / "kits19_preproc"))
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs",      type=int, default=5)
    parser.add_argument("--batch_size",  type=int, default=2)
    parser.add_argument("--amp",         action="store_true", default=True)
    parser.add_argument("--no_amp",      action="store_true", default=False)
    parser.add_argument("--dry_run",     action="store_true")
    parser.add_argument("--queue_poll_ms", type=int, default=100)
    parser.add_argument("--eval_cases_txt", default=str(MINATO_DIR / "evaluation_cases.txt"))
    args = parser.parse_args()

    amp_enabled = args.amp and not args.no_amp
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  MinatoLoader Baseline")
    print(f"  Workers={args.num_workers}  BS={args.batch_size}  Epochs={args.epochs}  AMP={amp_enabled}")
    print(f"  Device: {device}")
    print(f"  Data:   {args.data_dir}")
    print(f"{'='*60}\n")

    # ── Output dir ────────────────────────────────────────────
    ts       = int(time.time())
    dry_tag  = "_dry" if args.dry_run else ""
    run_name = f"{ts}_e{args.epochs}_bs{args.batch_size}_w{args.num_workers}{dry_tag}"
    out_dir  = BASE_DIR / "results" / "minato" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Output: {out_dir}\n")

    # ── System info ───────────────────────────────────────────
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
        "loader":          "minato",
        "num_workers":     args.num_workers,
        "batch_size":      args.batch_size,
        "epochs":          args.epochs,
        "amp":             amp_enabled,
        "dry_run":         args.dry_run,
        "data_dir":        args.data_dir,
        "queue_poll_ms":   args.queue_poll_ms,
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

    # ── Build MinatoLoader flags ──────────────────────────────
    flags = MinatoFlags(args)
    flags.data_dir = args.data_dir

    # ── Log file paths ────────────────────────────────────────
    sample_trace_path  = str(out_dir / "sample_trace.jsonl")
    queue_monitor_path = str(out_dir / "queue_monitor.jsonl")
    csv_path           = out_dir / "epoch_result.csv"
    log_path           = out_dir / "batch_logs.jsonl"

    # ── Shared mutable epoch reference ───────────────────────
    epoch_ref = [1]

    import multiprocessing as mp

    x_train = [os.path.join(args.data_dir, f"case_{c}_x.npy") for c in train_ids]
    y_train = [os.path.join(args.data_dir, f"case_{c}_y.npy") for c in train_ids]
    x_val   = [os.path.join(args.data_dir, f"case_{c}_x.npy") for c in val_ids]
    y_val   = [os.path.join(args.data_dir, f"case_{c}_y.npy") for c in val_ids]

    slow_queue = mp.Queue()
    slow_processed_queue = mp.Queue()

    # ── InstrumentedPytTrain (subclass — no code change to PytTrain) ──
    train_dataset_inst = InstrumentedPytTrain(
        x_train, y_train, slow_queue, slow_processed_queue,
        patch_size=flags.input_shape,
        oversampling=flags.oversampling,
        seed=flags.seed,
        sample_trace_path=sample_trace_path,
        epoch_ref=epoch_ref,
    )

    # ── Build AsynchronousLoader using their API ──────────────
    # AsynchronousLoader expects: dataset, flags, (optional) sampler
    # We pass our instrumented dataset as a drop-in replacement
    # Their loader will call dataset[idx] which invokes InstrumentedPytTrain.__getitem__
    from data_loading.pytorch_loader import PytVal

    val_dataset   = PytVal(x_val, y_val)

    # AsynchronousLoader signature (from their code):
    # AsynchronousLoader(dataset, batch_size, sampler, num_workers, ...)
    # Use their get_data_loaders if available, otherwise construct directly
    try:
        from data_loading.Asynchrnous_dataloader import AsynchronousLoader
        train_loader_raw = AsynchronousLoader(
            dataset=train_dataset_inst,
            device=device,
            shards=1,
            rank=0,
            slow_processed_queue=slow_processed_queue,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
    except TypeError:
        # Fallback: check their actual constructor signature
        import inspect
        sig = inspect.signature(AsynchronousLoader.__init__)
        print(f"  AsynchronousLoader signature: {sig}")
        raise

    # Wrap with our instrumented layer
    batch_extra_log = []
    train_loader    = InstrumentedAsyncLoader(train_loader_raw, batch_extra_log)

    # Validation uses standard PyTorch DataLoader (no async needed for val)
    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ── Queue Monitor (background thread) ────────────────────
    monitor = QueueMonitor(
        fast_queue=train_loader_raw.queue,
        slow_processed_queue=train_loader_raw.slow_processed_queue,
        log_path=queue_monitor_path,
        epoch_ref=epoch_ref,
        poll_interval_ms=args.queue_poll_ms,
    )

    # ── Model ─────────────────────────────────────────────────
    model     = Unet3D(in_channels=1, n_class=3, normalization="instancenorm", activation="relu").to(device)
    loss_fn   = DiceCELoss(to_onehot_y=True, use_softmax=True, layout=flags.layout, include_background=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.8, momentum=0.9, weight_decay=0.0)
    scaler    = GradScaler()

    # ── CSV header ────────────────────────────────────────────
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
        # Minato-exclusive
        "Slow_Sample_Ratio", "Mean_Batch_Assembly_Ms",
    ]

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(CSV_FIELDS)

    # ── Training loop ─────────────────────────────────────────
    monitor.start()
    try:
        for epoch in range(1, args.epochs + 1):
            print(f"\n[{time.strftime('%H:%M:%S')}] 🚀 Epoch {epoch}/{args.epochs}")

            # --- Train ---
            torch.cuda.reset_peak_memory_stats(device)
            batch_extra_log.clear()

            train_m = run_epoch(
                train_loader, model, loss_fn, optimizer, scaler,
                device, is_train=True, epoch=epoch, amp_enabled=amp_enabled,
                batch_extra_log_ref=batch_extra_log,
                epoch_ref=epoch_ref,
            )
            train_dataset_inst.flush()  # flush any buffered sample_trace records

            print(f"  Train | Loss: {train_m['Loss']:.4f} | "
                  f"Throughput: {train_m['Throughput_Img_Sec']:.2f} samp/s | "
                  f"GPU Starvation: {train_m['Total_GPU_Starvation_Sec']:.2f}s | "
                  f"GPU Busy: {train_m['GPU_Busy_Ratio_NoWarmup']*100:.1f}%")

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
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
                    train_m["Slow_Sample_Ratio"], train_m["Mean_Batch_Assembly_Ms"],
                ])

            with open(log_path, "a") as f:
                for entry in train_m["epoch_batch_logs"]:
                    f.write(json.dumps(entry) + "\n")

            # --- Validation ---
            torch.cuda.reset_peak_memory_stats(device)
            val_m = run_epoch(
                val_loader, model, loss_fn, optimizer, scaler,
                device, is_train=False, epoch=epoch, amp_enabled=amp_enabled,
                batch_extra_log_ref=[],
                epoch_ref=epoch_ref,
            )
            print(f"  Val   | Loss: {val_m['Loss']:.4f} | "
                  f"Throughput: {val_m['Throughput_Img_Sec']:.2f} samp/s")

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
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
                    None, None,   # Minato fields N/A for val
                ])

            with open(log_path, "a") as f:
                for entry in val_m["epoch_batch_logs"]:
                    f.write(json.dumps(entry) + "\n")

    finally:
        monitor.stop()
        train_dataset_inst.flush()

    print(f"\n✅ Done! Results in: {out_dir}")
    print(f"   epoch_result.csv   : {csv_path}")
    print(f"   batch_logs.jsonl   : {log_path}")
    print(f"   queue_monitor.jsonl: {queue_monitor_path}")
    print(f"   sample_trace.jsonl : {sample_trace_path}")
    print(f"   system_info.json   : {out_dir / 'system_info.json'}")


if __name__ == "__main__":
    main()
