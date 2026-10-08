import os
import glob
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def find_latest_run(base_dir):
    """Find the most recently created directory in the base_dir"""
    all_runs = glob.glob(os.path.join(base_dir, "*"))
    runs = [r for r in all_runs if os.path.isdir(r)]
    if not runs:
        return None
    latest_run = max(runs, key=os.path.getmtime)
    return latest_run

def plot_epoch_metrics(pytorch_csv, minato_csv, out_dir):
    """Plot epoch-level comparisons: Throughput, GPU Busy, etc."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    df_pyt = pd.read_csv(pytorch_csv) if pytorch_csv else None
    df_min = pd.read_csv(minato_csv) if minato_csv else None

    if df_pyt is None and df_min is None:
        return

    # Filter only Training rows
    if df_pyt is not None: df_pyt = df_pyt[df_pyt["Phase"] == "Train"]
    if df_min is not None: df_min = df_min[df_min["Phase"] == "Train"]

    # 1. Throughput
    ax = axes[0, 0]
    if df_pyt is not None:
        ax.plot(df_pyt["Epoch"], df_pyt["Throughput_Img_Sec"], marker='o', label='PyTorch Baseline', linewidth=2)
    if df_min is not None:
        ax.plot(df_min["Epoch"], df_min["Throughput_Img_Sec"], marker='s', label='MinatoLoader', linewidth=2)
    ax.set_title("Training Throughput (Images / Sec)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Images / Sec")
    ax.set_ylim(bottom=0)
    ax.legend()

    # 2. GPU Busy Ratio
    ax = axes[0, 1]
    if df_pyt is not None:
        ax.plot(df_pyt["Epoch"], df_pyt["GPU_Busy_Ratio_All"] * 100, marker='o', label='PyTorch Baseline', linewidth=2)
    if df_min is not None:
        ax.plot(df_min["Epoch"], df_min["GPU_Busy_Ratio_All"] * 100, marker='s', label='MinatoLoader', linewidth=2)
    ax.set_title("GPU Utilization (%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("GPU Busy %")
    ax.set_ylim(0, 105)
    ax.legend()

    # 3. Data Loading Wait (Starvation)
    ax = axes[1, 0]
    if df_pyt is not None:
        ax.plot(df_pyt["Epoch"], df_pyt["Total_GPU_Starvation_Sec"], marker='o', label='PyTorch Baseline', linewidth=2)
    if df_min is not None:
        ax.plot(df_min["Epoch"], df_min["Total_GPU_Starvation_Sec"], marker='s', label='MinatoLoader', linewidth=2)
    ax.set_title("Total GPU Starvation Time per Epoch (Seconds)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.set_ylim(bottom=0)
    ax.legend()

    # 4. RAM / VRAM
    ax = axes[1, 1]
    if df_pyt is not None:
        ax.plot(df_pyt["Epoch"], df_pyt["RAM_GB"], marker='o', linestyle='-', color='C0', label='PyTorch (RAM)')
        ax.plot(df_pyt["Epoch"], df_pyt["Peak_VRAM"], marker='^', linestyle='--', color='C0', label='PyTorch (VRAM)')
    if df_min is not None:
        ax.plot(df_min["Epoch"], df_min["RAM_GB"], marker='s', linestyle='-', color='C1', label='MinatoLoader (RAM)')
        ax.plot(df_min["Epoch"], df_min["Peak_VRAM"], marker='^', linestyle='--', color='C1', label='MinatoLoader (VRAM)')
    ax.set_title("Memory Consumption")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gigabytes (GB)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "epoch_metrics_comparison.png"), dpi=300)
    plt.close()


def plot_gpu_timeline(jsonl_path, title, out_path, num_batches=15):
    """Draw a Gantt-chart like timeline of Data Load Wait vs GPU Compute for the first N batches."""
    if not jsonl_path or not os.path.exists(jsonl_path):
        return

    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            if rec["phase"] == "train" and rec["epoch"] == 1:
                records.append(rec)
            if len(records) >= num_batches:
                break
                
    if not records:
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    
    y_pos = 0
    
    # We want to plot timeline from T=0 relative to the first batch
    t0 = records[0]["start_ts"]
    
    for idx, rec in enumerate(records):
        start_t = rec["start_ts"] - t0
        
        # In batch_logs, dataloader_wait_ms is the time spent waiting BEFORE GPU computation starts
        wait_s = rec["dataloader_wait_ms"] / 1000.0
        compute_s = rec["gpu_compute_ms"] / 1000.0 if rec["gpu_compute_ms"] else 0.0
        
        # Plot DataLoader wait
        ax.barh(y_pos, wait_s, left=start_t, height=0.5, color='salmon', alpha=0.8, 
                label='DataLoader Wait' if idx == 0 else "")
        
        # Plot GPU Compute
        ax.barh(y_pos, compute_s, left=start_t + wait_s, height=0.5, color='mediumseagreen', alpha=0.8,
                label='GPU Compute' if idx == 0 else "")
                
        # Annotate Batch ID
        ax.text(start_t + wait_s/2, y_pos, f"Wait", ha='center', va='center', fontsize=8, color='black')
        ax.text(start_t + wait_s + compute_s/2, y_pos, f"B{idx}", ha='center', va='center', fontsize=8, color='black')
        
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Detailed GPU Timeline (First {num_batches} Batches) - {title}")
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_minato_queues(queue_jsonl, sample_jsonl, out_dir):
    """Plot MinatoLoader exclusive queue behavior and fast/slow paths."""
    if not queue_jsonl or not os.path.exists(queue_jsonl):
        return

    df_q = pd.read_json(queue_jsonl, lines=True)
    if len(df_q) == 0:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Queue Sizes Timeline
    ax = axes[0]
    # Filter for first 120 seconds or so to see the pattern
    df_q_sub = df_q[df_q["elapsed_ms"] < 120000]
    time_s = df_q_sub["elapsed_ms"] / 1000.0
    
    ax.plot(time_s, df_q_sub["fast_q_size"], label='Fast Queue (Pre-processed)', color='blue')
    ax.plot(time_s, df_q_sub["slow_processed_q_size"], label='Slow Processed Queue', color='orange')
    ax.set_title("MinatoLoader Queue Dynamics (First 120s)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Items in Queue")
    ax.legend()
    
    # Fast vs Slow Distribution
    ax = axes[1]
    if sample_jsonl and os.path.exists(sample_jsonl):
        df_s = pd.read_json(sample_jsonl, lines=True)
        if len(df_s) > 0:
            counts = df_s["path"].value_counts()
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], startangle=90)
            ax.set_title("Augmentation Routing (Fast vs Slow Path)")
            
            # Print latency distribution
            print("\nPreprocess Latency (ms):")
            print(df_s.groupby("path")["preprocess_ms"].describe())
            
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "minato_queue_analysis.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dir", type=str, help="Specific PyTorch result dir (default: latest)")
    parser.add_argument("--minato_dir", type=str, help="Specific Minato result dir (default: latest)")
    parser.add_argument("--out_dir", type=str, default="../results/plots", help="Output dir for plots")
    args = parser.parse_args()

    base_results = Path(__file__).parent.parent / "results"
    
    pytorch_dir = args.pytorch_dir or find_latest_run(base_results / "pytorch")
    minato_dir  = args.minato_dir  or find_latest_run(base_results / "minato")
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"PyTorch run: {pytorch_dir}")
    print(f"Minato run:  {minato_dir}")
    print(f"Plots dir:   {out_dir}")

    # 1. Epoch Metrics Comparison
    pytorch_csv = os.path.join(pytorch_dir, "epoch_result.csv") if pytorch_dir else None
    minato_csv  = os.path.join(minato_dir, "epoch_result.csv") if minato_dir else None
    plot_epoch_metrics(pytorch_csv, minato_csv, out_dir)

    # 2. Detailed Timelines
    if pytorch_dir:
        plot_gpu_timeline(os.path.join(pytorch_dir, "batch_logs.jsonl"), "PyTorch Baseline", os.path.join(out_dir, "timeline_pytorch.png"))
    
    if minato_dir:
        plot_gpu_timeline(os.path.join(minato_dir, "batch_logs.jsonl"), "MinatoLoader", os.path.join(out_dir, "timeline_minato.png"))
        
        # 3. Minato Queues
        plot_minato_queues(
            os.path.join(minato_dir, "queue_monitor.jsonl"),
            os.path.join(minato_dir, "sample_trace.jsonl"),
            out_dir
        )

    print("✅ All plots generated.")


if __name__ == "__main__":
    main()
