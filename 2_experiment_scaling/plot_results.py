import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})

plt.style.use("dark_background")

FIRST_N_BATCHES = 10
# ============================================================
# LOAD
# ============================================================

def load_results(result_dir):

    csv_path = os.path.join(result_dir, "epoch_result.csv")
    json_path = os.path.join(result_dir, "batch_logs.jsonl")

    metrics = pd.read_csv(csv_path)

    batches = []
    with open(json_path) as f:
        for line in f:
            batches.append(json.loads(line))

    batches = pd.DataFrame(batches)

    return metrics, batches


# ============================================================
# PARSE FOLDER NAME
# 1781520069_e5_bs256_w2_tb4_vb2_dry
# ============================================================

def parse_run_name(run_dir):

    name = Path(run_dir).name

    m = re.match(
        r'(?P<ts>\d+)_e(?P<epochs>\d+)_bs(?P<bs>\d+)_w(?P<w>\d+)_tb(?P<tb>\d+)_vb(?P<vb>\d+)(?:_(?P<mode>.+))?$',
        name
    )

    if not m:
        return {}

    return {
        "epochs": int(m.group("epochs")),
        "batch_size": int(m.group("bs")),
        "workers": int(m.group("w")),
        "train_buffer": int(m.group("tb")),
        "val_buffer": int(m.group("vb")),
        "mode": m.group("mode")
    }


# ============================================================
# OUTPUT FOLDER
# ============================================================

def ensure_plot_dir(result_dir):

    save_dir = os.path.join(result_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    return save_dir


# ============================================================
# 1. PIPELINE COMPOSITION (%)
# ============================================================

def plot_pipeline_composition(metrics, save_dir):
    """
    Composition ใช้ Epoch_Time_Sec เป็นฐาน (total wall time)
    - Total_GPU_Compute_Sec  = GPU compute ทั้ง epoch (total)
    - Total_GPU_Starvation_Sec = GPU starvation ทั้ง epoch (total)
    - หมายเหตุ: Disk_Read_Sec / Transform_Sec ใน CSV คือ mean ต่อ batch
      ไม่ใช่ total จึงไม่นำมาบวกโดยตรง
      ใช้ epoch_time - compute - starvation แทน (= prefetch overhead)
    """

    train = metrics[
        metrics["Phase"].str.lower() == "train"
    ].copy()

    epoch_time = train["Epoch_Time_Sec"]
    compute    = train["Total_GPU_Compute_Sec"]
    starvation = train["Total_GPU_Starvation_Sec"]

    # overhead = prefetch / dataloader / pipeline gap
    overhead = (epoch_time - compute - starvation).clip(lower=0)
    total    = epoch_time  # = compute + starvation + overhead

    compute_pct    = compute    / total * 100
    starvation_pct = starvation / total * 100
    overhead_pct   = overhead   / total * 100

    epochs = train["Epoch"].tolist()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(epochs, compute_pct, label="GPU Compute")
    ax.bar(epochs, starvation_pct, bottom=compute_pct,
           label="GPU Starvation")
    ax.bar(epochs, overhead_pct, bottom=compute_pct + starvation_pct,
           label="Prefetch / Overhead")

    ax.set_title("Pipeline Time Composition (% of Wall Time)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Percent")
    ax.set_xticks(epochs)
    ax.set_ylim(0, 100)
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "pipeline_composition.png"),
        dpi=300
    )
    plt.close()


# ============================================================
# 2. GPU UTILIZATION
# ============================================================

def plot_gpu_util(metrics, save_dir):

    train = metrics[
        metrics["Phase"].str.lower() == "train"
    ]

    epochs = train["Epoch"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        epochs,
        train["GPU_Busy_Ratio_NoWarmup"] * 100,
        marker="o",
        linewidth=3
    )

    ax.set_ylabel("GPU Busy (%)")
    ax.set_xlabel("Epoch")
    ax.set_title("GPU Utilization")
    ax.set_xticks(epochs)
    ax.set_ylim(0, 100)
    ax.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "gpu_utilization.png"),
        dpi=300
    )
    plt.close()


# ============================================================
# 3. THROUGHPUT
# ============================================================

def plot_throughput(metrics, save_dir):

    train = metrics[
        metrics["Phase"].str.lower() == "train"
    ]

    epochs = train["Epoch"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        epochs,
        train["Throughput_Img_Sec"],
        marker="o",
        linewidth=3
    )

    ax.set_ylabel("Images / Sec")
    ax.set_xlabel("Epoch")
    ax.set_title("Training Throughput")
    ax.set_xticks(epochs)
    ax.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "throughput.png"),
        dpi=300
    )
    plt.close()


# ============================================================
# 4. GPU STARVATION
# ============================================================

def plot_gpu_starvation(metrics, save_dir):

    train = metrics[
        metrics["Phase"].str.lower() == "train"
    ]

    epochs = train["Epoch"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        epochs,
        train["Total_GPU_Starvation_Sec"],
        marker="o",
        linewidth=3
    )

    ax.set_ylabel("Seconds")
    ax.set_xlabel("Epoch")
    ax.set_title("GPU Starvation Time")
    ax.set_xticks(epochs)
    ax.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "gpu_starvation.png"),
        dpi=300
    )
    plt.close()


# ============================================================
# PIPELINE TIMELINE  (kept for reference, no longer used in plot)
# ============================================================

def build_real_pipeline(df, workers):

    worker_free = [0.0] * workers

    gpu_free = 0.0

    timeline = []

    df = df.sort_values("batch_id")

    for _, row in df.iterrows():

        worker_id = np.argmin(worker_free)

        worker_start = worker_free[worker_id]

        disk_end = worker_start + row.disk_ms

        trans_end = disk_end + row.trans_ms

        worker_free[worker_id] = trans_end

        ready_time = trans_end

        gpu_start = max(
            gpu_free,
            ready_time
        )

        starvation = max(
            0,
            gpu_start - gpu_free
        )

        gpu_end = gpu_start + row.gpu_compute_ms

        gpu_free = gpu_end

        timeline.append({

            "batch": int(row.batch_id),

            "worker": worker_id,

            "disk_start": worker_start,
            "disk_dur": row.disk_ms,

            "trans_start": disk_end,
            "trans_dur": row.trans_ms,

            "ready_time": ready_time,

            "gpu_start": gpu_start,
            "gpu_dur": row.gpu_compute_ms,

            "gpu_starvation": starvation
        })

    return timeline

def compute_epoch_stats(train, metric, epoch):
    """
    หมายเหตุ:
    - batch_id=0 มี gpu_starvation_ms=null (warmup) → ต้อง dropna() ก่อน idxmax()
    - ep_metric ต้อง filter เฉพาะ train phase ด้วย
    """

    ep = train[train.epoch == epoch]

    # filter train phase เท่านั้น (ไม่ให้ validation เข้ามาทำให้ mean ผิด)
    ep_metric = metric[
        (metric.Epoch == epoch) &
        (metric.Phase.str.lower() == "train")
    ]

    gpu_busy   = ep_metric["GPU_Busy_Ratio_NoWarmup"].mean() * 100
    gpu_idle   = 100 - gpu_busy
    throughput = ep_metric["Throughput_Img_Sec"].mean()

    # batch_id=0 → null starvation (warmup batch) → ข้ามออก
    ep_valid         = ep.dropna(subset=["gpu_starvation_ms"])
    starvation_total = ep_valid["gpu_starvation_ms"].sum()

    if ep_valid.empty:
        worst_batch_id   = -1
        worst_starvation = 0.0
    else:
        worst_row        = ep_valid.loc[ep_valid["gpu_starvation_ms"].idxmax()]
        worst_batch_id   = int(worst_row["batch_id"])
        worst_starvation = float(worst_row["gpu_starvation_ms"])

    return {
        "gpu_busy":         gpu_busy,
        "gpu_idle":         gpu_idle,
        "throughput":       throughput,
        "starvation":       starvation_total,
        "worst_batch":      worst_batch_id,
        "worst_starvation": worst_starvation,
    }

# ============================================================
# 5. GANTT TIMELINE
# ============================================================

def build_timeline_from_ts(df):
    """
    Build timeline items using real timestamps.
    If precise timestamps and worker IDs are present, use them directly.
    Otherwise, fall back to backward estimation from start_ts.
    """
    df = df.sort_values("batch_id")
    use_precise = "disk_start_rel_ms" in df.columns and "worker_id" in df.columns

    if use_precise:
        base_ts_ms = df.iloc[0]["gpu_start_rel_ms"]
    else:
        base_ts = df.iloc[0]["start_ts"]

    timeline = []
    for _, row in df.iterrows():
        if use_precise:
            gpu_start_ms = row["gpu_start_rel_ms"] - base_ts_ms
            gpu_dur      = float(row["gpu_compute_ms"])
            starvation   = float(row["gpu_starvation_ms"]) if pd.notna(row["gpu_starvation_ms"]) else 0.0

            disk_start_ms  = row["disk_start_rel_ms"] - base_ts_ms
            disk_dur       = float(row["disk_ms"])
            trans_start_ms = disk_start_ms + disk_dur
            trans_dur      = float(row["trans_ms"])
            data_ready_ms  = trans_start_ms + trans_dur
            worker_id      = int(row["worker_id"])
        else:
            gpu_start_ms   = (row["start_ts"] - base_ts) * 1000.0
            gpu_dur        = float(row["gpu_compute_ms"])
            starvation     = float(row["gpu_starvation_ms"]) if pd.notna(row["gpu_starvation_ms"]) else 0.0

            data_ready_ms  = gpu_start_ms - starvation
            trans_start_ms = data_ready_ms - float(row["trans_ms"])
            disk_start_ms  = trans_start_ms - float(row["disk_ms"])
            disk_dur       = float(row["disk_ms"])
            trans_dur      = float(row["trans_ms"])
            worker_id      = -1

        timeline.append({
            "batch":          int(row["batch_id"]),
            "disk_start":     disk_start_ms,
            "disk_dur":       disk_dur,
            "trans_start":    trans_start_ms,
            "trans_dur":      trans_dur,
            "ready_time":     data_ready_ms,
            "gpu_start":      gpu_start_ms,
            "gpu_dur":        gpu_dur,
            "gpu_starvation": starvation,
            "worker_id":      worker_id,
        })
    return timeline


def plot_global_timeline(
    batches,
    metric,
    save_dir,
    workers,
    batch_size
):
    train        = batches[batches.phase.str.lower() == "train"]
    train_metric = metric[metric.Phase.str.lower() == "train"]
    epochs       = sorted(train.epoch.unique())

    # ── Y-layout per epoch ─────────────────────────────────
    # [gpu_y]                 GPU lane     (LANE_H)
    # [gpu_y + LANE_H + GAP]  Data lane    (DATA_H)
    # Total per epoch (excl pad): LANE_H + GAP + DATA_H
    LANE_H       = 3
    SUB_LANE_GAP = 1
    GAP          = 4     # gap between GPU lane and Data lane
    EPOCH_PAD    = 10    # padding between epochs

    colors = {
        "disk":  "#d32f2f",
        "trans": "#d89257",
        "gpu":   "#4c6ef5",
        "wait":  "#6c757d",
        "warmup": "#212121",
    }

    fig, ax = plt.subplots(
        figsize=(28, max(10, len(epochs) * 4))
    )

    all_head_ends   = []
    all_tail_starts = []
    all_tail_ends   = []

    y_cursor = 0

    for epoch in epochs:

        ep   = train[train.epoch == epoch].sort_values("batch_id")
        head = ep.head(FIRST_N_BATCHES)
        tail = ep.tail(FIRST_N_BATCHES)
        has_gap = ep.shape[0] > 2 * FIRST_N_BATCHES

        # ── Build timelines from real timestamps ──
        head_timeline = build_timeline_from_ts(head)

        if has_gap:
            tail_timeline = build_timeline_from_ts(tail)   # also normalised to 0

            # Extents of head
            head_end_t = max(
                max(i["gpu_start"] + i["gpu_dur"]    for i in head_timeline),
                max(i["trans_start"] + i["trans_dur"] for i in head_timeline),
            )

            # Earliest point in tail
            tail_min_t = min(
                min(i["disk_start"] for i in tail_timeline),
                min(i["gpu_start"]  for i in tail_timeline),
            )

            # Place tail so tail_min lands at head_end + visual separator
            VISUAL_SEP   = 500.0   # ms gap between sections
            shift_offset = tail_min_t - (head_end_t + VISUAL_SEP)

            for item in tail_timeline:
                item["disk_start"]  -= shift_offset
                item["trans_start"] -= shift_offset
                item["ready_time"]  -= shift_offset
                item["gpu_start"]   -= shift_offset

            timeline        = head_timeline + tail_timeline
            tail_start_plot = head_end_t + VISUAL_SEP
            tail_end_plot   = max(i["gpu_start"] + i["gpu_dur"] for i in tail_timeline)

            all_head_ends.append(head_end_t)
            all_tail_starts.append(tail_start_plot)
            all_tail_ends.append(tail_end_plot)

        else:
            timeline = head_timeline
            has_gap  = False

        # ── Assign Data Lanes ──
        has_precise_workers = any(item.get("worker_id", -1) >= 0 for item in timeline)

        if has_precise_workers:
            # Directly use worker_id as the lane index
            for item in timeline:
                item["lane_idx"] = item.get("worker_id", 0)
            num_lanes = max(1, max(item["lane_idx"] for item in timeline) + 1)
        else:
            # Fall back to greedy completely dynamic lane assignment
            lanes_free_time = []
            for item in sorted(timeline, key=lambda x: x["disk_start"]):
                assigned = -1
                for i, free_time in enumerate(lanes_free_time):
                    if free_time <= item["disk_start"]:
                        assigned = i
                        lanes_free_time[i] = item["trans_start"] + item["trans_dur"]
                        break
                if assigned == -1:
                    assigned = len(lanes_free_time)
                    lanes_free_time.append(item["trans_start"] + item["trans_dur"])
                
                item["lane_idx"] = assigned
            num_lanes = max(1, len(lanes_free_time))

        DATA_H    = num_lanes * LANE_H + (num_lanes - 1) * SUB_LANE_GAP
        EPOCH_H   = LANE_H + GAP + DATA_H

        stats = compute_epoch_stats(train, train_metric, epoch)

        gpu_y  = y_cursor
        data_y = gpu_y + LANE_H + GAP

        # ── Epoch label ──────────────────────────────────────
        label_y = data_y + DATA_H + 2
        ax.text(
            0, label_y,
            f"Epoch {epoch}  [first+last {FIRST_N_BATCHES} batches]",
            fontsize=16, fontweight="bold"
        )

        # ── Lane labels ──────────────────────────────────────
        min_x = min(
            min(item["disk_start"] for item in timeline) if timeline else 0.0,
            min(item["gpu_start"] for item in timeline) if timeline else 0.0,
            0.0
        )
        label_x = min_x - 150.0

        ax.text(label_x, gpu_y  + LANE_H / 2, "GPU",
                fontsize=13, fontweight="bold", va="center", ha="right")
        
        if has_precise_workers:
            for lane_idx in range(num_lanes):
                w_y_center = data_y + lane_idx * (LANE_H + SUB_LANE_GAP) + LANE_H / 2
                label_text = "Main Thread" if workers == 0 else f"Worker {lane_idx}"
                ax.text(label_x, w_y_center, label_text,
                        fontsize=11, va="center", ha="right", color="#cccccc")
        else:
            ax.text(label_x, data_y + DATA_H / 2, "Data\n(all workers)",
                    fontsize=10, va="center", ha="right")

        # ── GPU lane ─────────────────────────────────────────
        for item in timeline:
            # starvation bar (GPU waiting for data)
            if item["gpu_starvation"] > 0:
                is_warmup = (item["batch"] == 0)
                color = colors["warmup"] if is_warmup else colors["wait"]
                ax.broken_barh(
                    [(item["gpu_start"] - item["gpu_starvation"],
                      item["gpu_starvation"])],
                    (gpu_y, LANE_H),
                    facecolors=color
                )
            # compute bar
            ax.broken_barh(
                [(item["gpu_start"], item["gpu_dur"])],
                (gpu_y, LANE_H),
                facecolors=colors["gpu"]
            )
            ax.text(
                item["gpu_start"] + item["gpu_dur"] / 2,
                gpu_y + LANE_H / 2,
                f"B{item['batch']}",
                fontsize=9, ha="center", va="center", color="white"
            )

        # ── Data lane (non-overlapping sub-lanes) ───
        for item in timeline:
            lane_idx = item["lane_idx"]
            wy = data_y + lane_idx * (LANE_H + SUB_LANE_GAP)

            # Disk bar
            ax.broken_barh(
                [(item["disk_start"],  item["disk_dur"])],
                (wy, LANE_H),
                facecolors=colors["disk"],  alpha=0.9
            )
            # Batch text in Disk bar
            if item["disk_dur"] > 50: # Only draw text if the bar is wide enough
                ax.text(
                    item["disk_start"] + item["disk_dur"] / 2,
                    wy + LANE_H / 2,
                    f"B{item['batch']}",
                    fontsize=8, ha="center", va="center", color="white", alpha=0.9
                )

            # Trans bar
            ax.broken_barh(
                [(item["trans_start"], item["trans_dur"])],
                (wy, LANE_H),
                facecolors=colors["trans"], alpha=0.9
            )
            # Batch text in Trans bar
            if item["trans_dur"] > 50: # Only draw text if the bar is wide enough
                ax.text(
                    item["trans_start"] + item["trans_dur"] / 2,
                    wy + LANE_H / 2,
                    f"B{item['batch']}",
                    fontsize=8, ha="center", va="center", color="white", alpha=0.9
                )

        # ── Break marker ─────────────────────────────────────
        if has_gap:
            gap_center = (head_end_t + tail_start_plot) / 2
            y_top      = data_y + DATA_H
            y_bot      = gpu_y

            # double dashed lines to signal axis discontinuity
            for xp in [gap_center - 25, gap_center + 25]:
                ax.plot([xp, xp], [y_bot, y_top],
                        color="#cccccc", linestyle="--", linewidth=1.5, alpha=0.8)

            skipped      = ep.shape[0] - len(head) - len(tail)
            gap_sec      = tail.iloc[0]["start_ts"] - head.iloc[-1]["start_ts"]
            tail_rel_sec = tail.iloc[0]["timestamp_relative"]

            ax.text(
                gap_center, (y_bot + y_top) / 2,
                f"···  {skipped} batches skipped\n"
                f"({gap_sec:.0f}s elapsed, tail starts @ {tail_rel_sec:.0f}s)  ···",
                color="#bbbbbb", fontsize=10, fontweight="bold",
                ha="center", va="center",
                bbox=dict(facecolor="#111111", edgecolor="#555555",
                          boxstyle="round,pad=0.4", linewidth=1)
            )

        # ── Stats panel (right side) ─────────────────────────
        if timeline:
            max_t  = max(i["gpu_start"] + i["gpu_dur"] for i in timeline)
            info_x = max_t * 1.01
        else:
            info_x = 0

        summary_text = (
            f"GPU Busy    : {stats['gpu_busy']:.1f}%\n"
            f"GPU Idle    : {stats['gpu_idle']:.1f}%\n"
            f"Throughput  : {stats['throughput']:.1f} img/s\n"
            f"Starvation  : {stats['starvation']:.0f} ms (total)\n\n"
            f"Worst Batch : B{stats['worst_batch']}\n"
            f"Max Stall   : {stats['worst_starvation']:.0f} ms"
        )
        ax.text(
            info_x, gpu_y + 1,
            summary_text,
            fontsize=11, va="top",
            bbox=dict(facecolor="black", alpha=0.7,
                      edgecolor="white", boxstyle="round,pad=0.5")
        )

        y_cursor += EPOCH_H + EPOCH_PAD

    # ── X-axis: head labels show ms from head-start (0),
    #            tail labels show ms from tail-start (0) ──────
    if all_head_ends:
        max_head = max(all_head_ends)
        min_tail = min(all_tail_starts)
        max_tail = max(all_tail_ends)
        step     = 500.0

        head_ticks  = list(np.arange(0, max_head + step, step))
        head_labels = [f"{t:.0f}" for t in head_ticks]

        tail_ticks  = list(np.arange(min_tail, max_tail + step, step))
        tail_labels = [f"{t - min_tail:.0f}" for t in tail_ticks]   # reset to 0

        ax.set_xticks(head_ticks + tail_ticks)
        ax.set_xticklabels(head_labels + tail_labels,
                           rotation=45, ha="right", fontsize=11)

        # Section annotations below x-axis
        ax.annotate(
            "← head (ms from batch-0)",
            xy=(max_head * 0.5, 0), xycoords=("data", "axes fraction"),
            fontsize=11, color="#aaaaaa", ha="center", va="top",
            xytext=(0, -32), textcoords="offset points"
        )
        ax.annotate(
            "tail (ms from last-batch-0) →",
            xy=((min_tail + max_tail) * 0.5, 0), xycoords=("data", "axes fraction"),
            fontsize=11, color="#aaaaaa", ha="center", va="top",
            xytext=(0, -32), textcoords="offset points"
        )

    # ── Legend ───────────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=colors["gpu"],  label="GPU Compute"),
        Patch(facecolor=colors["wait"], label="GPU Starvation (wait)"),
        Patch(facecolor=colors["warmup"], label="Pipeline Warmup (wait)"),
        Patch(facecolor=colors["disk"], label="Disk Read"),
        Patch(facecolor=colors["trans"],label="Transform"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=13, framealpha=0.8)

    ax.set_title(
        f"Data Pipeline Timeline\n"
        f"Workers={workers}  BatchSize={batch_size}  "
        f"[GPU from real timestamps · disk/transform derived]",
        fontsize=20
    )
    ax.set_xlabel(
        "Time (ms)  —  head & tail each start at 0",
        fontsize=14
    )
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "global_timeline.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

# ============================================================
# MASTER
# ============================================================

def generate_report(result_dir):

    print("Loading...")

    metrics, batches = load_results(
        result_dir
    )

    save_dir = ensure_plot_dir(
        result_dir
    )

    meta = parse_run_name(
        result_dir
    )

    print("Generating plots...")

    plot_pipeline_composition(
        metrics,
        save_dir
    )

    plot_gpu_util(
        metrics,
        save_dir
    )

    plot_throughput(
        metrics,
        save_dir
    )

    plot_gpu_starvation(
        metrics,
        save_dir
    )

    workers = meta.get("workers")
    batch_size = meta.get("batch_size")

    plot_global_timeline(
        batches,
        metrics,
        save_dir,
        workers=workers,
        batch_size=batch_size
    )

    print("Done.")
    print(save_dir)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    generate_report(
        "/home/mew/Desktop/mew/study/Master degree/thesis/2_experiment_scaling/thesis_results_real/1781672285_e5_bs256_w2_tb8_vb4_dry"
    )
    generate_report(
        "/home/mew/Desktop/mew/study/Master degree/thesis/2_experiment_scaling/thesis_results_real/1781672334_e5_bs256_w0_tb8_vb4_dry"
    )
    generate_report(
        "/home/mew/Desktop/mew/study/Master degree/thesis/2_experiment_scaling/thesis_results_real/1781672404_e5_bs256_w4_tb8_vb4_dry"
    )