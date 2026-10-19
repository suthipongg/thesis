#!/usr/bin/env python3
"""
Plots the Worker Scaling Curves (Workers 1-8) for Thesis
=========================================================
Highlights the 4 P-Core hardware inflection point and performance degradation.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

SCRIPT_DIR = Path(__file__).parent
CSV_PATH = SCRIPT_DIR / "results" / "worker_sweep_summary.csv"
OUT_DIR = SCRIPT_DIR / "results" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} does not exist. Run run_worker_sweep.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    # Use steady-state epochs (e.g. Epoch >= 2 if multi-epoch, to remove cold-start artifacts)
    if df['Epoch'].max() > 1:
        steady_df = df[df['Epoch'] >= 2].copy()
    else:
        steady_df = df.copy()

    # Aggregate by System and Workers
    agg = steady_df.groupby(['System', 'Workers']).agg({
        'Throughput_Img_Sec': ['mean', 'std'],
        'GPU_Busy_Ratio': 'mean',
        'Total_Starvation_Sec': 'mean',
        'Avg_Wait_Ms': 'mean',
        'RAM_GB': 'max',
        'CPU_Percent': 'mean'
    }).reset_index()

    # Flatten multi-index
    agg.columns = ['System', 'Workers', 'Throughput_Mean', 'Throughput_Std', 'GPU_Busy', 'Starvation_Sec', 'Avg_Wait_Ms', 'RAM_GB', 'CPU_Percent']
    agg['System_Label'] = agg['System'].map({'pytorch': 'PyTorch DataLoader', 'minato': 'MinatoLoader'})

    palette = {'PyTorch DataLoader': '#3498db', 'MinatoLoader': '#e67e22'}

    # -------------------------------------------------------------
    # 1. Primary Thesis Plot: Throughput Scaling Curve (Workers 1-8)
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 7))
    ax = sns.lineplot(
        data=agg, x='Workers', y='Throughput_Mean', hue='System_Label',
        marker='o', markersize=10, linewidth=3.5, palette=palette
    )
    plt.title('Worker Scaling Curve: Throughput under Heavy 3D Augmentations\n(Intel Core i5-12450HX: 4 P-Cores + 4 E-Cores)', fontsize=15, pad=15)
    plt.xlabel('Number of DataLoader Workers', fontsize=13)
    plt.ylabel('Throughput (Images / Second)', fontsize=13)
    plt.xticks(range(1, 9))

    # Add vertical reference line for 4 P-Cores
    plt.axvline(x=4, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8)
    plt.text(4.05, agg['Throughput_Mean'].min() * 1.05, '4 P-Cores Limit\n(Sweet Spot)', color='#c0392b', fontweight='bold', fontsize=11)

    # Shaded degradation area (Workers > 4)
    plt.axvspan(4, 8, color='#f1948a', alpha=0.15, label='E-Core & Over-subscription Zone')

    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    out_plot1 = OUT_DIR / "worker_throughput_scaling.png"
    plt.savefig(out_plot1, dpi=300)
    plt.close()
    print(f"✅ Saved: {out_plot1}")

    # -------------------------------------------------------------
    # 2. Comprehensive 2x2 Dashboard
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # [0,0] Throughput
    sns.lineplot(data=agg, x='Workers', y='Throughput_Mean', hue='System_Label', marker='o', ax=axes[0, 0], palette=palette, linewidth=2.5)
    axes[0, 0].set_title('Throughput (Images / Sec)')
    axes[0, 0].set_ylabel('Images / Sec')
    axes[0, 0].axvline(x=4, color='red', linestyle='--', alpha=0.6)
    axes[0, 0].set_xticks(range(1, 9))

    # [0,1] GPU Starvation (Seconds)
    sns.lineplot(data=agg, x='Workers', y='Starvation_Sec', hue='System_Label', marker='s', ax=axes[0, 1], palette=palette, linewidth=2.5)
    axes[0, 1].set_title('GPU Starvation Time per Epoch (Seconds)')
    axes[0, 1].set_ylabel('Wait Time (s)')
    axes[0, 1].axvline(x=4, color='red', linestyle='--', alpha=0.6)
    axes[0, 1].set_xticks(range(1, 9))

    # [1,0] Batch Wait Latency (ms)
    sns.lineplot(data=agg, x='Workers', y='Avg_Wait_Ms', hue='System_Label', marker='^', ax=axes[1, 0], palette=palette, linewidth=2.5)
    axes[1, 0].set_title('Average DataLoader Wait Latency (ms/batch)')
    axes[1, 0].set_ylabel('Latency (ms)')
    axes[1, 0].axvline(x=4, color='red', linestyle='--', alpha=0.6)
    axes[1, 0].set_xticks(range(1, 9))

    # [1,1] Memory Usage (RAM GB)
    sns.lineplot(data=agg, x='Workers', y='RAM_GB', hue='System_Label', marker='D', ax=axes[1, 1], palette=palette, linewidth=2.5)
    axes[1, 1].set_title('Peak System RAM Consumption (GB)')
    axes[1, 1].set_ylabel('RAM (GB)')
    axes[1, 1].axvline(x=4, color='red', linestyle='--', alpha=0.6)
    axes[1, 1].set_xticks(range(1, 9))

    for ax in axes.flat:
        ax.get_legend().remove()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02), frameon=True)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out_plot2 = OUT_DIR / "worker_sweep_dashboard.png"
    plt.savefig(out_plot2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out_plot2}")

if __name__ == "__main__":
    main()
