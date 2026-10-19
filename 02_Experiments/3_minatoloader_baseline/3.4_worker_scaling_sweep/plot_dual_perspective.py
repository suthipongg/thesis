import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

SCRIPT_DIR = Path(__file__).resolve().parent
CONTINUOUS_CSV = SCRIPT_DIR / "results/worker_sweep_summary.csv"
WITH_VAL_CSV = SCRIPT_DIR / "results_with_val/worker_sweep_with_val_summary.csv"
C1_CSV = SCRIPT_DIR / "../3.3_adaptive_scheduler/results/continuous_C1/epoch_result.csv"
C2_CSV = SCRIPT_DIR / "../3.3_adaptive_scheduler/results/continuous_C2/epoch_result.csv"
OUT_DIR = SCRIPT_DIR / "results/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_dual_dashboard():
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), dpi=300)

    # ----------------------------------------------------
    # Subplot 1: Continuous Streaming Scaling (Mode B)
    # ----------------------------------------------------
    ax1 = axes[0]
    if CONTINUOUS_CSV.exists():
        df_cont = pd.read_csv(CONTINUOUS_CSV)
        df_cont_steady = df_cont[df_cont['Epoch'] >= 2].groupby(['System', 'Workers'])['Throughput_Img_Sec'].mean().reset_index()

        pt_data = df_cont_steady[df_cont_steady['System'] == 'pytorch'].sort_values('Workers')
        mi_data = df_cont_steady[df_cont_steady['System'] == 'minato'].sort_values('Workers')

        ax1.plot(pt_data['Workers'], pt_data['Throughput_Img_Sec'], marker='o', linewidth=2.5,
                 color='#1f77b4', label='PyTorch DataLoader')
        ax1.plot(mi_data['Workers'], mi_data['Throughput_Img_Sec'], marker='s', linewidth=2.5,
                 color='#2ca02c', label='MinatoLoader (Continuous)')

        if C1_CSV.exists():
            df_c1 = pd.read_csv(C1_CSV)
            c1_tput = df_c1[df_c1['Epoch'] >= 2]['Throughput_Img_Sec'].mean()
            ax1.scatter([4], [c1_tput], color='#ff7f0e', s=130, zorder=5, marker='^',
                        label=f'Minato + C1 Scheduler ({c1_tput:.1f} img/s)')
        if C2_CSV.exists():
            df_c2 = pd.read_csv(C2_CSV)
            c2_tput = df_c2[df_c2['Epoch'] >= 2]['Throughput_Img_Sec'].mean()
            ax1.scatter([4], [c2_tput], color='#9467bd', s=130, zorder=5, marker='d',
                        label=f'Minato + C2 Adaptive ({c2_tput:.1f} img/s)')

        ax1.axvline(x=4, color='gray', linestyle='--', alpha=0.7, label='Hardware P-Core Limit (4 Cores)')
        ax1.set_title("Mode B: Continuous Streaming Throughput\n(Steady State / No Inter-Epoch Pause)", fontsize=13, fontweight='bold', pad=12)
        ax1.set_xlabel("Number of Workers", fontweight='bold')
        ax1.set_ylabel("Throughput (images / sec)", fontweight='bold')
        ax1.set_xticks(range(1, 9))
        ax1.legend(loc='upper left', frameon=True, fontsize=10)
        ax1.grid(True, alpha=0.3)

    # ----------------------------------------------------
    # Subplot 2: With-Validation Scaling & Tipping Point (Mode A)
    # ----------------------------------------------------
    ax2 = axes[1]
    if WITH_VAL_CSV.exists():
        df_val = pd.read_csv(WITH_VAL_CSV)
        df_val_e2 = df_val[df_val['Epoch'] >= 2].groupby('Workers')['Throughput_Img_Sec'].mean().reset_index().sort_values('Workers')
        
        ax2.plot(df_val_e2['Workers'], df_val_e2['Throughput_Img_Sec'], marker='D', linewidth=2.5,
                 color='#d62728', label='Minato (With Validation Pause)')
        
        # Shade P-Core vs E-Core zones
        ax2.axvspan(0.8, 4.2, color='#2ca02c', alpha=0.08, label='P-Core Safe Zone (W=1..4)')
        ax2.axvspan(4.2, 8.2, color='#d62728', alpha=0.08, label='E-Core & Contention Zone (W=5..8)')
        ax2.axvline(x=4.2, color='gray', linestyle='--', alpha=0.7)

        # Highlight values
        for _, row in df_val_e2.iterrows():
            w = int(row['Workers'])
            t = row['Throughput_Img_Sec']
            y_offset = 6 if w <= 4 else -14
            ax2.annotate(f"{t:.1f}", (w, t), textcoords="offset points", xytext=(0, y_offset),
                         ha='center', fontweight='bold', fontsize=9.5)

        ax2.set_title("Mode A: Throughput With Validation Pause\n(Pre-buffering Plateau vs E-Core Cliff)", fontsize=13, fontweight='bold', pad=12)
        ax2.set_xlabel("Number of Workers", fontweight='bold')
        ax2.set_ylabel("Throughput (images / sec)", fontweight='bold')
        ax2.set_xticks(range(1, 9))
        ax2.set_ylim(18, 33)
        ax2.legend(loc='lower left', frameon=True, fontsize=10)
        ax2.grid(True, alpha=0.3)

    # ----------------------------------------------------
    # Subplot 3: Lock Contention / Wait Latency (Mode A)
    # ----------------------------------------------------
    ax3 = axes[2]
    if WITH_VAL_CSV.exists():
        df_wait = df_val[df_val['Epoch'] >= 2].groupby('Workers')['Avg_Wait_Ms'].mean().reset_index().sort_values('Workers')
        
        colors = ['#2ca02c' if w <= 4 else '#d62728' for w in df_wait['Workers']]
        bars = ax3.bar(df_wait['Workers'], df_wait['Avg_Wait_Ms'], color=colors, alpha=0.85, width=0.6)
        
        ax3.axhline(y=10, color='gray', linestyle=':', alpha=0.7, label='Optimal Latency Threshold (<10ms)')
        ax3.set_title("Root Cause: Dataloader Wait Latency (Mode A)\n(Proving IPC Multi-Processing Lock Contention)", fontsize=13, fontweight='bold', pad=12)
        ax3.set_xlabel("Number of Workers", fontweight='bold')
        ax3.set_ylabel("Average Wait Latency per Batch (ms)", fontweight='bold')
        ax3.set_xticks(range(1, 9))
        
        for bar in bars:
            h = bar.get_height()
            ax3.annotate(f'{h:.1f}ms', xy=(bar.get_x() + bar.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                         fontweight='bold', fontsize=9.5)

        ax3.legend(loc='upper left', frameon=True, fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = OUT_DIR / "dual_perspective_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Dual-Perspective Dashboard successfully saved to: {out_path}")

if __name__ == "__main__":
    generate_dual_dashboard()
