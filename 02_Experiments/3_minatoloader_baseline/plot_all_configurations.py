#!/usr/bin/env python3
"""
Generates the overarching thesis comparison plots across all 6 configurations:
A1, A2, B1, B2 (from 3.2_heavy_comparison) and C1, C2 (from 3.3_adaptive_scheduler).
"""
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "results" / "plots" / "new_phase2_thesis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

configs = [
    (BASE_DIR / "3.2_heavy_comparison" / "results" / "heavy_out_A1", "PyTorch (4 Workers)"),
    (BASE_DIR / "3.2_heavy_comparison" / "results" / "heavy_out_A2", "PyTorch (8 Workers)"),
    (BASE_DIR / "3.2_heavy_comparison" / "results" / "heavy_out_B1", "Minato (4 Workers)"),
    (BASE_DIR / "3.2_heavy_comparison" / "results" / "heavy_out_B2", "Minato (8 Workers)"),
    (BASE_DIR / "3.3_adaptive_scheduler" / "results" / "heavy_out_C1", "Minato Scheduled (8W)"),
    (BASE_DIR / "3.3_adaptive_scheduler" / "results" / "heavy_out_C2", "Minato Scheduled+Adaptive (8W)")
]

def load_data(exp_dir, name):
    csv_file = exp_dir / "epoch_result.csv"
    if not csv_file.exists():
        print(f"Warning: {csv_file} not found")
        return None
    df = pd.read_csv(csv_file)
    df = df[df['Phase'] == 'Val'].copy()
    df['System'] = name
    return df

dfs = []
for p, name in configs:
    df = load_data(p, name)
    if df is not None:
        dfs.append(df)

if not dfs:
    print("No data found!")
    exit(1)

all_df = pd.concat(dfs, ignore_index=True)

# 1. Throughput
plt.figure(figsize=(11, 6))
sns.barplot(data=all_df, x='Epoch', y='Throughput_Img_Sec', hue='System')
plt.title('Throughput (Images/Second) under Heavy Augmentations')
plt.ylabel('Throughput (img/s)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(OUT_DIR / 'thesis_throughput.png', dpi=300)
plt.close()

# 2. Wait Time
plt.figure(figsize=(11, 6))
sns.barplot(data=all_df, x='Epoch', y='Total_Starvation_Sec', hue='System')
plt.title('Total GPU Wait Time (Seconds per Epoch)')
plt.ylabel('Wait Time (Seconds)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(OUT_DIR / 'thesis_wait_time.png', dpi=300)
plt.close()

# 3. RAM Consumption
plt.figure(figsize=(11, 6))
sns.barplot(data=all_df, x='Epoch', y='RAM_GB', hue='System')
plt.title('System RAM Consumption')
plt.ylabel('RAM (GB)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(OUT_DIR / 'thesis_ram.png', dpi=300)
plt.close()

print(f"✅ All 3 overarching thesis plots generated successfully in {OUT_DIR}")
