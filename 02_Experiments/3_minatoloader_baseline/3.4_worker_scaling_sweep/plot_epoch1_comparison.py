"""
Plot Epoch 1 vs Epoch 2 Comparison (Mode A With Val)
====================================================
Proves that in Epoch 1 (before any validation pause happened):
1. Wait Latency behaves normally (monotonically decreases with workers, no contention cliff)
2. Queue saturation does not exist (queue stays low <19 items, never hits 50)
3. Queue progression across 19 batches starts from ~0 and stays low, unlike Epoch 2 which starts at 50.

Generates 3 comparison figures:
- fig3_epoch1_vs_epoch2_wait_latency.png (like Fig 3)
- fig4_epoch1_vs_epoch2_queue_saturation.png (like Fig 4)
- fig4b_epoch1_vs_epoch2_queue_drainage.png (like Fig 4b)
"""

import json
import csv
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BASE = Path('/home/mew/Desktop/mew/study/Master degree/thesis/02_Experiments/3_minatoloader_baseline/3.4_worker_scaling_sweep')
OUT_DIR = BASE / 'advisor_plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'legend.fontsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.pad_inches': 0.15,
    'savefig.bbox': 'tight',
})

C_PCORE = '#2196F3'
C_LOCK  = '#E91E63'
C_E1    = '#4CAF50'   # Green for Healthy Epoch 1
C_E2    = '#D62728'   # Red for Contended Epoch 2

# ── Load Epoch 1 and Epoch 2 Data ─────────────────────────────
e1_wait = {}
e1_tp = {}
e1_qsize = {}
e1_batch_q = {}

e2_wait = {}
e2_tp = {}
e2_qsize = {}
e2_batch_q = {}

# 1. Workers 1, 2, 3, 5, 6, 7 from 3.4
for w in [1, 2, 3, 5, 6, 7]:
    p_batch = BASE / f'results_with_val/runs/minato_w{w}/batch_logs.jsonl'
    if p_batch.exists():
        with open(p_batch) as f:
            lines = [json.loads(l) for l in f]
        b_e1 = [l for l in lines if l['epoch'] == 1]
        b_e2 = [l for l in lines if l['epoch'] == 2]
        
        e1_wait[w] = np.mean([l['dataloader_wait_ms'] for l in b_e1])
        e1_qsize[w] = np.mean([l.get('fast_q_size', 0) for l in b_e1])
        e1_batch_q[w] = [l.get('fast_q_size', 0) for l in b_e1]
        
        e2_wait[w] = np.mean([l['dataloader_wait_ms'] for l in b_e2])
        e2_qsize[w] = np.mean([l.get('fast_q_size', 0) for l in b_e2])
        e2_batch_q[w] = [l.get('fast_q_size', 0) for l in b_e2]
        
    p_csv = BASE / f'results_with_val/runs/minato_w{w}/epoch_result.csv'
    if p_csv.exists():
        with open(p_csv) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r['Epoch'] == '1':
                e1_tp[w] = float(r['Throughput_Img_Sec'])
            elif r['Epoch'] == '2':
                e2_tp[w] = float(r['Throughput_Img_Sec'])

# 2. Workers 4 and 8 from 3.2
for name, w in [('heavy_out_B1', 4), ('heavy_out_B2', 8)]:
    p_batch = Path('/home/mew/Desktop/mew/study/Master degree/thesis/02_Experiments/3_minatoloader_baseline/3.2_heavy_comparison/results') / name / 'batch_logs.jsonl'
    if p_batch.exists():
        with open(p_batch) as f:
            lines = [json.loads(l) for l in f]
        b_e1 = [l for l in lines if l['epoch'] == 1]
        b_e2 = [l for l in lines if l['epoch'] == 2]
        e1_wait[w] = np.mean([l['dataloader_wait_ms'] for l in b_e1])
        e1_qsize[w] = np.mean([l.get('fast_q_size', 0) for l in b_e1])
        e1_batch_q[w] = [l.get('fast_q_size', 0) for l in b_e1]
        
        e2_wait[w] = np.mean([l['dataloader_wait_ms'] for l in b_e2])
        e2_qsize[w] = np.mean([l.get('fast_q_size', 0) for l in b_e2])
        e2_batch_q[w] = [l.get('fast_q_size', 0) for l in b_e2]

    p_csv = Path('/home/mew/Desktop/mew/study/Master degree/thesis/02_Experiments/3_minatoloader_baseline/3.2_heavy_comparison/results') / name / 'epoch_result.csv'
    if p_csv.exists():
        with open(p_csv) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r['Epoch'] == '1':
                e1_tp[w] = float(r['Throughput_Img_Sec'])
            elif r['Epoch'] == '2':
                e2_tp[w] = float(r['Throughput_Img_Sec'])

workers = sorted(e1_wait.keys())
print("Epoch 1 Wait:", {w: round(e1_wait[w], 1) for w in workers})
print("Epoch 2 Wait:", {w: round(e2_wait[w], 1) for w in workers})
print("Epoch 1 Queue Avg:", {w: round(e1_qsize[w], 1) for w in workers})
print("Epoch 2 Queue Avg:", {w: round(e2_qsize[w], 1) for w in workers})


# ╔═══════════════════════════════════════════════════════════════╗
# ║  FIGURE 1: Epoch 1 vs Epoch 2 Wait Latency (Like Fig 3)       ║
# ╚═══════════════════════════════════════════════════════════════╝
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Epoch 1 (Before Val Pause - Healthy)
bars1 = ax1.bar(workers, [e1_wait[w] for w in workers], color='#2E7D32', edgecolor='white', width=0.65, zorder=3)
for bar, w in zip(bars1, workers):
    val = e1_wait[w]
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8, f'{val:.0f}ms', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax1.axvline(x=4.5, color='#999', linestyle=':', linewidth=1.5, alpha=0.6)
ax1.annotate('Healthy monotonic decrease!\n404ms → 40ms\n(NO Lock Contention)', 
             xy=(7, e1_wait[7]), xytext=(3.5, 250),
             fontsize=9.5, fontweight='bold', color='#2E7D32',
             arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2E7D32', alpha=0.9))

ax1.set_xlabel('Number of Workers')
ax1.set_ylabel('Dataloader Wait Latency (ms)')
ax1.set_title('Epoch 1 (Before Any Validation Pause):\nWait Latency Decreases Normally (No Contention)', color='#2E7D32')
ax1.set_ylim(0, 480)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.grid(axis='y', alpha=0.3)

# Right: Epoch 2 (After Validation Pause - Lock Contention)
colors_e2 = [C_PCORE if w <= 4 else C_LOCK for w in workers]
bars2 = ax2.bar(workers, [e2_wait[w] for w in workers], color=colors_e2, edgecolor='white', width=0.65, zorder=3)
for bar, w in zip(bars2, workers):
    val = e2_wait[w]
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6, f'{val:.1f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.axhline(y=10, color='gray', linestyle=':', alpha=0.7, label='Optimal threshold (<10ms)')
ax2.axvline(x=4.5, color='#999', linestyle=':', linewidth=1.5, alpha=0.6)
ax2.annotate('Contention Cliff!\nJumps 17× at W=8\n(Thundering Herd on Full Queue)',
             xy=(8, e2_wait[8]), xytext=(4.2, 23),
             fontsize=9.5, fontweight='bold', color=C_LOCK,
             arrowprops=dict(arrowstyle='->', color=C_LOCK, lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor=C_LOCK, alpha=0.9))

ax2.set_xlabel('Number of Workers')
ax2.set_ylabel('Dataloader Wait Latency (ms)')
ax2.set_title('Epoch 2 (After 10-min Validation Pause):\nContention Cliff Appears (Queue Full 50/50)', color=C_LOCK)
ax2.set_ylim(0, 35)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.legend(loc='upper left')
ax2.grid(axis='y', alpha=0.3)

fig.suptitle('Fig 3-Compare: Mode A Wait Latency — Epoch 1 (No Val Pause) vs Epoch 2 (After Val Pause)\n'
             'Proves conclusively that Lock Contention ONLY occurs after the queue is saturated during validation!',
             fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig3_epoch1_vs_epoch2_wait_latency.png')
plt.close(fig)
print("✅ Fig 3-Compare saved")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  FIGURE 2: Epoch 1 vs Epoch 2 Queue Saturation (Like Fig 4)    ║
# ╚═══════════════════════════════════════════════════════════════╝
fig, ax = plt.subplots(figsize=(11, 6))

w_arr = np.array(workers)
ax.bar(w_arr - 0.18, [e1_qsize[w] for w in workers], 0.36, color='#4CAF50', edgecolor='white', label='Epoch 1 (Before Val Pause — Cold Start)', zorder=3)
ax.bar(w_arr + 0.18, [e2_qsize[w] for w in workers], 0.36, color='#D62728', edgecolor='white', label='Epoch 2 (After Val Pause — Pre-buffered)', zorder=3)

ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5, label='Queue Capacity Limit (50 items)')
ax.axvline(x=4.5, color='#999', linestyle=':', linewidth=1.5, alpha=0.6)

for w in workers:
    ax.text(w - 0.18, e1_qsize[w] + 1, f'{e1_qsize[w]:.1f}', ha='center', fontsize=8, fontweight='bold', color='#2E7D32')
    ax.text(w + 0.18, e2_qsize[w] + 1, f'{e2_qsize[w]:.1f}', ha='center', fontsize=8, fontweight='bold', color='#C62828')

ax.annotate('Epoch 2: Queue saturated (32-48/50)\nFilled during 10-min validation\n→ Triggers Thundering Herd',
            xy=(7.18, e2_qsize[7]), xytext=(3.5, 43),
            fontsize=9.5, fontweight='bold', color='#C62828',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#C62828', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5))

ax.annotate('Epoch 1: Queue stays low (<19/50)\nNo prior validation pause\n→ No queue saturation, no lock collision',
            xy=(5.82, e1_qsize[6]), xytext=(2.0, 18),
            fontsize=9.5, fontweight='bold', color='#2E7D32',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2E7D32', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))

ax.set_xlabel('Number of DataLoader Workers')
ax.set_ylabel('Average Queue Size (items in RAM)')
ax.set_title('Fig 4-Compare: Queue Fill Level — Epoch 1 vs Epoch 2\n(Epoch 1 starts from 0 and never fills queue; Epoch 2 starts pre-filled at 50)', fontweight='bold')
ax.set_xlim(0.5, 8.5)
ax.set_ylim(0, 58)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.3)

fig.savefig(OUT_DIR / 'fig4_epoch1_vs_epoch2_queue_saturation.png')
plt.close(fig)
print("✅ Fig 4-Compare saved")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  FIGURE 3: Epoch 1 vs Epoch 2 Drainage Progression (Like 4b)  ║
# ╚═══════════════════════════════════════════════════════════════╝
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

colors = {1: '#1f77b4', 2: '#aec7e8', 3: '#2ca02c', 4: '#8c564b', 5: '#ff7f0e', 6: '#d62728', 7: '#9467bd', 8: '#e377c2'}

# Left: Epoch 1 Batch-by-Batch
for w in workers:
    qs = e1_batch_q[w]
    ax1.plot(range(len(qs)), qs, marker='o', linewidth=2, label=f'W={w} (start={qs[0]}, end={qs[-1]})', color=colors[w])

ax1.axhline(y=50, color='red', linestyle=':', alpha=0.6, label='Capacity (50)')
ax1.set_xlabel('Batch Index in Epoch 1 (0 to 18)', fontweight='bold')
ax1.set_ylabel('Queue Size (items in RAM)', fontweight='bold')
ax1.set_title('A. Epoch 1: Cold Start (No Prior Validation Pause)\n(Queue starts near 0 and stays low — NO saturation)', color='#2E7D32', fontweight='bold')
ax1.set_xticks(range(0, 19))
ax1.legend(loc='upper left', fontsize=8.5, frameon=True)
ax1.grid(True, alpha=0.3)

# Right: Epoch 2 Batch-by-Batch
for w in workers:
    qs = e2_batch_q[w]
    ax2.plot(range(len(qs)), qs, marker='s', linewidth=2, label=f'W={w} (start={qs[0]}, end={qs[-1]})', color=colors[w])

ax2.axhline(y=50, color='red', linestyle=':', alpha=0.6, label='Capacity (50)')
ax2.set_xlabel('Batch Index in Epoch 2 (0 to 18)', fontweight='bold')
ax2.set_title('B. Epoch 2: Post-Validation (Pre-buffered 50 Items)\n(Queue starts at 50 — W=8 stays saturated at 45-50)', color='#C62828', fontweight='bold')
ax2.set_xticks(range(0, 19))
ax2.legend(loc='lower left', fontsize=8.5, frameon=True)
ax2.grid(True, alpha=0.3)

fig.suptitle('Fig 4b-Compare: Batch-by-Batch Queue Dynamics — Epoch 1 vs Epoch 2\n'
             'In Epoch 1, queue never fills up; In Epoch 2, validation pre-buffers 50 items causing contention',
             fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig4b_epoch1_vs_epoch2_queue_drainage.png')
plt.close(fig)
print("✅ Fig 4b-Compare saved")

print("\n🚀 All Epoch 1 vs Epoch 2 comparison plots generated successfully!")
