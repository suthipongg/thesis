import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set global matplotlib styles (still good for font sizes, etc.)
plt.rcParams.update({
    'font.size': 160,
    'axes.titlesize': 160,
    'axes.labelsize': 160,
    'xtick.labelsize': 160,
    'ytick.labelsize': 160,
    'legend.fontsize': 100,
    'lines.markersize': 20,
    'lines.linewidth': 12,
    'legend.loc': 'upper left',
    'figure.titlesize': 26,
})

colors = {
    'PyTorch': '#1f77b4',
    'DALI': '#ff7f0e',
    'Speedy': '#2ca02c',
    'PECAN': '#d62728'
}

# ---------- Load Data ----------
# Speedy
df_speedy = pd.read_csv("/projects/I20240005/rnouaj/image-segmentation/async/results_metrics/speedy_A100_batch4_50epochs.csv")
for col in ['throughput(MBs)', 'iteration_time', 'time_diff', 'iter_persec']:
    df_speedy[col] = pd.to_numeric(df_speedy[col], errors='coerce')
df_avg_speedy = df_speedy.groupby(['epoch', 'iteration']).agg({
    'throughput(MBs)': 'mean',
    'iteration_time': 'min',
}).reset_index()

# PyTorch
df_pytorch = pd.read_csv("/projects/I20240005/rnouaj/image-segmentation/imseg/results_metrics/pytorch_A100_4GPUs_pytorch_50ep_pref2.csv")
for col in ['throughput(MBs)', 'iteration_time', 'time_diff', 'iter_persec']:
    df_pytorch[col] = pd.to_numeric(df_pytorch[col], errors='coerce')
df_avg_pytorch = df_pytorch.groupby(['epoch', 'iteration']).agg({
    'throughput(MBs)': 'mean',
    'iteration_time': 'min',
}).reset_index()
df_avg_pytorch['throughput(MBs)'] *= 2
df_avg_pytorch = df_avg_pytorch[df_avg_pytorch['iteration_time'] < 900]

# DALI
df_dali = pd.read_csv("/projects/I20240005/rnouaj/image-segmentation/async/A100_b4_4GPUs_dali_50epochs_test2.csv")
df_avg_dali = df_dali.groupby(['epoch', 'iteration'], as_index=False).agg({
    'throughput(MBs)': 'mean',
    'iteration_time': 'min',
}).reset_index()
df_avg_dali.loc[df_avg_dali['epoch'] == 1, 'throughput(MBs)'] *= 8

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(46, 34))

# Plot each system
ax.plot(df_avg_pytorch['iteration_time'], df_avg_pytorch['throughput(MBs)'],
        marker='o', linestyle='-', label='Pytorch', color=colors['PyTorch'])

ax.plot(df_avg_dali['iteration_time'], df_avg_dali['throughput(MBs)'],
        marker='o', linestyle='-', label='DALI', color=colors['DALI'])

ax.plot(df_avg_speedy['iteration_time'], df_avg_speedy['throughput(MBs)'],
        marker='o', linestyle='-', label='SpeedyLoader', color=colors['Speedy'])


ax.tick_params(axis='both', pad=40)

# Add vertical lines at last iteration_time
x_speedy = df_avg_speedy['iteration_time'].max()
x_pytorch = df_avg_pytorch['iteration_time'].max()
x_dali = df_avg_dali['iteration_time'].max()

for xpos in [x_speedy, x_pytorch, x_dali]:
    ax.axvline(x=xpos, color='black', linestyle='--', linewidth=2)

# Styling
# ax.set_xlabel("Time (s)")
ax.set_ylabel("Throughput (MB/s)")

ax.grid(True, axis='y', linestyle='--', alpha=1)
ax.set_ylim(bottom=0)
# ax.legend(loc='best', ncol=1)

plt.tight_layout()
plt.savefig("imseg_A100.svg",bbox_inches='tight', dpi=300)
plt.show()
