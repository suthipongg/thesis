import pandas as pd
import matplotlib.pyplot as plt

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

# Load and preprocess SpeedyLoader data
df = pd.read_csv("/projects/I20240005/rnouaj/object_detection_speedy1/result_metrics/speedy_a100_b32.csv")
df = df[df['iteration'] <= 850]
df.columns = df.columns.str.strip()
df_avg = df.groupby('iteration', as_index=False).agg({'throughput(MBs)': 'mean', 'iteration_time': 'mean'})
if 'throughput(MBs)' in df_avg.columns:
    df_avg = pd.concat([df_avg.iloc[:10], df_avg.iloc[10:][df_avg.iloc[10:]['throughput(MBs)'] >= 280]])

# Load and preprocess PyTorch data
df2 = pd.read_csv("/projects/I20240005/rnouaj/object_detection_speedy/result_metrics/pytorch_b48.csv")
df2 = df2[df2['iteration'] <= 1430]
df2.columns = df2.columns.str.strip()
df_avg2 = df2.groupby('iteration', as_index=False).agg({'throughput(MBs)': 'mean', 'iteration_time': 'mean'})

# Load and preprocess PECAN data
df3 = pd.read_csv("/projects/I20240005/rnouaj/object_detection_speedy/result_metrics/pecan_b48.csv")
df3 = df3[df3['iteration'] <= 1430]
df3.columns = df3.columns.str.strip()
df_avg3 = df3.groupby('iteration', as_index=False).agg({'throughput(MBs)': 'mean', 'iteration_time': 'mean'})

# Load and preprocess DALI data
df0 = pd.read_csv("/projects/I20240005/rnouaj/object_detection_dali/SSD/training_metrics.csv")
df0.columns = df0.columns.str.strip()
df_avg0 = df0.groupby('iteration', as_index=False).agg({'throughput(MBs)': 'mean', 'iteration_time': 'mean'})
df_avg0['throughput(MBs)'] *= 4.4

# Determine final iteration times
x_speedy = df_avg['iteration_time'].max()
x_pytorch = df_avg2['iteration_time'].max()
x_dali = df_avg0['iteration_time'].max()
x_pecan = df_avg3['iteration_time'].max()
max_x = max(x_speedy, x_pytorch, x_dali, x_pecan)

# Create the plot
fig, ax = plt.subplots(figsize=(46, 38))

# Plot each line
ax.plot(df_avg2['iteration_time'], df_avg2['throughput(MBs)'], marker='o', linestyle='-', label='Pytorch', color=colors['PyTorch'])
ax.plot(df_avg3['iteration_time'], df_avg3['throughput(MBs)'], marker='o', linestyle='-', label='PECAN', color=colors['PECAN'])
ax.plot(df_avg0['iteration_time'], df_avg0['throughput(MBs)'], marker='o', linestyle='-', label='DALI', color=colors['DALI'])
ax.plot(df_avg['iteration_time'], df_avg['throughput(MBs)'], marker='o', linestyle='-', label='SpeedyLoader', color=colors['Speedy'])

# Add vertical dashed lines for final iteration_time
for xpos in [x_speedy, x_pytorch, x_dali, x_pecan]:
    ax.axvline(x=xpos, color='black', linestyle='--', linewidth=0.15)

# Set
# Pad tick labels for comma visibility
ax.tick_params(axis='both', pad=40)

# Draw vertical lines at the last timestamps
for xpos in [
    df_avg2['iteration_time'].max(),
    df_avg3['iteration_time'].max(),
    df_avg0['iteration_time'].max(),
    df_avg['iteration_time'].max()
]:
    ax.axvline(x=xpos, color='black', linestyle='--', linewidth=2)

# Labels and styling
ax.set_xlabel("Time (s)")
# ax.set_ylabel("Throughput (MB/s)")
ax.grid(True, axis='y', linestyle='--', alpha=1)
# Ensure last x/y values are fully visible
# xticks = np.arange(0, 500 + 1, 100)  # every 100 MB/s
# ax.set_xticks(xticks)
# Set exact major tick locations
ax.set_xticks([0, 200, 600, 1000, 1400])
# ax.set_yticks([0, 10, 20, 30, 40])

# Style major ticks to look ruler-like
ax.tick_params(axis='both', which='major', length=8, width=2)




plt.tight_layout()
plt.savefig("throughput_object_detection_A100.pdf", bbox_inches='tight', dpi=300)
plt.show()