




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
    'legend.fontsize': 150,
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

# Load the CSV file (update the filename)
df = pd.read_csv("/projects/I20240005/rnouaj/Speech_recognition/rnnt/pytorch/results_diff_#GPUs_dali/results_3s/4gpu_dali.csv")  # Replace with actual file
df = df[df['iteration'] <= 900]

# Compute the average for each iteration
df_avg = df.groupby('iteration', as_index=False).agg({'iteration_time': 'mean', 'throughput(MBs)': 'mean'})
df_avg['throughput(MBs)'] = df_avg['throughput(MBs)'] * 2.5
df2 = pd.read_csv("/projects/I20240005/rnouaj/Speech_recognition/rnnt/pytorch/results_metrics/speech_speedyA100_3s/speedy3s_test2.csv")
# Print the first few rows to verify
print("heads", df2.head())

# Filter the DataFrame for 'iteration' <= 900
df2 = df2[df2['iteration'] <= 500]



# Compute the average for each iteration
df_avg2 = df2.groupby('iteration', as_index=False).agg({'iteration_time': 'mean', 'throughput(MBs)': 'mean'})

# Add (0, 0) to the start of the data
df_avg2 = pd.concat([pd.DataFrame({'iteration': [0], 'iteration_time': [0], 'throughput(MBs)': [0]}), df_avg2]).reset_index(drop=True)

# Load the third CSV file
df1 = pd.read_csv("/projects/I20240005/rnouaj/Speech_recognition/rnnt/pytorch/results_diff_#GPUs_pytorch/results_speech3s/4gpu_pytorch.csv"
)
df1 = df1[df1['iteration'] <= 1000]

# Compute the average for each iteration
df_avg1 = df1.groupby('iteration', as_index=False).agg({'iteration_time': 'mean', 'throughput(MBs)': 'mean'})

# Load and filter data
df_pecan = pd.read_csv("/projects/I20240005/rnouaj/Speech_recognition/rnnt/pytorch/results_diff_#GPUs_pytorch/results_speech3s/4gpu_pecan.csv")
df_pecan = df_pecan[df_pecan['iteration'] <= 1000]

# Compute averages per iteration
df_avgpecan = df_pecan.groupby('iteration', as_index=False).agg({'iteration_time': 'mean', 'throughput(MBs)': 'mean'})
# # Plot the results
# # Add (0, 0) to the start of the data
# df_avgpecan = df_avgpecan._append({'iteration': 0, 'iteration_time': 0, 'throughput(MBs)': 0}, ignore_index=True)
# df_avgpecan = df_avgpecan.sort_values(by='iteration')
# df_avg1 = df_avg1._append({'iteration': 0, 'iteration_time': 0, 'throughput(MBs)': 0}, ignore_index=True)
# df_avg1 = df_avg1.sort_values(by='iteration')
# df_avg2 = df_avg2._append({'iteration': 0, 'iteration_time': 0, 'throughput(MBs)': 0}, ignore_index=True)
# df_avg2 = df_avg2.sort_values(by='iteration')
# df_avg = df_avg._append({'iteration': 0, 'iteration_time': 0, 'throughput(MBs)': 0}, ignore_index=True)
# df_avg = df_avg.sort_values(by='iteration')
# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(46, 38))

# Plot each system with formatted lines and markers


ax.plot(df_avg1['iteration_time'], df_avg1['throughput(MBs)'], linestyle='-', label='PyTorch', color=colors['PyTorch'])

ax.plot(df_avgpecan['iteration_time'], df_avgpecan['throughput(MBs)'],
        linestyle='-', label='Pecan', color=colors['PECAN'])


ax.plot(df_avg['iteration_time'], df_avg['throughput(MBs)'],
        linestyle='-', label='DALI', color=colors['DALI'])

ax.plot(df_avg2['iteration_time'], df_avg2['throughput(MBs)'],
    linestyle='-', label='SpeedyLoader', color=colors['Speedy'])

for xpos in [
    df_avg['iteration_time'].max(),
    df_avg2['iteration_time'].max(),
    df_avg1['iteration_time'].max(),
    df_avgpecan['iteration_time'].max()
]:
    ax.axvline(x=xpos, color='black', linestyle='--', linewidth=2)

# Labels and styling
# ax.set_ylabel("Throughput (MB/s)")
ax.set_xlabel("Time (s)")
ax.grid(True, axis='y', linestyle='--', alpha=1)
ymax = df_avg2['throughput(MBs)'].max()
ax.set_ylim(top=40 * 1.05)  # add 5% headroom

# Style major ticks to look ruler-like
ax.set_xticks([0, 100, 200, 300, 400, 500])
ax.set_yticks([0, 10, 20, 30,40])

ax.tick_params(axis='x', length=16, pad=24)  # length in points, pad in points
ax.tick_params(axis='y', length=16, pad=24)  # length in points, pad in points


# ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig("throughput_speech_3s_A100.pdf", bbox_inches='tight', dpi=300)
plt.show()





