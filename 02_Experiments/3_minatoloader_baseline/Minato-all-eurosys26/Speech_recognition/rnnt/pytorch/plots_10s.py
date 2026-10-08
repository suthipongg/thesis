import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load the CSV file (update the filename)



# Set global matplotlib style for readability
# Set global matplotlib style for readability
plt.rcParams.update({
    'figure.figsize': (18, 10),     # Bigger figure
    'font.size': 32,                # Base font size
    'axes.titlesize': 32,           # Title font
    'axes.labelsize': 32,           # Axis labels
    'xtick.labelsize': 30,          # X tick labels
    'ytick.labelsize': 30,          # Y tick labels
    'legend.fontsize': 25,          # Legend text
    'lines.markersize': 6,         # Bigger markers
    'lines.linewidth': 3,           # Thicker lines
    'legend.loc': 'upper left',     # Legend location (optional here)
    'figure.titlesize': 26,         # Global figure title
})

df = pd.read_csv("/projects/I20240005/rnouaj/Speech_recognition/rnnt/pytorch/results_metrics/speech_daliA100_10s/speech_daliA100_10s_good.csv")  # Replace with actual file
df = df[df['iteration'] <= 1000]

# Compute the average for each iteration
df_avg = df.groupby('iteration', as_index=False).agg({'iteration_time': 'mean', 'throughput(MBs)': 'mean'})
df_avg['throughput(MBs)'] = df_avg['throughput(MBs)'] * 3.75
# Read the CSV file, skipping the first row which contains repeated column headers
df2 = pd.read_csv("/projects/I20240005/rnouaj/Speech_recognition/rnnt/pytorch/results_metrics/speech_speedyA100_10s/speedy10s_test2.csv")

# Print the first few rows to verify
print("heads", df2.head())

df2 = df2[df2['iteration'] <= 1000]



# Compute the average for each iteration
df_avg2 = df2.groupby('iteration', as_index=False).agg({'iteration_time': 'mean', 'throughput(MBs)': 'mean'})

df1 = pd.read_csv("/projects/I20240005/rnouaj/Speech_recognition/rnnt/pytorch/results_metrics/speech_pytorchA100_10s/speech_pytorchA100_10s_test2.csv"
)
df1 = df1[df1['iteration'] <= 1000]

# Compute the average for each iteration
df_avg1 = df1.groupby('iteration', as_index=False).agg({'iteration_time': 'mean', 'throughput(MBs)': 'mean'})

# Add (0, 0) to the start of the data
# df_avg1 = pd.concat([pd.DataFrame({'iteration': [0], 'iteration_time': [0], 'throughput(MBs)': [0]}), df_avg1]).reset_index(drop=True)

# Load and filter data
df_pecan = pd.read_csv("/projects/I20240005/rnouaj/Speech_recognition/rnnt/pytorch/results_metrics/speech_pytorchA100_10s/speech_pecantestA100_10s.csv")
df_pecan = df_pecan[df_pecan['iteration'] <= 1000]

# # Compute averages per iteration
df_avgpecan = df_pecan.groupby('iteration', as_index=False).agg({'iteration_time': 'mean', 'throughput(MBs)': 'mean'})

# Plot the results



plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# Get the final iteration_time for each plot
# Get the final iteration_time for each plot
x_speedy = df_avg['iteration_time'].max()
x_pytorch = df_avg2['iteration_time'].max()
x_dali = df_avg1['iteration_time'].max()
x_pecan = df_avgpecan['iteration_time'].max()

# Expand x-axis to avoid cutoff
max_x = max(x_speedy, x_pytorch, x_dali, x_pecan)
fig, ax = plt.subplots()

# Plot the lines
ax.plot(df_avg['iteration_time'], df_avg['throughput(MBs)'], marker='o', linestyle='-', label='DALI')
ax.plot(df_avg2['iteration_time'], df_avg2['throughput(MBs)'], marker='o', linestyle='-', label='SpeedyLoader')
ax.plot(df_avg1['iteration_time'], df_avg1['throughput(MBs)'], marker='o', linestyle='-', label='Pytorch DataLoader')
ax.plot(df_avgpecan['iteration_time'], df_avgpecan['throughput(MBs)'], marker='o', linestyle='-', label='PECAN')

# Grid and axis limits
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_xlim(right=max_x + 380)

# Add vertical lines and text (using relative transform for y)
ax.axvline(x_speedy, color='C0', linestyle=':', linewidth=2)
ax.text(x_speedy, 0.6, ' 1k iterations ', transform=ax.get_xaxis_transform(), color='C0',
        verticalalignment='bottom', horizontalalignment='right')

ax.axvline(x_pytorch, color='C1', linestyle=':', linewidth=2)
ax.text(x_pytorch, 0.7, ' 1k iterations', transform=ax.get_xaxis_transform(), color='C1',
        verticalalignment='bottom', horizontalalignment='left')

ax.axvline(x_dali, color='C2', linestyle=':', linewidth=2)
ax.text(x_dali, 0.4, ' 1k iterations', transform=ax.get_xaxis_transform(), color='C2',
        verticalalignment='bottom', horizontalalignment='left')

ax.axvline(x_pecan, color='C3', linestyle=':', linewidth=2)
ax.text(x_pecan, 0.55, '  1k iterations', transform=ax.get_xaxis_transform(), color='C3',
        verticalalignment='bottom', horizontalalignment='left')

# Labels and legend
ax.set_xlabel("Time (s)")
ax.set_ylabel("Throughput (MB/s)")
# ax.set_title("Throughput MB/s Speech recognition (0.5s and heavy 3s) A100")
ax.legend(loc='upper right')

# Save and show
plt.savefig("Throughput_speech10s_A100_test2.pdf", bbox_inches='tight', dpi=300)  # Save the plot to a file
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()
