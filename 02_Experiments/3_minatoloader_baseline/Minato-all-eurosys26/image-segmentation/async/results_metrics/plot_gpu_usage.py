import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set global matplotlib style for readability
# Set global matplotlib style for readability
plt.rcParams.update({
    'figure.figsize': (18, 10),     # Bigger figure
    'font.size': 38,                # Base font size
    'axes.titlesize': 38,           # Title font
    'axes.labelsize': 38,           # Axis labels
    'xtick.labelsize': 35,          # X tick labels
    'ytick.labelsize': 35,          # Y tick labels
    'legend.fontsize': 40,          # Legend text
    'lines.markersize': 9,         # Bigger markers
    'lines.linewidth': 6,           # Thicker lines
    'legend.loc': 'best',     # Legend location (optional here)
    'figure.titlesize': 26,         # Global figure title
})

# Load the log file
df = pd.read_csv('/projects/I20240005/rnouaj/image-segmentation/async/results_metrics/GPU_A100async_b4.log',delim_whitespace=True)
# df = pd.read_csv('/projects/I20240005/rnouaj/image-segmentation/imseg/results_metrics/GPUusage.log',delim_whitespace=True)





# Convert 'Time(s)' column to integer seconds
df['Time(s)'] = df['Time(s)'].astype(str).str.replace('s', '', regex=False).astype(int)
print(df.columns)
df["CPU(%)"] = df["CPU(%)"] * 128 / 24
print('length of cpu', len(df["CPU(%)"]))
print('length of gpu', len(df["GPU(%)"]))
print("length of time before", len(df["Time(s)"]))

# Replace Time(s) with values from 5 to 250, step 1
new_time = np.linspace(5, 200, num=64).astype(int)


# Assign the new time values
df['Time(s)'] = new_time
print("length of time before", len(df["Time(s)"]))

print(df.head())

# df["CPU(%)"] = df["CPU(%)"] 


# df = df[df['Time(s)'] < 200]
# Plot
plt.plot(df['Time(s)'], df['CPU(%)'], label='CPU (%)', marker='o')
plt.plot(df['Time(s)'], df['GPU(%)'], label='GPU (%)', marker='o')
# plt.plot(df['Time(s)'], df['LGPU(%)'], label='LGPU (%)', marker='^')

plt.xlabel('Time (s)')
plt.ylabel('Usage (%)')
plt.legend(loc='best')
plt.grid(True)

plt.savefig('speedy_imseg_gpuusage_A100test1.pdf', bbox_inches="tight", dpi=300)
plt.tight_layout()
plt.show()
