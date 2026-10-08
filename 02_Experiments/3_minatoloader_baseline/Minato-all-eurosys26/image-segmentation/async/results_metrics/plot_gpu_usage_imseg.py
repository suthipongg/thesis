import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
# df = pd.read_csv('/projects/I20240005/rnouaj/image-segmentation/async/results_metrics/GPU_A100async_b4.log',delim_whitespace=True)
df = pd.read_csv('/projects/I20240005/rnouaj/image-segmentation/imseg/results_metrics/GPU_imseg_pytorch_pref24.log',delim_whitespace=True)





# Convert 'Time(s)' column to integer seconds
df['Time(s)'] = df['Time(s)'].astype(str).str.replace('s', '', regex=False).astype(int)
print(df.columns)


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

plt.savefig('pytorch_imseg_gpuusage_A100test1.pdf', bbox_inches="tight", dpi=300)
plt.tight_layout()
plt.show()
