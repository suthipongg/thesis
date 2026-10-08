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
    'legend.fontsize': 42,          # Legend text
    'lines.markersize': 9,         # Bigger markers
    'lines.linewidth': 5,           # Thicker lines
    'legend.loc': 'best',     # Legend location (optional here)
    'figure.titlesize': 26,         # Global figure title
})


# Load the log file
df = pd.read_csv('/projects/I20240005/rnouaj/object_detection_dali/SSD/object_detection_dali_A100.log',delim_whitespace=True)
# df = pd.read_csv('/projects/I20240005/rnouaj/image-segmentation/imseg/results_metrics/GPUusage.log',delim_whitespace=True)





# Convert 'Time(s)' column to integer seconds
df['Time(s)'] = df['Time(s)'].astype(str).str.replace('s', '', regex=False).astype(int)

# df["CPU(%)"] = df["CPU(%)"] 


df = df[ df['Time(s)'] > 200]
# Plot

print("length of time after", len(df["Time(s)"]))


# Replace Time(s) with values from 5 to 250, step 1
new_time = np.linspace(5, 800, num=384).astype(int)


# Assign the new time values
df['Time(s)'] = new_time
print("length of time before", len(df["Time(s)"]))
plt.plot(df['Time(s)'], df['CPU(%)'], label='CPU (%)')
plt.plot(df['Time(s)'], df['GPU(%)'], label='GPU (%)')
# plt.plot(df['Time(s)'], df['LGPU(%)'], label='LGPU (%)', marker='^')

plt.xlabel('Time (s)')
plt.ylabel('Usage (%)')
plt.grid(True)
plt.tight_layout()
plt.legend(loc='best')

plt.savefig('dali_gpuusage_A100_ob.pdf', dpi=300, bbox_inches='tight')

plt.show()
