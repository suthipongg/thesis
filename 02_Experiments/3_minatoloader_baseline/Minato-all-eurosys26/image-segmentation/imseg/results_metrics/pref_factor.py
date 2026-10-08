import pandas as pd
import matplotlib.pyplot as plt

# --- List your CSV files ---
csv_files = [
    "/projects/I20240005/rnouaj/image-segmentation/imseg/results_metrics/pytorch_A100_4GPUs_pytorch_50ep_pref2.csv",
    "/projects/I20240005/rnouaj/image-segmentation/imseg/results_metrics/pytorch_A100_4GPUs_pytorch_50ep_pref4.csv",
    "/projects/I20240005/rnouaj/image-segmentation/imseg/results_metrics/pytorch_A100_4GPUs_pytorch_50ep_pref6.csv",
    "/projects/I20240005/rnouaj/image-segmentation/imseg/results_metrics/pytorch_A100_4GPUs_pytorch_50ep_pref8.csv",
    "/projects/I20240005/rnouaj/image-segmentation/imseg/results_metrics/pytorch_A100_4GPUs_pytorch_50ep_pref24.csv",
]

labels = ["Prefetch_factor=2", "Prefetch_factor=4", "Prefetch_factor=6", "Prefetch_factor=8", "Prefetch_factor=24"]

# --- Start Plot ---
plt.rcParams.update({
    'figure.figsize': (18, 12),     # Bigger figure
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


#  --- Loop through the CSVs ---
for csv_file, label in zip(csv_files, labels):
    # Load the CSV
    df = pd.read_csv(csv_file)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Group by epoch and iteration first, then average over epoch
    df_avg_pytorch = df.groupby(['epoch', 'iteration']).agg({
    'throughput(MBs)': 'mean',
    'iteration_time': 'min',
    }).reset_index()




    # Plot
    plt.plot(df_avg_pytorch['epoch'], df_avg_pytorch[ 'throughput(MBs)']*2, marker='o', label=label)


    # Plot

# --- Final touches ---
plt.xlabel("Epoch")
plt.ylabel("Throughput (MB/s)")
plt.grid(True, axis='y')
plt.legend()
plt.tight_layout()
plt.savefig("throughput_over_epochs.pdf", bbox_inches='tight')
plt.show()
