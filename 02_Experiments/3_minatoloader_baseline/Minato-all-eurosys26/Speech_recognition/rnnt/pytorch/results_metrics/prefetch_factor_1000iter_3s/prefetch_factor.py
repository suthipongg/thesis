import pandas as pd
import matplotlib.pyplot as plt

# --- List your CSV files ---
csv_files = [
"/projects/I20240005/rnouaj/Speech_recognition/rnnt/pytorch/results_metrics/prefetch_factor_1000iter_3s/speech_pytorchA100_3s_b32_pref4.csv",
"/projects/I20240005/rnouaj/Speech_recognition/rnnt/pytorch/results_metrics/prefetch_factor_1000iter_3s/speech_pytorchA100_3s_b32_pref6.csv",
"/projects/I20240005/rnouaj/Speech_recognition/rnnt/pytorch/results_metrics/prefetch_factor_1000iter_3s/speech_pytorchA100_3s_b32_pref8.csv",
"/projects/I20240005/rnouaj/Speech_recognition/rnnt/pytorch/results_metrics/prefetch_factor_1000iter_3s/speech_pytorchA100_3s_b32_pref10.csv",
"/projects/I20240005/rnouaj/Speech_recognition/rnnt/pytorch/results_metrics/prefetch_factor_1000iter_3s/speech_pytorchA100_3s_b32_pref12.csv",

]

labels = ["Prefetch_factor=2", "Prefetch_factor=4", "Prefetch_factor=8"]

# --- Start Plot ---
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

# --- Loop through the CSVs ---
for csv_file, label in zip(csv_files, labels):
    # Load the CSV
    df = pd.read_csv(csv_file)

    # Clean up (if needed) - remove extra spaces
    df.columns = df.columns.str.strip()

    # Group by epoch and calculate average throughput(MBs)
    epoch_throughput = df.groupby('iteration')['throughput(MBs)'].mean()


    # Plot
    plt.plot(epoch_throughput.index, epoch_throughput.values, marker='o', label=label)

# --- Final touches ---
plt.xlabel("Epoch")
plt.ylabel("Throughput (MB/s)")
plt.grid(True, axis='y')
plt.legend()
plt.tight_layout()
plt.savefig("prefetch_factor_speech.pdf", bbox_inches='tight')
plt.show()
