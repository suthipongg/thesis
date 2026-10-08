import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def load_data(exp_dir, name):
    if not os.path.exists(f"{exp_dir}/epoch_result.csv"):
        return None
    df = pd.read_csv(f"{exp_dir}/epoch_result.csv")
    df = df[df['Phase'] == 'Val'].copy()
    df['System'] = name
    return df

exps = [
    ("heavy_out_A1", "PyTorch (4 Workers)"),
    ("heavy_out_A2", "PyTorch (8 Workers)"),
    ("heavy_out_B1", "Minato (4 Workers)"),
    ("heavy_out_B2", "Minato (8 Workers)"),
    ("heavy_out_C1", "Minato Scheduled (8W)"),
    ("heavy_out_C2", "Minato Scheduled+Adaptive (8W)")
]

dfs = []
for d, n in exps:
    df = load_data(d, n)
    if df is not None:
        dfs.append(df)

if not dfs:
    print("No data found!")
    exit(1)

all_df = pd.concat(dfs, ignore_index=True)

# Plot Throughput
plt.figure(figsize=(10, 6))
sns.barplot(data=all_df, x='Epoch', y='Throughput_Img_Sec', hue='System')
plt.title('Throughput (Images/Second) under Heavy Augmentations')
plt.ylabel('Throughput (img/s)')
plt.savefig('../results/plots/thesis_throughput.png')

# Plot Wait Time
plt.figure(figsize=(10, 6))
sns.barplot(data=all_df, x='Epoch', y='Total_Starvation_Sec', hue='System')
plt.title('Total GPU Wait Time (Seconds per Epoch)')
plt.ylabel('Wait Time (Seconds)')
plt.savefig('../results/plots/thesis_wait_time.png')

# Plot RAM
plt.figure(figsize=(10, 6))
sns.barplot(data=all_df, x='Epoch', y='RAM_GB', hue='System')
plt.title('System RAM Consumption')
plt.ylabel('RAM (GB)')
plt.savefig('../results/plots/thesis_ram.png')

print("Plots generated in ../results/plots/")
