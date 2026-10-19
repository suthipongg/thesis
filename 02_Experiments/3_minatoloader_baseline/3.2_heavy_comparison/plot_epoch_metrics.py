import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def load_data(exp_dir, name):
    if not os.path.exists(f"{exp_dir}/epoch_result.csv"):
        return None
    df = pd.read_csv(f"{exp_dir}/epoch_result.csv")
    df = df[df['Phase'] == 'Val'].copy()
    df['System'] = name
    return df

exps = [
    ("results/heavy_out_A1", "A1: PyTorch (4W)"),
    ("results/heavy_out_A2", "A2: PyTorch (8W)"),
    ("results/heavy_out_B1", "B1: Minato (4W)"),
    ("results/heavy_out_B2", "B2: Minato (8W)")
]

dfs = []
for d, n in exps:
    df = load_data(d, n)
    if df is not None:
        dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
systems = all_df['System'].unique()
colors = sns.color_palette("husl", len(systems))
palette = dict(zip(systems, colors))

# 1. Throughput
sns.lineplot(data=all_df, x='Epoch', y='Throughput_Img_Sec', hue='System', marker='o', palette=palette, ax=axes[0, 0], linewidth=3, markersize=10)
axes[0, 0].set_title("Training Throughput (Images / Sec)")
axes[0, 0].set_ylabel("Images / Sec")

# 2. GPU Busy Ratio
sns.lineplot(data=all_df, x='Epoch', y='GPU_Busy_Ratio', hue='System', marker='s', palette=palette, ax=axes[0, 1], linewidth=3, markersize=10)
axes[0, 1].set_title("GPU Utilization (%)")
axes[0, 1].set_ylabel("GPU Busy Ratio")

# 3. Data Loading Wait (Starvation)
sns.lineplot(data=all_df, x='Epoch', y='Total_Starvation_Sec', hue='System', marker='^', palette=palette, ax=axes[1, 0], linewidth=3, markersize=10)
axes[1, 0].set_title("Total GPU Starvation Time per Epoch (Sec)")
axes[1, 0].set_ylabel("Seconds")

# 4. RAM
sns.lineplot(data=all_df, x='Epoch', y='RAM_GB', hue='System', marker='D', palette=palette, ax=axes[1, 1], linewidth=3, markersize=10)
axes[1, 1].set_title("System Memory (RAM) Consumption")
axes[1, 1].set_ylabel("Gigabytes (GB)")

# Remove individual legends and create one shared legend at the bottom
for ax in axes.flat:
    ax.get_legend().remove()

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(rect=[0, 0.05, 1, 1])
out_dir = "results/plots"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/thesis_epoch_metrics_A_B.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved to {out_path}")
