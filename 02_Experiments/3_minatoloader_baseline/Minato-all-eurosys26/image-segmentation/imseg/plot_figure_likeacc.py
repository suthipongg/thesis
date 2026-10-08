import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# --- Global style ---
plt.rcParams.update({
    'figure.figsize': (24, 16),
    'font.size': 92,
    'axes.titlesize': 92,
    'axes.labelsize': 92,
    'xtick.labelsize': 90,
    'ytick.labelsize': 90,
    'legend.fontsize': 94,
    'lines.markersize': 6,
    'lines.linewidth': 12,
    'legend.loc': 'best',
    'figure.titlesize': 26,
})

# ---------- Config ----------
paths = {
    'PyTorch': Path('/projects/I20240005/rnouaj/backup_copy/Results_workloads_deucalion/image-segmentation/imseg/pytorch_tags_log.csv'),
    'Minato': Path('/projects/I20240005/rnouaj/backup_copy/Results_workloads_deucalion/image-segmentation/imseg/pytorch_tags_log2.csv'),
}

# Create timestamped output folder
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = Path(f"results_{timestamp}")
out_dir.mkdir(parents=True, exist_ok=True)

# ---------- Loader ----------
import csv
import ast
import pandas as pd

def load_tags_csv(path, name):
    rows = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            if len(row) < 3:
                continue
            epoch = int(row[0].strip())
            iteration = int(row[1].strip())
            # join everything from col2 onward into the tags string
            tag_str = ",".join(row[2:]).strip()
            try:
                tags = ast.literal_eval(tag_str)
            except Exception:
                tags = []
            rows.append((epoch, iteration, tags))
      # compute stats
    df = pd.DataFrame(rows, columns=["epoch", "iteration", "tags"])
    df["batch_size"] = df["tags"].apply(len)
    df["num_long"] = df["tags"].apply(lambda t: sum(1 for x in t if str(x).upper().startswith("L")))
    df["num_short"] = df["tags"].apply(lambda t: sum(1 for x in t if str(x).upper().startswith("S")))
    df["frac_long"] = df["num_long"] / df["batch_size"].replace(0, np.nan)
    df["frac_short"] = df["num_short"] / df["batch_size"].replace(0, np.nan)



    df["loader"] = name
    return df

# ---------- Load all ----------
dfl = [load_tags_csv(path, name) for name, path in paths.items()]
# Keep only the first 10 epochs
for i in range(len(dfl)):
    dfl[i] = dfl[i][dfl[i]["epoch"] < 20]   # keep epoch 0–9
data = pd.concat(dfl, ignore_index=True)
print("Loaded data sample:\n", data.head())


# ---------- Summary ----------
def summarize(df):
    rows = []
    for loader, sub in df.groupby(['loader']):
        x = sub['frac_long'].dropna().to_numpy()
        rows.append({
            'loader': loader,
            'n_batches': int(len(x)),
            'mean_frac_long': float(np.mean(x)) if len(x) else np.nan,
            'std_frac_long': float(np.std(x)) if len(x) else np.nan,
            'mean_batch_size': float(sub['batch_size'].mean()),
            'avg_num_S': float(np.mean(sub['num_short'])),
            'std_num_S': float(np.std(sub['num_short'])),
            'avg_num_L': float(np.mean(sub['num_long'])),
            'std_num_L': float(np.std(sub['num_long'])),
        })
    return pd.DataFrame(rows)

summary = summarize(data)
print("\n=== Batch Composition Summary ===")
print(summary.to_string(index=False))
summary.to_csv(out_dir / "summary.csv", index=False)
colors = {
    "PyTorch": "tab:blue",
    "Minato": "tab:orange",
 
}



fig, ax = plt.subplots(figsize=(24, 16))
# ---------- Plots ----------
# 1. Category distribution
prop_rows = []
for loader, sub in data.groupby('loader'):
    counts = sub['num_long'].value_counts().sort_index()
    total = counts.sum()
    for k in range(int(data['batch_size'].max()) + 1):
        v = counts.get(k, 0)
        prop_rows.append({'loader': loader, 'k_long': k, 'prop': float(v/total)})

prop = pd.DataFrame(prop_rows)


all_k = sorted(prop['k_long'].unique().tolist())
width = 0.35
x = np.arange(len(all_k))
for i, name in enumerate(["PyTorch", "Minato"]):   # desired order
    sub = prop[prop['loader'] == name]
    color = colors.get(name, None)
    aligned = [sub[sub['k_long'] == k]['prop'].values[0] if k in sub['k_long'].values else 0.0 for k in all_k]
    ax.bar(x + (i - 0.5) * width, aligned, width=width, label=name, color=color)

ax.set_xticks(x)
ax.set_xticklabels([str(k) for k in all_k])

# after plotting bars
ymax = prop['prop'].max()

# round up ymax to the nearest 0.2
ytop = np.ceil(ymax / 0.2) * 0.2  

# force ticks at fixed step of 0.2
yticks = np.arange(0, ytop + 0.001, 0.2)

# round ticks to 1 decimal place (so 0.6 instead of 0.599999)
yticks = np.round(yticks, 1)
ax.set_ylim(0, ytop)
ax.set_yticks(yticks)
ax.set_xlabel('# of slow samples', labelpad=20)
ax.set_ylabel('Dist. of batches', labelpad=20)



# Make aspect ratio consistent

ax.tick_params(axis="both", which="major", pad=30, length=15, width=4, direction="inout")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(framealpha=0.5, fontsize=96, loc="upper right")
# plt.legend()
plt.tight_layout()
plt.savefig(out_dir / 'fig_batchcomp3dunet_categories.pdf', dpi=300, bbox_inches="tight")
plt.close()

# ---------- Plot 2: Timeline (continuous iterations across epochs, with inline avg lines) ----------
max_iter = data["iteration"].max()
data["global_iter"] = data["epoch"] * (max_iter + 1) + data["iteration"]

series = data.groupby(["loader", "global_iter"], as_index=False)["frac_long"].mean()

# restrict to iteration window
iter_min, iter_max = 140, 180
series = series[(series["global_iter"] >= iter_min) & (series["global_iter"] <= iter_max)]


plt.margins(x=0.1)   # add 5% margin on both sides of x-axis


for loader, sub in series.groupby("loader"):
    print("loader", loader)
    color = colors.get(loader, None)
    # main line
    plt.plot(sub["global_iter"], sub["frac_long"],  alpha=0.9, color = color)
    # average line in same color
    avg_frac = sub["frac_long"].mean()

    plt.axhline(y=avg_frac, linestyle="--", color=color, alpha=0.8)
    #   # update legend to show avg value
    # plt.plot([], [], color=color, linestyle="--", 
    #          label=f"{loader} avg: {avg_frac:.2f}")
   
ymax = series["frac_long"].max()
# pick the next "round" bound (0.05 step granularity)
ytop = np.ceil(ymax * 20) / 20.0  

# define ticks manually (nice steps up to ytop)
yticks = np.arange(0, ytop + 0.01, 0.1)   # 0.0, 0.1, 0.2, ...
if ytop > 0.4:  # add 0.5 if needed
    yticks = np.unique(np.append(yticks, 0.5))

plt.ylim(0, ytop)
plt.yticks(yticks)

plt.xlabel("Iteration", labelpad=20)
plt.ylabel("Prop. slow", labelpad=20)
ax = plt.gca()
ax.tick_params(axis="both", which="major", pad=40, length=15, width=4, direction="inout")
# ax.legend(framealpha=0.2, fontsize=96, loc="best")

plt.grid(True, linestyle="--", alpha=0.4)
# plt.legend(ncol=1, framealpha=0.5, loc = 'lower right')  # now only has loader names
plt.tight_layout()
plt.savefig(out_dir / f"fig_batchcomp_3dunet_timeline_{iter_min}_{iter_max}.pdf", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nResults saved under: {out_dir}")





