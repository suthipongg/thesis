import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def extract_workers(jsonl_path):
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Only track training phase
            if "active_workers" in data and data.get("phase", "train") == "train":
                records.append({
                    "Batch": len(records),
                    "Epoch": data.get("epoch", 1),
                    "Active Workers": data["active_workers"]
                })
    return pd.DataFrame(records)

df_c1 = extract_workers("results/heavy_out_C1/batch_logs.jsonl")
df_c2 = extract_workers("results/heavy_out_C2/batch_logs.jsonl")

df_c1['System'] = 'C1 (Scheduled)'
df_c2['System'] = 'C2 (Scheduled + Adaptive)'

df_all = pd.concat([df_c1, df_c2])

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_all, x='Batch', y='Active Workers', hue='System', drawstyle='steps-post', linewidth=3)
plt.title("Worker Scheduler Auto-Tuning Timeline")
plt.xlabel("Batch Index (Progress over time)")
plt.ylabel("Number of Active Workers")
plt.ylim(0, 10)
plt.axhline(y=8, color='red', linestyle='--', alpha=0.5, label='Max Workers (8)')
plt.axhline(y=4, color='green', linestyle='--', alpha=0.5, label='Sweet Spot (4)')
plt.legend()
plt.tight_layout()
out_dir = "results/plots"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/worker_scaling_timeline.png"
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Plot saved to {out_path}")
