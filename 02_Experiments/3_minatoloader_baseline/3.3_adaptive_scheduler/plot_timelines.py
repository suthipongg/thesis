import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def plot_gpu_timeline(jsonl_path, title, out_path, num_batches=15):
    if not jsonl_path or not os.path.exists(jsonl_path):
        return

    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            # Only use epoch 1
            if rec.get("epoch", 1) == 1:
                records.append(rec)
            if len(records) >= num_batches:
                break
                
    if not records:
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    y_pos = 0
    t0 = records[0]["t_dataloader_start"]
    
    for idx, rec in enumerate(records):
        start_t = rec["t_dataloader_start"] - t0
        wait_s = rec["dataloader_wait_ms"] / 1000.0
        compute_s = rec["gpu_compute_ms"] / 1000.0 if rec["gpu_compute_ms"] else 0.0
        
        ax.barh(y_pos, wait_s, left=start_t, height=0.5, color='salmon', alpha=0.8, 
                label='DataLoader Wait' if idx == 0 else "")
        ax.barh(y_pos, compute_s, left=start_t + wait_s, height=0.5, color='mediumseagreen', alpha=0.8,
                label='GPU Compute' if idx == 0 else "")
                
        ax.text(start_t + wait_s/2, y_pos, f"Wait", ha='center', va='center', fontsize=8, color='black')
        ax.text(start_t + wait_s + compute_s/2, y_pos, f"B{idx}", ha='center', va='center', fontsize=8, color='black')
        
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Detailed GPU Timeline (First {num_batches} Batches) - {title}")
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    out_dir = "results/plots"
    os.makedirs(out_dir, exist_ok=True)
    
    plot_gpu_timeline("results/heavy_out_C1/batch_logs.jsonl", "Minato Scheduled (8 Workers)", f"{out_dir}/timeline_C1_scheduled.png")
    plot_gpu_timeline("results/heavy_out_C2/batch_logs.jsonl", "Minato Scheduled + Adaptive (8 Workers)", f"{out_dir}/timeline_C2_adaptive.png")
    
    print("Timeline plots generated successfully.")
