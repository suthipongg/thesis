#!/usr/bin/env bash
#SBATCH --job-name=3dunet_soeedy           # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # Tasks per node
#SBATCH --cpus-per-task=128          # CPUs per task (32 per GPU × 4 GPUs)
#SBATCH --output=speedy_3dunet_tag.txt  # Output file
#SBATCH --error=speedy_3dunet_tag.err   # Error file
#SBATCH --gpus=4
#SBATCH --account=i20240005g         # Account for resource allocation
#SBATCH --time=2:00:00               # Time limit
#SBATCH --partition=normal-a100-40    # GPU partition
   

module load Python
echo "Current working directory: $(pwd)"

# # Run monitoring script in background
# bash gpu_script.sh > gpu_usage.log 2>&1 &
# PIDS=($!)   # store PID

# echo "Start monitoring"
# for i in $(seq 0 3); do
#     nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory \
#                --id=$i --format=csv -lms=100 > 3dunet-gpu-${i}.csv &
#     PIDS+=($!)   # add PID to list
# done

# # Run main job
singularity exec --nv \
    --bind /projects:/projects \
    --bind /projects/I20240005/rnouaj/backup_copy/Results_workloads_deucalion/image-segmentation/async/ckpts:/projects/I20240005/rnouaj/backup_copy/Results_workloads_deucalion/image-segmentation/async/ckpts \
    --bind /projects/I20240005/rnouaj/backup_copy/Results_workloads_deucalion/image-segmentation/async/output:/projects/I20240005/rnouaj/backup_copy/Results_workloads_deucalion/image-segmentation/async/results \
    --bind /projects/I20240005/raw-data-dir/kits19/data:/projects/I20240005/raw-data-dir/kits19/data \
    /projects/I20240005/rnouaj/singularity_images/singularity_async.sif \
    bash run_and_time.sh

# Kill all monitoring jobs after training finishes
for pid in "${PIDS[@]}"; do
    kill $pid 2>/dev/null
done

