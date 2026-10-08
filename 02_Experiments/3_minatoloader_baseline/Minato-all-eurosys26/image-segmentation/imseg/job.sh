#!/usr/bin/env bash
#SBATCH --job-name=3dunet           # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # Tasks per node
#SBATCH --cpus-per-task=128          # CPUs per task (32 per GPU × 4 GPUs)
#SBATCH --output=3dunet_tag.txt  # Output file
#SBATCH --error=3dunet_tag.err   # Error file
#SBATCH --gpus=4
#SBATCH --account=i20240005g         # Account for resource allocation
#SBATCH --time=2:00:00               # Time limit
#SBATCH --partition=normal-a100-40    # GPU partition
   

# Load required modules
module load Python
echo "Current working directory: $(pwd)"

# # Run the monitoring script in the background & save output to a file
# bash gpu_script.sh > /projects/I20240005/rnouaj/image-segmentation/imseg/GPUusage_16worker.csv 2>&1 & 
# MONITOR_PID=$!   # Save the process ID of the monitoring script

# Run the script inside the Singularity container

singularity exec --nv \
    --bind /projects:/projects \
    --bind /projects/I20240005/rnouaj/backup_copy/Results_workloads_deucalion/image-segmentation/imseg/ckpts:/projects/I20240005/rnouaj/backup_copy/Results_workloads_deucalion/image-segmentation/imseg/ckpts \
    --bind /projects/I20240005/rnouaj/backup_copy/Results_workloads_deucalion/image-segmentation/imseg/output:/projects/I20240005/rnouaj/backup_copy/Results_workloads_deucalion/image-segmentation/imseg/results \
    --bind /projects/I20240005/raw-data-dir/kits19/data:/projects/I20240005/raw-data-dir/kits19/data \
    /projects/I20240005/rnouaj/singularity_images/singularity_imseg.sif \
    bash run_and_time.sh \


# Stop the monitoring script after training finishes
kill $MONITOR_PID
