#!/usr/bin/env bash
#SBATCH --job-name=object-geepafs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=waft_object_geepafs.txt
#SBATCH --error=waft_object_geepafs.err
#SBATCH --gpus=4
#SBATCH --account=i20240005g
#SBATCH --time=2:00:00
#SBATCH --partition=normal-a100-40       

# Load required modules
module load Python
echo "Current working directory: $(pwd)"

# # Run the monitoring script in the background & save output to a file
# bash gpu_script.sh > monitor_output_speedybtest_accuracy2.log 2>&1 & 
# MONITOR_PID=$!   # Save the process ID of the monitoring script

# Run the script inside the Singularity container
singularity exec --nv \
    --bind /projects:/projects \
    --bind /projects/I20240005/coco:/projects/I20240005/coco \
    /projects/I20240005/rnouaj/singularity_images/singularity_object_detection_speedy.sif \
    bash -c "./run_and_time.sh"

# # Stop the monitoring script after training finishes
# kill $MONITOR_PID
