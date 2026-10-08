#!/usr/bin/env bash
#SBATCH --job-name=speech           # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # Tasks per node
#SBATCH --cpus-per-task=128          # CPUs per task (32 per GPU × 4 GPUs)
#SBATCH --output=dali_frac25.txt  # Output file
#SBATCH --error=dali_frac25.err   # Error file
#SBATCH --gpus=4
#SBATCH --account=i20240005g         # Account for resource allocation
#SBATCH --time=2:00:00               # Time limit
#SBATCH --partition=normal-a100-40    # GPU partition
   
# Load required modules
module load Python
echo "Current working directory: $(pwd)"

# # Run the monitoring script in the background & save output to a file
# bash gpu_script.sh > /projects/I20240005/rnouaj/Speech_recognition/rnnt/pytorch/results_metrics/speech_daliA100_3s/GPU_speech3s_test4.log 2>&1 & 
# MONITOR_PID=$!   # Save the process ID of the monitoring script

# Run the script inside the Singularity container

singularity exec --nv \
    --bind /projects:/projects \
    --bind /projects/I20240005/LibriSpeech:/projects/I20240005/LibriSpeech \
    /projects/I20240005/rnouaj/singularity_images/singularity_speech_prot.sif \
    bash -c "./scripts/train.sh"


# Stop the monitoring script after training finishes
kill $MONITOR_PID
