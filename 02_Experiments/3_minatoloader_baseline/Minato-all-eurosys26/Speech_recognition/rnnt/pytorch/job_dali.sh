#!/bin/bash
#SBATCH --job-name=pytorch            # Job name
#SBATCH --output=output_dali_10s_b32.log   # Standard output log
#SBATCH --error=error_%j.log          # Error log file
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --partition=dev-a100-40       # Partition name
#SBATCH --time=4:00:00                # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=100           # Number of CPU cores
#SBATCH --account=i20240005g          # Account to charge

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
