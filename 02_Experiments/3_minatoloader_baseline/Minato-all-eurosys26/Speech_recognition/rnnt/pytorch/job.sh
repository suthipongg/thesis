#!/bin/bash
#SBATCH --job-name=sxxy4gpu            # Job name
#SBATCH --output=sxx4gpu.log   # Standard output log
#SBATCH --error=error_%j.log          # Error log file
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --partition=dev-a100-40       # Partition name
#SBATCH --time=1:00:00                # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=100           # Number of CPU cores
#SBATCH --account=i20240005g          # Account to charge

# Load required modules
module load Python
echo "Current working directory: $(pwd)"

# Run the monitoring script in the background & save output to a file

# Run the script inside the Singularity container

singularity exec --nv \
    --bind /projects:/projects \
    --bind /projects/I20240005/LibriSpeech:/projects/I20240005/LibriSpeech \
    /projects/I20240005/rnouaj/singularity_images/singularity_speech_prot.sif \
    bash -c "./scripts/train.sh"


# Stop the monitoring script after training finishes
kill $MONITOR_PID
