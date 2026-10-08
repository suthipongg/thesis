#!/bin/bash
#SBATCH --job-name=download            # Job name
#SBATCH --output=output_download.log   # Standard output log
#SBATCH --error=error_%j.log          # Error log file
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --partition=dev-arm     # Partition name
#SBATCH --time=2:00:00                # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=20          # Number of CPU cores
#SBATCH --account=i20240005a          # Account to charge

source /share/env/module_select.sh
module load Python
pip install sox
source venv/bin/activate


bash scripts/preprocess_librispeech.sh
