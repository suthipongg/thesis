#!/bin/bash

cd "/home/mew/Desktop/mew/study/Master degree/thesis/02_Experiments/3_minatoloader_baseline/src"
source "/home/mew/Desktop/mew/study/Master degree/thesis/02_Experiments/3_minatoloader_baseline/.venv_minato/bin/activate"

# B2: MinatoLoader, 8 workers
echo "Running B2: Minato, 8 workers"
python run_heavy_experiments.py --system minato --workers 8 --batch_size 2 --epochs 5 --out_dir heavy_out_B2
echo "B2 finished!"

# C1 & C2
bash run_phase2.sh

echo "======================================"
echo " RESUMED PIPELINE COMPLETED SUCCESSFULLY! "
echo "======================================"
