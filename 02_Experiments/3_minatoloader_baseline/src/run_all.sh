#!/bin/bash

source "/home/mew/Desktop/mew/study/Master degree/thesis/02_Experiments/3_minatoloader_baseline/.venv_minato/bin/activate"

echo "======================================"
echo " Starting Phase 1 Heavy Experiments"
echo "======================================"

cd "/home/mew/Desktop/mew/study/Master degree/thesis/02_Experiments/3_minatoloader_baseline/src"

# Define parameters
EPOCHS=5
BATCH_SIZE=2

# A1: PyTorch (4 workers)
echo "Running A1: PyTorch, 4 workers"
python run_heavy_experiments.py --system pytorch --workers 4 --batch_size $BATCH_SIZE --epochs $EPOCHS --out_dir heavy_out_A1
echo "A1 finished!"

# A2: PyTorch (8 workers)
echo "Running A2: PyTorch, 8 workers"
python run_heavy_experiments.py --system pytorch --workers 8 --batch_size $BATCH_SIZE --epochs $EPOCHS --out_dir heavy_out_A2
echo "A2 finished!"

# B1: MinatoLoader (4 workers)
echo "Running B1: MinatoLoader, 4 workers"
python run_heavy_experiments.py --system minato --workers 4 --batch_size $BATCH_SIZE --epochs $EPOCHS --out_dir heavy_out_B1
echo "B1 finished!"

# B2: MinatoLoader (8 workers)
echo "Running B2: MinatoLoader, 8 workers"
python run_heavy_experiments.py --system minato --workers 8 --batch_size $BATCH_SIZE --epochs $EPOCHS --out_dir heavy_out_B2
echo "B2 finished!"

echo "======================================"
echo " All Phase 1 Experiments Completed!"
echo "======================================"
