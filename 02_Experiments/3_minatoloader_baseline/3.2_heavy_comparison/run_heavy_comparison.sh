#!/bin/bash
# ==============================================================================
# 3.2 Heavy Augmentation Comparison (CPU Bottleneck Benchmark)
# ==============================================================================
# Compares PyTorch vs MinatoLoader under Heavy 3D Augmentations:
# - A1: PyTorch (4 workers)
# - A2: PyTorch (8 workers)
# - B1: MinatoLoader (4 workers)
# - B2: MinatoLoader (8 workers)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.venv_minato/bin/activate"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo " Starting 3.2 Heavy Experiments (A1, A2, B1, B2)"
echo "=========================================================="

EPOCHS=5
BATCH_SIZE=2
mkdir -p results/plots

# A1: PyTorch (4 workers)
echo ">>> [1/4] Running A1: PyTorch, 4 workers <<<"
python run_heavy_experiments.py --system pytorch --workers 4 --batch_size $BATCH_SIZE --epochs $EPOCHS --out_dir results/heavy_out_A1
echo "A1 finished!"

# A2: PyTorch (8 workers)
echo ">>> [2/4] Running A2: PyTorch, 8 workers <<<"
python run_heavy_experiments.py --system pytorch --workers 8 --batch_size $BATCH_SIZE --epochs $EPOCHS --out_dir results/heavy_out_A2
echo "A2 finished!"

# B1: MinatoLoader (4 workers)
echo ">>> [3/4] Running B1: MinatoLoader, 4 workers <<<"
python run_heavy_experiments.py --system minato --workers 4 --batch_size $BATCH_SIZE --epochs $EPOCHS --out_dir results/heavy_out_B1
echo "B1 finished!"

# B2: MinatoLoader (8 workers)
echo ">>> [4/4] Running B2: MinatoLoader, 8 workers <<<"
python run_heavy_experiments.py --system minato --workers 8 --batch_size $BATCH_SIZE --epochs $EPOCHS --out_dir results/heavy_out_B2
echo "B2 finished!"

echo ">>> Generating Comparison Plots <<<"
python plot_epoch_metrics.py
python plot_phase2_timelines.py
python plot_epoch5_timelines.py

echo "=========================================================="
echo " 3.2 Heavy Comparison Completed! Results in results/ "
echo "=========================================================="
