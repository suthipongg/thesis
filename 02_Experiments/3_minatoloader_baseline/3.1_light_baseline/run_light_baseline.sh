#!/bin/bash
# ==============================================================================
# 3.1 Initial Lightweight Baseline (No Heavy Augmentations)
# ==============================================================================
# Tests standard PyTorch DataLoader vs original MinatoLoader under light load:
# 2 workers, batch_size=2, 5 epochs.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.venv_minato/bin/activate"
cd "$SCRIPT_DIR"

echo "=== [1/2] Running PyTorch Light Baseline ==="
python run_pytorch_baseline.py --num_workers 2 --epochs 5 --batch_size 2

echo "=== [2/2] Running MinatoLoader Light Baseline ==="
python run_minato_baseline.py --num_workers 2 --epochs 5 --batch_size 2

echo "=== Generating Comparison Plots ==="
python plot_results.py

echo "=== 3.1 Light Baseline Finished! Results in 3.1_light_baseline/results/ ==="
