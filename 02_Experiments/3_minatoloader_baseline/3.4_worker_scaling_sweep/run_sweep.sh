#!/bin/bash
# ==============================================================================
# 3.4 Worker Scaling Sweep Benchmark (Workers 1 to 8)
# ==============================================================================
# Sweeps worker counts across PyTorch and MinatoLoader to find the exact inflection point.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.venv_minato/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SCRIPT_DIR"

echo "=========================================================="
echo " 🚀 Running Worker Sweep: 1 to 8 Workers (PyTorch & Minato)"
echo "=========================================================="

python run_worker_sweep.py --workers 1,2,3,4,5,6,7,8 --systems pytorch,minato --epochs 3 --batch_size 2

echo "=========================================================="
echo " 📈 Generating Worker Scaling Plots"
echo "=========================================================="

python plot_sweep.py

echo "=========================================================="
echo " ✅ Sweep Completed! Plots saved in results/plots/"
echo "=========================================================="
