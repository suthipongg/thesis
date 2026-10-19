#!/bin/bash
# ==============================================================================
# 3.3 Dynamic WorkerScheduler & Adaptive Timeout (Our Custom Implementation)
# ==============================================================================
# Runs our custom implementation of the missing features from the EuroSys paper:
# - C1: MinatoLoader + WorkerScheduler (Dynamic scaling 2->8 based on CPU)
# - C2: MinatoLoader + WorkerScheduler + Adaptive Timeout (Rolling P75 threshold)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.venv_minato/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SCRIPT_DIR"

echo "=========================================================="
echo " Starting 3.3 Scheduled & Adaptive Experiments (C1, C2)"
echo "=========================================================="

EPOCHS=5
BATCH_SIZE=2
mkdir -p results/plots

# C1: MinatoLoader + WorkerScheduler
echo ">>> [1/2] Running C1: MinatoLoader + WorkerScheduler <<<"
python run_scheduled_experiments.py --batch_size $BATCH_SIZE --epochs $EPOCHS --out_dir results/heavy_out_C1
echo "C1 finished!"

# C2: MinatoLoader + WorkerScheduler + Adaptive Timeout
echo ">>> [2/2] Running C2: MinatoLoader + WorkerScheduler + Adaptive Timeout <<<"
python run_scheduled_experiments.py --adaptive_timeout --batch_size $BATCH_SIZE --epochs $EPOCHS --out_dir results/heavy_out_C2
echo "C2 finished!"

echo ">>> Generating Plots <<<"
python plot_worker_scaling.py
python plot_timelines.py

echo "=========================================================="
echo " 3.3 Experiments Completed! Results in results/ "
echo "=========================================================="
