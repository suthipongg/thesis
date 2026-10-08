#!/bin/bash

source "/home/mew/Desktop/mew/study/Master degree/thesis/02_Experiments/3_minatoloader_baseline/.venv_minato/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "======================================"
echo " Starting Phase 2 Scheduled Experiments"
echo "======================================"

cd "/home/mew/Desktop/mew/study/Master degree/thesis/02_Experiments/3_minatoloader_baseline/src"

# Define parameters
EPOCHS=5
BATCH_SIZE=2

# C1: MinatoLoader + WorkerScheduler
echo "Running C1: MinatoLoader + WorkerScheduler"
python run_scheduled_experiments.py --batch_size $BATCH_SIZE --epochs $EPOCHS --out_dir heavy_out_C1
echo "C1 finished!"

# C2: MinatoLoader + WorkerScheduler + Adaptive Timeout
echo "Running C2: MinatoLoader + WorkerScheduler + Adaptive Timeout"
python run_scheduled_experiments.py --adaptive_timeout --batch_size $BATCH_SIZE --epochs $EPOCHS --out_dir heavy_out_C2
echo "C2 finished!"

echo "======================================"
echo " All Phase 2 Experiments Completed!"
echo "======================================"
