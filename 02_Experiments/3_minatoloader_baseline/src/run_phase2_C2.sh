#!/bin/bash

source "/home/mew/Desktop/mew/study/Master degree/thesis/02_Experiments/3_minatoloader_baseline/.venv_minato/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "/home/mew/Desktop/mew/study/Master degree/thesis/02_Experiments/3_minatoloader_baseline/src"

echo "Running C2: MinatoLoader + WorkerScheduler + Adaptive Timeout"
python run_scheduled_experiments.py --batch_size 2 --epochs 5 --out_dir heavy_out_C2 --adaptive_timeout
echo "C2 finished!"
