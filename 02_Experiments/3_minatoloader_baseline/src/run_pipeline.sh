#!/bin/bash

cd "/home/mew/Desktop/mew/study/Master degree/thesis/02_Experiments/3_minatoloader_baseline/src"

echo "======================================"
echo " STARTING FULL PIPELINE (PHASE 1 + 2) "
echo "======================================"

bash run_all.sh
bash run_phase2.sh

echo "======================================"
echo " PIPELINE COMPLETED SUCCESSFULLY! "
echo "======================================"
