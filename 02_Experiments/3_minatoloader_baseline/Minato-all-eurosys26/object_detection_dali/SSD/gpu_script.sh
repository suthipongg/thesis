#!/bin/bash

# This script monitors CPU, GPU, and Disk usage in real-time.
echo -e "Time(s)\tCPU(%)\tGPU(%)\tMem(GB)"

# Start monitoring indefinitely
start_time=$(date +%s)

while true; do
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))

    # Get CPU usage (user CPU percentage)
    cpuUsage=$(top -bn1 | awk '/Cpu/ {print $2}')
    memUsage=$(free -g | awk '/Mem/{print $3}')



    # Get GPU utilizations
    gpuUsages=($(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits))
    if [ ${#gpuUsages[@]} -gt 0 ]; then
        total=0
        count=0
        for usage in "${gpuUsages[@]}"; do
            total=$((total + usage))
            count=$((count + 1))
        done
        if [ "$count" -gt 0 ]; then
            gpuUsage=$((total / count))
        else
            gpuUsage="0"
        fi
    else
        gpuUsage="0"
    fi

    # Get Disk usage metrics
    # tps=$(iostat -d sdb 1 2 | awk 'NR > 7 {print $2}')
    # kB_read_s=$(iostat -d sdb 1 2 | awk 'NR > 7 {print $3}')
    # kB_wrtn_s=$(iostat -d sdb 1 2 | awk 'NR > 7 {print $4}')

    # Output all metrics
    echo -e "${elapsed_time}s\t${cpuUsage}\t${gpuUsage}\t${memUsage}"
    
    sleep 1
done
