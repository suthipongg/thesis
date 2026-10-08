#!/bin/bash


# This script monitors CPU, memory, and GPU usage in real-time.
echo -e "Time(s)\tCPU(%) GPU(%) LGPU(%)"

# Start monitoring indefinitely
start_time=$(date +%s)  # Get the start time in seconds


while true; do
    current_time=$(date +%s)  # Get the current time in seconds

    # Calculate the elapsed time in seconds
    elapsed_time=$((current_time - start_time))


    # Get CPU usage using mpstat (average CPU usage)
    #cpuUsage=$(mpstat 1 1 | awk '/Average:/ {print 100 - $12}')

    # Get CPU usage (for user CPU usage specifically, %usr)
    cpuUsage=$(top -bn1 | awk '/Cpu/ {print $2}')



    # Get memory usage (in MB)
    memUsage=$(free -m | awk '/Mem/{print $3}')

    # Get GPU utilization for all running GPUs and store in an array
    gpuUsages=($(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits))

    # Exclude the GPU with the lowest usage
    if [ ${#gpuUsages[@]} -gt 0 ]; then
        # Find the lowest GPU usage and its index
        lowest_usage=${gpuUsages[0]}
        lowest_index=0
        
        for i in "${!gpuUsages[@]}"; do
            if [ "${gpuUsages[i]}" -lt "$lowest_usage" ]; then
                lowest_usage="${gpuUsages[i]}"
                lowest_index=$i
            fi
        done

        # Remove the lowest usage from the array
        unset gpuUsages[lowest_index]

        # Calculate the average GPU utilization
        total=0
        count=0
        for usage in "${gpuUsages[@]}"; do
            total=$((total + usage))
            count=$((count + 1))
        done

        # Calculate the average if there are remaining GPUs
        if [ "$count" -gt 0 ]; then
            average=$((total / count))
            gpuUsage="${average}"
        else
            gpuUsage="0"
        fi
    else
        gpuUsage="0"
        lowest_usage="N/A"  # If no GPUs are found, set lowest usage to N/A
    fi

    # Print the usage including the time column, lowest GPU usage, and average GPU usage
    echo -e "${elapsed_time}s\t${cpuUsage}\t${gpuUsage}\t${lowest_usage}"
    sleep 1
done