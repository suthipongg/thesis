#!/bin/bash

echo -e "Time(s)\tCPU(%)\tGPU(%)\tLGPU(%)\tGPU Power(W)"

start_time=$(date +%s)

while true; do
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))

    # Get CPU usage (User CPU %)
    cpuUsage=$(top -bn1 | awk '/Cpu/ {print $2}')

    # Get memory usage (in MB)
    memUsage=$(free -m | awk '/Mem/{print $3}')

    # Get GPU utilization and power draw
    gpuUsages=($(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits))
    gpuPowers=($(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits))

    if [ ${#gpuUsages[@]} -gt 0 ]; then
        lowest_usage=${gpuUsages[0]}
        lowest_index=0
        total_usage=0
        total_power=0
        count=0

        for i in "${!gpuUsages[@]}"; do
            if [ "${gpuUsages[i]}" -lt "$lowest_usage" ]; then
                lowest_usage="${gpuUsages[i]}"
                lowest_index=$i
            fi
        done

        # Remove the lowest usage from the array
        unset gpuUsages[lowest_index]
        unset gpuPowers[lowest_index]

        for i in "${!gpuUsages[@]}"; do
            total_usage=$(echo "$total_usage + ${gpuUsages[i]}" | bc)
            total_power=$(echo "$total_power + ${gpuPowers[i]}" | bc)
            count=$((count + 1))
        done

        if [ "$count" -gt 0 ]; then
            avg_usage=$(echo "scale=2; $total_usage / $count" | bc)
            avg_power=$(echo "scale=2; $total_power / $count" | bc)
        else
            avg_usage="0"
            avg_power="0"
        fi
    else
        avg_usage="0"
        lowest_usage="N/A"
        avg_power="0"
    fi

    echo -e "${elapsed_time}s\t${cpuUsage}\t${avg_usage}\t${lowest_usage}%\t${avg_power}"
    sleep 1
done
