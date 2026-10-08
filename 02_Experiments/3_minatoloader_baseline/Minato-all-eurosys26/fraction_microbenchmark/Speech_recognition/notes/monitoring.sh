#!/usr/bin/env bash
# !/bin/bash
# This script monitors CPU and memory usage

echo "CPU     GPU"

var=$(pidof time)

while [ -z "$var" ]
do
	var=$(pidof time)
done

while [ ! -z "$(pidof time)" ]
#while :
do 
  # Get the current usage of CPU and memory
  cpuUsage=$(top -bn1 | awk '/Cpu/ { print $2}')
  memUsage=$(free -m | awk '/Mem/{print $3}')
  gpuUsage=$(nvidia-smi | tail -n +10 | awk '{print $13}' | sed -n 1p)

  # Print the usage
  # echo "CPU Usage: $cpuUsage%"
  # echo "GPU Usage: $gpuUsage"
  #echo "Memory Usage: $memUsage MB"

  echo "$cpuUsage%	$gpuUsage"
 
  # Sleep for 1 second
  sleep 1
  #usleep 1000
done
