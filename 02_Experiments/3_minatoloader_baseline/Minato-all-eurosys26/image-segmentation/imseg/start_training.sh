#!/bin/bash

SCRIPT_DIR=$(dirname -- "$(readlink -f -- "$0")")
echo $SCRIPT_DIR

CURRENT_USER=$(whoami)
echo $CURRENT_USER
echo $SCRIPT_DIR
ls -ld "$SCRIPT_DIR"
mkdir -p "${SCRIPT_DIR}/output"
mkdir -p "${SCRIPT_DIR}/ckpts"

NUM_GPUS=${1:-3}
CONTAINER_NAME=${2:-train_imseg}
BATCH_SIZE=${3:-2}
DOCKER_MEMORY=${4:-""}

DOCKER_MEMORY_PARAM=

if [ ! -z "$DOCKER_MEMORY" ]; then
    DOCKER_MEMORY_PARAM="-m $((DOCKER_MEMORY * 1024 * 1024 * 1024))"
fi

sudo docker run -it --ipc=host --name="$CONTAINER_NAME" --rm --runtime=nvidia $DOCKER_MEMORY_PARAM \
  	-v /raid/data/imseg/raw-data/kits19/data/:/raw_data \
    -v .:/workspace \
    -v /raid/data/imseg/raw-data/kits19/preproc-data/:/data \
    -v "${SCRIPT_DIR}/output":/results \
    -v "${SCRIPT_DIR}/ckpts":/ckpts \
    unet3d:rahma bash


#-v /raid/data/unet3d/29gb-npy-prepp/:/data \
