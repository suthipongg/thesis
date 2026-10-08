#!/bin/bash

SCRIPT_DIR=$( dirname -- "$( readlink -f -- "$0"; )" )
mkdir -p ${SCRIPT_DIR}/output
mkdir -p ${SCRIPT_DIR}/ckpts


CONTAINER_NAME=${2:train_imseg}

DOCKER_MEMORY=${4:-""}

DOCKER_MEMORY_PARAM=

if [ ! -z "$DOCKER_MEMORY" ]
then
	DOCKER_MEMORY_PARAM="-m ${DOCKER_MEMORY}g"
fi


docker run --ipc=host --name=$CONTAINER_NAME  --rm  $DOCKER_MEMORY_PARAM \
   -v /raid/data/imseg/raw-data/kits19/data/:/raw_data \
   -v /raid/data/unet3d/29gb-npy-prepp/:/data \
   -v ${SCRIPT_DIR}/output:/results \
   -v ${SCRIPT_DIR}/ckpts:/ckpts \
  unet3d:rahma mprof run python3 preprocess.py --data_dir /raw_data  --results_dir /data
