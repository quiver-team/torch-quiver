#!/bin/sh
set -e

cd $(dirname $0)/..

export HOROVOD_GPU_OPERATIONS=NCCL
pip3 install -U horovod
