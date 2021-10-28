#!/bin/sh
set -e

cd $(dirname $0)/..

CUDA_CAP=$(./auto/detect-cuda-cap)

echo "CUDA_CAP: $CUDA_CAP"

REQUIREMENTS=auto/cuda/$CUDA_CAP/requirements.torch.txt
if [ ! -f $REQUIREMENTS ]; then
    exit 1
fi

python3 -m pip install -r $REQUIREMENTS -U -f https://download.pytorch.org/whl/torch_stable.html
