#!/bin/sh
set -e

cd $(dirname $0)/..

if [ ! -d KungFu ]; then
    git clone https://github.com/lsds/KungFu.git
fi

cd KungFu
. ./scripts/utils/measure.sh

reinstall() {
    # KUNGFU_ENABLE_NCCL=1
    pip3 install --no-index -U .
    rm setup.py
    ln -s setup_pytorch.py setup.py
    pip3 install --no-index -U .
}

measure reinstall
