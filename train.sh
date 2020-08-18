#!/bin/sh
set -e

notify() {
    if [ -f ~/.slack/notify ]; then
        ~/.slack/notify "$@"
    fi
}

p=$(basename $(pwd))
notify "Running $0"

# export ENABLE_TRACE=1

. ./scripts/measure.sh

mkdir -p logs

train_origin() {
    python3 benchmarks/ogbn_products_sage/origin.py >logs/train.origin.out.txt 2>logs/train.origin.err.txt
}

train_cuda_sample() {
    python3 benchmarks/ogbn_products_sage/cuda_sample.py >logs/train.out.txt 2>logs/train.err.txt
}

measure train_origin
measure train_cuda_sample

notify "finished $0"
