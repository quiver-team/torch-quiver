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

# python3 benchmarks/ogbn_products_sage/origin.py
python3 benchmarks/ogbn_products_sage/cuda_sample.py >train.out.txt 2>train.err.txt
# python3 benchmarks/ogbn_products_sage/cuda_sample.py

notify "finished $0"
