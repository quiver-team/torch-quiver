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
./benchmarks/bench-pyg-product.py 2>err-new.txt >out-new.txt
notify "finished $0"
