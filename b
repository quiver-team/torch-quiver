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
./benchmarks/bench-pyg-product.py >logs/bench.out.txt 2>logs/bench.err.txt

notify "finished $0"
