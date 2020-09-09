#!/bin/sh
set -e

notify() {
    if [ -f ~/.slack/notify ]; then
        ~/.slack/notify "$@"
    fi
}

export PYTHONPATH=$PWD/srcs/python

kungfu_run_flags() {
    echo -q
    echo -logdir logs
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

main() {
    for nw in $(echo 16 15 14 13 12 11 10 9 8 7); do
        for np in $(echo 1 2 3 4); do
            echo "np=$np, nw=$nw"
            kungfu_run -np $np python3 benchmarks/ogbn_products_sage/origin.py --num-workers $nw --runs 1 --epochs 2
            echo
        done
    done
}

notify "running $0"
main
notify "done $0"

# extract log:
# for line in open('out.log'):
#     line = line.strip()
#     if 'np=' in line:
#         print(line)
#     if 'train one epoch took' in line:
#         print(line)
