#!/bin/sh
set -e

notify() {
    if [ -f ~/.slack/notify ]; then
        ~/.slack/notify "$@"
    fi
}

cd $(dirname $0)
export PYTHONPATH=$PWD/../../srcs/python

. ../../scripts/measure.sh

kungfu_run_flags() {
    echo -q
    echo -logdir logs
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

run_origin() {
    # single
    # python3 origin.py --runs 1 --epochs 1 --num-workers 1

    # kungfu distributed
    kungfu_run -np 4 python3 origin.py --runs 1 --epochs 1 --num-workers 1 --distribute kungfu

    # horovod distributed
    horovodrun -np 4 python3 origin.py --runs 1 --epochs 1 --num-workers 1 --distribute horovod
    true
}

run_quiver() {
    # single
    # python3 cuda.py --runs 1 --epochs 1

    # kungfu distributed
    kungfu_run -np 4 python3 cuda.py --runs 1 --epochs 1 --distribute kungfu

    # horovod distributed
    horovodrun -np 4 python3 cuda.py --runs 1 --epochs 1 --distribute horovod
}

measure run_origin
# measure run_quiver
