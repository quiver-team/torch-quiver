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

run_origin() {
    python3 origin.py
}

run_quiver() {
    python3 cuda.py
}

measure run_origin
echo
measure run_quiver
