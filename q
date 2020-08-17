#!/bin/sh
set -e

notify() {
    if [ -f ~/.slack/notify ]; then
        ~/.slack/notify "$@"
    fi
}

. ./scripts/measure.sh

quick_build() {
    ./configure
    make -j 8
}

p=$(basename $(pwd))
notify "Building $p from $(hostname)"

git add -A
measure quick_build
git clean -fdx

# [done] q $ pip3 install --no-index -U . took 42s
measure pip3 install --no-index -U .

measure pytest tests
measure python3 examples/e1.py
notify "Built $p from $(hostname)"
