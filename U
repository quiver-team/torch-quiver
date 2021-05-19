#!/bin/sh
set -e

notify() {
    if [ -f ~/.slack/notify ]; then
        ~/.slack/notify "$@"
    fi
}

. ./scripts/measure.sh

# [done] q $ pip3 install --no-index -U . took 42s
# measure pip3 install --no-index -U .

p=$(basename $(pwd))
notify "Building $p from $(hostname)"
measure python3 -m pip install --no-index -U .
notify "Built $p from $(hostname)"
