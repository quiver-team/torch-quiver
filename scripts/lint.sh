#!/bin/sh
set -e

cd $(dirname $0)/..

list_py_srcs() {
    find ./benchmarks -type f | grep .py$
    find ./srcs/python -type f | grep .py$
}

fmt_py() {
    # autoflake -i --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys $1
    # autoflake -i $1
    # isort -y $1
    yapf -i $1
}

fmt_all_py() {
    for src in $(list_py_srcs); do
        echo "fmt_py $src"
        fmt_py $src
    done
}

# TODO: also format c++ code
fmt_all_py
