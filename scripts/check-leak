#!/bin/sh
set -e

rebuild() {
    ./configure
    make -j8
}

rebuild

valgrind --leak-check=full --show-leak-kinds=all ./bin/quiver-example >out.txt 2>err.txt
