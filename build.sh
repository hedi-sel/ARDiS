#!/bin/bash
cd $(dirname $0)

cd build/ &&
rm -f CMakeCache.txt &&
make -j 8