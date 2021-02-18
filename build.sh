#!/bin/bash
cd $(dirname $0)

cd build/ &&
cmake .. &&
rm -f CMakeCache.txt &&
make -j 8