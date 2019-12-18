#!/bin/bash
cd $(dirname $0)

cd build/ &&
make &&
cd .. &&
python3 src/test.py
