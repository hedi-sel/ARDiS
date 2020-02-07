#!/bin/bash
cd $(dirname $0)

./build.sh &&
python3 src/pythonScripts/test.py
