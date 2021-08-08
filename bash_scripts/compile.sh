#!/bin/bash

rm ../spdz/Compiler/lr.py
cp lr.py ../spdz/Compiler/lr.py

rm ../spdz/Programs/Source/runLR.mpc
cp runLR.mpc ../spdz/Programs/Source/runLR.mpc
./../spdz/compile.py -R 64 -Z 2 runLR