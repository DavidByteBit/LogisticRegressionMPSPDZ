#!/bin/bash

rm ../spdz/Programs/Compiler/lr.py
cp lr.py ../spdz/Programs/Compiler/lr.py

rm ../spdz/Programs/Source/runLR.mpc
cp runLR.mpc ../spdz/Programs/Source/runLR.mpc
./../spdz/compile.py -R 64 -Z 2 runLR