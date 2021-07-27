#!/bin/bash

rm ../spdz/Programs/Source/runLR.mpc
cp runLR.mpc ../spdz/Programs/Source/runLR.mpc
./../spdz/compile.py -B 64 -Z 2 runLR