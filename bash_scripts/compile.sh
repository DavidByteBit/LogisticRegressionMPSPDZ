#!/bin/bash

compile=$1

rm ../spdz/Programs/Source/runLR.mpc
cp runLR.mpc ../spdz/Programs/Source/runLR.mpc
./../spdz/compile.py -R 64 -Z 2 runLR