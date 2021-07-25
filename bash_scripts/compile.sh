#!/bin/bash

rm ../spdz/Programs/Source/fair.mpc
cp runLR.mpc ../spdz/Programs/Source/runLR.mpc
./../spdz/compile.py -R 72 runLR