#!/bin/bash

rm ../spdz/Programs/Source/fair.mpc
mv runLR.mpc ../spdz/Programs/Source/runLR.mpc
./../spdz/compile.py -R 64 runLR