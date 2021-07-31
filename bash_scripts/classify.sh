#!/bin/bash

rm ../spdz/Programs/Source/classifyLR.mpc
cp runLR.mpc ../spdz/Programs/Source/classifyLR.mpc
./../spdz/compile.py -R 64 -Z 2 classifyLR