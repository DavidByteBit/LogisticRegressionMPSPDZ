#!/bin/bash

rm ../spdz/Programs/Source/classifyLR.mpc
cp classifyLR.mpc ../spdz/Programs/Source/classifyLR.mpc
./../spdz/compile.py -R 64 -Z 2 classifyLR