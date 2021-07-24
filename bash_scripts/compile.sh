#!/bin/bash

rm ../spdz/Programs/Source/fair.mpc
mv secure_fairness.py ../spdz/Programs/Source/fair.mpc
./../spdz/compile.py -R 64 runLR