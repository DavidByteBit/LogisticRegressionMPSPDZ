#!/bin/bash

compile=$1

rm ../spdz/Programs/Source/runLR.mpc
cp runLR.mpc ../spdz/Programs/Source/runLR.mpc
echo running ./../spdz/compile.py "$compile"
./../spdz/compile.py $compile