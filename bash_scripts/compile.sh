#!/bin/bash

compile=$1

rm ../spdz/Programs/Source/runLR.mpc
cp runLR.mpc ../spdz/Programs/Source/runLR.mpc
cp generate_noise.py ../spdz/Programs/Source/generate_noise.py
echo running ./../spdz/compile.py "$compile"
./../spdz/compile.py $compile