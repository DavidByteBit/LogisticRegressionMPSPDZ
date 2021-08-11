#!/bin/bash

compile=$1

rm ../spdz/Programs/Source/runLR.mpc
cp runLR.mpc ../spdz/Programs/Source/runLR.mpc
echo ./../spdz/compile.py "$compile"