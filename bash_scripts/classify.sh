#!/bin/bash

n_features=$1
alice_examples=$2
bob_examples=$3

rm ../spdz/Programs/Source/classifyLR.mpc
cp runLR.mpc ../spdz/Programs/Source/classifyLR.mpc
./../spdz/compile.py -R 64 -Z 2 classifyLR "$n_features" "$alice_examples" "$bob_examples"