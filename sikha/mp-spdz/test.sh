#!/bin/bash
  
echo "Importing training data"
fold=1
echo > Player-Data/Input-P0-0
input_file="train$fold.txt"
cat Player-Data/$input_file > Player-Data/Input-P0-0

n_examples_train=1369
n_features=1874
n_epochs=200
batch_size=128
lambda=1
n_times=5


results="wts_$n_examples_train-$n_features-$n_epochs-$batch_size-$lambda.txt"
echo "Results for fold $fold in $results with regularization and min batch sgd"

for i in $(seq 1 $n_times)
do
    ./compile.py -Y -R 64 LRtrain $n_examples_train $n_features $n_epochs $batch_size $lambda >> $results
    Scripts/ring.sh LRtrain-$n_examples_train-$n_features-$n_epochs-$batch_size-$lambda >> $results
done

echo "Time $timestamp - $results" >> times.txt
cat $results | grep Time100 >> times.txt

cat times.txt
