#!/bin/bash

timestamp=$(date +%d%m%Y%H%M%S)

echo "LAMBDA $lambda"
echo "protocol=$protocol"
echo "n_epochs=$n_epochs"
echo "batch_size=$batch_size"
echo "party=$party"
echo "ip_source=$ip_source"
echo "port $port"
echo "data_path $data_path"
echo "save_folder $save_folder"
echo "process_labels $process_labels"

results="logs/results_$timestamp.txt"

python3  Step1_Preprocess.py $data_path $save_folder $process_labels

/opt/app/MP-SPDZ/Scripts/../$protocol $party lr_training-$n_examples_train-$n_examples_test-$n_features-$n_epochs-$batch_size-$lambda -pn $port -h $ip_source > $results

python3 Step5_classification.py $path_to_model $test_data_folder $prediction_file_path $process_labels 

