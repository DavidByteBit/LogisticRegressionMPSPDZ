#!/bin/bash

timestamp=$(date +%d%m%Y%H%M%S)

echo "N1=$N1"
echo "N2=$N2"
echo "lambda=$lambda"
echo "epsilon=$epsilon"
echo "protocol=$protocol"
echo "num_epochs=$num_epochs"
echo "num_features=$num_features"
echo "batch_size=$batch_size"
echo "party=$party"
echo "ip_source=$ip_source"
echo "port=$port"
echo "data_path=$data_path"
echo "save_folder=$save_folder"
echo "process_labels=$process_labels"
echo "path_to_model=$path_to_model" 
echo "test_data_folder=$test_data_folder" 
echo "prediction_file_path=$prediction_file_path"

results="logs/results_$timestamp.txt"

python3  Step1_Preprocess.py $data_path $save_folder $process_labels

/opt/app/MP-SPDZ/Scripts/../$protocol $party lr_training-$N1-$N2-$num_features-$num_epochs-$batch_size-$lambda-$epsilon -pn $port -h $ip_source > $results

python3 Step5_classification.py $path_to_model $test_data_folder $prediction_file_path $process_labels 
