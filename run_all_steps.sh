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
echo "task_training_testing=$task_training_testing"

if [ $task_training_testing == "training" ]
then
    c_rehash Player-Data

    python3  Step1_Preprocess.py $data_path $save_folder $process_labels $party

    ./gen_files_mpc.sh # generate Input-P$party-0

    ./compile.py -R 64 -Y lr_training $N1 $N2 $num_features $num_epochs $batch_size $lambda $epsilon

    if [ $party -eq '0' ]
    then
        results="$save_folder/results$timestamp.txt" 
        ./$protocol $party lr_training-$N1-$N2-$num_features-$num_epochs-$batch_size-$lambda-$epsilon -pn $port -h $ip_source > $results
        cat $results | grep Bias | sed 's/Bias //g' > $path_to_model
        cat $results | grep Weight | sed 's/Weight //g' | tr -d "\n" >> $path_to_model
    else
        ./$protocol $party lr_training-$N1-$N2-$num_features-$num_epochs-$batch_size-$lambda-$epsilon -pn $port -h $ip_source
    fi
else
    # "testing"
    python3  Step1_Preprocess.py $data_path $test_data_folder $process_labels $party
    python3 Step5_classification.py $path_to_model $test_data_folder $prediction_file_path $process_labels
fi

