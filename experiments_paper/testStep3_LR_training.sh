program=Step3_LR_training

results=results.txt

Scripts/setup-ssl.sh 3
./compile.py -R 64 -Y $program 
Scripts/ring.sh $program > $results

process_labels="true"
data_path="Alice/data/party_0.csv" 
save_folder="Alice/save/"
path_to_model="Alice/save/weights.txt"
test_data_folder="Alice/test/"
prediction_file_path="Alice/test/predictions.txt" 

cat $results | grep Bias | sed 's/Bias //g' > $path_to_model
cat $results | grep Weight | sed 's/Weight //g' | tr -d "\n" >> $path_to_model

python3  Step1_Preprocess.py $data_path $test_data_folder $process_labels
python3 Step5_classification.py $path_to_model $test_data_folder $prediction_file_path $process_labels
