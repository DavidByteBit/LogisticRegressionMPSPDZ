docker pull ricardojmmaia/idash2021
 
docker run -p 1730:1730 -e port=1729 -e party=1 -e ip_source="192.168.0.13" \
   -e protocol="semi2k-party.x"  \
   -e process_labels="true" \
   -e data_path="/opt/app/MP-SPDZ/data/party_1.csv" \
   -e save_folder="/opt/app/MP-SPDZ/save/" \
   -e path_to_model="/opt/app/MP-SPDZ/save/weigths.txt"  \
   -e test_data_folder="/opt/app/MP-SPDZ/test/" \
   -e prediction_file_path="/opt/app/MP-SPDZ/test/predictions.txt" \
   -e epsilon=1 -e lambda=1 -e num_epochs=300 -e batch_size=128 \
   -e N1=882 -e N2=831 -e num_features=1874 \
   -e task_training_testing="training" \
   -v $(pwd)/save:/opt/app/MP-SPDZ/save \
   -v $(pwd)/data:/opt/app/MP-SPDZ/data \
   -v $(pwd)/Player-Data:/opt/app/MP-SPDZ/Player-Data \
   -v $(pwd)/test:/opt/app/MP-SPDZ/test \
   ricardojmmaia/idash2021 
