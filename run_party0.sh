
docker run -p 1728:1728 -p 1729:1729 -e port=1729 -e party=0 -e ip_source="192.168.0.13" \
   -e protocol="replicated-ring-party.x"  \
   -e process_labels="true" \
   -e data_path="/opt/app/MP-SPDZ/data/IDASH-ALICE/party_2.csv" \
   -e save_folder="/opt/app/MP-SPDZ/save/" \
   -e path_to_model="/opt/app/MP-SPDZ/save/"  \
   -e test_data_folder="/opt/app/MP-SPDZ/save/" \
   -e prediction_file_path="/opt/app/MP-SPDZ/save/" \
   -e epsilon=1 -e lambda=1 -e num_epochs=300 -e batch_size=128 \
   -e N1=832 -e N2=832 -e num_features=1784 \
   -v $(pwd)/save:/opt/app/MP-SPDZ/save \
   -v $(pwd)/PPMLRobots:/opt/app/MP-SPDZ/data \
   ricardojmmaia/idash2021 sh /opt/app/MP-SPDZ/run_all_steps.sh 
