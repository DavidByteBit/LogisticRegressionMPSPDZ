party=0

docker run -p 1728:1728 -p 1729:1729 -e port=1729 -e party=0 \
   -e ip_source="192.168.0.13" -e protocol="replicated-ring-party.x"  \
   -e data_path="/opt/app/MP-SPDZ/data/IDASH-ALICE/party_2.csv" \
   -e save_folder="save_folder" -e process_labels="true" \
   -e lambda=1 -e n_epochs=10 -e batch_size=128 -e n_folds=5 \
   -v $pwd/PPMLRobots/:/opt/app/MP-SPDZ/data/
   -v $pwd/logs/:/opt/app/MP-SPDZ/logs/
   ricardojmmaia/idash2021 sh /opt/app/MP-SPDZ/run_all_steps.sh 
