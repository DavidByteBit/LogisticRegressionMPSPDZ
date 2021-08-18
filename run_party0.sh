party=0

docker run -p 1728:1728 -p 1729:1729 -e port=1729 -e party=0 \
   -e ip_source="192.168.0.13" -e protocol="semi2k-party.x"  \
   -e data_path="data_path" -e save_folder="save_folder" -e process_labels="True" \
   -e lambda=1 -e n_epochs=10 -e batch_size=128 -e n_folds=5 \
   ricardojmmaia/idash2021 sh /opt/app/MP-SPDZ/run_all_steps.sh 
