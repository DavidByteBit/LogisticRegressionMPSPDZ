
docker run -p 1730:1730 -e port=1729 -e party=1 -e ip_source="192.168.0.13" \
   -e data_path="data_path" -e save_folder="save_folder" \
   -e process_labels="true" \
   -e protocol="replicated-ring-party.x" -e lambda=1 -e n_epochs=10 \
   -e batch_size=128 \
   -v $(pwd)/logs:/opt/app/MP-SPDZ/logs \
   -v $(pwd)/results:/opt/app/MP-SPDZ/results \
   -v $(pwd)/PPMLRobots:/opt/app/MP-SPDZ/data \
   ricardojmmaia/idash2021 sh /opt/app/MP-SPDZ/run_all_steps.sh  &
