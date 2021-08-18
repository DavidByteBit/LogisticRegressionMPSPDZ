docker stop $(docker ps)

docker rmi $(docker images) -f

docker build . -t ricardojmmaia/idash2021
