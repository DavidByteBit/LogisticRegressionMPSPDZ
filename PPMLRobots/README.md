
# SUBMISSION to IDASH TRACK III : Confidential Computing 
--------------------
## Overview

The federated setup assumes two parties - Alice and Bob. Bob holds his training data in <party_1.csv> and Alice holds her training data in <party_2.csv>. They locally perform training of a machine learning model and add noise to the model parameters at their end. They then exchange the noisy parameters, and perform local avergaing of the noisy model parameters. At the end of this federated setup, each party holds the aggregated trained model which is differentially private.

## Project Structure
##### Both Alice and Bob have the following files with them
- main.py - Script that accepts the arguments and initiates the process
- processData.py - Script to preprocess train and test data - generating l2 norm data
- trainIDASH.py - Script to locally train a model, add noise and aggregate the models
- inferIDASH.py - Script to locally infer the test set with the aggregated model
- party_<i>.csv - Sample training/testing files
- numpysocket.py - Script to send and recieve numpy objects
- Dockerfile - To launch the dockers with required installations and files
##### System Requirements
- python3
- numpy
- pandas
- sklearn
- joblib
- Networking

## Program
1. Ensure that both the nodes have all requirements installed or have been setup using the docker.
2. Ensure that both the nodes are connected. Note the ip address of each node. 

The main program accepts arguments as shown below:
```sh
usage: main.py [-h] -i <absolute_path_input> -t train/test -m path to model -e <epsilon> [-o <absolute_path_output>] -ip
               <this-ip-address> -oip <other-ip-address> [-p <ip-port>]

optional arguments:
  -h, --help            show this help message and exit
  -i <absolute_path_input>, --input_path <absolute_path_input>
                        Absolute path of the input file - data for train or test
  -t train/test, --task train/test
                        Task to be performed by program - whether a training program or testing program
  -m path to model, --model path to model
                        Specify path of trained model - if train path to store trained model, else path of the trained
                        model for inference.
  -e <epsilon>, --epsilon <epsilon>
                        epsilon in DP definition
  -o <absolute_path_output>, --output_path <absolute_path_output>
                        Absolute path to store output file - if test, path to store inference
  -ip <this-ip-address>, --my_ip <this-ip-address>
                        Ip address of this machine
  -oip <other-ip-address>, --other_ip <other-ip-address>
                        Ip address of other machine
  -p <ip-port>, --port <ip-port>
                        PORT number, PORT will be dafault to 3000
```

## Training the model
To train a model in federated setup with 2 nodes:
- Mandatory parameters for training : -i -t -m -e -ip -oip
- Optional parameters for training : -o -p
- On Alice-node:
```sh
python3 main.py -i <abs_path_of_train_data> -m <abs_path_to_save_model> -e <value_of_epsilon> -t train -ip <alice-ip> -oip <bob-ip> -p <port>
```
- On Bob-node:
```sh
python3 main.py -i <abs_path_of_train_data> -m <abs_path_to_save_model> -e <value_of_epsilon> -t train -ip <bob-ip> -oip <alice-ip> -p <port>
```

## Testing the model
To test the model on a node that has the final model:
- Mandatory parameters for testing : -i -t -m 
- Optional parameters for testing : -o 
```sh
python3 main.py -i <abs_path_of_test_data> -m <abs_path_of_saved_model> -o <abs_path_to_save_predicted_results>
```
## TODO
Above is okay for submission, Things to make evaluation more easier
- Create entry point script to launch docker and run program with parameters
- Use threading to have the same code for (server and client) considering the nodes as a part of distributed system
