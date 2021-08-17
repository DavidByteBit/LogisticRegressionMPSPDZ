#######################################################################
# Track: Track III - Confidential Computing
# Team: PPMLRobots
# Contact: sikha@uw.edu, _mence40@uw.edu, mdecock@uw.edu
# Description: Performs local training on data and adds noise to generated model
#              and runs federated averaging with other parties
# Assumptions:
#            1. Both parties know the protocol to follow
#            2. Both parties will use the same classifier with agreed parameters
#######################################################################

import threading
from pathlib import Path

import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, SGDClassifier
import joblib
import socket
import sys
from _thread import *
import traceback
from numpysocket import NumpySocket
import time


class trainFederated:

    def __init__(self, X, y, myIP, otherIP, port=3000, epsilon=1.0):
        self.epsilon = epsilon
        self.mylambda = 1
        self.X = X
        self.y = y
        self.classifier = LogisticRegression(solver='liblinear', random_state=10000, max_iter=100, C=1 / self.mylambda)
        self.myIP = myIP
        self.otherIP = otherIP
        self.PORT = port
        self.noisy_weights_received = None

    def get_noise(self, d, n):
        try:
            ## Noise generation
            # Step 1: Generating d dimension normal vector
            x_guass_vec = np.random.normal(0., 1, d)
            x_guass_vec = x_guass_vec.reshape(1, -1)
            # Step 2: Normalize with l2 norm
            x_guass_vec_norm = preprocessing.normalize(x_guass_vec, norm='l2')
            # Step 3: Sample gamma from gamma distribution
            scale = 2 / (n * self.epsilon * self.mylambda)
            gamma_val = np.random.gamma(d, scale)
            # Step 4: mul gamma to Step 2
            noise_to_add = x_guass_vec_norm * gamma_val
            return noise_to_add
        except:
            print("Some error occurred. Please contact team with below information")
            print("Unexpected error:", sys.exc_info()[0])
            print("Unexpected error:", sys.exc_info()[1])
            print(traceback.print_exc())

    def train_local_with_noise(self):
        try:
            print("3.1 Train classifier locally")
            # clf_local = SGDClassifier(loss='log',random_state=10000, max_iter=50)
            self.classifier.fit(self.X, self.y)
            print("3.2 Add noise locally")
            d = len(self.classifier.coef_[0]) + len(self.classifier.intercept_)
            noise_to_add = self.get_noise(d, len(self.X))
            weights = np.concatenate((self.classifier.coef_[0], self.classifier.intercept_))
            noisy_weights = weights + noise_to_add[0]
            self.classifier.coef_ = noisy_weights[:-1].reshape(1, -1)
            self.classifier.intercept_ = np.array([noisy_weights[-1]])
        except:
            print("Some error occurred. Please contact team with below information")
            print("Unexpected error:", sys.exc_info()[0])
            print("Unexpected error:", sys.exc_info()[1])
            print(traceback.print_exc())


    def exchange_noisy_models(self):
        try:

            npSocket_client = NumpySocket()
            npSocket_server = NumpySocket()

            npSocket_client.startClient(self.PORT)
            noisy_weights_received2 = npSocket_client.recieveNumpy()

            npSocket_server.startServer(self.otherIP, self.PORT)
            noisy_weights_to_share = np.concatenate((self.classifier.coef_[0], self.classifier.intercept_))
            npSocket_server.sendNumpy(noisy_weights_to_share)

            print(type(noisy_weights_received2),' recieved of shape ',noisy_weights_received2.shape)

            npSocket_server.endServer()
            npSocket_client.endClient()
            return noisy_weights_received2


        except:
            print("Some error occurred. Please contact team with below information")
            print("Unexpected error:", sys.exc_info()[0])
            print("Unexpected error:", sys.exc_info()[1])
            print(traceback.print_exc())

    def create_final_model(self, output_path=Path('')):
        try:
            print("3.3 Exchange noisy weights")
            noisy_weights_received2 = self.exchange_noisy_models()
            print("3.4 Average the noisy models")
            noisy_coeff_received = noisy_weights_received2[:-1].reshape(1, -1)
            noisy_intercept_received = np.array([noisy_weights_received2[-1]])
            avg_coeff = (self.classifier.coef_ + noisy_coeff_received) / 2
            avg_inter = (self.classifier.intercept_ + noisy_intercept_received) / 2
            self.classifier.coef_ = avg_coeff
            self.classifier.intercept_ = avg_inter
            print("3.5 Store the model on disk")
            if (output_path):
                joblib.dump(self.classifier, Path.joinpath(output_path, 'federatedClassifier.joblib'))

        except Exception as e:
            print("Some error occurred. Please contact team with below information")
            print("Unexpected error:", sys.exc_info()[0])
            print("Unexpected error:", sys.exc_info()[1])
            print(traceback.print_exc())

    #### Attempt to have uniform nodes for as distributed system
    '''
    class Server:
        def __init__(self, myIP, myPORT):
            self.S_HOST = myIP
            self.S_PORT = myPORT

        def connect_and_send(self, data):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.S_HOST, self.S_PORT))
                s.listen()
                print('...Server Listening...')
                # while True:
                conn, addr = s.accept()
                print("Connected to ::", addr)
                conn.sendall(data)
                conn.close()
                s.close()
                print('...Server Closed...')

        def npServer(self,data):
            try:
                npSocket = NumpySocket()
                print("...Start sending data now...")
                npSocket.startServer(self.S_HOST, self.S_PORT)
                npSocket.sendNumpy(data)
                npSocket.endServer()
            except:
                traceback.print_exc()


    class Client:
        def __init__(self, serverIP, serverPORT):
            self.C_HOST = serverIP
            self.C_PORT = serverPORT

        def connect_and_recieve(self):
            data = None
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                #s.setblocking(0)
                print('...Connecting to Server...')
                s.connect((self.C_HOST, self.C_PORT))
                data = s.recv(1024)
                print('...Receive data...', data.shape)
                s.close()
                print('...Closing Client...')
            return data

        def npClient(self):
            try:
                npSocket = NumpySocket()
                npSocket.startClient(self.C_HOST,self.C_PORT)
                data = npSocket.recieveNumpy()
                npSocket.endClient()
                return data
            except:
                traceback.print_exc()

    def share_noisy_model(self):
        try:
            noisy_weights_to_share = np.concatenate((self.classifier.coef_[0], self.classifier.intercept_))
            my_server = self.Server(myIP=self.myIP, myPORT=self.PORT)
            #my_server.connect_and_send(noisy_weights_to_share)
            my_server.npServer(noisy_weights_to_share)

        except:
            print("Some error occurred. Please contact team with below information")
            print("Unexpected error:", sys.exc_info()[0])

    def recieve_noisy_model(self):
        try:
            my_client = self.Client(serverIP=self.otherIP, serverPORT=self.PORT+1)
            #noisy_weights_received = my_client.connect_and_recieve()
            noisy_weights_received = my_client.npClient()
            print(type(noisy_weights_received),' recieved of shape ',noisy_weights_received.shape)
            return noisy_weights_received
        except:
            print("Some error occurred. Please contact team with below information")
            print("Unexpected error:", sys.exc_info()[0])

    def npClient_outer(self):
        npSocket_client = NumpySocket()
        npSocket_client.startClient(self.PORT)
        self.noisy_weights_received = npSocket_client.recieveNumpy()
        print(type(self.noisy_weights_received),' recieved of shape ',self.noisy_weights_received.shape)
        npSocket_client.endClient()

    def exchange_noisy_models(self):
        try:
            thread_client = threading.Thread(target=self.npClient_outer())
            thread_client.start()
            #npSocket_client = NumpySocket()
            npSocket_server = NumpySocket()
            #npSocket_client.startClient(self.PORT)
            npSocket_server.startServer(self.otherIP, self.PORT+1)
            noisy_weights_to_share = np.concatenate((self.classifier.coef_[0], self.classifier.intercept_))
            npSocket_server.sendNumpy(noisy_weights_to_share)
            #noisy_weights_received = npSocket_client.recieveNumpy()
            #print(type(noisy_weights_received),' recieved of shape ',noisy_weights_received.shape)
            npSocket_server.endServer()
            #npSocket_client.endClient()
            #return self.noisy_weights_received
            thread_client.join()

        except:
            print("Some error occurred. Please contact team with below information")
            print("Unexpected error:", sys.exc_info()[0])
            print("Unexpected error:", sys.exc_info()[1])
            print(traceback.print_exc())
        '''


