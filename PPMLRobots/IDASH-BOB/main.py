#! /usr/bin/python

#######################################################################
# Track: Track III - Confidential Computing
# Team: PPMLRobots
# Contact: sikha@uw.edu, _mence40@uw.edu, mdecock@uw.edu
# Description: Main file checks the command line arguments
#              and makes function call to train models with DP or
#              generate inferences on differentially private model.
#######################################################################

import pathlib
import sys
import traceback

sys.path.append('/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages')

import datetime
import trainIDASH
import inferIDASH
import argparse
import processData

# Main
# Arguments - input_filepath, train/test, epsilon, output_filepath
if __name__ == '__main__':
    try:
        print("<=== TRACK III, TEAM: PPMLRobots ===>")
        print()
        print("<=== Processing Started @", datetime.datetime.now(), " ===>")
        print("1. Parsing Arguments")
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--input_path',
                            required=True,
                            type=pathlib.Path,
                            default='./party.csv',
                            dest="input_path",
                            metavar="<absolute_path_input>",
                            help="Absolute path of the input file - data for train or test")
        parser.add_argument('-t', '--task',
                            required=True,
                            type=str,
                            default='train',
                            dest="task",
                            metavar="train/test",
                            help="Task to be performed by program - whether a training program or testing program")
        parser.add_argument('-m', '--model',
                            required=True,
                            type=pathlib.Path,
                            default='./federatedClassifier.joblib',
                            dest="model_path",
                            metavar="path to model",
                            help="Specify path of trained model - if train path to store trained model, else path of the trained model for inference.")
        parser.add_argument('-e', '--epsilon',
                            required=False,
                            type=float,
                            default=1.0,
                            dest="epsilon",
                            metavar="<epsilon>",
                            help="epsilon in DP definition")
        parser.add_argument('-o', '--output_path',
                            required=False,
                            type=pathlib.Path,
                            default='./',
                            dest="output_path",
                            metavar="<absolute_path_output>",
                            help="Absolute path to store output file - if test, path to store inference ")
        parser.add_argument('-ip', '--my_ip',
                            required=False,
                            type=str,
                            default='127.0.0.1',
                            dest="my_ip",
                            metavar="<this-ip-address>",
                            help="Ip address of this machine")
        parser.add_argument('-oip', '--other_ip',
                            required=False,
                            type=str,
                            default='127.0.0.1',
                            dest="other_ip",
                            metavar="<other-ip-address>",
                            help="Ip address of other machine")
        parser.add_argument('-p', '--port',
                            required=False,
                            type=int,
                            default=3000,
                            dest="port",
                            metavar="<ip-port>",
                            help="PORT number, PORT will be dafault to 3000 ")

        args = parser.parse_args()
        print("2. Preprocess input data")
        print(args.input_path)
        preprocessor = processData.processData(data_path=args.input_path, task=args.task)
        X, y = preprocessor.load_data()
        print("3. Performing task")
        if (args.task == 'train'):
            print("Training")
            # myip = ipaddress.ip_address(args.my_ip)
            # oip = ipaddress.ip_address(args.other_ip)
            trainer = trainIDASH.trainFederated(X, y, myIP=args.my_ip, otherIP=args.other_ip,port=args.port, epsilon=args.epsilon
                                                )
            trainer.train_local_with_noise()
            trainer.create_final_model(args.model_path)
        if (args.task == 'test'):
            print("Testing")
            tester = inferIDASH.testFederated(X)
            print("3.1 Load Model")
            tester.load_classifier(args.model_path)
            print("3.2 Classify test data")
            tester.classify_data_local(X, args.output_path)
        print("<=== Processing Ended @", datetime.datetime.now(), " ===>")
    except:
        print("Some error occurred. Please contact team with below information")
        print("Unexpected error:", sys.exc_info()[0])
        print("Unexpected error:", sys.exc_info()[1])
        print(traceback.print_exc())
