#######################################################################
# Track: Track III - Confidential Computing
# Team: PPMLRobots
# Contact: sikha@uw.edu, _mence40@uw.edu, mdecock@uw.edu
# Description: Performs testing using the model
#
# Assumptions:
#            1. Party knows which classifier was used for training
#
#######################################################################
import sys
import traceback
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
import joblib


class testFederated:

    def __init__(self, X):
        self.X = X
        self.classifier = None

    def load_classifier(self, model_path):
        try:
            self.classifier = joblib.load(model_path)
        except:
            print("Some error occurred. Please contact team with below information")
            print("Unexpected error:", sys.exc_info()[0])
            print("Unexpected error:", sys.exc_info()[1])
            print(traceback.print_exc())

    def classify_data_local(self, X, output_path=Path('')):
        try:
            predicted_classes = self.classifier.predict(X)
            predicted_classes = np.array(predicted_classes)
            print("3.3 Store the results on disk")
            with open(Path.joinpath(output_path, 'predicted_output.txt'), 'w') as outfile:
                np.savetxt(outfile, predicted_classes, delimiter='\n', fmt="%d")
        except:
            print("Some error occurred. Please contact team with below information")
            print("Unexpected error:", sys.exc_info()[0])
            print("Unexpected error:", sys.exc_info()[1])
            print(traceback.print_exc())
