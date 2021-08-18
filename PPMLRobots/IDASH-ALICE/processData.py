#######################################################################
# Track: Track III - Confidential Computing
# Team: PPMLRobots
# Contact: sikha@uw.edu, _mence40@uw.edu, mdecock@uw.edu
# Description: Loads and generates the dataframe for training or inference
#              and makes function call to train models with DP or
#              generate inferences on differentially private model.
#######################################################################
import traceback

import numpy as np
import pandas as pd
from sklearn import preprocessing
import sys

class processData:

    def __init__(self, data_path='./party.csv',task='train'):
        self.data_path = data_path
        self.task = task

    def pre_process_data(self,dirty_df):
        try:
            
            if self.task == 'train':
                print('Train Program - Dropping patient id,cohort_type')
                dirty_df = dirty_df.drop(['patient_id','cohort_type'], axis = 1)
                target_map = {u'1': 1, u'0': 0}
                dirty_df['__target__'] = dirty_df['cohort_flag'].map(str).map(target_map)
                dirty_df = dirty_df.drop(['cohort_flag'], axis = 1)
                clean_X = dirty_df.drop('__target__', axis=1)
                print(clean_X.shape)
                clean_y = np.array(dirty_df['__target__'])
            else:
                print('Test Program - Dropping only patient id')
                clean_X = dirty_df.drop(['patient_id'], axis = 1)
                print(clean_X.shape)
                #clean_X = dirty_df.drop(['cohort_flag'], axis = 1)
                clean_y = np.array([])
            clean_X = clean_X.to_numpy()
            clean_X = preprocessing.normalize(clean_X, norm='l2')
            return clean_X, clean_y
        except:
            print("Some error occurred. Please contact team with below information")
            print("Unexpected error:", sys.exc_info()[0])
            print("Unexpected error:", sys.exc_info()[1])
            print(traceback.print_exc())

    def load_data(self):
        try:
            #assert self.suffix == '.csv'
            print("2.1 Loading data")
            df = pd.read_csv(self.data_path)
            print("Found data of shape ", df.shape)
            print("2.2 Preprocessing data")
            X, y = self.pre_process_data(df)
            print("Pre-processed data shape ", X.shape)
            return X, y
        except:
            print("Some error occurred. Please contact team with below information")
            print("Unexpected error:", sys.exc_info()[0])
            print("Unexpected error:", sys.exc_info()[1])
            print(traceback.print_exc())





