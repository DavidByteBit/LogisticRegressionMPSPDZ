# date: July 22, 2021
# name: Martine De Cock
# description: Training ML models on IDASH2021, Track 3 data

# DP LR

import numpy as np
import pandas as pd
from sklearn import preprocessing
from numpy import savetxt
import sys


##################################################################################

def preprocess(dirty_df):
    dirty_df = dirty_df.drop(['patient_id', 'cohort_type'], axis=1)
    target_map = {u'1': 1, u'0': 0}
    dirty_df['__target__'] = dirty_df['cohort_flag'].map(str).map(target_map)
    dirty_df = dirty_df.drop(['cohort_flag'], axis=1)
    clean_X = dirty_df.drop('__target__', axis=1)
    clean_X = clean_X.to_numpy()
    clean_X = preprocessing.normalize(clean_X, norm='l2')
    clean_y = np.array(dirty_df['__target__'])

    return clean_X, clean_y

##################################################################################

# Load the data
data_path = sys.argv[1]
save_folder = sys.argv[2]

df1 = pd.read_csv(data_path)

X1, y1 = preprocess(df1)

savetxt(save_folder + "/train_features.csv", X1, delimiter=',', fmt='%s')
savetxt(save_folder + "/train_labels.csv", y1, delimiter=',', fmt='%s')

