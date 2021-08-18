# date: August 17, 2021
# name: David
# description: Training ML models on IDASH2021, Track 3 data

# DP LR

import numpy as np
import pandas as pd
from sklearn import preprocessing
from numpy import savetxt
import sys


##################################################################################

def preprocess(dirty_df, has_labels):

    if has_labels:
        dirty_df = dirty_df.drop(['patient_id', 'cohort_type'], axis=1)
        target_map = {u'1': 1, u'0': 0}
        dirty_df['__target__'] = dirty_df['cohort_flag'].map(str).map(target_map)
        dirty_df = dirty_df.drop(['cohort_flag'], axis=1)
        clean_X = dirty_df.drop('__target__', axis=1)
        clean_X = clean_X.to_numpy()
        clean_X = preprocessing.normalize(clean_X, norm='l2')
        clean_y = np.array(dirty_df['__target__'])

        return clean_X, clean_y

    else:
        dirty_df = dirty_df.drop(['patient_id'], axis=1)
        clean_X = dirty_df.to_numpy()
        clean_X = preprocessing.normalize(clean_X, norm='l2')

        return clean_X


##################################################################################

# Load the data
data_path = sys.argv[1]
save_folder = sys.argv[2]
process_labels = sys.argv[3]

if process_labels.lower() == "true":
    process_labels = True
else:
    process_labels = False

print(process_labels)

df1 = pd.read_csv(data_path)

X1, y1 = preprocess(df1, process_labels)

if process_labels:
    savetxt(save_folder + "/features.csv", X1, delimiter=',', fmt='%s')
    savetxt(save_folder + "/labels.csv", y1, delimiter=',', fmt='%s')
else:
    savetxt(save_folder + "/features.csv", X1, delimiter=',', fmt='%s')
