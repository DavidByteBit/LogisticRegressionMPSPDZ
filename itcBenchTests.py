# date: Aug 12, 2021
# name: Sikha
# description: Training ML models on IDASH2021, Track 3 data

# DP LR

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import preprocessing




##################################################################################

def preprocess(dirty_df):
    dirty_df = dirty_df.drop(['patient_id','cohort_type'], axis = 1)
    target_map = {u'1': 1, u'0': 0}
    dirty_df['__target__'] = dirty_df['cohort_flag'].map(str).map(target_map)
    dirty_df = dirty_df.drop(['cohort_flag'], axis = 1)
    clean_X = dirty_df.drop('__target__', axis=1)
    clean_X = clean_X.to_numpy()
    clean_X = preprocessing.normalize(clean_X, norm='l2')
    clean_y = np.array(dirty_df['__target__'])
    return clean_X, clean_y

def getnoise(d,n,epsilon=1,mylambda=1):
    ## Noise generation
    # Step 1: Generating d dimension normal vector
    x_guass_vec = np.random.normal(0., 1, d)
    x_guass_vec = x_guass_vec.reshape(1,-1)
    # Step 2: Normalize with l2 norm
    x_guass_vec_norm = preprocessing.normalize(x_guass_vec, norm='l2')
    # Step 3: Sample gamma from gamma distribution
    scale = 2/(n * epsilon * mylambda)
    gamma_val = np.random.gamma(d, scale)
    # Step 4: mul gamma to Step 2
    noise_to_add = x_guass_vec_norm * gamma_val
    return noise_to_add


##################################################################################


# Load the data
df1 = pd.read_csv('data/alice_data.csv')
df2 = pd.read_csv('data/bob_data.csv')

print(df1.shape)

X1, y1 = preprocess(df1)
X2, y2 = preprocess(df2)

#LRresults = np.zeros((5, 5))
accP1_dict = {}
accP2_dict = {}
accAll_dict = {}
accAvg_dict = {}

accP1DP_dict = {}
accP2DP_dict = {}
accAllDP_dict = {}
accAvgDP_dict = {}
accAvgDP2_dict = {}
accAvgDP3_dict = {}

kf1 = KFold(n_splits=5,shuffle = True,random_state = 42)
kf2 = KFold(n_splits=5,shuffle = True,random_state = 42)

epsilon = 1
mylambda = 1
i = 0

for (train1_indices, test1_indices), (train2_indices, test2_indices) in zip(kf1.split(X1,y1),kf2.split(X2,y2)):
    print("FOLD ", i+1)

    X1_train, X1_test = X1[train1_indices,:], X1[test1_indices,:]
    y1_train, y1_test = y1[train1_indices], y1[test1_indices]
    X2_train, X2_test = X2[train2_indices,:], X2[test2_indices,:]
    y2_train, y2_test = y2[train2_indices], y2[test2_indices]


    X_train = np.append(X1_train,X2_train,axis=0)
    y_train = np.append(y1_train,y2_train)
    X_test = np.append(X1_test,X2_test,axis=0)
    y_test = np.append(y1_test,y2_test)


    ########## Train and test logistic regression models #################

    #### Without Noise

    clf1 = SGDClassifier(tol=0.001, loss='log', random_state=10000, max_iter=300, alpha=0.001)
    clf2 = SGDClassifier(tol=0.001, loss='log', random_state=10000, max_iter=300, alpha=0.001)
    clf = SGDClassifier(tol=0.001, loss='log', random_state=10000, max_iter=300, alpha=0.001)
    clf_dummy = SGDClassifier(tol=0.001, loss='log', random_state=10000, max_iter=300, alpha=0.001)

    clf1.fit(X1_train, y1_train)
    accP1 = accuracy_score(y_test, clf1.predict(X_test))

    clf2.fit(X2_train, y2_train)
    accP2 = accuracy_score(y_test, clf2.predict(X_test))

    clf.fit(X_train, y_train)
    accALL = accuracy_score(y_test, clf.predict(X_test))

    clf_dummy.fit(X_train, y_train)
    accdummy = accuracy_score(y_test, clf.predict(X_test))

    avg_coeff = (clf1.coef_ + clf2.coef_)/2
    avg_inter = (clf1.intercept_ + clf2.intercept_)/2
    clf_dummy.coef_ = avg_coeff
    clf_dummy.intercept_ = avg_inter
    accAVG = accuracy_score(y_test,clf_dummy.predict(X_test))

    accP1_dict[i] = accP1
    accP2_dict[i] = accP2
    accAll_dict[i] = accALL
    accAvg_dict[i] = accAVG

    #### With noise
    num_runs = 1000 #1000

    accAvgDP2 = 0
    for j in range(num_runs):
        d1 = len(clf1.coef_[0])+len(clf1.intercept_)
        # Step 5: add noise to weights
        noise_to_add = getnoise(d1,len(X1_train),epsilon,mylambda)
        weights = np.concatenate((clf1.coef_[0],clf1.intercept_))
        noisy_weights = weights + noise_to_add[0]
        noisy_coeff1 = noisy_weights[:-1].reshape(1,-1)
        noisy_intercept1 = np.array([noisy_weights[-1]])

        d2 = len(clf2.coef_[0])+len(clf2.intercept_)
        # Step 5: add noise to weights
        noise_to_add = getnoise(d2,len(X2_train),epsilon,mylambda)
        weights = np.concatenate((clf2.coef_[0],clf2.intercept_))
        noisy_weights = weights + noise_to_add[0]
        noisy_coeff2 = noisy_weights[:-1].reshape(1,-1)
        noisy_intercept2 = np.array([noisy_weights[-1]])

        avg_coeff = (noisy_coeff1 + noisy_coeff2)/2
        avg_inter = (noisy_intercept1 + noisy_intercept2)/2

        clf_dummy.coef_ = avg_coeff
        clf_dummy.intercept_ = avg_inter
        accAvgDP2 = accAvgDP2 + accuracy_score(y_test,clf_dummy.predict(X_test))
    accAvgDP2_dict[i] = accAvgDP2/num_runs

    i = i + 1


# Printing the averages over the 5 folds


print("Results for SGD")
np.set_printoptions(precision=2)
print("Acc P1: ",np.mean(list(accP1_dict.values())))
print("Acc P2: ",np.mean(list(accP2_dict.values())))
# print("Acc All: ",np.mean(list(accAll_dict.values())))
# print("Acc Averaged: ",np.mean(list(accAvg_dict.values())))
# print("Acc P1+DP: ",np.mean(list(accP1DP_dict.values())))
# print("Acc P2+DP: ",np.mean(list(accP2DP_dict.values())))
# print("Acc All+DP: ",np.mean(list(accAllDP_dict.values())))
# print("Acc Averaged+DP: ",np.mean(list(accAvgDP_dict.values())))
print("Acc Avergaed(P1+DP,P2+DP): ", np.mean(list(accAvgDP2_dict.values())))
#print("Acc Avergaed(P1+DP,P2+DP) -prev: ",np.mean(list(accAvgDP3_dict.values())))

