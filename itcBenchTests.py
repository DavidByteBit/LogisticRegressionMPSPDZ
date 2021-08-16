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
import lightgbm as lgb
# import category_encoders as ce



##################################################################################

def preprocess(dirty_df):
    dirty_df = dirty_df.drop(['patient_id','cohort_type'], axis = 1)
    target_map = {u'1': 1, u'0': 0}
    dirty_df['__target__'] = dirty_df['cohort_flag'].map(str).map(target_map)
    dirty_df = dirty_df.drop(['cohort_flag'], axis = 1)
    clean_X = dirty_df.drop('__target__', axis=1)
    #m_est = ce.m_estimate.MEstimateEncoder(cols=list(clean_X.columns), m=10)
    #m_est_fit = m_est.fit(clean_X,dirty_df['__target__'])
    #clean_X = m_est_fit.transform(clean_X)
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

#print(df1.shape)

X1, y1 = preprocess(df1)
X2, y2 = preprocess(df2)

#LRresults = np.zeros((5, 5))
accP1_dict = {}
accP2_dict = {}
accAll_dict = {}
accAvg_dict = {}
#accLGBAvg_dict = {}

accP1DP_dict = {}
accP2DP_dict = {}
accAllDP_dict = {}
accAvgDP_dict = {}
accAvgDP2_dict = {}
accAvgDP3_dict = {}

kf1 = KFold(n_splits=5,shuffle = True,random_state = 42)
kf2 = KFold(n_splits=5,shuffle = True,random_state = 42)

epsilon = 1
mylambda_ser = np.linspace(1, 1, 1, endpoint=True)
print(mylambda_ser)
acc_plot_avg_9 = []
lambdas_plot = []
var_plot_avg_9 = []
#np.logspace(1,100,num=100)
for mylambda in mylambda_ser:
    i = 0
    mylambda = int(round(mylambda))
    print(mylambda)
    for (train1_indices, test1_indices), (train2_indices, test2_indices) in zip(kf1.split(X1,y1),kf2.split(X2,y2)):
        #print("FOLD ", i+1)

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

        #clf1 = SGDClassifier(loss='log',random_state=10000, max_iter=50,alpha=mylambda)
        #clf2 = SGDClassifier(loss='log',random_state=10000, max_iter=50)
        #clf = SGDClassifier(loss='log',random_state=10000, max_iter=50)
        #clf_dummy = SGDClassifier(loss='log',random_state=10000, max_iter=50)

        clf1 = LogisticRegression(solver='liblinear',random_state=10000, max_iter=200,C=1/mylambda)
        clf2 = LogisticRegression(solver='liblinear',random_state=10000, max_iter=200,C=1/mylambda)
        clf = LogisticRegression(solver='liblinear',random_state=10000, max_iter=200,C=1/mylambda)
        clf_dummy = LogisticRegression(solver='liblinear',random_state=10000, max_iter=200,C=1/mylambda)


        clf1.fit(X1_train, y1_train)
        accP1 = accuracy_score(y_test,clf1.predict(X_test))

        clf2.fit(X2_train, y2_train)
        accP2 = accuracy_score(y_test,clf2.predict(X_test))

        clf.fit(X_train, y_train)
        accALL = accuracy_score(y_test,clf.predict(X_test))

        clf_dummy.fit(X_train, y_train)


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


        # AVG(p1,p2) +DP
        avg_coeff = (clf1.coef_ + clf2.coef_)/2
        avg_inter = (clf1.intercept_ + clf2.intercept_)/2
        accMERG = 0
        for j in range(num_runs):
            ## Noise generation
            d = len(avg_coeff[0])+len(avg_inter)
            # Step 5: add noise to weights
            noise_to_add = getnoise(d,min(len(X1_train),len(X2_train)),epsilon,mylambda)
            # Step 5: add noise to weights
            weights = np.concatenate((avg_coeff[0],avg_inter))
            noisy_weights = weights + noise_to_add[0]
            clf_dummy.coef_ = noisy_weights[:-1].reshape(1,-1)
            clf_dummy.intercept_ = np.array([noisy_weights[-1]])
            accMERG = accMERG + accuracy_score(y_test,clf_dummy.predict(X_test))
        accAvgDP_dict[i] = accMERG/num_runs


        #### P1 + DP
        accP1DP = 0
        for j in range(num_runs):
            d = len(clf1.coef_[0])+len(clf1.intercept_)
            # Step 5: add noise to weights
            noise_to_add = getnoise(d,len(X1_train),epsilon,mylambda)
            weights = np.concatenate((clf1.coef_[0],clf1.intercept_))
            noisy_weights = weights + noise_to_add[0]
            clf_dummy.coef_ = noisy_weights[:-1].reshape(1,-1)
            clf_dummy.intercept_ = np.array([noisy_weights[-1]])
            accP1DP = accP1DP + accuracy_score(y_test,clf_dummy.predict(X_test))
        accP1DP_dict[i] = accP1DP/num_runs

        #### P2 + DP
        accP2DP = 0
        for j in range(num_runs):
            d = len(clf2.coef_[0])+len(clf2.intercept_)
            # Step 5: add noise to weights
            noise_to_add = getnoise(d,len(X2_train),epsilon,mylambda)
            weights = np.concatenate((clf2.coef_[0],clf2.intercept_))
            noisy_weights = weights + noise_to_add[0]
            clf_dummy.coef_ = noisy_weights[:-1].reshape(1,-1)
            clf_dummy.intercept_ = np.array([noisy_weights[-1]])
            accP2DP = accP2DP + accuracy_score(y_test,clf_dummy.predict(X_test))
        accP2DP_dict[i] = accP2DP/num_runs


        #### All + DP
        accAllDP = 0
        for j in range(num_runs):
            d = len(clf.coef_[0])+len(clf.intercept_)
            # Step 5: add noise to weights
            noise_to_add = getnoise(d,len(X_train),epsilon,mylambda)
            weights = np.concatenate((clf.coef_[0],clf.intercept_))
            noisy_weights = weights + noise_to_add[0]
            clf_dummy.coef_ = noisy_weights[:-1].reshape(1,-1)
            clf_dummy.intercept_ = np.array([noisy_weights[-1]])
            accAllDP = accAllDP + accuracy_score(y_test,clf_dummy.predict(X_test))
        accAllDP_dict[i] = accAllDP/num_runs

        #### (Avg(P1+DP, P2+DP))
        #avg_coeff = (clf1.coef_ + clf2.coef_)/2 #already noise is added
        #avg_inter = (clf1.intercept_ + clf2.intercept_)/2  #already noise is added
        #clf_dummy.coef_ = avg_coeff
        #clf_dummy.intercept_ = avg_inter
        #accAvgDP3_dict[i] = accuracy_score(y_test,clf_dummy.predict(X_test))

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

    print("=============")
    print()
    print("Results for LR with lambda=", mylambda, "and epsilon=",epsilon)
    print("=============")
    print()
    print("Model 1: Acc P1: ",np.mean(list(accP1_dict.values())))
    print("Model 2: Acc P2: ",np.mean(list(accP2_dict.values())))
    print("Model 3: Acc All: ",np.mean(list(accAll_dict.values())))
    #print("Acc All LGB: ",np.mean(list(accLGBAvg_dict.values())))
    print("Model 4: Acc Averaged: ",np.mean(list(accAvg_dict.values())))
    print("Model 5: Acc P1+DP: ",np.mean(list(accP1DP_dict.values())))
    print("Model 6: Acc P2+DP: ",np.mean(list(accP2DP_dict.values())))
    print("Model 7: Acc Averaged+DP: ",np.mean(list(accAvgDP_dict.values())))
    print("Model 8: Acc All+DP: ",np.mean(list(accAllDP_dict.values())))
    print("Model 9: Acc Avergaed(P1+DP,P2+DP): ",np.mean(list(accAvgDP2_dict.values())))
    print("=============")
    print()

    print("For Model 7")
    accs = list(accAvgDP_dict.values())
    print("Best Accuracy:", max(accs))
    print("Worst Accuracy:", min(accs))
    print("Mean:",np.mean(accs))
    print("Variance",np.var(accs))
    print("=============")
    print()
    print("For Model 7")
    accs = list(accAllDP_dict.values())
    print("Best Accuracy:", max(accs))
    print("Worst Accuracy:", min(accs))
    print("Mean:",np.mean(accs))
    print("Variance",np.var(accs))
    print("=============")
    print()
    print("For Model 9")
    accs = list(accAvgDP2_dict.values())
    print("Best Accuracy:", max(accs))
    print("Worst Accuracy:", min(accs))
    print("Mean:",np.mean(accs))
    print("Variance",np.var(accs))
    print("=============")
    print()
    acc_plot_avg_9.append(np.mean(list(accAvgDP2_dict.values())))
    var_plot_avg_9.append(np.var(list(accAvgDP2_dict.values())))
    lambdas_plot.append(mylambda)


from matplotlib import pyplot as plt
plt.plot(lambdas_plot, acc_plot_avg_9,
         lambdas_plot, var_plot_avg_9)
plt.show()