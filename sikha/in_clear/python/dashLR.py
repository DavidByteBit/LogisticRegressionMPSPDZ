# date: July 22, 2021
# name: Martine De Cock
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

  
  
##################################################################################

  
# Load the data
df1 = pd.read_csv('party_1.csv')
df2 = pd.read_csv('party_2.csv')

print(df1.shape)

X1, y1 = preprocess(df1)
X2, y2 = preprocess(df2)
'''
with open('Player-0-all.txt', 'a+') as outfile:
  np.savetxt(outfile,X1, delimiter='\n',fmt='%14.13f')

with open('Player-0-all.txt', 'a+') as outfile:
  np.savetxt(outfile,y1, delimiter='\n',fmt='%d')


with open('Player-1-all.txt', 'a+') as outfile:
  np.savetxt(outfile,X2, delimiter='\n',fmt='%14.13f')

with open('Player-1-all.txt', 'a+') as outfile:
  np.savetxt(outfile,y2, delimiter='\n',fmt='%d')


'''

# This will hold 4 accuracy results for each of 5 folds
# (1) accuracy of model trained on data from P1
# (2) accuracy of model trained on data from P2
# (3) accuracy of model trained on data from P1 and from P2
# (4) accuracy of aggregation of model (1) and (2) from above
LRresults = np.zeros((5, 5))


kf1 = KFold(n_splits=5,shuffle = True,random_state = 42)
kf2 = KFold(n_splits=5,shuffle = True,random_state = 42)

epsilon = 1
mylambda = 0.5
  
i = 0

for (train1_indices, test1_indices), (train2_indices, test2_indices) in zip(kf1.split(X1,y1),kf2.split(X2,y2)):
  print("FOLD ", i+1)

  X1_train, X1_test = X1[train1_indices,:], X1[test1_indices,:]
  y1_train, y1_test = y1[train1_indices], y1[test1_indices]
  X2_train, X2_test = X2[train2_indices,:], X2[test2_indices,:]
  y2_train, y2_test = y2[train2_indices], y2[test2_indices]

  with open('train_Alice'+str(i+1)+'.txt', 'a+') as outfile:
    np.savetxt(outfile,X1_train, delimiter='\n',fmt='%14.13f')
  with open('train_Alice'+str(i+1)+'.txt', 'a+') as outfile:
      np.savetxt(outfile,y1_train, delimiter='\n',fmt='%d')
  with open('train_Bob'+str(i+1)+'.txt', 'a+') as outfile:
    np.savetxt(outfile,X2_train, delimiter='\n',fmt='%14.13f')
  with open('train_Bob'+str(i+1)+'.txt', 'a+') as outfile:
    np.savetxt(outfile,y2_train, delimiter='\n',fmt='%d')

  X_train = np.append(X1_train,X2_train,axis=0)
  y_train = np.append(y1_train,y2_train)
  X_test = np.append(X1_test,X2_test,axis=0)
  y_test = np.append(y1_test,y2_test)


  print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
  np.save('train_X'+str(i+1)+'.npy',X_train)
  np.save('train_y'+str(i+1)+'.npy',y_train)
  np.save('test_X'+str(i+1)+'.npy',X_test)
  np.save('test_y'+str(i+1)+'.npy',y_test)

  with open('train'+str(i+1)+'.txt', 'a+') as outfile:
    np.savetxt(outfile,X_train, delimiter='\n',fmt='%14.13f')

  with open('train'+str(i+1)+'.txt', 'a+') as outfile:
    np.savetxt(outfile,y_train, delimiter='\n',fmt='%d')

  with open('test'+str(i+1)+'.txt', 'a+') as outfile:
      np.savetxt(outfile,X_test, delimiter='\n',fmt='%14.13f')

  with open('test'+str(i+1)+'.txt', 'a+') as outfile:
    np.savetxt(outfile,y_test, delimiter='\n',fmt='%d')

  
  ########## Train and test logistic regression models #################

  #clf1 = LogisticRegression(solver='liblinear',random_state=10000, max_iter=10)
  #clf2 = LogisticRegression(solver='liblinear',random_state=10000, max_iter=10)
  #clf = LogisticRegression(solver='liblinear',random_state=10000, max_iter=10)

  clf1 = SGDClassifier(loss='log',random_state=10000, max_iter=30)
  clf2 = SGDClassifier(loss='log',random_state=10000, max_iter=30)
  clf = SGDClassifier(loss='log',random_state=10000, max_iter=30,verbose = 2)
  
  clf1.fit(X1_train, y1_train)
  accP1  = accuracy_score(y_test,clf1.predict(X_test))
  
  clf2.fit(X2_train, y2_train)
  accP2 = accuracy_score(y_test,clf2.predict(X_test))
  
  clf.fit(X_train, y_train)
  accALL = accuracy_score(y_test,clf.predict(X_test))
  
  print("P1:", accP1, clf1.n_iter_)
  print("P2:", accP2, clf2.n_iter_)
  print("All:", accALL, clf.n_iter_)

  num_runs = 100 

  # Adding noise to All model
  clean_coeff = clf.coef_
  clean_inter = clf.intercept_
  myscale = 2/(len(X_train) * epsilon * mylambda)
  accAllDP = 0
  for j in range(num_runs):
    clf.coef_ = clean_coeff + np.random.laplace(0., myscale, len(clean_coeff))
    clf.intercept = clean_inter + np.random.laplace(0., myscale, 1)
    accAllDP = accAllDP + accuracy_score(y_test,clf.predict(X_test))   
  accAllDP = accAllDP/num_runs  
  print("Acc All + DP:", accAllDP)


  # Merging of LR models
  avg_coeff = (clf1.coef_ + clf2.coef_)/2
  avg_inter = (clf1.intercept_ + clf2.intercept_)/2 
  myscale = 2/(min(len(X1_train),len(X2_train)) * epsilon * mylambda)
  accMERG = 0
  for j in range(num_runs):
    clf1.coef_ = avg_coeff + np.random.laplace(0., myscale, len(avg_coeff))
    clf1.intercept = avg_inter + np.random.laplace(0., myscale, 1)
    accMERG = accMERG + accuracy_score(y_test,clf1.predict(X_test))   
  accMERG = accMERG/num_runs  
  print("Acc P1&P2 + DP:", accMERG)



  LRresults[i] = [accP1,accP2,accALL,accMERG,accAllDP]

  print("==========completed")
  i = i + 1


# Printing the averages over the 5 folds
print("          P1,   P2, All, P1&P2 + DP, All + DP")
np.set_printoptions(precision=2)
print("LR:     ",np.mean(LRresults, axis=0))
