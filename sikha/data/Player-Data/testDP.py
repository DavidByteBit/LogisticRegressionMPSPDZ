import diffprivlib
import numpy as np
from sklearn.metrics import accuracy_score

X_train = np.load('train_X1.npy')
y_train = np.load('train_y1.npy')
X_test = np.load('test_X1.npy')
y_test = np.load('test_y1.npy')



from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='liblinear',random_state=10000, max_iter=10)
clf.fit(X_train, y_train)
accALL = accuracy_score(y_test,clf.predict(X_test))
print("Non-private test accuracy: %.2f%%" % (accALL * 100))

import diffprivlib.models as dp

dp_clf = dp.LogisticRegression()
dp_clf.fit(X_train, y_train)



