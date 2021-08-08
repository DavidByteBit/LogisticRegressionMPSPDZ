from sklearn.linear_model import LogisticRegression


def populate(path, a, t):

    if t == "t":
        with open(path, 'r') as f:
            for line in f:
                a.append(line.replace("\n", "").split(","))
    else:
        with open(path, 'r') as f:
            for line in f:
                a.append(line.replace("\n", ""))


X_train = []
y_train = []
X_test = []
y_test = []

path = "data/Alice/train_X_fold0.csv"
populate(path, X_train, "t")
path = "data/Bob/train_X_fold0.csv"
populate(path, X_train, "t")

path = "data/Alice/train_y_fold0.csv"
populate(path, y_train, "y")
path = "data/Bob/train_y_fold0.csv"
populate(path, y_train, "y")

path = "data/Alice/test_X_fold0.csv"
populate(path, X_test, "t")
path = "data/Bob/test_X_fold0.csv"
populate(path, X_test, "t")

path = "data/Alice/test_y_fold0.csv"
populate(path, y_test, "y")
path = "data/Bob/test_y_fold0.csv"
populate(path, y_test, "y")


clf = LogisticRegression(penalty='l2', tol=0.001, C=1.0, random_state=42,
                         solver='sag', max_iter=8).fit(X_train, y_train)
pred = clf.predict(X_test)

print("done training")

correct = 0
incorrect = 0

for i in range(len(pred)):
    if int(float(pred[i])) == int(float(y_test[i])):
        correct += 1
    else:
        incorrect += 1

print(correct)

print(incorrect)
