import sys
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_model(file_path):
    bias = 0
    weights = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.replace("\n", "").split(",")
            bias = line[0]
            weights = line[1:]

    return bias, weights


def load_test_data(directory):
    feature_path = directory + "/features.csv"
    label_path = directory + "/labels.csv"

    test_features = []
    test_labels = []

    with open(feature_path, 'r') as f:
        for line in f:
            test_features.append(line.replace("\n", "").split(","))

    with open(label_path, 'r') as f:
        for line in f:
            test_labels.append(line.replace("\n", ""))

    test_labels = [float(x) for x in test_labels]
    test_features = [[float(y) for y in x] for x in test_features]

    test_labels = np.array(test_labels)
    test_features = np.array(test_features)

    return test_features, test_labels


# Currently assumes model is a comma separated string with the bias first, then all of the weights
path_to_model = sys.argv[1]
test_data_folder = sys.argv[2]

data, labels = load_test_data(test_data_folder)

# Initialize model
clf_dummy = LogisticRegression(max_iter=1)
clf_dummy.fit(data, labels)

b, W = load_model(path_to_model)

# Overwrite model
for i in range(len(clf_dummy.coef_[0])):
    clf_dummy.coef_[0][i] = W[i]
clf_dummy.intercept_[0] = b

# Start classifying
clf_dummy.predict(data)
accAVG = accuracy_score(labels, clf_dummy.predict(data))

print("Accuracy = {n}%".format(n=accAVG))
