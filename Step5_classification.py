import sys
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_model(file_path):
    bias = 0
    weights = []
    str = ""
    with open(file_path, 'r') as f:
        for line in f:
            str = str + line

    str = str.split(",")
    bias = str[0]
    weights = str[1:]

    return bias, weights


def load_test_data(directory, has_labels):
    feature_path = directory + "/features.csv"
    label_path = directory + "/labels.csv"

    test_features = []
    test_labels = []

    with open(feature_path, 'r') as f:
        for line in f:
            test_features.append(line.replace("\n", "").split(","))

    if has_labels:
        with open(label_path, 'r') as f:
            for line in f:
                test_labels.append(line.replace("\n", ""))
    else:
        # We have to train for one epoch with two different labels, so we can just populate
        # test_labels with 0,1,0,1... to get the job done
        for j in range(len(test_features)):
            test_labels.append(j % 2)

    test_labels = [float(x) for x in test_labels]
    test_features = [[float(y) for y in x] for x in test_features]

    test_labels = np.array(test_labels)
    test_features = np.array(test_features)

    return test_features, test_labels


# Currently assumes model is a comma separated string with the bias first, then all of the weights
path_to_model = sys.argv[1]
test_data_folder = sys.argv[2]
prediction_file_path = sys.argv[3]
process_labels = sys.argv[4]

if process_labels.lower() == "true":
    process_labels = True
else:
    process_labels = False

data, labels = load_test_data(test_data_folder, process_labels)

# Initialize model - Has to train to be initialized properly. Train for one epoch
clf_dummy = LogisticRegression(max_iter=1)
clf_dummy.fit(data, labels)

b, W = load_model(path_to_model)

# Overwrite model
for i in range(len(clf_dummy.coef_[0])):
    clf_dummy.coef_[0][i] = W[i]
clf_dummy.intercept_[0] = b

# Start classifying
pred = clf_dummy.predict(data)

if process_labels:
    accAVG = accuracy_score(labels, pred)
    print("Accuracy = {n}%".format(n=accAVG))

with open(prediction_file_path, 'w') as f:
    pred = [str(i) for i in pred]
    f.write(",".join(pred))
