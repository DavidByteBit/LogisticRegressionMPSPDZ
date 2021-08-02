import yaml
import sys


def parse_settings():
    settings_map = None

    with open(sys.argv[1], 'r') as stream:
        try:
            settings_map = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return settings_map


def sig(x):
    return 1 / (1 + 1 / (2.71 ** x))


def dp(a, b):
    assert (len(a) == len(b))

    res = 0

    for i in range(len(a)):
        res += a[i] * b[i]

    return res


def load_model(file_path):
    output = []
    delim = 0

    i = 0
    # Find model weights
    with open(file_path, 'r') as f:
        for line in f:
            output.append(line)
            if "Training finished" in line:
                delim = i
            i += 1

    bias = float(output[delim + 2].replace("\n", ""))
    weights_mid = output[delim + 3]

    weights_mid = weights_mid.replace("[", "").replace("]", "").replace("\n", "")

    weights = weights_mid.split(",")

    print(weights[0:20])

    weights = [[float(x) for x in r] for r in weights]

    return bias, weights


def load_test_data(directory, fold):
    feature_path = directory + "/test_X_fold{n}.csv".format(n=fold)
    label_path = directory + "/test_y_fold{n}.csv".format(n=fold)

    data = []
    label = []

    with open(feature_path, 'r') as f:
        for line in f:
            data.append(line.replace("\n", "").split(","))

    with open(label_path, 'r') as f:
        for line in f:
            label.append(line.replace("\n", ""))

    return data, label


settings_map = parse_settings()

alice_data_path = settings_map["alice_data_folder"]
bob_data_path = settings_map["bob_data_folder"]
folds = settings_map["fold"]
model_path = settings_map["path_to_this_repo"]

data = []
labels = []

d, la = load_test_data(alice_data_path, folds)

data.append(d)
labels.append(la)

d, la = load_test_data(bob_data_path, folds)

data.append(d)
labels.append(la)

b, W = load_model(model_path)

# Start classifying

correct = 0

incorrect = 0

threshold = 0.5

for i in range(len(data)):
    row = data[i]
    classification_intermediate = sig(dp(W, row) + b)

    label = 0

    if classification_intermediate >= threshold:
        label = 1

    true_label = labels[i]

    if true_label == label:
        correct += 1
    else:
        incorrect += 1


print("correct: {a}\n incorrect: {b}\n ratio: {c}".
      format(a=correct, b=incorrect, c=(correct / (correct + incorrect)) * 100.0))
