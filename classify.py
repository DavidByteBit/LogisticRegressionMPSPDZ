

def sig(x):
    return 1 / (1 + 1 / (2.71 ** x))

def dp(a, b):

    assert (len(a) == len(b))

    res = 0

    for i in range(len(a)):
        res += a[i] * b[i]

    return res

def parse_file(file_path):

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

    bias = output[delim + 2].replace("\n", "")
    weights_mid = output[delim + 3]

    weights_mid = weights_mid.replace("[", "").replace("]", "").replace("\n", "")

    weights = weights_mid.split(",")

    return bias, weights


def load_test_data(directory, fold):

    feature_path = directory + "test_X_fold{n}.csv".format(n=fold)
    label_path = directory + "test_y_fold{n}.csv".format(n=fold)

    data = []
    label = []

    with open(feature_path, 'r') as f:
        for line in f:
            data.append(line.replace("\n", "").split(","))

    with open(label_path, 'r') as f:
        for line in f:
            label.append(line.replace("\n", ""))

    return data, label
