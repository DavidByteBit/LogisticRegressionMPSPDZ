import yaml
import subprocess
import sys
import math

# Step 0,1: Setting everything up and re-writing the .mpc file
mpc_file_path = "runLR.mpc"
mpc_classify_file_path = "classifyLR.mpc"
settings_map = None

with open(sys.argv[1], 'r') as stream:
    try:
        settings_map = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

folds = settings_map["folds"]

create_data = settings_map["create_data"]

if create_data.lower() == "true":
    subprocess.call(settings_map['path_to_this_repo'] + "/bash_scripts/create_data.sh %s" % folds, shell=True)

alice_data = []
bob_data = []

# 'command line arguments' for our .mpc file
n_epochs = settings_map['n_epochs']
folds = settings_map['folds']
alice_examples = 0
bob_examples = 0
n_features = 0
test_ratio = 0

path = settings_map['alice_data_folder']

x_train = path + "/train_X_fold{n}.csv".format(n=0)
y_train = path + "/train_y_fold{n}.csv".format(n=0)
x_test = path + "/test_X_fold{n}.csv".format(n=0)
y_test = path + "/test_y_fold{n}.csv".format(n=0)

paths = [x_train, y_train, x_test, y_test]

feature_length_not_found = True

for p in paths:
    with open(p, 'r') as stream:
        row = 0
        for line in stream:
            if feature_length_not_found and "train_X_fold" in p:
                feature_length_not_found = False
                n_features = len(line.replace("\n", "").split(","))
            row += 1
            alice_data.extend(line.replace("\n", "").split(","))

        if "train_X_fold" in p:
            alice_examples = row


path = settings_map['bob_data_folder']

x_train = path + "/train_X_fold{n}.csv".format(n=0)
y_train = path + "/train_y_fold{n}.csv".format(n=0)
x_test = path + "/test_X_fold{n}.csv".format(n=0)
y_test = path + "/test_y_fold{n}.csv".format(n=0)

paths = [x_train, y_train, x_test, y_test]

for p in paths:
    with open(p, 'r') as stream:
        row = 0
        for line in stream:
            row += 1
            bob_data.extend(line.replace("\n", "").split(","))

        if "train_X_fold" in p:
            bob_examples = row


file = []
found_delim = False
start_of_delim = 0

i = 0
with open(mpc_file_path, 'r') as stream:
    for line in stream:

        if not found_delim and "@args" in line:
            start_of_delim = i
            found_delim = True
        i += 1

        file.append(line)

file[start_of_delim + 1] = "alice_examples = {n}\n".format(n=alice_examples)
file[start_of_delim + 2] = "bob_examples = {n}\n".format(n=bob_examples)
file[start_of_delim + 3] = "n_features = {n}\n".format(n=n_features)
file[start_of_delim + 4] = "n_epochs = {n}\n".format(n=n_epochs)
file[start_of_delim + 5] = "folds = {n}\n".format(n=folds)

# file as a string
file = ''.join([s for s in file])

# print(file)

with open(mpc_file_path, 'w') as stream:
    stream.write(file)

################### Next one ######################

file = []
found_delim = False
start_of_delim = 0

i = 0
with open(mpc_classify_file_path, 'r') as stream:
    for line in stream:

        if not found_delim and "@args" in line:
            start_of_delim = i
            found_delim = True
        i += 1

        file.append(line)

test_ratio = 1 / folds
alice_test_examples = int(math.ceil(alice_examples * test_ratio))
bob_test_examples = int(bob_examples * test_ratio)

file[start_of_delim + 1] = "alice_examples = {n}\n".format(n=n_features)
file[start_of_delim + 2] = "bob_examples = {n}\n".format(n=alice_test_examples)
file[start_of_delim + 3] = "n_features = {n}\n".format(n=bob_test_examples)

# file as a string
file = ''.join([s for s in file])

# print(file)

with open(mpc_classify_file_path, 'w') as stream:
    stream.write(file)


# Step 2: write to Alice's and Bobs private input files
with open(settings_map['alice_private_input_path'], 'w') as stream:

    s = " ".join(alice_data)

    stream.write(s)


print("Alice has {n} many private values".format(n=len(alice_data)))

with open(settings_map['bob_private_input_path'], 'w') as stream:

    s = " ".join(bob_data)

    stream.write(s)


print("Bob has {n} many private values".format(n=len(bob_data)))

# Step 3: Compile .mpc program
subprocess.call(settings_map['path_to_this_repo'] + "/bash_scripts/compile.sh")
subprocess.call(settings_map['path_to_this_repo'] + "/bash_scripts/classify.sh")

input("Press enter to populate Alice's and Bobs data with the next fold")

# Not working, need to change it like I did for fold 0
for f in range(folds - 1):

    path = settings_map['alice_data_folder']

    x_train = path + "/test_X_fold{n}.csv".format(n=f+1)
    y_train = path + "/test_X_fold{n}.csv".format(n=f+1)
    x_test = path + "/test_X_fold{n}.csv".format(n=f+1)
    y_test = path + "/test_X_fold{n}.csv".format(n=f+1)

    paths = [x_train, y_train, x_test, y_test]

    for p in paths:
        with open(p, 'r') as stream:
            for line in stream:
                alice_data.extend(line.split(","))

# Not working, need to change it like I did for fold 0
for f in range(folds - 1):

    path = settings_map['bob_data_folder']

    x_train = path + "/train_X_fold{n}.csv".format(n=f+1)
    y_train = path + "/train_y_fold{n}.csv".format(n=f+1)
    x_test = path + "/test_X_fold{n}.csv".format(n=f+1)
    y_test = path + "/test_y_fold{n}.csv".format(n=f+1)

    paths = [x_train, y_train, x_test, y_test]

    for p in paths:
        with open(p, 'r') as stream:
            for line in stream:
                bob_data.extend(line.split(","))

    with open(settings_map['alice_private_input_path'], 'w') as stream:

        s = ""

        # Should just be one row I think, so I may clean this up a bit
        for row in alice_data:
            s += " ".join(row)

        stream.write(s)

    print("Alice has {n} many private values".format(n=len(alice_data)))

    with open(settings_map['bob_private_input_path'], 'w') as stream:

        s = ""

        # Should just be one row I think, so I may clean this up a bit
        for row in bob_data:
            s += " ".join(row)

        stream.write(s)

    print("Bob has {n} many private values".format(n=len(bob_data)))

    if f != folds - 1:
        input("Press enter to populate Alice's and Bobs data with the next fold")


