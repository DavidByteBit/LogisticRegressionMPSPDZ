import yaml
import subprocess
import sys

def transpose(list):
    return [[row[i] for row in list] for i in range(len(list[0]))]


# Step 0,1: Setting everything up and re-writing the .mpc file
mpc_file_path = "runLR.mpc"
settings_map = None

with open(sys.argv[1], 'r') as stream:
    try:
        settings_map = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

folds = settings_map["folds"]

create_data = bool(settings_map["create_data"])

if create_data:
    subprocess.call(settings_map['path_to_this_repo'] + "/bash_scripts/create_data.sh", folds)

alice_data = []
bob_data = []

for f in range(folds):

    path = settings_map['alice_data_folder']

    x_train = path + "/test_X_fold{n}".format(n=f)
    y_train = path + "/test_X_fold{n}".format(n=f)
    x_test = path + "/test_X_fold{n}".format(n=f)
    y_test = path + "/test_X_fold{n}".format(n=f)

    paths = [x_train, y_train, x_test, y_test]

    for p in paths:
        with open(p, 'r') as stream:
            for line in stream:
                alice_data.extend(line.split(","))

for f in range(folds):

    path = settings_map['bob_data_folder']

    x_train = path + "/test_X_fold{n}".format(n=f)
    y_train = path + "/test_X_fold{n}".format(n=f)
    x_test = path + "/test_X_fold{n}".format(n=f)
    y_test = path + "/test_X_fold{n}".format(n=f)

    paths = [x_train, y_train, x_test, y_test]

    for p in paths:
        with open(p, 'r') as stream:
            for line in stream:
                alice_data.extend(line.split(","))

# 'command line arguments' for our .mpc file
alice_examples = len(alice_data)
bob_examples = len(bob_data)
n_features = len(alice_data[0])
n_epochs = settings_map['n_epochs']
folds = settings_map['folds']

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


# Step 2: write to Alice's and Bobs private input files
with open(settings_map['alice_private_input_path'], 'w') as stream:

    str = ""

    # Should just be one row I think, so I may clean this up a bit
    for row in alice_data:
        str += " ".join(row)

    stream.write(str)


print("Alice has {n} many private values".format(n=len(alice_data)))

with open(settings_map['bob_private_input_path'], 'w') as stream:

    str = ""

    # Should just be one row I think, so I may clean this up a bit
    for row in bob_data:
        str += " ".join(row)

    stream.write(str)


print("Alice has {n} many private values".format(n=len(bob_data)))

# Step 3: Compile .mpc program
subprocess.call(settings_map['path_to_this_repo'] + "/bash_scripts/compile.sh")

