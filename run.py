import yaml
import subprocess
import sys
import math

# These files are apart of the repo and are assumed to remain in the relative path they were initially defined to be in
mpc_file_path = "runLR.mpc"
mpc_classify_file_path = "classifyLR.mpc"


# Entry point to run the script
def run():
    # Parse settings
    settings_map = parse_settings()

    # Grab settings
    folds = settings_map["folds"]

    create_data = settings_map["create_data"]

    # Determine if we need to create folds
    if create_data.lower() == "true":
        subprocess.call(settings_map['path_to_this_repo'] + "/bash_scripts/create_data.sh %s" % folds, shell=True)

    # Upload Alice/Bob data to their respective private files
    params = write_data(settings_map)

    # Edit .mpc source code for compilation
    edit_source_code(settings_map, params)

    # Compile .mpc program
    c = settings_map["compiler"]
    subprocess.check_call([settings_map['path_to_this_repo'] + "/bash_scripts/compile.sh", c])
    # subprocess.call(settings_map['path_to_this_repo'] + "/bash_scripts/classify.sh")


def edit_source_code(settings_map, params):
    # 'command line arguments' for our .mpc file
    n_epochs = settings_map['n_epochs']
    folds = settings_map['folds']
    lambda_ = settings_map['lambda']
    batch = settings_map['batch']
    alice_examples = params[0]
    bob_examples = params[1]
    n_features = params[2]
    # test_ratio = 0

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
    file[start_of_delim + 6] = "lambda_ = {n}\n".format(n=lambda_)
    file[start_of_delim + 6] = "batch = {n}\n".format(n=batch)

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

    file[start_of_delim + 1] = "alice_examples = {n}\n".format(n=alice_test_examples)
    file[start_of_delim + 2] = "bob_examples = {n}\n".format(n=bob_test_examples)
    file[start_of_delim + 3] = "n_features = {n}\n".format(n=n_features)

    # file as a string
    file = ''.join([s for s in file])

    # print(file)

    with open(mpc_classify_file_path, 'w') as stream:
        stream.write(file)


def write_data(settings_map):
    path = settings_map['alice_data_folder']

    fold = settings_map['fold']

    x_train = path + "/train_X_fold{n}.csv".format(n=fold)
    y_train = path + "/train_y_fold{n}.csv".format(n=fold)

    alice_data = []
    bob_data = []

    data = []

    with open(x_train, 'r') as stream:
        for line in stream:
            data.append(line.replace("\n", "").split(","))

    alice_examples = len(data)
    n_features = len(data[0])


    for row in data:
        alice_data.extend(row)

    with open(y_train, 'r') as stream:
        for line in stream:
            alice_data.extend(line.replace("\n", "").split(","))

    data = []

    x_train = path + "/train_X_fold{n}.csv".format(n=fold)
    y_train = path + "/train_y_fold{n}.csv".format(n=fold)

    paths = [x_train, y_train]

    with open(x_train, 'r') as stream:
        for line in stream:
            data.append(line.replace("\n", "").split(","))

    bob_examples = len(data)

    for row in data:
        bob_data.extend(row)

    with open(y_train, 'r') as stream:
        for line in stream:
            bob_data.extend(line.replace("\n", "").split(","))

    with open(settings_map['alice_private_input_path'], 'w') as stream:

        s = " ".join(alice_data)

        stream.write(s)

    print("Alice has {n} many private values".format(n=len(alice_data)))

    with open(settings_map['bob_private_input_path'], 'w') as stream:

        s = " ".join(bob_data)

        stream.write(s)

    print("Bob has {n} many private values".format(n=len(bob_data)))

    return [alice_examples, bob_examples, n_features]


def parse_settings():
    settings_map = None

    with open(sys.argv[1], 'r') as stream:
        try:
            settings_map = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return settings_map


run()

