

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

    bias = output[delim + 2]
    weights_mid = output[delim + 3]

