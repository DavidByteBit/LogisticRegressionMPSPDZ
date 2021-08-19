

path = "models/models0"
n_features = 1874

with open(path, 'w') as f:
    model = ['0.0'] * (n_features + 1)
    f.write(",".join(model))
