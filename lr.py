import math
import re

from Compiler import mpc_math, util
from Compiler.types import *
from Compiler.types import _unreduced_squant
from Compiler.library import *
from Compiler.util import is_zero, tree_reduce
from Compiler.comparison import CarryOutRawLE
from Compiler.GC.types import sbitint
from functools import reduce
import math

def dp(a, b):
    res = sfix(0)

    for i in range(len(b)):
        res += a[i] * b[i]

    return res

def dp_batch(vec, matrix, b=0):
    z = sfix.Array(len(matrix))
    assert(len(vec) == len(matrix[0]))

    for i in range(len(matrix)):
        z[i] += dp(vec, matrix[i]) + b

    return z

# TODO: optimize?
def clipped_relu(z):

    lt = z < - 0.5
    gt = z > 0.5
    eq = z + 0.5

    return cfix(0.0) * lt + cfix(1.0) * gt + eq * (1 - lt) * (1 - gt)

class LogisticRegression:


    def __init__(self, examples, labels, iterations=13, learning_rate=0.001):
        # self.df = df
        self.examples = examples
        self.labels = labels
        self.iterations = iterations
        self.learning_rate = learning_rate

    def train(self):
        # We initialize our W and b as zeros
        w = sfix.Array(len(self.examples[0]))
        w_delta = sfix.Array(len(self.examples[0]) + 1)
        b = sfix(0)

        X = self.examples
        y = self.labels
        m = len(X)  # Number of samples

        loss = []  # Keeping track of the cost function values

        for i in range(self.iterations):

            # Computes our predictions
            z = dp_batch(w, X, b=b)
            pred = sfix.Array(len(self.examples[0]))

            for j in range(len(self.examples[0])):
                pred[i] = clipped_relu(z[i])

            # update bias
            for k in range(len(y)):
                w_delta[0] = w_delta[0] + self.learning_rate * (y[k] - pred[k])

            # update weights
            for j in range(len(self.examples[0])):
                for k in range(len(y)):
                    w_delta[j + 1] = w_delta[j + 1] + self.learning_rate * (y[k] - pred[k]) * X[k][j]

            b = b + w_delta[0]

            for j in range(len(self.examples[0])):
                w[j] = w[j] + w_delta[j]

            # # Computes our cost function
            # cost = (-1 / m) * np.sum(np.dot(y, np.log(pred).T) + np.dot(1 - y, np.log(1 - pred).T))
            # loss.append(cost)  # Computes the gradient
            # dw = (1 / m) * np.dot(X, (pred - y).T)
            # db = (1 / m) * np.sum(pred - y, axis=1)
            #
            # # Updates the W and b
            # w = w - self.learning_rate * dw
            # b = b - self.learning_rate * db
            # return {"W": w, "b": b, "loss": loss}

        return w, b


