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


def dp_batch(vec, matrix, b=0):
    z = sfix.Array(len(matrix))
    assert (len(vec) == len(matrix[0]))

    @for_range_opt(len(matrix))
    def _(i):
        z[i] = sfix.dot_product(vec, matrix[i]) + b

    return z


def clipped_relu(z):
    a = z < -0.5
    b = z > 0.5
    return a.if_else(0, b.if_else(1, 0.5 + z))


class LogisticRegression:

    def __init__(self, examples, labels, iterations=13, learning_rate=0.0001, b=0):
        # self.df = df
        self.examples = examples
        self.labels = labels
        self.iterations = iterations
        self.learning_rate = learning_rate

    def train(self):
        # We initialize our W and b as zeros
        w = sfix.Array(len(self.examples[0]))
        w_delta = sfix.Array(len(self.examples[0]) + 1)
        b = sfix.Array(1)

        X = self.examples
        y = self.labels
        m = len(X)  # Number of samples

        loss = []  # Keeping track of the cost function values

        @for_range_opt(self.iterations)
        def _(i):

            print_ln("iteration %s", i)
            time()

            # Computes our predictions
            z = dp_batch(w, X, b=b[0])
            pred = sfix.Array(len(self.examples))
            print_ln("dot product complete")

            @for_range_opt(len(self.examples))
            def _(j):
                pred[j] = clipped_relu(z[j])

            print_ln("classifications complete")

            # update bias
            @for_range_opt(len(y))
            def _(k):
                w_delta[0] = w_delta[0] + self.learning_rate * (y[k] - pred[k])

            print_ln("delta update for bias complete")

            # update weights
            @for_range_opt(len(self.examples[0]))
            def _(j):
                print_ln("delta update for feature %s complete", j)
                @for_range_opt(len(y))
                def _(k):
                    w_delta[j + 1] = w_delta[j + 1] + self.learning_rate * (y[k] - pred[k]) * X[k][j]

            b[0] = b[0] + w_delta[0]

            for j in range(len(self.examples[0])):
                w[j] = w[j] + w_delta[j + 1]

            print_ln("\n\n\tepoch %s complete\n\n", i)

        return w, b[0]
