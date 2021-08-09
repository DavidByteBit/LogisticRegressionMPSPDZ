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


def sig5(x):
    cuts = [-5, -2.5, 2.5, 5]
    le = [0] + [x <= cut for cut in cuts] + [1]
    select = [le[i + 1] - le[i] for i in range(5)]
    outputs = [cfix(10 ** -4),
               0.02776 * x + 0.145,
               0.17 * x + 0.5,
               0.02776 * x + 0.85498,
               cfix(1 - 10 ** -4)]
    return sum(a * b for a, b in zip(select, outputs))


class LogisticRegression:

    def __init__(self, examples, labels, iterations=13, learning_rate=0.0001):
        # self.df = df
        self.examples = examples
        self.labels = labels
        self.iterations = iterations
        self.learning_rate = learning_rate

    def train(self):
        X = self.examples
        y = self.labels
        m = len(X)  # Number of samples
        feat = len(X[0])  # Number of features

        # We initialize our W and b as zeros
        w = sfix.Matrix(1, feat)

        @for_range(m)
        def _(i):
            w[0][i] = 0.0

        b = sfix.Array(1)

        b[0] = 0.0

        @for_range_opt(self.iterations)
        def _(i):

            w_delta = sfix.Array(feat + 1)

            @for_range(m + 1)
            def _(i):
                w_delta[i] = 0.0

            print_ln("iteration %s", i)
            time()

            # Computes our predictions
            z = dp_batch(w[0], X, b=b[0])
            pred = sfix.Array(m)

            @for_range(m)
            def _(i):
                pred[i] = 0.0

            # print_ln("%s", pred.reveal_nested())

            print_ln("dot product complete")

            @for_range_opt(m)
            def _(k):
                pred[k] = clipped_relu(z[k])
                # print_ln("%s", pred[k].reveal())

            print_ln("classifications complete")

            # update bias
            @for_range_opt(m)
            def _(k):
                w_delta[0] = w_delta[0] + self.learning_rate * (y[k] - pred[k]) # + momentum * w_delta[0]

            print_ln("delta update for bias complete")

            @for_range(m)
            def _(k):
                save = sfix.Array(1)
                save[0] = self.learning_rate * (y[k] - pred[k])
                @for_range(feat)
                def _(j):
                    w_delta[j + 1] += save[0] * X[k][j]


            b[0] = b[0] + w_delta[0] - self.learning_rate * b[0]

            print_ln("%s", w_delta.reveal_nested())

            @for_range(m)
            def _(j):
                w[0][j] = w[0][j] + w_delta[j + 1] - self.learning_rate * w[0][j]

            # print_ln("%s", w[0].reveal_nested())

            print_ln("\n\n\tepoch %s complete\n\n", i)

        return w[0], b[0]
