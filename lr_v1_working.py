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
                w_delta[0] = w_delta[0] + self.learning_rate * (y[k] - pred[k])

            print_ln("delta update for bias complete")

            counter = sfix.Array(2)

            # # update weights
            # @for_range(feat)
            # def _(j):
            #     # print_ln("delta update for feature %s complete", j)
            #     @for_range(m)
            #     def _(k):
            #         counter[0] += X[k][0]
            #         counter[1] += self.learning_rate * (y[k] - pred[k])
            #         # print_ln("%s", X[k][j].reveal())
            #         w_delta[j + 1] = w_delta[j + 1] + self.learning_rate * (y[k] - pred[k]) * X[k][j]


            @for_range(m)
            def _(k):
                save = sfix.Array(1)
                save[0] = self.learning_rate * (y[k] - pred[k])
                @for_range(feat)
                def _(j):
                    w_delta[j + 1] += save[0] * X[k][j]


            print_ln("here %s", counter[0].reveal())

            b[0] = b[0] + w_delta[0]

            print_ln("%s", w_delta[1:].reveal_nested())

            w[0] = w[0] + w_delta[1:]

            # print_ln("%s", w[0].reveal_nested())

            print_ln("\n\n\tepoch %s complete\n\n", i)

        return w[0], b[0]
