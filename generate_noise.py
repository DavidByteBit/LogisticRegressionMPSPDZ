from Compiler.mpc_math import log_fx
from Compiler.mpc_math import cos
from Compiler.mpc_math import sin
from Compiler.mpc_math import sqrt
import math

e = math.e
pi = math.pi

def produce_Guassian_noise_(n):

    assert n >= 1

    gaussian_vec = sint.Array(n)

    # div two since the Box-Mueller transform produces 2 samples
    @for_range(n // 2)
    def _(i):
        # TODO: test range
        A = sfix.get_random(0, 1)
        B = sfix.get_random(0, 1)

        C = sqrt(-2 * log_fx(A, e))

        trig_arg = (2 * pi) * B
        cosine = cos(trig_arg)
        sine = sin(trig_arg)

        r1 = C * cosine
        r2 = C * sine

        gaussian_vec[2 * i] = r1
        gaussian_vec[(2 * i) + 1] = r2

    # if n is odd, our vec is one element short. Obtain one more sample
    if n % 2 == 1:
        gaussian_vec[n - 1] = produce_Guassian_noise_(2)[0]

    return gaussian_vec


def normalize_(vec):

    n = vec.len()

    L2_norm = sfix(0)
    L2_norm_vec_intermediate = sint.Array(n)

    @for_range(n)
    def _(i):
        L2_norm_vec_intermediate[i] = vec[i]

    L2_norm_vec_intermediate = L2_norm_vec_intermediate * L2_norm_vec_intermediate

    s = sint(0)

    for i in range(n):
        s += L2_norm_vec_intermediate[i]

    s = sqrt(s)

    for i in range(n):
        L2_norm_vec_intermediate[i] = L2_norm_vec_intermediate[i]/s

    return L2_norm_vec_intermediate


def gen_noise(n):
    vec = produce_Guassian_noise_(n)
    vec = normalize_(vec)


gen_noise(16)
