from Compiler.mpc_math import log_fx
from Compiler.mpc_math import cos
from Compiler.mpc_math import sin
from Compiler.mpc_math import sqrt
import math

e = math.e
pi = math.pi


def gen_samples_(d):

    print_ln("generating samples")

    assert d >= 1

    gaussian_vec = sfix.Array(d)

    # div two since the Box-Mueller transform produces 2 samples
    @for_range_opt(d // 2)
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
    if d % 2 == 1:
        gaussian_vec[d - 1] = gen_samples_(2)[0]

    return gaussian_vec


def normalize_(vec, d):

    print_ln("normalizing samples")

    L2_norm_vec_intermediate = sfix.Array(d)
    L2_norm_vec = sfix.Array(d)

    @for_range_opt(d)
    def _(i):
        L2_norm_vec_intermediate[i] = vec[i] * vec[i]

    s = sfix.Array(1)
    s[0] = sfix(0)

    @for_range(d)
    def _(i):
        s[0] += L2_norm_vec_intermediate[i]

    # for i in range(d):
    #     s += L2_norm_vec_intermediate[i]

    L2_norm = sqrt(s[0])

    @for_range_opt(d)
    def _(i):
        L2_norm_vec[i] = L2_norm_vec_intermediate[i] / L2_norm

    return L2_norm_vec


def gen_gamma_dis(d, n, epsilon, lamb):

    print_ln("generating gamma dis samples")

    samples = gen_samples_(d)
    samples_log = sfix.Array(d)

    @for_range_opt(d)
    def _(i):
        samples_log[i] = -1 * log_fx(samples[i], e)

    gamma_dis = sfix.Array(d)

    # equivalent to dividing by 2/n*epsilon*lambda
    norm_const = n * epsilon * lamb * 0.5

    @for_range(d)
    def _(i):
        gamma_dis[i] = samples_log[i] * norm_const

    return gamma_dis

#### added by sikha - pseudo

# exponential distribution with rate 1 (Exp(1))
def generate_exp_distribution():
    print_ln("generating exp distribution")
    U = sfix.get_random(0, 1)
    exp_sample = -1 * log_fx(U, e)
    return exp_sample


def gen_gamma_dis2(n, epsilon=1, lamb=1):

    print_ln("generating gamma dis samples")

    #gamma_samples = sfix.Array(d)
    final_gamma = 0

    @for_range_opt(d)
    def _(i):
        final_gamma = final_gamma + generate_exp_distribution()


    # equivalent to dividing by 2/n*epsilon*lambda
    norm_const = n * epsilon * lamb
    div = 2/norm_const

    final_gamma = final_gamma * div

    return final_gamma
#### end


def gen_noise(d, n, epsilon, lamb):

    gaussian_vec = gen_samples_(d)
    gaussian_vec_normalized = normalize_(gaussian_vec, d)

    #### added by sikha
    gamma = gen_gamma_dis2(d, n, epsilon, lamb)
    noise_vector.assign_vector(gaussian_vec_normalized.get_vector() * gamma)

    return noise_vector
    #### end


#d_ = 1700
#n_ = 1800
#epsilon_ = 1
#lamb_ = 1

#noise_vector = gen_noise(d_, n_, epsilon_, lamb_)

