from random_walk import feynman_kac_sample

import numpy as np


def test_phi(x, y):
    return np.sin(x)*np.sin(y)


def feynman_kac_test_sol(x, y, N):
    expectation = 0
    Var = 0

    for i in range(N):
        sample = feynman_kac_sample(10, x, y, test_phi, test_phi)
        expectation += sample
        Var += sample**2

    expectation /= N
    Var /= N
    Var -= expectation**2

    print("found ", expectation,  " with variance ", Var, " expected ",
          np.sin(y)*np.sin(x))
    return expectation


feynman_kac_test_sol(.5, .5, 4000)

feynman_kac_test_sol(.2, .2, 4000)
