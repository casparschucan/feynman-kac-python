from random_walk import feynman_kac_sample

import numpy as np


def test_phi(x, y):
    return np.sin(np.pi * x)*np.sin(np.pi * y)


def test_bound(x, y):
    return 0


def test_rhs(x, y):
    return np.pi**2 * (np.sin(np.pi * x)*np.sin(np.pi * y))


def feynman_kac_test_sol(x, y, N):
    expectation = 0
    Var = 0

    for i in range(N):
        if i % 100 == 0:
            print("done with ", i, " iterations")

        sample = feynman_kac_sample(10, x, y, test_bound, test_rhs)
        expectation += sample
        Var += sample**2

    expectation /= N
    Var /= N
    Var -= expectation**2

    print("found ", expectation,  " with variance ", Var, " expected ",
          test_phi(x, y))
    return expectation


feynman_kac_test_sol(.5, .5, 3000)

feynman_kac_test_sol(.1, .7, 3000)
