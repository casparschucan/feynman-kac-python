from random_walk import feynman_kac_sample

from multiprocessing import Pool
import matplotlib.pyplot as plt

import numpy as np


def test_phi(x, y):
    return np.sin(np.pi * x)*np.sin(np.pi * y)


def test_bound(x, y):
    return 0


def test_rhs(x, y):
    return np.pi**2 * (np.sin(np.pi * x)*np.sin(np.pi * y))


def feynman_kac_test_sol(x, y, N, dt):
    expectation = 0
    Var = 0
    if __name__ == '__main__':

        samples = np.zeros(N)
        sample_results = []
        n_procs = 10
        with Pool(processes=n_procs) as pool:
            for i in range(N):
                sample_results.append(pool.apply_async(feynman_kac_sample,
                                                       args=(x,
                                                             y,
                                                             test_bound,
                                                             test_rhs,
                                                             dt)))

            samples = np.array([r.get() for r in sample_results])

        expectation = samples.mean()

    return expectation


print("starting loop")


def eval_num_samples(N):
    average = 0
    Var = 0

    for i in range(40):
        temp = feynman_kac_test_sol(.5, .5, N, .00001)
        average += temp
        Var += temp**2

        if (i+1) % 10 == 0:
            error = Var/(i+1) - ((average/(i+1))**2)
            print("After ", i+1,
                  " iterations, we have reached a standard deviation of: ", error)

    Var /= 40
    average /= 40
    Var -= average**2

    return Var, average


errs = np.zeros(6)

vals = np.zeros(6)

Ns = np.zeros(6)

for i in range(6):
    N = 100 * (2**i)
    Ns[i] = N
    errs[i], vals[i] = eval_num_samples(N)
    print("N: ", N, " error: ", errs[i], "val: ", vals[i])

plt.loglog(Ns, errs)
plt.savefig("err_plot.png")

plt.loglog(Ns, vals)
plt.savefig("val_plot.png")
