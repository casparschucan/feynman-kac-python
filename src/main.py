from random_walk import feynman_kac_sample, feynman_kac_correlated
from analyze_data import check_convergence
from mlmc import mlmc

from multiprocessing import Pool
import matplotlib.pyplot as plt

import numpy as np


def non_hom_test(x, y):
    return np.sin(x)*np.sin(y)


def test_phi(x, y):
    return np.sin(np.pi*x)*np.sin(np.pi*y)


def test_bound(x, y):
    return 0


def test_rhs(x, y):
    return np.pi**2 * (np.sin(np.pi * x)*np.sin(np.pi * y))


def generate_samples(x, y, N, dt):
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

    return samples


def feynman_kac_eval(x, y, N, dt):
    Ns = 100 * np.logspace(0, 5, 7, base=2)

    Monte_Carlo_ideal = np.logspace(0, 5, 7, base=(1/np.sqrt(2)))

    samples = generate_samples(x, y, N, dt)

    errs = check_convergence(samples, Ns)

    run_id = str(N) + "_" + str(dt) + "_" + str(x) + "_" + str(y)

    np.savetxt(run_id + "_data.csv", samples,  delimiter=",")

    plt.loglog(Ns, errs)
    plt.loglog(Ns, Monte_Carlo_ideal)
    plt.savefig(run_id + "_plot.png")

    print("At position ", x, y, " we estimate a value of: ", samples.mean(),
          " vs the true value of: ", test_phi(x, y),
          " with a difference of: ", samples.mean() - test_phi(x, y))
    plt.show()


test_positions = np.linspace(0.1, 0.5, 5)

N = 25600
dt = 0.00001

x = .5
y = .5

# feynman_kac_eval(x, y, N, dt)
print(mlmc(x, y, test_bound, test_rhs, .025, .001), " vs ", test_phi(x, y))
# samples = generate_samples(x, y, 100000, .0001)
# print(samples.mean())
# args = (x, y, test_bound, test_rhs, 0.005, 2)
# feynman_kac_correlated(args)
