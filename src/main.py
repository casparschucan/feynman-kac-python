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


print("starting loop")


def eval_num_samples(N, x, y):
    average = 0
    Var = 0

    for i in range(40):
        temp = generate_samples(x, y, N, .00001)
        average += temp
        Var += temp**2

        if (i+1) % 10 == 0:
            error = Var/(i+1) - ((average/(i+1))**2)
            print("After ", i+1,
                  " iterations, we have reached a standard deviation of: ",
                  error)

    Var /= 40
    average /= 40
    Var -= average**2

    return np.sqrt(Var), average


def eval_samples(N: int, samples):
    # get standard deviation of the averages of subarrays of length N
    N_tot = len(samples)

    Var = 0
    average = samples.mean()

    for i in range(int(N_tot / N)):
        start_index: int = int(i*N)
        end_index: int = int((i+1)*N)
        Var += (samples[start_index:end_index].mean() - average)**2

    return np.sqrt(Var)


def check_convergence(x, y, N, dt):
    errs = np.zeros(6)
    Ns = 100 * np.logspace(0, 5, 6)

    Monte_Carlo_ideal = np.zeros(6)

    samples = generate_samples(x, y, N, dt)

    pos_string = str(x) + "_" + str(y)

    np.savetxt(pos_string + "_data.csv", samples,  delimiter=",")

    for i in range(6):
        N_cur = Ns[i]
        Monte_Carlo_ideal[i] = 1/(np.sqrt(2)**i)

        errs[i] = eval_samples(N_cur, samples)

    plt.loglog(Ns, errs)
    plt.loglog(Ns, Monte_Carlo_ideal)
    plt.savefig(pos_string + "_plot.png")

    print("At position ", x, y, " we estimate a value of: ", samples.mean(),
          " vs the true value of: ", test_phi(x, y),
          " with a differente of: ", samples.mean() - test_phi(x, y))


test_positions = np.linspace(0.1, 0.5, 5)

for i in range(5):
    for j in range(i, 5):
        N = 128000
        dt = 0.00001
        x = test_positions[i]
        y = test_positions[j]
        check_convergence(x, y, N, dt)
