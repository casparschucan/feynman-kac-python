from random_walk import feynman_kac_sample, feynman_kac_correlated
from analyze_data import check_convergence
from mlmc import mlmc
from test_functions import test_cos, test_cos_rhs
from test_functions import test_bound, test_rhs, test_phi
from test_functions import non_hom_test

from multiprocessing import Pool
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

import numpy as np


def generate_samples(x, y, N, dt, f=test_phi, g=test_rhs):

    samples = np.zeros((N, 2))
    sample_results = []
    n_procs = 10
    with Pool(processes=n_procs) as pool:
        for i in range(N):
            sample_results.append(pool.apply_async(feynman_kac_sample,
                                                   args=(x,
                                                         y,
                                                         f,
                                                         g,
                                                         dt)))

        samples = np.array([r.get() for r in tqdm(sample_results, total=N)])

    return samples


def feynman_kac_eval(x, y, N, dt0, f=test_phi, g=test_rhs):
    n = 8
    Ns = 1000 * np.logspace(0, n-1, n, base=2)

    Monte_Carlo_ideal = np.logspace(0, n-1, n, base=(1/np.sqrt(2)))

    dts = dt0 * np.logspace(0, 5, 6, base=1/2)
    print(dts)

    for dt in dts:
        samples_work = generate_samples(x, y, N, dt, f, g)

        samples = samples_work[:, 0]

        errs = check_convergence(samples, Ns, x, y, f)

        run_id = str(N) + "_" + str(dt) + "_" + str(x) + "_" + str(y)

        np.savetxt(run_id + "_data.csv", samples,  delimiter=",")

        plt.loglog(Ns, errs, label="average errors dt="+str(dt))
        plt.scatter(Ns, errs)

        print("At position ", x, y, " we estimate a value of: ",
              samples.mean(), " vs the true value of: ", f(x, y),
              " with a difference of: ", samples.mean() - f(x, y))

    plt.loglog(Ns, Monte_Carlo_ideal, label="ideal convergence")
    plt.xlabel("N samples")
    plt.ylabel("average error")
    plt.legend()
    plt.savefig(run_id + "_plot.png")
    plt.show()


def check_mlmc_speedup(N, epsilon, dt0, x=.5, y=.5):
    errs = np.zeros(N)
    expectation = np.zeros(N)
    cost = np.zeros(N)
    max_level = np.zeros(N)
    speedup = np.zeros(N)
    for i in range(N):
        expectation[i], cost[i], max_level[i], max_cost = mlmc(x, y,
                                                               test_bound,
                                                               test_rhs,
                                                               dt0, epsilon)
        errs[i] = abs(test_phi(x, y) - expectation[i])

        dt = dt0 * 2**int(-max_level[i])
        Ntest = 10000
        samples_work = generate_samples(x, y, Ntest, dt)

        samples = samples_work[:, 0]
        work = samples_work[:, 1]

        varL = samples.std()**2
        NlCl = work.sum()
        costs_sd = NlCl * varL / epsilon**2
        speedup[i] = costs_sd / cost[i]

    return speedup.mean()


def plot_mlmc_speedup(N, dt0, x=.5, y=.5):
    epsilons = [.1, .05, .025, .0125]
    speedups = []

    for eps in epsilons:
        speedups.append(check_mlmc_speedup(N, eps, dt0))

    # Log-Log plot
    plt.loglog(epsilons, speedups, label="speedup")
    plt.title("average speedup")
    plt.xlabel("target epsilon")
    plt.ylabel("estimated speedup")
    plt.legend()

    print("average measured speedups:\n", speedups)

    # Show plot
    plt.show()


if __name__ == "__main__":
    # Set up argument parsing
    desc = 'Simulate feynman-kac poisson solver'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-e', '--epsilon', type=float, default=.01)
    parser.add_argument('-s', '--standard_mc', action='store_true')
    parser.add_argument('-d', '--dt0', type=float, default=.01)
    parser.add_argument('-N', '--N_samples', type=int, default=256000)
    parser.add_argument('-x', '--x', type=float, default=.5)
    parser.add_argument('-y', '--y', type=float, default=.5)
    parser.add_argument('--non_homogeneous', action='store_true')
    parser.add_argument('-w', '--plot_walks', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--speedup', action='store_true')

    args = parser.parse_args()

    if args.standard_mc:
        if args.non_homogeneous:
            feynman_kac_eval(args.x, args.y, args.N_samples,
                             args.dt0, test_cos, test_cos_rhs)
        else:
            feynman_kac_eval(args.x, args.y, args.N_samples, args.dt0)
    elif args.plot_walks:
        feynman_kac_correlated((args.x, args.y, test_bound, test_rhs,
                                args.dt0, 1),
                               plot_walks=True)
    elif args.non_homogeneous:
        res, cost, max_level, _ = mlmc(args.x, args.y,
                                       non_hom_test, non_hom_test,
                                       args.dt0, args.epsilon,
                                       debug=args.debug)
        print(res, " vs. ", non_hom_test(args.x, args.y))
        print("had to generate ", cost, " random numbers")
        print("and went up to level ", max_level)
    elif args.speedup:
        plot_mlmc_speedup(args.N_samples, args.dt0)
    else:
        res, cost, max_level, uncorrelated_ratios = mlmc(args.x, args.y,
                                                         test_bound, test_rhs,
                                                         args.dt0,
                                                         args.epsilon,
                                                         debug=args.debug)
        print(res, " vs. ", test_phi(args.x, args.y))
        print("had to generate ", cost, " random numbers")
        print("and went up to level ", max_level)
        print("uncorrelation ratios are:\n", uncorrelated_ratios)
