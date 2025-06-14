from random_walk import feynman_kac_sample, feynman_kac_correlated
from analyze_data import check_convergence
from mlmc import mlmc
from walk_on_spheres import walk_on_spheres, walk_on_spheres_correlated
from test_functions import cos, cos_rhs
from test_functions import sin, sin_rhs
from test_functions import non_hom_test, sq_cos, sq_cos_rhs
from test_functions import gaussian, gaussian_rhs
from test_functions import exp, exp_rhs
from test_functions import poly, poly_rhs
from test_functions import paper, paper_rhs
from benchmarks import benchmark_mlmc, create_large_parameter_set

from multiprocessing import Pool
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import pandas as pd

import numpy as np


def generate_samples(x, y, N, dt, f=sin, g=sin_rhs):

    samples = np.zeros((N, 2))
    sample_results = []
    n_procs = 10
    with Pool(processes=n_procs) as pool:
        for i in range(N):
            sample_results.append(pool.apply_async(walk_on_spheres,
                                                   args=(x,
                                                         y,
                                                         f,
                                                         g,
                                                         dt)))

        samples = np.array([r.get() for r in tqdm(sample_results, total=N)])

    return samples


def feynman_kac_eval(x, y, N, dt0, f=sin, g=sin_rhs):
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

        # np.savetxt(run_id + "_data.csv", samples,  delimiter=",")

        plt.loglog(Ns, errs, label="average errors dt="+str(dt))
        plt.scatter(Ns, errs)

        print("At position ", x, y, " we estimate a value of: ",
              samples.mean(), " vs the true value of: ", f(x, y),
              " with a difference of: ", samples.mean() - f(x, y))

    plt.loglog(Ns, Monte_Carlo_ideal, label="ideal convergence")
    plt.xlabel("N samples")
    plt.ylabel("average error")
    plt.legend()
    # plt.savefig(run_id + "_plot.png")
    plt.show()


def check_mlmc_speedup(N, epsilon, dt0, x=.5, y=.5):
    errs = np.zeros(N)
    expectation = np.zeros(N)
    cost = np.zeros(N)
    max_level = np.zeros(N)
    speedup = np.zeros(N)
    print("N", N)
    for i in range(N):
        expectation[i], cost[i], max_level[i], max_cost = mlmc(x, y,
                                                               sin,
                                                               sin_rhs,
                                                               dt0, epsilon)
        errs[i] = abs(sin(x, y) - expectation[i])

        dt = dt0 * 16**int(-max_level[i])
        Ntest = 10000
        samples_work = generate_samples(x, y, Ntest, dt)

        samples = samples_work[:, 0]
        work = samples_work[:, 1]

        varL = samples.std()**2
        NlCl = work.sum()
        costs_sd = NlCl * varL / epsilon**2 / Ntest
        speedup[i] = costs_sd / cost[i]

    return speedup.mean()


def plot_mlmc_speedup(N, dt0, x=.5, y=.5, phi=sin, rhs=sin_rhs):
    epsilons = [.1, 0.025, .01, 0.0025, .001]
    speedups = []

    for eps in epsilons:
        speedups.append(check_mlmc_speedup(N, eps, dt0))

    # Log-Log plot
    plt.loglog(epsilons, speedups, label="speedup")
    plt.scatter(epsilons, speedups)
    plt.xticks(epsilons)
    plt.title("average speedup of MLMC")
    plt.xlabel("target error")
    plt.grid()
    plt.ylim(0.1, 10)
    plt.ylabel("estimated speedup")
    plt.legend()

    print("average measured speedups:\n", speedups)

    # Show plot
    plt.show()


def output_mlmc_single_run(res, cost, max_level, uncorrelated_ratios, phi):
    delta = abs(res - phi(args.x, args.y))
    print(res, " vs. ", phi(args.x, args.y), " with a delta: ", delta)
    print("had to generate ", cost, " random numbers")
    print("and went up to level ", max_level)
    print("uncorrelation ratios are:\n", uncorrelated_ratios)


if __name__ == "__main__":
    # Set up argument parsing
    desc = 'Simulate feynman-kac poisson solver'

    # Available test functions:
    options = ["sin", "cos", "sq_cos", "poly", "gauss", "exp", "paper"]
    solution_dict = {"sin": sin,
                     "cos": cos,
                     "sq_cos": sq_cos,
                     "poly": poly,
                     "gauss": gaussian,
                     "exp": exp,
                     "paper": paper}
    rhs_dict = {"sin": sin_rhs,
                "cos": cos_rhs,
                "sq_cos": sq_cos_rhs,
                "poly": poly_rhs,
                "gauss": gaussian_rhs,
                "exp": exp_rhs,
                "paper": paper_rhs}

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('phi', choices=options)
    parser.add_argument('-e', '--epsilon', type=float, default=.01)
    parser.add_argument('-s', '--standard_mc', action='store_true')
    parser.add_argument('-d', '--dt0', type=float, default=.01)
    parser.add_argument('-N', '--N_samples', type=int, default=256000)
    parser.add_argument('-x', '--x', type=float, default=.5)
    parser.add_argument('-y', '--y', type=float, default=.5)
    parser.add_argument('-w', '--plot_walks', action='store_true')
    parser.add_argument('--single_sample', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--speedup', action='store_true')
    parser.add_argument('-b', '--benchmark', action='store_true')

    args = parser.parse_args()

    phi = solution_dict[args.phi]
    rhs = rhs_dict[args.phi]

    if args.plot_walks:
        if args.standard_mc:
            feynman_kac_correlated((args.x, args.y,
                                    phi, rhs,
                                    args.dt0, 1, 2), plot_walks=True)
        else:
            walk_on_spheres(args.x, args.y,
                            phi, rhs,
                            args.dt0,
                            plot_walk=True)
    elif args.standard_mc:
        if args.single_sample:
            walk_on_spheres(args.x, args.y, phi, rhs, args.dt0)
        feynman_kac_eval(args.x, args.y, args.N_samples, args.dt0, phi, rhs)
    elif args.speedup:
        plot_mlmc_speedup(args.N_samples, args.dt0, args.x, args.y, phi, rhs)
    elif args.benchmark:
        parameter_sets = create_large_parameter_set()
        res = benchmark_mlmc(parameter_sets, 10)
        res.to_csv("benchmark_data.csv")
    else:
        res, cost, max_level, uncorrelated_ratios = mlmc(args.x, args.y,
                                                         phi, rhs,
                                                         args.dt0,
                                                         args.epsilon,
                                                         debug=args.debug)
        output_mlmc_single_run(res, cost, max_level,
                               uncorrelated_ratios, phi)
