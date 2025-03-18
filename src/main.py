from random_walk import feynman_kac_sample, feynman_kac_correlated
from analyze_data import check_convergence
from mlmc import mlmc

from multiprocessing import Pool
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

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

            samples = np.array([r.get() for r in tqdm(sample_results, total=N)])

    return samples


def feynman_kac_eval(x, y, N, dt0):
    Ns = 100 * np.logspace(0, 5, 6, base=2)

    Monte_Carlo_ideal = np.logspace(0, 5, 6, base=(1/np.sqrt(2)))

    dts = dt0 * np.logspace(0, 5, 6, base=1/2)
    print(dts)

    for dt in dts:
        samples = generate_samples(x, y, N, dt)

        errs = check_convergence(samples, Ns)

        run_id = str(N) + "_" + str(dt) + "_" + str(x) + "_" + str(y)

        np.savetxt(run_id + "_data.csv", samples,  delimiter=",")

        plt.loglog(Ns, errs, label="average errors dt="+str(dt))
        plt.scatter(Ns, errs)

        print("At position ", x, y, " we estimate a value of: ", samples.mean(),
              " vs the true value of: ", test_phi(x, y),
              " with a difference of: ", samples.mean() - test_phi(x, y))

    plt.loglog(Ns, Monte_Carlo_ideal, label="ideal convergence")
    plt.xlabel("N samples")
    plt.ylabel("average error")
    plt.legend()
    plt.savefig(run_id + "_plot.png")
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

    args = parser.parse_args()

    if args.standard_mc:
        feynman_kac_eval(args.x, args.y, args.N_samples, args.dt0)
    elif args.plot_walks:
        feynman_kac_correlated((args.x, args.y, test_bound, test_rhs, args.dt0, 1),
                               plot_walks=True)
    elif args.non_homogeneous:
        res, cost, max_level = mlmc(args.x, args.y,
                                    non_hom_test, non_hom_test,
                                    args.dt0, args.epsilon,
                                    debug=args.debug)
        print(res, " vs. ", non_hom_test(args.x, args.y))
        print("had to generate ", cost, " random numbers")
        print("and went up to level ", max_level)
    else:
        res, cost, max_level = mlmc(args.x, args.y,
                                    test_bound, test_rhs,
                                    args.dt0, args.epsilon,
                                    debug=args.debug)
        print(res, " vs. ", test_phi(args.x, args.y))
        print("had to generate ", cost, " random numbers")
        print("and went up to level ", max_level)
