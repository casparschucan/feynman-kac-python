from random_walk import feynman_kac_correlated

import numpy as np
from multiprocessing import Pool


def generate_mlmc_data(x: float,
                       y: float,
                       f,
                       g,
                       dt_fine: float,
                       level: int,
                       N_samples: int,
                       debug=False):
    work = 0
    sample_sum = 0
    sample_sum_sq = 0
    counter = 0
    if debug:
        print("generating ", N_samples, " samples")

    n_procs = 10

    chunk_size = max(N_samples // n_procs, 10)
    chunk_size = min(chunk_size, 100000)

    with Pool(processes=n_procs) as pool:

        sample_results = pool.imap_unordered(feynman_kac_correlated,
                                         ((x, y, f, g, dt_fine, level) for _ in range(N_samples)),
                                         chunksize=chunk_size)

        for sample in sample_results:
            sample_sum += sample[0]
            sample_sum_sq += sample[0]**2
            work += sample[1]
            counter += 1
            if (counter % chunk_size == 0) and debug:
                print("done with another chunk_size: ", chunk_size)

    return sample_sum, sample_sum_sq, work


def mlmc(x: float, y: float, f, g, dt0: float, epsilon: float, debug=False):
    max_level = 3
    N_start = 1000
    N_samples = np.full(max_level, N_start)
    N_samples_diff = np.full(max_level, N_start)

    converged = False
    costs = np.zeros(max_level)
    sample_sums = np.zeros(max_level)
    sample_sums_sq = np.zeros(max_level)

    sample_sums[0], sample_sums_sq[0], costs[0] = generate_mlmc_data(x, y,
                                                                     f, g,
                                                                     dt0, 0,
                                                                     N_samples[0],
                                                                     debug=debug)

    sample_sums[1], sample_sums_sq[1], costs[1] = generate_mlmc_data(x, y,
                                                                     f, g,
                                                                     dt0/2, 1,
                                                                     N_samples[1],
                                                                     debug=debug)

    while not converged:
        dt_finest = dt0/(2**(max_level - 1))
        sample_sum, sample_sum_sq, work = generate_mlmc_data(x, y,
                                                             f, g,
                                                             dt_finest,
                                                             max_level,
                                                             N_samples_diff[max_level - 1],
                                                             debug=debug)
        costs[max_level - 1] = work
        sample_sums[max_level - 1] = sample_sum
        sample_sums_sq[max_level - 1] = sample_sum_sq

        # array containing the cost per sample for every level
        cost_at_level = costs / N_samples
        # array containing the variance for every level
        variances = (sample_sums_sq / N_samples
                     - (sample_sums / N_samples)**2)

        # the sum $\sum_{l=0}^{L}\sqrt(V_l*C_l)$
        var_cost_sq_sum = np.sum(np.sqrt(variances * cost_at_level))
        if debug:
            print("Samples per level:", N_samples)
            print("Variance per level: ", variances)
            print("Cost per sample per level: ", cost_at_level)
        # check how many samples are needed for each level
        for level in range(max_level):
            # optimal numbers of samples per Lagrange multiplier
            optimal_n_samples = int(2/(epsilon**2)
                                    * np.sqrt(variances[level]
                                              / cost_at_level[level])
                                    * var_cost_sq_sum)
            N_samples_diff[level] = max(0, optimal_n_samples - N_samples[level])
            N_samples[level] = max(N_samples[level], optimal_n_samples)

            dt = dt0/2**level
            # add the needed samples
            sample_sum, sample_sum_sq, work = generate_mlmc_data(x, y,
                                                                 f, g,
                                                                 dt, level,
                                                                 N_samples_diff[level],
                                                                 debug=debug)
            costs[level] += work
            sample_sums[level] += sample_sum
            sample_sums_sq[level] += sample_sum_sq

        # find convergence by linear fit
        x_conv = np.linspace(2, max_level, max_level-1)
        y_conv = np.log2(np.abs(sample_sums[1:max_level]/N_samples[1:max_level]))
        a, b = np.polyfit(x_conv, y_conv, 1)

        if debug:
            print("Measured convergence rate ", -a)

        # check convergence
        conv_lhs = np.abs(sample_sums[max_level-1]) / N_samples[max_level-1]
        conv_lhs /= (2.0**(-a) - 1)
        if debug:
            print("the estimated error is: ", conv_lhs * np.sqrt(2))
        if conv_lhs < epsilon/np.sqrt(2):
            converged = True
        else:
            max_level += 1
            N_samples = np.append(N_samples, N_start)
            N_samples_diff = np.append(N_samples_diff, N_start)
            sample_sums = np.append(sample_sums, 0)
            sample_sums_sq = np.append(sample_sums_sq, 0)
            costs = np.append(costs, 0)

    if debug:
        print("Expectation value per level: ", sample_sums/N_samples)
    expectation = np.sum(sample_sums/N_samples)
    work = costs.sum()
    return expectation, work, max_level
