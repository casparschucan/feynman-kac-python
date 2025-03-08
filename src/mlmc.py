from random_walk import feynman_kac_correlated

import numpy as np
from multiprocessing import Pool


def generate_mlmc_data(x: float,
                       y: float,
                       f,
                       g,
                       dt_fine: float,
                       level: int,
                       N_samples: int):
    work = 0
    sample_sum = 0
    sample_sum_sq = 0

    print("generating ", N_samples, " samples")

    sample_results = []
    n_procs = 10

    with Pool(processes=n_procs) as pool:
        for i in range(N_samples):
            if (i+1) % (N_samples // 10) == 0:
                print("done with another 10%")

            sample_results.append(pool.apply_async(feynman_kac_correlated,
                                                   args=(x, y,
                                                         f, g,
                                                         dt_fine,
                                                         level)))

        results = np.array([r.get() for r in sample_results])

        sample_sum = results[:, 0].sum()
        sample_sum_sq = np.linalg.vector_norm(results[:, 0])**2
        work = results[:, 1].sum()
        # for r in sample_results:
            # sample = r.get()
            # if sample != 0:
                # print("Alright that's a start")
            # sample_sum += sample
            # sample_sum_sq += sample**2

    # for i in range(N_samples):
        # sample = feynman_kac_correlated(x, y, f, g, dt_fine, level)
        # work += 1
        # sample_sum += sample
        # sample_sum_sq += sample**2

    return sample_sum, sample_sum_sq, work


def mlmc(x: float, y: float, f, g, dt0: float, epsilon: float):
    max_level = 3
    N_samples = np.full(max_level, 100)
    N_samples_diff = np.full(max_level, 100)

    converged = False
    costs = np.zeros(max_level)
    sample_sums = np.zeros(max_level)
    sample_sums_sq = np.zeros(max_level)

    sample_sums[0], sample_sums_sq[0], costs[0] = generate_mlmc_data(x, y,
                                                                     f, g,
                                                                     dt0, 0,
                                                                     N_samples[0])

    sample_sums[1], sample_sums_sq[1], costs[1] = generate_mlmc_data(x, y,
                                                                     f, g,
                                                                     dt0/2, 1,
                                                                     N_samples[1])

    while not converged:
        dt_finest = dt0/(2**(max_level - 1))
        sample_sum, sample_sum_sq, work = generate_mlmc_data(x, y,
                                                             f, g,
                                                             dt_finest,
                                                             max_level,
                                                             N_samples[max_level - 1])
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

        print(variances)
        print(cost_at_level)
        # check how many samples are needed for each level
        for level in range(max_level):
            # optimal numbers of samples per Lagrange multiplier
            optimal_n_samples = int(2/(epsilon**2)
                                    * np.sqrt(variances[level]
                                              / cost_at_level[level])
                                    * var_cost_sq_sum)
            N_samples_diff = max(0, optimal_n_samples - N_samples[level])
            N_samples[level] = max(N_samples[level], optimal_n_samples)

            dt = dt0/2**level
            # add the needed samples
            sample_sum, sample_sum_sq, work = generate_mlmc_data(x, y,
                                                                 f, g,
                                                                 dt, level,
                                                                 N_samples[level])
            costs[level] += work
            sample_sums[level] += sample_sum
            sample_sums_sq[level] += sample_sum_sq

        # find convergence by linear fit
        x = np.linspace(2, max_level, max_level-1)
        y = np.log(np.abs(sample_sums[1:max_level]/N_samples[1:max_level]))
        a, b = np.polyfit(x, y, 1)

        # check convergence
        conv_lhs = sample_sums[max_level-1] / N_samples[max_level-1]
        conv_lhs /= (2.0**(-b) - 1)
        if conv_lhs < epsilon/np.sqrt(2):
            converged = True
        else:
            max_level += 1
            N_samples = np.append(N_samples, 100)
            N_samples_diff = np.append(N_samples_diff, 100)
            sample_sums = np.append(sample_sums, 0)
            sample_sums_sq = np.append(sample_sums_sq, 0)
            costs = np.append(costs, 0)

    expectation = np.sum(sample_sums/N_samples)
    return expectation
