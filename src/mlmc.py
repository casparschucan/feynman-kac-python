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
                       dt_ratio,
                       debug=False):
    work = 0
    sample_sum = 0
    sample_sum_sq = 0
    uncorrelated_sum = 0
    if debug:
        print("generating ", N_samples, " samples")

    n_procs = 10

    chunk_size = max(N_samples // (10 * n_procs), 10)
    chunk_size = min(chunk_size, 10000)

    with Pool(processes=n_procs) as pool:

        sample_results = pool.imap_unordered(feynman_kac_correlated,
                                             ((x, y, f, g,
                                               dt_fine, level, dt_ratio)
                                              for _ in range(N_samples)),
                                             chunksize=chunk_size)

        for sample in sample_results:
            sample_sum += sample[0]
            sample_sum_sq += sample[0]**2
            work += sample[1]
            uncorrelated_sum += sample[2]

    return sample_sum, sample_sum_sq, work, uncorrelated_sum


def mlmc(x: float, y: float, f, g, dt0: float, epsilon: float,
         debug=False, dt_ratio=2):
    max_level = 3
    N_start = 1000
    N_samples = np.full(max_level, N_start)
    N_samples_diff = np.full(max_level, N_start)

    converged = False
    costs = np.zeros(max_level)
    sample_sums = np.zeros(max_level)
    sample_sums_sq = np.zeros(max_level)
    uncor_sums = np.zeros(max_level)

    while not converged:

        # generate data based on last optimal estimate
        for level in range(max_level):
            dt = dt0/dt_ratio**level
            # add the needed samples
            ss, ss_sq, work, uncor = generate_mlmc_data(x, y,
                                                        f, g,
                                                        dt, level,
                                                        N_samples_diff[level],
                                                        dt_ratio,
                                                        debug=debug)
            costs[level] += work
            sample_sums[level] += ss
            sample_sums_sq[level] += ss_sq
            uncor_sums[level] += uncor

        # array containing the cost per sample for every level
        cost_at_level = costs / N_samples
        # array containing the variance for every level with Bessel correction
        variances = ((sample_sums_sq / N_samples -
                      (sample_sums / N_samples)**2)
                     * N_samples / (N_samples - 1))
        # the sum $\sum_{l=0}^{L}\sqrt(V_l*C_l)$
        var_cost_sq_sum = np.sum(np.sqrt(variances * cost_at_level))
        if debug:
            # print("Samples per level:", N_samples)
            print("Variance per level:\n", variances)
            # print("Expectation:\n", sample_sums/N_samples)

        # check how many samples are needed for each level and add samples
        # when necessary
        for level in range(max_level):
            # optimal numbers of samples per Lagrange multiplier
            optimal_n_samples = int(1/(epsilon**2)
                                    * np.sqrt(variances[level]
                                              / cost_at_level[level])
                                    * var_cost_sq_sum)
            # update additional samples needed
            N_samples_diff[level] = max(0,
                                        optimal_n_samples - N_samples[level])
            N_samples[level] = max(N_samples[level], optimal_n_samples)

            dt = dt0/dt_ratio**level
            # add the needed samples
            ss, ss_sq, work, uncor = generate_mlmc_data(x, y,
                                                        f, g,
                                                        dt, level,
                                                        N_samples_diff[level],
                                                        dt_ratio,
                                                        debug=debug)
            costs[level] += work
            sample_sums[level] += ss
            sample_sums_sq[level] += ss_sq
            uncor_sums[level] += uncor

        # array containing the cost per sample for every level
        cost_at_level = costs / N_samples
        # array containing the variance for every level with Bessel correction
        variances = ((sample_sums_sq / N_samples -
                      (sample_sums / N_samples)**2)
                     * N_samples / (N_samples - 1))
        # find convergence by linear fit
        # Specifically convergence rate alpha of the error,
        # beta of the variance and gamma of the cost
        x_conv = np.linspace(1, max_level-1, max_level-1)
        y_conv = np.log2(np.abs(sample_sums[1:max_level]
                                / N_samples[1:max_level]))
        alpha, _ = np.polyfit(x_conv, y_conv, 1)
        alpha = min(alpha, -.5)

        y_var = -np.log2(variances[1:])
        beta, _ = np.polyfit(x_conv, y_var, 1)
        beta = max(beta, .5)

        y_cost = np.log2(cost_at_level[1:])
        gamma, _ = np.polyfit(x_conv, y_cost, 1)
        gamma = max(gamma, .5)

        if debug:
            print("Measured convergence rate ", -alpha)

        # check convergence
        conv_lhs = np.abs(sample_sums[max_level-1]) / N_samples[max_level-1]
        conv_lhs /= (2.0**(-alpha) - 1)
        if debug:
            print("the estimated error is: ", conv_lhs * np.sqrt(2))
        if conv_lhs < epsilon/2:
            converged = True
        else:
            variances = np.append(variances,
                                  max(variances[max_level-1] / (2**beta),
                                      1e-10))
            cost_at_level = np.append(cost_at_level,
                                      cost_at_level[max_level-1] * (2**gamma))
            var_cost_sq_sum += np.sqrt(cost_at_level[max_level]
                                       * variances[max_level])

            max_level += 1
            N_samples = np.append(N_samples, 0)
            N_samples_diff = np.append(N_samples_diff, 0)
            sample_sums = np.append(sample_sums, 0)
            sample_sums_sq = np.append(sample_sums_sq, 0)
            costs = np.append(costs, 0)
            uncor_sums = np.append(uncor_sums, 0)

            for level in range(max_level):
                # optimal numbers of samples per Lagrange multiplier
                optimal_n_samples = int(1/(epsilon**2)
                                        * np.sqrt(variances[level]
                                                  / cost_at_level[level])
                                        * var_cost_sq_sum)
                optimal_n_samples = max(optimal_n_samples, 500)
                N_samples_diff[level] = max(0, (optimal_n_samples
                                                - N_samples[level]))

                N_samples[level] = max(N_samples[level], optimal_n_samples)

    if debug:
        print("Expectation value per level: ", sample_sums/N_samples)
    expectation = np.sum(sample_sums/N_samples)
    work = costs.sum()
    return expectation, work, max_level, cost_at_level[max_level-1]
