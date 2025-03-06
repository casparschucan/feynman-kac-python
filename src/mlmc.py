from random_walk import feynman_kac_correlated

import numpy as np


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

    for i in range(N_samples):
        sample, sample_work = feynman_kac_correlated(x, y, f, g, dt_fine, level)
        work += sample_work
        sample_sum += sample
        sample_sum_sq += sample**2

    return sample_sum, sample_sum_sq, work


def mlmc(x: float, y: float, f, g, dt0: float, epsilon: float):
    max_level = 2
    N_samples = np.full(max_level + 1, 100)
    N_samples_diff = np.full(max_level + 1, 100)

    converged = False
    work_at_level = np.zeros(max_level + 1)
    sample_sum_at_level = np.zeros(max_level + 1)
    sample_sum_sq_at_level = np.zeros(max_level + 1)

    sample_sum_at_level[0], sample_sum_sq_at_level[0], work_at_level[0] = generate_mlmc_data(x, y,
                                                                                                 f, g,
                                                                                                 dt0, 0,
                                                                                                 N_samples[0])

    sample_sum_at_level[1], sample_sum_sq_at_level[1], work_at_level[1] = generate_mlmc_data(x, y,
                                                                                                 f, g,
                                                                                                 dt0/2, 1,
                                                                                                 N_samples[1])

    while not converged:
        dt_finest = dt0/(2**(max_level - 1))
        sample_sum, sample_sum_sq, work = generate_mlmc_data(x, y,
                                                                 f, g,
                                                                 dt_finest, max_level,
                                                                 N_samples[max_level])
        work_at_level[max_level] = work
        sample_sum_at_level[max_level] = sample_sum
        sample_sum_sq_at_level[max_level] = sample_sum_sq

        # array containing the cost per sample for every level
        cost_at_level = work_at_level / N_samples
        # array containing the variance for every level
        var_at_level = (sample_sum_sq_at_level / N_samples
                        - (sample_sum_at_level / N_samples)**2)

        # the sum $\sum_{l=0}^{L}\sqrt(V_l*C_l)$
        var_cost_sq_sum = np.sum(np.sqrt(var_at_level * cost_at_level))

        # check how many samples are needed for each level
        for level in range(max_level):
            # optimal numbers of samples per Lagrange multiplier
            optimal_n_samples = int(2/(epsilon**2)
                                    * np.sqrt(var_at_level[level]
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
            work_at_level[level] += work
            sample_sum_at_level += sample_sum
            sample_sum_sq += sample_sum_sq

        # find convergence by linear fit
        x = np.linspace(2, max_level, max_level-1)
        y = np.log(np.abs(sample_sum_at_level[1:max_level]/N_samples[1:max_level]))
        a, b = np.polyfit(x, y, 1)

        # check convergence
        conv_lhs = sample_sum_at_level[max_level]/N_samples[max_level]/(2.0**(-b) - 1)
        if conv_lhs < epsilon/np.sqrt(2):
            converged = True
        else:
            max_level += 1
            N_samples.append(100)
            N_samples_diff.append(100)
            sample_sum_at_level.append(0)
            sample_sum_sq_at_level.append(0)
            work_at_level.append(0)

    expectation = np.sum(sample_sum_at_level/N_samples)/max_level
    return expectation
