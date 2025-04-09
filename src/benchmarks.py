import pandas as pd
from mlmc import mlmc
from test_functions import *
from multiprocessing import Pool
import time
import numpy as np
from tqdm import tqdm


test_functions = {"Sine": sin,
                  "Cosine": cos,
                  "Cosine^2": sq_cos,
                  "Polynomial": poly,
                  "Gaussian": gaussian,
                  "Exponential": exp}
test_rhs = {"Sine": sin_rhs,
            "Cosine": cos_rhs, 
            "Cosine^2": sq_cos_rhs,
            "Polynomial": poly_rhs,
            "Gaussian": gaussian_rhs,
            "Exponential": exp_rhs}


def run_single_benchmark(params):
        total_time = 0
        total_cost = 0
        avg_error = 0
        max_levels = []
        runs = 10
        solution = params["f"]
        x = params["x"]
        y = params["y"]
        for _ in range(runs):
            start_time = time.time()
            expectation, cost, max_level, _ = mlmc(**params)
            end_time = time.time()

            total_time += (end_time - start_time)
            total_cost += cost
            max_levels.append(max_level)
            error = abs(expectation - solution(x, y))
            avg_error += error

        avg_error /= runs
        f_name = "Sine"
        for name in test_functions.keys():
            if test_functions[name] == params["f"]:
                f_name = name
                break

        current_result = {
                          **params,
                          'avg_execution_time': total_time / runs,
                          'avg_computational_cost': total_cost / runs,
                          'avg_max_level': sum(max_levels) / len(max_levels),
                          'expectation': expectation,  # Last computed value
                          'avg_error': avg_error,
                          'test_name': f_name
                         }

        return current_result

def benchmark_mlmc(parameter_sets, runs=5):
    """
    Benchmark the mlmc function with different parameter sets.

    Args:
        parameter_sets: List of dictionaries, each containing parameters for mlmc
        runs: Number of times to run each parameter set for averaging

    Returns:
        DataFrame with benchmark results
    """
    results = []

    parameter_sets = create_large_parameter_set()
    n_procs_outer = 10
    with Pool(n_procs_outer) as pool:
        sample_results = pool.imap_unordered(run_single_benchmark,
                                             (params for params in parameter_sets)
                                             )
        for result in tqdm(sample_results):
            results.append(result)

    return pd.DataFrame(results)


def create_large_parameter_set():
    parameter_sets = []
    coord_range = np.linspace(.1, .9, 9)
    for f in test_functions.keys():
        for x in coord_range:
            for y in coord_range:
                parameter_sets.append({
                    'x': x, 'y': y, 'f': test_functions[f], 'g': test_rhs[f],
                    'dt0': 1e-3, 'epsilon': 0.01
                })
    return parameter_sets
