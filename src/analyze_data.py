import numpy as np


def test_phi(x, y):
    return np.sin(np.pi*x)*np.sin(np.pi*y)


def calculate_std(N: int, samples):
    # get standard deviation of the averages of subarrays of length N
    N_tot = len(samples)

    Var = 0
    average = samples.mean()

    for i in range(int(N_tot / N)):
        start_index: int = int(i*N)
        end_index: int = int((i+1)*N)
        Var += (samples[start_index:end_index].mean() - average)**2

    Var /= N_tot/N
    return np.sqrt(Var)


def calculate_mean_err(N: int, samples):
    mean_err = 0

    N_tot = len(samples)

    if N > N_tot or N_tot % N != 0:
        print("Oh no that's not good")
        return 1e10

    for i in range(int(N_tot / N)):
        start_index: int = int(i*N)
        end_index: int = int((i+1)*N)
        mean_err += abs(samples[start_index:end_index].mean() - test_phi(.5, .5))

    mean_err /= N_tot/N
    return mean_err


def check_convergence(samples, Ns):
    errs = np.zeros(len(Ns))

    for i in range(len(Ns)):
        N_cur = Ns[i]

        errs[i] = calculate_mean_err(N_cur, samples)

    return errs
