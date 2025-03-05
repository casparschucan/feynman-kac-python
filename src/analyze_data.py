import numpy as np


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


def check_convergence(samples, Ns):
    errs = np.zeros(len(Ns))

    for i in range(len(Ns)):
        N_cur = Ns[i]

        errs[i] = calculate_std(N_cur, samples)

    return errs
