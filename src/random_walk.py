import numpy as np
from visualize import visualize_random_walk
import os
import secrets
import threading


# Create a global dictionary to store thread-local RNGs
_thread_local_rngs = {}


def get_rng():
    """Get a thread-local random number generator."""
    thread_id = threading.get_ident()

    if thread_id not in _thread_local_rngs:
        # Create a new RNG for this thread
        # Use secrets for cryptographically strong randomness
        seed = secrets.randbits(64) ^ os.getpid()
        _thread_local_rngs[thread_id] = np.random.default_rng(seed)

    return _thread_local_rngs[thread_id]
# a function samples a random walk starting from a specific point
# and returns the needed values for feynman-kac poisson eval on the unit square


def feynman_kac_sample_with_work(x0: float, y0: float, f, g, dt):

    x = x0
    y = y0
    integral = 0
    num_steps = 0

    # Convert timestamp to microseconds (integer)
    gen = get_rng()

    while x > 0 and y > 0 and x < 1 and y < 1:
        eps_x = gen.normal(scale=np.sqrt(dt))

        eps_y = gen.normal(scale=np.sqrt(dt))
        if not is_in_domain(x + eps_x, y + eps_y):
            t, x_edge, y_edge = calculate_exit(x, y, eps_x, eps_y)
            integral += t * dt * g(x, y)
            x = x_edge
            y = y_edge
            break

        integral += g(x, y)*dt
        y += eps_y
        x += eps_x

        num_steps += 1

    integral += f(x, y)

    assert (abs(x - 1) < 1e-10) or (abs(x) < 1e-10) or (abs(y - 1) < 1e-10) or (abs(y) < 1e-10)
    return integral, num_steps, False


def feynman_kac_sample(x0: float, y0: float, f, g, dt):
    integral, num_steps, _ = feynman_kac_sample_with_work(x0, y0, f, g, dt)
    ret = np.array([integral, num_steps])
    return ret


def is_in_domain(x, y):
    return x > 0 and y > 0 and x < 1 and y < 1


def project_to_domain_edge(x, y):
    if not is_in_domain(x, y):
        if x < 0:
            x = 0
        elif x > 1:
            x = 1

        if y < 0:
            y = 0
        elif y > 1:
            y = 1
    elif x < y:
        if x < 1-y:
            x = 0
        elif y < .5:
            y = 0
        else:
            y = 1
    else:
        if y < 1-x:
            y = 0
        elif x < 0.5:
            x = 0
        else:
            x = 1

    return x, y


def generate_fine_random(N_samples, dt):
    gen = get_rng()
    dx = np.array([gen.normal(scale=np.sqrt(dt)) for _ in range(N_samples)])
    dy = np.array([gen.normal(scale=np.sqrt(dt)) for _ in range(N_samples)])
    return dx, dy


def calculate_exit(x, y, delta_x, delta_y):
    t = 1
    if delta_x + x <= 0:
        t = min(t, -x/delta_x)
    elif delta_x + x >= 1:
        t = min(t, (1-x)/delta_x)

    if delta_y + y <= 0:
        t = min(t, -y/delta_y)
    elif delta_y + y >= 1:
        t = min(t, (1-y)/delta_y)

    if t < 0:
        print("nooooo ", t)

    assert (t <= 1) and (t >= 0)
    return t, x + delta_x * t, y + delta_y * t


def feynman_kac_correlated(args, plot_walks=False):
    x0, y0, f, g, dt_fine, level, dt_ratio = args
    if (level == 0):
        return feynman_kac_sample_with_work(x0, y0, f, g, dt_fine)

    x_fine = x0
    x_coarse = x0
    y_fine = y0
    y_coarse = y0

    coarse_integral = 0
    fine_integral = 0

    dt_coarse = dt_ratio*dt_fine
    num_steps = 0

    fine_in = True
    coarse_in = True
    uncorrelated = False

    steps1_x = [x0]
    steps2_x = [x0]
    steps1_y = [y0]
    steps2_y = [y0]

    while fine_in or coarse_in:
        if not fine_in:
            uncorrelated = True
        eps_x, eps_y = generate_fine_random(dt_ratio, dt_fine)
        num_steps += dt_ratio
        for i in range(dt_ratio):
            if not fine_in:
                break
            if not is_in_domain(x_fine + eps_x[i], y_fine + eps_y[i]):
                t, x, y = calculate_exit(x_fine, y_fine, eps_x[i], eps_y[i])
                fine_in = False
                fine_integral += t * dt_fine * g(x_fine, y_fine)
                x_fine = x
                y_fine = y
                if plot_walks:
                    steps1_x.append(x_fine)
                    steps1_y.append(y_fine)
                break

            fine_integral += g(x_fine, y_fine)*dt_fine
            x_fine += eps_x[i]
            y_fine += eps_y[i]
            if plot_walks:
                steps1_x.append(x_fine)
                steps1_y.append(y_fine)

            fine_in = is_in_domain(x_fine, y_fine)

        eps_x_coarse = eps_x.sum()

        eps_y_coarse = eps_y.sum()

        if not is_in_domain(x_coarse + eps_x_coarse, y_coarse + eps_y_coarse):
            t, x, y = calculate_exit(x_coarse, y_coarse, eps_x_coarse, eps_y_coarse)
            coarse_in = False
            coarse_integral += t * dt_coarse * g(x_coarse, y_coarse)
            x_coarse = x
            y_coarse = y
            if plot_walks:
                steps2_x.append(x_coarse)
                steps2_y.append(y_coarse)

        if coarse_in:
            coarse_integral += g(x_coarse, y_coarse)*dt_coarse
            x_coarse += eps_x_coarse
            y_coarse += eps_y_coarse
            if plot_walks:
                steps2_x.append(x_coarse)
                steps2_y.append(y_coarse)

        if not is_in_domain(x_coarse, y_coarse):
            coarse_in = False

    if plot_walks:
        visualize_random_walk(steps1_x, steps1_y, steps2_x, steps2_y)

    assert (abs(x_fine - 1) < 1e-10) or (abs(x_fine) < 1e-10) or (abs(y_fine - 1) < 1e-10) or (abs(y_fine) < 1e-10)
    assert (abs(x_coarse - 1) < 1e-10) or (abs(x_coarse) < 1e-10) or (abs(y_coarse - 1) < 1e-10) or (abs(y_coarse) < 1e-10)
    fine_integral += f(x_fine, y_fine)
    coarse_integral += f(x_coarse, y_coarse)


    if plot_walks:
        steps1_x.pop(-1)
        steps1_y.pop(-1)
        steps2_x.pop(-1)
        steps2_y.pop(-1)
        steps1_x.append(x_fine)
        steps1_y.append(y_fine)
        steps2_x.append(x_coarse)
        steps2_y.append(y_coarse)
    integral = fine_integral - coarse_integral

    return integral, num_steps, uncorrelated
