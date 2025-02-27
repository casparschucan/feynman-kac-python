import numpy as np


# a function samples a random walk starting from a specific point
# and returns the needed values for feynman-kac poisson eval on the unit square


def feynman_kac_sample(N: int, x0: float, y0: float, f, g):

    x = x0
    y = y0
    integral = 0
    num_steps = 0

    delta_x = 0.001

    gen = np.random.default_rng()

    while x > 0 and y > 0 and x < 1 and y < 1:
        integral += g(x, y)
        eps_x = gen.standard_normal()
        x += eps_x * delta_x

        eps_y = gen.standard_normal()
        y += eps_y * delta_x

        num_steps += 1

    if x < 0:
        x = 0
    elif x > 1:
        x = 1

    if y < 0:
        y = 0
    elif y > 1:
        y = 1

    integral /= num_steps

    integral += f(x, y)

    return integral
