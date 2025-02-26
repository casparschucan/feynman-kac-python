import numpy as np


# a function samples a random walk starting from a specific point
# and returns the needed values for feynman-kac poisson eval on
# a 2-D grid discretization of the unit square [0,1]^2


def feynman_kac_sample(N: int, x0: float, y0: float, f, g):

    x = x0
    y = y0
    integral = 0
    total_dist = 0

    delta_x = 0.02

    gen = np.random.default_rng()

    while x > 0 and y > 0 and x < 1 and y < 1:
        eps_x = gen.standard_normal()
        x += eps_x * delta_x

        eps_y = gen.standard_normal()
        y += eps_y * delta_x

        dist = np.sqrt((eps_x*delta_x)**2 + (eps_y*delta_x)**2)
        integral += g(x, y)*dist

        total_dist += dist

    if x < 0:
        x = 0
    elif x > 1:
        x = 1

    if y < 0:
        y = 0
    elif y > 1:
        y = 1

    integral += f(x, y)

    integral /= total_dist

    return integral
