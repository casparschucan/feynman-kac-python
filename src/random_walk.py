import numpy as np


# a function samples a random walk starting from a specific point
# and returns the needed values for feynman-kac poisson eval on the unit square


def feynman_kac_sample(N: int, x0: float, y0: float, f, g):

    x = x0
    y = y0
    integral = 0
    num_steps = 0

    dt = 0.00005

    gen = np.random.default_rng()

    while x > 0 and y > 0 and x < 1 and y < 1:
        integral += g(x, y)*dt
        eps_x = gen.normal(scale=np.sqrt(dt))
        x += eps_x

        eps_y = gen.normal(scale=np.sqrt(dt))
        y += eps_y

        num_steps += 1

    if x < 0:
        x = 0
    elif x > 1:
        x = 1

    if y < 0:
        y = 0
    elif y > 1:
        y = 1

    integral += f(x, y)

    return integral
