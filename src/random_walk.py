import numpy as np


# a function samples a random walk starting from a specific point
# and returns the needed values for feynman-kac poisson eval on the unit square


def feynman_kac_sample_with_work(x0: float, y0: float, f, g, dt):

    x = x0
    y = y0
    integral = 0
    num_steps = 0

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

    return integral, num_steps


def feynman_kac_sample(x0: float, y0: float, f, g, dt):
    integral, _ = feynman_kac_sample_with_work(x0, y0, f, g, dt)
    return integral


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


def feynman_kac_correlated(x0: float,
                           y0: float,
                           f,
                           g,
                           dt_fine: float,
                           level: int):
    if (level == 0):
        return feynman_kac_sample_with_work(x0, y0, f, g, dt_fine)

    x_fine = x0
    x_coarse = x0
    y_fine = y0
    y_coarse = y0

    coarse_integral = 0
    fine_integral = 0

    dt_coarse = 2*dt_fine
    num_steps = 0

    gen = np.random.default_rng()
    while is_in_domain(x_fine, y_fine) and is_in_domain(x_coarse, y_coarse):
        fine_integral += g(x_fine, y_fine)*dt_fine
        eps_x1 = gen.normal(scale=np.sqrt(dt_fine))
        x_fine += eps_x1

        eps_y1 = gen.normal(scale=np.sqrt(dt_fine))
        y_fine += eps_y1

        num_steps += 1

        if not is_in_domain(x_fine, y_fine):
            break

        fine_integral += g(x_fine, y_fine)*dt_fine
        eps_x2 = gen.normal(scale=np.sqrt(dt_fine))
        x_fine += eps_x2

        eps_y2 = gen.normal(scale=np.sqrt(dt_fine))
        y_fine += eps_y2

        coarse_integral += g(x_coarse, y_coarse)*dt_coarse
        eps_x_coarse = (eps_x1 + eps_x2)/np.sqrt(2)
        x_coarse += eps_x_coarse

        eps_y_coarse = (eps_y1 + eps_y2)/np.sqrt(2)
        y_coarse += eps_y_coarse

        num_steps += 1

    x_fine, y_fine = project_to_domain_edge(x_fine, y_fine)
    x_coarse, y_coarse = project_to_domain_edge(x_coarse, y_coarse)

    integral = fine_integral - coarse_integral

    return integral, num_steps
