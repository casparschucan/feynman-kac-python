import numpy as np
import scipy

from visualize import visualize_correlated_random_walk, visualize_walk_on_spheres
from rng import get_rng
from random_walk import project_to_domain_edge


def distance_to_edge(x, y):
    distance = min(x, y, (1-x), (1-y))
    return distance


def Green_density(r, d):
    return 4 * r / ((d**2)) * np.log(d/r)


def Green_probability(r, d):
    if (r == 0):
        return 0
    return 2*r*r/(d*d) * (0.5 + np.log(d/r))


def Green_probability_2(r, d):
    if (r == 0):
        return 0
    return r/d * (1 + np.log(d/r))


def Ball_area(d):
    return (d**2)/4


def sample_radius_reject(d):
    rng = get_rng()
    max = 4/(d*np.e)
    sample_x = rng.random()*d
    sample_y = rng.random() * max
    while sample_y > Green_density(sample_x, d):
        sample_x = rng.random()*d
        sample_y = rng.random() * max

    return sample_x


def sample_radius_transform(d):
    rng = get_rng()
    u = rng.random()

    def func(x):
        return Green_probability_2(x, 1) - u
    sol = scipy.optimize.root_scalar(func,
                                     bracket=[0, 1], method='brentq')
    if not sol.converged:
        print("Oh no u = ", u, "with max possible = ", Green_probability(1, 1))
        raise Exception("test")
    return sol.root * d


def walk_on_spheres_with_work(x0: float, y0: float, f, g, delta: float,
                              debug=False, plot_walk=False):
    x = x0
    y = y0
    rng = get_rng()
    num_steps = 0

    x_steps = [x0]
    y_steps = [y0]

    # the recorded integral
    integral = 0

    while distance_to_edge(x, y) > delta:
        # the biggest possible step size ensuring we don't leave the domain
        step_radius = distance_to_edge(x, y)
        direction = rng.random() * 2 * np.pi  # a direction in form of an angle

        num_steps += 1

        dx = step_radius * np.cos(direction)
        dy = step_radius * np.sin(direction)

        # uniformly sample disk of the current step
        green_radius = sample_radius_reject(step_radius)
        green_direction = rng.random() * 2 * np.pi
        green_x = green_radius * np.cos(green_direction) + x
        green_y = green_radius * np.sin(green_direction) + y

        # introduce factor 2 compared to paper due to differen poisson
        # formulation
        integral += 2*Ball_area(step_radius) * g(green_x, green_y)

        x += dx
        y += dy

        if plot_walk:
            x_steps.append(x)
            y_steps.append(y)

    x, y = project_to_domain_edge(x, y)
    if plot_walk:
        visualize_walk_on_spheres(x_steps, y_steps)
    integral += f(x, y)

    return integral, num_steps


def walk_on_spheres(x0: float, y0: float, f, g, delta: float,
                    debug=False, plot_walk=False):
    integral, work = walk_on_spheres_with_work(x0, y0, f, g, delta,
                                               debug, plot_walk)
    return np.array([integral, work])

# def feynman_kac_correlated(args, plot_walks=False):
    # x0, y0, f, g, dt_fine, level, dt_ratio = args


def walk_on_spheres_correlated(args, plot_walk=False):
    x0, y0, f, g, delta_fine, level, delta_ratio = args
    x_fine = x0
    y_fine = y0
    x_coarse = x0
    y_coarse = y0
    rng = get_rng()
    num_steps = 0

    delta_coarse = delta_fine*delta_ratio

    x_steps_fine = [x0]
    y_steps_fine = [y0]
    x_steps_coarse = [x0]
    y_steps_coarse = [y0]

    # the recorded integral
    integral_fine = 0
    integral_coarse = 0

    uncor = 0

    coarse_in = True

    if level == 0:
        integral, num_steps = walk_on_spheres_with_work(x0, y0, f, g, delta_fine)
        return integral, num_steps, 0

    while True:
        # the biggest possible step size ensuring we don't leave the domain
        step_radius = distance_to_edge(x_fine, y_fine)
        direction = rng.random() * 2 * np.pi  # a direction in form of an angle

        num_steps += 1

        dx = step_radius * np.cos(direction)
        dy = step_radius * np.sin(direction)

        if distance_to_edge(x_coarse + dx, y_coarse + dy) <= delta_coarse:
            x_coarse += dx
            y_coarse += dy
            coarse_in = False

        # break loop if we are close enough to the edge
        if distance_to_edge(x_fine+dx, y_fine+dy) <= delta_fine:
            x_fine += dx
            y_fine += dy
            break

        if plot_walk:
            x_steps_fine.append(x_fine)
            y_steps_fine.append(y_fine)
            if coarse_in:
                x_steps_coarse.append(x_coarse)
                y_steps_coarse.append(y_coarse)

        # uniformly sample disk of the current step
        green_radius = sample_radius_reject(step_radius)
        green_direction = rng.random() * 2 * np.pi
        green_x = green_radius * np.cos(green_direction) + x_fine
        green_y = green_radius * np.sin(green_direction) + y_fine

        # introduce factor 2 compared to paper due to differen poisson
        # formulation
        integral_fine += 2*Ball_area(step_radius) * g(green_x, green_y)

        x_fine += dx
        y_fine += dy

        if coarse_in:
            integral_coarse += 2*Ball_area(step_radius)*g(green_x, green_y)
            x_coarse += dx
            y_coarse += dy
        else:
            uncor = 1

    x_fine, y_fine = project_to_domain_edge(x_fine, y_fine)
    x_coarse, y_coarse = project_to_domain_edge(x_coarse, y_coarse)
    if plot_walk:
        x_steps_fine.append(x_fine)
        y_steps_fine.append(y_fine)
        x_steps_coarse.append(x_coarse)
        y_steps_coarse.append(y_coarse)
        visualize_correlated_random_walk(x_steps_fine, y_steps_fine,
                              x_steps_coarse, y_steps_coarse)

    integral_fine += f(x_fine, y_fine)
    integral_coarse += f(x_coarse, y_coarse)

    return integral_fine - integral_coarse, num_steps, uncor
