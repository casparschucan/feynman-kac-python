import numpy as np

from visualize import visualize_random_walk
from rng import get_rng
from random_walk import project_to_domain_edge


def distance_to_edge(x, y):
    distance = min(x, y, (1-x), (1-y))
    return distance


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
        # the step size ensuring we don't leave the domain
        step_radius = distance_to_edge(x, y)
        direction = rng.random() * 2 * np.pi  # a direction in form of an angle

        num_steps += 1

        dx = step_radius * np.cos(direction)
        dy = step_radius * np.sin(direction)
        x += dx
        y += dy
        if plot_walk:
            x_steps.append(x)
            y_steps.append(y)

    x, y = project_to_domain_edge(x, y)
    if plot_walk:
        x_steps.append(x)
        y_steps.append(y)
        visualize_random_walk(x_steps, y_steps, [], [])
    integral += f(x, y)
