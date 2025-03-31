import numpy as np

from visualize import visualize_random_walk
from rng import get_rng
from random_walk import project_to_domain_edge


def distance_to_edge(x, y):
    distance = min(x, y, (1-x), (1-y))
    return distance


def Green_density(r, d):
    return 2 * r / (np.pi * (d**2)) * np.log(d/r)


def Ball_area(d):
    return (d**2)/4


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

    while True:
        # the step size ensuring we don't leave the domain
        step_radius = distance_to_edge(x, y)
        direction = rng.random() * 2 * np.pi  # a direction in form of an angle

        num_steps += 1

        dx = step_radius * np.cos(direction)
        dy = step_radius * np.sin(direction)

        if distance_to_edge(x+dx, y+dy) <= delta:
            x += dx
            y += dy
            break

        if plot_walk:
            x_steps.append(x)
            y_steps.append(y)

        green_radius = step_radius * (rng.random()**.5)
        green_direction = rng.random() * 2 * np.pi
        green_x = green_radius * np.cos(green_direction) + x
        green_y = green_radius * np.sin(green_direction) + y

        integral += 2*Ball_area(step_radius) * g(green_x, green_y)

        x += dx
        y += dy

    x, y = project_to_domain_edge(x, y)
    if plot_walk:
        x_steps.append(x)
        y_steps.append(y)
        visualize_random_walk(x_steps, y_steps, [], [])
    integral += f(x, y)

    return integral, num_steps


def walk_on_spheres(x0: float, y0: float, f, g, delta: float,
                    debug=False, plot_walk=False):
    integral, work = walk_on_spheres_with_work(x0, y0, f, g, delta,
                                               debug, plot_walk)
    return np.array([integral, work])
