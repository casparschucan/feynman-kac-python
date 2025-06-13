import matplotlib.pyplot as plt
import numpy as np


def distance(x, y):
    return min(1-x, x, y, 1-y)


def visualize_walk_on_spheres(steps_x, steps_y):

    # Plot the two random walks
    plt.figure(figsize=(12, 12))
    plt.plot(steps_x, steps_y, color="blue", linewidth=2, zorder=2)
    plt.scatter(steps_x, steps_y, color="orange", linewidth=2, zorder=3,
                label="steps")

    ax = plt.gca()
    for x, y in zip(steps_x, steps_y):
        d = distance(x, y)
        if d > 1e-2:
            circle_i = plt.Circle((x, y), d, fill=False, color="blue",
                                  linestyle="--", linewidth=2, zorder=1)
            ax.add_patch(circle_i)

    start = plt.Circle((steps_x[0], steps_y[0]), 0.01, fill=True,
                       color="green", zorder=5, label="starting point")
    ax.add_patch(start)
    # Formatting
    plt.axis('scaled')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("X", fontsize="large")
    plt.ylabel("Y", fontsize="large")
    plt.xticks(fontsize="large")
    plt.yticks(fontsize="large")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize="large")
    plt.grid(True)
    # Show the plot
    plt.tight_layout()
    plt.show()


def visualize_correlated_random_walk(steps1_x, steps1_y, steps2_x, steps2_y):

    # Plot the two random walks
    plt.figure(figsize=(12, 12))
    plt.axis('scaled')
    plt.plot(steps1_x, steps1_y, color="blue",
             alpha=0.7, zorder=1, linewidth=2)
    plt.plot(steps2_x, steps2_y, color="orange",
             alpha=0.7, zorder=3, linewidth=2)

    sizes_fine = np.ones(len(steps1_x))*64
    sizes_coarse = np.ones(len(steps2_x))*16
    plt.scatter(steps1_x, steps1_y, label="fine steps",
                zorder=2, sizes=sizes_fine, color="blue")
    plt.scatter(steps2_x, steps2_y, label="coarse steps",
                zorder=4, sizes=sizes_coarse, color="orange")

    start = plt.Circle((steps1_x[0], steps1_y[0]), 0.01, fill=True,
                       color="green", zorder=5, label="starting point")
    ax = plt.gca()
    ax.add_patch(start)

    # Formatting
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("X", fontsize="large")
    plt.ylabel("Y", fontsize="large")
    plt.xticks(fontsize="large")
    plt.yticks(fontsize="large")
    plt.legend(fontsize="large")
    plt.grid(True)
    # Show the plot
    plt.tight_layout()
    plt.show()
