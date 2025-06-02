import matplotlib.pyplot as plt


def distance(x, y):
    return min(1-x, x, y, 1-y)


def visualize_walk_on_spheres(steps_x, steps_y):

    # Plot the two random walks
    plt.figure(figsize=(12, 12))
    plt.plot(steps_x, steps_y, color="blue", linewidth=2, zorder=2)
    plt.scatter(steps_x, steps_y, color="orange", linewidth=2, zorder=3)

    ax = plt.gca()
    for x, y in zip(steps_x, steps_y):
        d = distance(x, y)
        if d > 1e-2:
            circle_i = plt.Circle((x, y), d, fill=False, color="blue",
                                  linestyle="--", linewidth=2, zorder=1)
            ax.add_patch(circle_i)

    # Formatting
    plt.axis('scaled')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("X", fontsize="large")
    plt.ylabel("Y", fontsize="large")
    plt.xticks(fontsize="large")
    plt.yticks(fontsize="large")
    plt.grid(True)
    # Show the plot
    plt.show()


def visualize_correlated_random_walk(steps1_x, steps1_y, steps2_x, steps2_y):

    # Plot the two random walks
    plt.figure(figsize=(12, 12))
    plt.axis('scaled')
    plt.plot(steps1_x, steps1_y, label="fine", color="blue", alpha=0.7, zorder=1)
    plt.plot(steps2_x, steps2_y, label="coarse", color="red", alpha=0.7, zorder=1)
    plt.scatter(steps1_x, steps1_y, label="Start/End RW1", zorder=2)
    plt.scatter(steps2_x, steps2_y, label="Start/End RW2", zorder=2)

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
    plt.show()
