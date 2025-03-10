import matplotlib.pyplot as plt


def visualize_random_walk(steps1_x, steps1_y, steps2_x, steps2_y):

    # Plot the two random walks
    plt.figure(figsize=(6, 6))
    plt.plot(steps1_x, steps1_y, label="fine", color="blue", alpha=0.7)
    plt.plot(steps2_x, steps2_y, label="coarse", color="red", alpha=0.7)
    plt.scatter(steps1_x, steps1_y, label="Start/End RW1")
    plt.scatter(steps2_x, steps2_y, label="Start/End RW2")

    # Formatting
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Two Correlated Random Walks on Unit Square")
    plt.legend()
    plt.grid(True)
    # Show the plot
    plt.show()
