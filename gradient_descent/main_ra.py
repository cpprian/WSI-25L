import numpy as np
import matplotlib.pyplot as plt
from lib import gradient_descent, gradient_rastrigin, rastrigin

if __name__ == "__main__":
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    x, y = np.meshgrid(x, y)
    z = rastrigin(x, y)

    beta = [0.001, 0.005, 0.01, 0.05]
    point = np.random.uniform(-5, 5, 2)
    
    for b in beta:
        path = gradient_descent(point, b, 0.0001, gradient_rastrigin)

        fig = plt.figure()
        ax = plt.subplot(projection="3d", computed_zorder=False)
        ax.plot_surface(x, y, z, cmap="viridis", alpha=0.7)
        ax.scatter(point[0], point[1], rastrigin(point[0], point[1]), color="red", linewidth=2, label="Start Point", zorder=3)
        ax.scatter(path[:, 0], path[:, 1], rastrigin(path[:, 0], path[:, 1]), color="gray", linewidth=2, label="Path", zorder=1)
        ax.scatter(path[-1, 0], path[-1, 1], rastrigin(path[-1, 0], path[-1, 1]), color="yellow", linewidths=2, label="End Point", zorder=2)
        ax.set_title(f"Gradient Descent [Rastrigin] with beta={b}")
        ax.legend()
        plt.savefig(f"img/rastrigin_beta_{b}.png")
        plt.close(fig)