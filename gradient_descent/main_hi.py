import numpy as np
import matplotlib.pyplot as plt
from lib import gradient_descent, gradient_himmelblau, himmelblau


if __name__ == "__main__":
    x = np.arange(-5, 5, 0.01)
    y = np.arange(-5, 5, 0.01)
    x, y = np.meshgrid(x, y)
    z = himmelblau(np.array([x, y]))

    beta = [0.001, 0.005, 0.01, 0.05]
    start = np.random.uniform(-5, 5, 2)
    for b in beta:
        path = gradient_descent(start, b, 0.0001, gradient_himmelblau)

        fig = plt.figure()
        ax = plt.subplot(projection="3d", computed_zorder=False)
        ax.plot_surface(x, y, z, cmap="viridis", zorder=1)
        ax.scatter(
            path[:, 0],
            path[:, 1],
            himmelblau(path.T),
            color="magenta",
            linewidth=2,
            label="Path",
            zorder=2,
        )
        ax.scatter(start[0], start[1], color="green", label="Start point", zorder=3)
        ax.scatter(
            path[-1, 0], path[-1, 1], color="yellow", label="End point", zorder=4
        )
        ax.set_title(f"Gradient Descent [Himmelblau] with beta={b}")
        ax.legend()
        plt.savefig(f"img/himmelblau_beta_{b}.png")
        plt.close(fig)
