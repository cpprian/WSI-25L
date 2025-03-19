import numpy as np
import matplotlib.pyplot as plt
from lib import gradient_descent, gradient_rastrigin, rastrigin


if __name__ == "__main__":
    x = np.arange(-5, 5, 0.01)
    y = rastrigin(x)

    beta = [0.001, 0.005, 0.01, 0.05]
    point = np.random.uniform(-5, 5, 1)
    for b in beta:
        path = gradient_descent([point], b, 0.001, gradient_rastrigin)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.scatter(path, rastrigin(path), color="gray", linewidth=2, label="Path")
        ax.set_title(f"Gradient Descent [Rastrigin] with beta={b}")
        plt.savefig(f"img/rastrigin_beta_{b}.png")
        plt.close(fig)
