import numpy as np


def himmelblau(point: np.ndarray) -> np.ndarray:
    x, y = point
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def gradient_himmelblau(point: np.ndarray) -> int:
    x, y = point
    dx = 2 * (-7 + x + y**2 + 2 * x * (-11 + x**2 + y))
    dy = 2 * (-11 + x**2 + y + 2 * y * (-7 + x + y**2))
    return np.array([dx, dy])


def rastrigin(*X) -> np.ndarray:
    return 10 * len(X) + sum([x**2 - 10 * np.cos(2 * np.pi * x) for x in X])


def gradient_rastrigin(x: int) -> int:
    x = np.array(x)
    return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)


def stop(gradient: callable, path: np.ndarray, epsilon: float, beta: float) -> bool:
    grad_norm = np.linalg.norm(gradient(path[-1]))
    next_step = beta * grad_norm
    return grad_norm < epsilon or next_step < epsilon


def gradient_descent(
    start: np.ndarray, beta: float, epsilon: float, gradient: callable, n: int = 1000
) -> np.ndarray:
    current = start
    path = [current]
    for _ in range(n):
        grad = gradient(current)
        next_point = current - beta * grad
        path.append(next_point)
        if stop(gradient, path, epsilon, beta):
            print(f"Stopped at {next_point}")
            break
        current = next_point

    return np.array(path)
