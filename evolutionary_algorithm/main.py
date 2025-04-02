import numpy as np
import matplotlib.pyplot as plt


def himmelblau(point: np.ndarray) -> np.ndarray:
    x, y = point
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

class EvolutionaryAlgorithm:
    def __init__(self, _population_size: int, _mutation_rate: float, _generations: int, _fun: callable, _cross_rate: float = 0.7):
        self.population_size = _population_size
        self.mutation_rate = _mutation_rate
        self.generations = _generations
        self.func = _fun
        self.crossover_rate = _cross_rate
        self.track_best_solution = []
        self.track_best_fitness = []
        self.first_best_solution = None
        self.first_best_fitness = None

    def initialize_population(self, bounds: tuple) -> np.ndarray:
        return np.random.uniform(bounds[0], bounds[1], (self.population_size, len(bounds)))
    
    def fitness(self, population: np.ndarray) -> np.ndarray:
        return np.array([self.func(ind) for ind in population])
    
    def select_parents_ranking(self, population: np.ndarray, fitness_values: np.ndarray) -> np.ndarray:
        sorted_indices = np.argsort(fitness_values)
        selected_indices = sorted_indices[:self.population_size // 2]
        return population[selected_indices]
    
    def select_parents_tournament(self, population: np.ndarray, fitness_values: np.ndarray) -> np.ndarray:
        selected_indices = np.random.choice(len(population), size=self.population_size // 2, replace=True)
        selected_fitness = fitness_values[selected_indices]
        best_indices = selected_indices[np.argsort(selected_fitness)[:self.population_size // 4]]
        return population[best_indices]
    
    def select_parents_roulette(self, population: np.ndarray, fitness_values: np.ndarray) -> np.ndarray:
        fitness_sum = np.sum(fitness_values)
        selection_probs = fitness_values / fitness_sum
        selected_indices = np.random.choice(len(population), size=self.population_size // 2, p=selection_probs)
        return population[selected_indices]
    
    def crossover_single_point(self, parents: np.ndarray) -> np.ndarray:
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents) and np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, len(parents[i]))
                child1 = np.concatenate((parents[i][:crossover_point], parents[i + 1][crossover_point:]))
                child2 = np.concatenate((parents[i + 1][:crossover_point], parents[i][crossover_point:]))
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(parents[i])
                if i + 1 < len(parents):
                    offspring.append(parents[i + 1])
        return np.array(offspring)
    
    def crossover_arithmetic(self, parents: np.ndarray) -> np.ndarray:
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents) and np.random.rand() < self.crossover_rate:
                alpha = np.random.rand()
                child1 = alpha * parents[i] + (1 - alpha) * parents[i + 1]
                child2 = alpha * parents[i + 1] + (1 - alpha) * parents[i]
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(parents[i])
                if i + 1 < len(parents):
                    offspring.append(parents[i + 1])
        return np.array(offspring)
    
    def mutate(self, population: np.ndarray) -> np.ndarray:
        for i in range(len(population)):
            if np.random.rand() < self.mutation_rate:
                mutation_vector = np.random.uniform(-1, 1, size=population[i].shape)
                population[i] += mutation_vector
        return population
    
    def run(self, bounds: tuple) -> np.ndarray:
        population = self.initialize_population(bounds)
        # population = np.array(population)
        best_solution = None
        best_fitness = float('inf')

        for _ in range(self.generations):
            fitness_values = self.fitness(population)
            best_idx = np.argmin(fitness_values)
            if fitness_values[best_idx] < best_fitness:
                best_fitness = fitness_values[best_idx]
                best_solution = population[best_idx]
            if self.first_best_fitness is None and self.first_best_solution is None:
                self.first_best_fitness = fitness_values[best_idx]
                self.first_best_solution = population[best_idx]

            self.track_best_solution.append(population[best_idx])
            self.track_best_fitness.append(fitness_values[best_idx])
            parents = self.select_parents_roulette(population, fitness_values)
            offspring = self.crossover_single_point(parents)
            offspring = self.mutate(offspring)

            population = np.vstack((parents, offspring))

        return best_solution

def main():
    random_seed = 139565
    bounds = (-5, 5)
    population_size = 50
    mutation_rate = 0.5
    generations = 15
    cross_rate = 1.0

    np.random.seed(random_seed)

    x = np.arange(-5, 5, 0.01)
    y = np.arange(-5, 5, 0.01)
    x, y = np.meshgrid(x, y)
    z = himmelblau(np.array([x, y]))

    ea = EvolutionaryAlgorithm(population_size, mutation_rate, generations, himmelblau, cross_rate)
    best_solution = ea.run(bounds=bounds)

    fig = plt.figure(figsize=(10, 7))
    ax = plt.subplot(projection="3d", computed_zorder=False)
    ax.plot_surface(x, y, z, cmap="viridis", zorder=1)
    x_vals = [p[0] for p in ea.track_best_solution[:-1]]
    y_vals = [p[1] for p in ea.track_best_solution[:-1]]
    z_vals = ea.track_best_fitness[:-1]

    ax.scatter(x_vals, y_vals, z_vals, label="All best solutions", color="red", zorder=2)
    ax.scatter(best_solution[0], best_solution[1], himmelblau(best_solution) ,color="green", label="Best solution from all generations", zorder=3)
    ax.scatter(ea.first_best_solution[0], ea.first_best_solution[1], ea.first_best_fitness, color="yellow", label="First best solution", zorder=4)
    ax.set_title("EA select roulette")
    ax.legend()
    plt.savefig(f"img/himmelblau_select_roulette.png", dpi=100, bbox_inches='tight')
    plt.close(fig)

    print(f"min: {np.min(ea.track_best_fitness)}")
    print(f"max: {np.max(ea.track_best_fitness)}")
    print(f"avg: {np.mean(ea.track_best_fitness)}")
    print(f"stdev: {np.std(ea.track_best_fitness)}")

if __name__ == "__main__":
    main()