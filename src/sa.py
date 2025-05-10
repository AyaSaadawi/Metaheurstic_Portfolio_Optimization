import numpy as np
import time
from fitness_function import fitness_function

def simulated_annealing_optimize(returns_df, initial_temp=100, cooling_rate=0.95, max_iter=1000):
    start_time = time.time()

    
    num_assets = returns_df.shape[1]

    current = np.random.dirichlet(np.ones(num_assets))
    best = current.copy()

    current_fitness = fitness_function(current, returns_df)
    best_fitness = current_fitness

    temp = initial_temp
    history = []
    trajectory = [current.copy()]  # Track all candidate solutions

    for _ in range(max_iter):
        candidate = current + np.random.uniform(-0.05, 0.05, size=num_assets)
        candidate = np.clip(candidate, 0.0, 1.0)
        candidate /= np.sum(candidate)

        candidate_fitness = fitness_function(candidate, returns_df)

        if candidate_fitness < current_fitness or np.random.rand() < np.exp(-(candidate_fitness - current_fitness) / temp):
            current = candidate
            current_fitness = candidate_fitness

            if candidate_fitness < best_fitness:
                best = candidate
                best_fitness = candidate_fitness

        temp *= cooling_rate
        history.append(-best_fitness)
        trajectory.append(current.copy())

    end_time = time.time()
    exec_time = end_time - start_time

    return best, -best_fitness, history, exec_time, trajectory