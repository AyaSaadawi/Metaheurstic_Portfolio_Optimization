import numpy as np
import time
import random
from fitness_function import fitness_function

def genetic_algorithm_optimize(returns_df, pop_size=30, crossover_rate=0.7, mutation_rate=0.1, max_iter=100):
    start_time = time.time()

    
    num_assets = returns_df.shape[1]

    # Initialize population
    population = [np.random.dirichlet(np.ones(num_assets)) for _ in range(pop_size)]

    best = None
    best_score = float('inf')
    history = []
    trajectory = []  # To track all candidate solutions per iteration

    for _ in range(max_iter):
        scores = [fitness_function(p, returns_df) for p in population]
        sorted_pairs = sorted(zip(scores, population), key=lambda x: x[0])
        population = [x[1] for x in sorted_pairs[:pop_size]]

        next_gen = []

        while len(next_gen) < pop_size:
            p1, p2 = random.sample(population[:10], 2)

            # Crossover
            if np.random.rand() < crossover_rate:
                alpha = np.random.rand()
                child = alpha * p1 + (1 - alpha) * p2
            else:
                child = p1.copy()

            # Mutation
            if np.random.rand() < mutation_rate:
                child += np.random.uniform(-0.05, 0.05, size=num_assets)
                child = np.clip(child, 0.0, 1.0)

            child /= np.sum(child)
            next_gen.append(child)

        population = next_gen
        scores = [fitness_function(p, returns_df) for p in population]
        best_idx = np.argmin(scores)
        best_candidate = population[best_idx]
        best_candidate_score = scores[best_idx]

        # Track best solution
        if best_candidate_score < best_score:
            best_score = best_candidate_score
            best = best_candidate

        # Track history and trajectory
        history.append(-best_score)
        trajectory.append(best_candidate)

    end_time = time.time()
    exec_time = end_time - start_time

    return best, -best_score, history, exec_time, trajectory