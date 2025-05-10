import numpy as np
import time
from fitness_function import fitness_function
import random

def hybrid_pso_ga_optimize(returns_df, num_particles=30, max_iter=100, w=0.5, c1=1.5, c2=1.5, ga_interval=10):
    start_time = time.time()

    
    num_assets = returns_df.shape[1]

    # Initialize particles
    particles = [np.random.dirichlet(np.ones(num_assets)) for _ in range(num_particles)]
    velocities = [np.random.uniform(-0.1, 0.1, num_assets) for _ in range(num_particles)]
    personal_best = particles.copy()
    personal_best_scores = [fitness_function(p, returns_df) for p in particles]

    global_best_idx = np.argmin(personal_best_scores)
    global_best = personal_best[global_best_idx].copy()
    global_best_score = personal_best_scores[global_best_idx]

    history = []
    trajectory = [global_best.copy()]  # Track best solution each iteration

    for iter in range(max_iter):
        for i in range(num_particles):
            # Update velocity and position
            r1, r2 = np.random.rand(num_assets), np.random.rand(num_assets)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best[i] - particles[i])
                + c2 * r2 * (global_best - particles[i])
            )

            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0.0, 1.0)
            particles[i] /= np.sum(particles[i])  # Normalize

            score = fitness_function(particles[i], returns_df)

            if score < personal_best_scores[i]:
                personal_best[i] = particles[i]
                personal_best_scores[i] = score

                if score < global_best_score:
                    global_best = particles[i]
                    global_best_score = score

        # Periodically apply GA to top-performing particles
        if iter % ga_interval == 0 and iter != 0:
            top_indices = np.argsort(personal_best_scores)[:10]
            selected = [personal_best[i] for i in top_indices]
            new_particles = []

            while len(new_particles) < num_particles:
                p1, p2 = random.sample(selected, 2)
                alpha = np.random.rand()
                child = alpha * p1 + (1 - alpha) * p2

                # Mutation
                if np.random.rand() < 0.1:
                    child += np.random.uniform(-0.05, 0.05, num_assets)
                    child = np.clip(child, 0.0, 1.0)

                child /= np.sum(child)
                new_particles.append(child)

            particles = new_particles
            velocities = [np.random.uniform(-0.1, 0.1, num_assets) for _ in range(num_particles)]

        history.append(-global_best_score)
        trajectory.append(global_best.copy())

    end_time = time.time()
    exec_time = end_time - start_time

    return global_best, -global_best_score, history, exec_time, trajectory