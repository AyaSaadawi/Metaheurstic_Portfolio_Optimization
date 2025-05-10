from fitness_function import fitness_function
import numpy as np
import time

def pso_optimize(returns_df, n_particles=30, max_iter=100, w=0.5, c1=1.5, c2=1.5):
    start_time = time.time()

    num_assets = returns_df.shape[1]

    # Initialize particles and velocities
    particles = np.random.dirichlet(np.ones(num_assets), size=n_particles)
    velocities = np.random.uniform(-0.1, 0.1, size=(n_particles, num_assets))

    # Initialize personal and global bests
    personal_best_positions = particles.copy()
    personal_best_scores = np.array([fitness_function(p, returns_df) for p in particles])

    global_best_index = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_score = personal_best_scores[global_best_index]

    convergence_history = []
    trajectory = [global_best_position.copy()]  # Start with initial best

    for _ in range(max_iter):
        for i in range(n_particles):
            r1, r2 = np.random.rand(), np.random.rand()

            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (personal_best_positions[i] - particles[i]) +
                             c2 * r2 * (global_best_position - particles[i]))

            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0.0, 1.0)
            particles[i] /= np.sum(particles[i])

            score = fitness_function(particles[i], returns_df)

            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particles[i].copy()

        best_idx = np.argmin(personal_best_scores)
        if personal_best_scores[best_idx] < global_best_score:
            global_best_score = personal_best_scores[best_idx]
            global_best_position = personal_best_positions[best_idx].copy()

        convergence_history.append(-global_best_score)
        trajectory.append(global_best_position.copy())

    exec_time = time.time() - start_time

    return global_best_position, -global_best_score, convergence_history, exec_time, trajectory