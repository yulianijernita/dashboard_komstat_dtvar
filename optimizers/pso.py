"""Particle Swarm Optimization (PSO)."""
import time
import numpy as np


class PSO:
    def __init__(self, objective_func, objective_batch, bounds,
                 pop_size=50, max_iter=500, c1=2.0, c2=2.0, w=1.0, rho=0.99):
        self.objective_func = objective_func
        self.objective_batch = objective_batch
        self.lb = np.asarray(bounds[0], dtype=float)
        self.ub = np.asarray(bounds[1], dtype=float)
        self.pop_size, self.max_iter, self.dim = pop_size, max_iter, len(self.lb)

        self.pos = np.random.rand(pop_size, self.dim)
        self.vel = np.zeros((pop_size, self.dim))
        self.pbest_pos = self.pos.copy()
        self.pbest_score = np.full(pop_size, np.inf)
        self.gbest_pos = np.zeros(self.dim)
        self.gbest_score = np.inf

        self.c1, self.c2, self.w, self.rho = c1, c2, w, rho

    def optimize(self):
        convergence_curve, time_curve = [], []

        for _ in range(self.max_iter):
            iter_start = time.perf_counter()

            # Posisi normalized [0,1] -> parameter asli, evaluasi batch
            X = self.lb + self.pos * (self.ub - self.lb)
            scores = np.asarray(self.objective_batch(X), dtype=float)

            # PBest
            mask = scores < self.pbest_score
            self.pbest_score[mask] = scores[mask]
            self.pbest_pos[mask] = self.pos[mask]

            # GBest
            best_idx = np.argmin(scores)
            if scores[best_idx] < self.gbest_score:
                self.gbest_score = scores[best_idx]
                self.gbest_pos = self.pos[best_idx].copy()

            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)

            # Velocity & posisi
            self.vel = (self.w * self.vel
                        + self.c1 * r1 * (self.pbest_pos - self.pos)
                        + self.c2 * r2 * (self.gbest_pos - self.pos))
            self.pos = np.clip(self.pos + self.vel, 0, 1)

            # Inertia
            self.w = self.rho * self.w

            convergence_curve.append(self.gbest_score)
            time_curve.append(time.perf_counter() - iter_start)

        return (self.lb + self.gbest_pos * (self.ub - self.lb),
                self.gbest_score, convergence_curve, time_curve)