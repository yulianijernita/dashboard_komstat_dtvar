"""Hybrid PSO-DE: kandidat PSO di-crossover dengan mutant DE/rand/1."""
import time
import numpy as np

from optimizers.de import de_trial


class Hybrid_PSO_DE:
    def __init__(self, objective_func, objective_batch, bounds,
                 pop_size=500, max_iter=500, F=0.5, CR=0.9,
                 c1=2.0, c2=2.0, w=1.0, rho=0.99):
        self.objective_func = objective_func
        self.objective_batch = objective_batch
        self.lb = np.asarray(bounds[0], dtype=float)
        self.ub = np.asarray(bounds[1], dtype=float)
        self.pop_size, self.max_iter, self.dim = pop_size, max_iter, len(self.lb)

        # DE
        self.F, self.CR = F, CR
        # PSO
        self.c1, self.c2, self.w, self.rho = c1, c2, w, rho

        self.pos = np.random.rand(pop_size, self.dim)
        self.vel = np.zeros((pop_size, self.dim))
        self.pbest_pos = self.pos.copy()
        self.pbest_score = np.full(pop_size, np.inf)
        self.gbest_pos = np.zeros(self.dim)
        self.gbest_score = np.inf

    def optimize(self):
        convergence_curve, time_curve = [], []

        # Evaluasi populasi awal
        X = self.lb + self.pos * (self.ub - self.lb)
        scores = np.asarray(self.objective_batch(X), dtype=float)
        self.pbest_score = scores.copy()
        self.pbest_pos = self.pos.copy()
        best_idx = np.argmin(scores)
        self.gbest_score = scores[best_idx]
        self.gbest_pos = self.pos[best_idx].copy()

        for _ in range(self.max_iter):
            iter_start = time.perf_counter()

            # 1. Komponen PSO
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            self.vel = (self.w * self.vel
                        + self.c1 * r1 * (self.pbest_pos - self.pos)
                        + self.c2 * r2 * (self.gbest_pos - self.pos))
            pso_candidate = np.clip(self.pos + self.vel, 0, 1)

            # 2-3. Mutasi DE/rand/1 + crossover terhadap kandidat PSO
            trial = de_trial(self.pos, self.pop_size, self.dim,
                             self.F, self.CR, target=pso_candidate)
            trial = np.clip(trial, 0, 1)

            # 4. Evaluasi batch
            X_trial = self.lb + trial * (self.ub - self.lb)
            trial_scores = np.asarray(self.objective_batch(X_trial), dtype=float)

            # 5. Selection
            better = trial_scores <= self.pbest_score
            self.pos[better] = trial[better]
            self.pbest_score[better] = trial_scores[better]
            self.pbest_pos[better] = trial[better]

            # 6. GBest
            best_idx = np.argmin(self.pbest_score)
            if self.pbest_score[best_idx] < self.gbest_score:
                self.gbest_score = self.pbest_score[best_idx]
                self.gbest_pos = self.pbest_pos[best_idx].copy()

            convergence_curve.append(self.gbest_score)
            time_curve.append(time.perf_counter() - iter_start)

            self.w = self.rho * self.w

        return (self.lb + self.gbest_pos * (self.ub - self.lb),
                self.gbest_score, convergence_curve, time_curve)