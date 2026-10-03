"""Differential Evolution (DE/rand/1/bin) + helper yang dipakai juga oleh Hybrid PSO-DE."""
import time
import numpy as np


def distinct_indices(pop_size):
    """3 index berbeda (dan != i) untuk setiap individu."""
    r = np.empty((pop_size, 3), dtype=int)
    idx = np.arange(pop_size)
    for i in range(3):
        r[:, i] = np.random.randint(0, pop_size, size=pop_size)
        same = r[:, i] == idx
        while np.any(same):
            r[same, i] = np.random.randint(0, pop_size, size=np.sum(same))
            same = r[:, i] == idx
        if i > 0:
            dup = np.zeros(pop_size, dtype=bool)
            for prev in range(i):
                dup |= r[:, i] == r[:, prev]
            while np.any(dup):
                r[dup, i] = np.random.randint(0, pop_size, size=np.sum(dup))
                dup = r[:, i] == idx
                for prev in range(i):
                    dup |= r[:, i] == r[:, prev]
    return r[:, 0], r[:, 1], r[:, 2]


def de_trial(base, pop_size, dim, F, CR, target):
    """Mutasi DE/rand/1 + koreksi batas + binomial crossover terhadap `target`."""
    r1, r2, r3 = distinct_indices(pop_size)
    mutant = base[r1] + F * (base[r2] - base[r3])
    invalid = (mutant < 0) | (mutant > 1)
    mutant = np.where(invalid, np.random.rand(pop_size, dim), mutant)
    crossover = np.random.rand(pop_size, dim) < CR
    j_rand = np.random.randint(0, dim, size=pop_size)
    crossover[np.arange(pop_size), j_rand] = True  # minimal 1 gen dari mutant
    return np.where(crossover, mutant, target)


class DE:
    def __init__(self, objective_func, objective_batch, bounds,
                 pop_size=500, max_iter=500, F=0.5, CR=0.9):
        self.objective_func = objective_func
        self.objective_batch = objective_batch
        self.lb = np.asarray(bounds[0], dtype=float)
        self.ub = np.asarray(bounds[1], dtype=float)
        self.pop_size, self.max_iter, self.dim = pop_size, max_iter, len(self.lb)
        self.F, self.CR = F, CR
        self.pop = np.random.rand(pop_size, self.dim)
        self.fit = np.full(pop_size, np.inf, dtype=float)

    def optimize(self):
        convergence_curve, time_curve = [], []

        # Populasi awal
        X = self.lb + self.pop * (self.ub - self.lb)
        self.fit = np.asarray(self.objective_batch(X), dtype=float).copy()

        for _ in range(self.max_iter):
            iter_start = time.perf_counter()

            trials = de_trial(self.pop, self.pop_size, self.dim,
                              self.F, self.CR, target=self.pop)

            X_trial = self.lb + trials * (self.ub - self.lb)
            trial_scores = np.asarray(self.objective_batch(X_trial), dtype=float)

            # Selection
            better = trial_scores <= self.fit
            self.pop[better] = trials[better]
            self.fit[better] = trial_scores[better]

            convergence_curve.append(np.min(self.fit))
            time_curve.append(time.perf_counter() - iter_start)

        best_idx = np.argmin(self.fit)
        best_pos = self.lb + self.pop[best_idx] * (self.ub - self.lb)
        return best_pos, self.fit[best_idx], convergence_curve, time_curve