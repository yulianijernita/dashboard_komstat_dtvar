"""Hybrid PSO-DE dengan mekanisme yang disamakan dengan DTVaRHPSODE."""

import time
import numpy as np

# Hapus import de_trial dari sini karena kita akan menggunakan vektorisasi langsung

def inisialisasi_ulang_acak(x):
    """
    Mengembalikan nilai yang berada di luar [0, 1]
    dengan nilai acak baru pada [0, 1].
    """
    x = np.asarray(x, dtype=float).copy()
    outside = (x < 0.0) | (x > 1.0)
    x[outside] = np.random.rand(np.sum(outside))
    return x


class Hybrid_PSO_DE:

    def __init__(
        self,
        objective_func,
        objective_batch,
        bounds,
        pop_size=500,
        max_iter=500,
        F=0.5,
        CR=0.9,
        c1=2.0,
        c2=2.0,
        w=1.0,
        rho=0.9
    ):

        self.objective_func = objective_func
        self.objective_batch = objective_batch

        self.lb = np.asarray(bounds[0], dtype=float)
        self.ub = np.asarray(bounds[1], dtype=float)

        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = len(self.lb)

        # =====================================================
        # Parameter DE
        # =====================================================
        self.F = F
        self.CR = CR

        # =====================================================
        # Populasi awal
        # =====================================================
        self.pos = np.random.rand(
            self.pop_size,
            self.dim
        )

        self.fit = np.full(
            self.pop_size,
            np.inf
        )

        # =====================================================
        # Komponen PSO
        # =====================================================
        self.vel = np.zeros(
            (self.pop_size, self.dim)
        )

        self.pbest_pos = self.pos.copy()

        self.pbest_score = np.full(
            self.pop_size,
            np.inf
        )

        self.gbest_pos = np.zeros(
            self.dim
        )

        self.gbest_score = np.inf

        # =====================================================
        # Parameter PSO
        # =====================================================
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.rho = rho

    # =========================================================
    # OPTIMIZATION
    # =========================================================

    def optimize(self):

        convergence_curve = []

        start_waktu = time.perf_counter()

        # =====================================================
        # Evaluasi populasi awal
        # =====================================================

        X = (
            self.lb
            + self.pos * (self.ub - self.lb)
        )

        self.fit = np.asarray(
            self.objective_batch(X),
            dtype=float
        ).copy()

        # =====================================================
        # Inisialisasi pbest
        # =====================================================

        self.pbest_score = self.fit.copy()
        self.pbest_pos = self.pos.copy()

        # =====================================================
        # Inisialisasi gbest
        # =====================================================

        best_idx = np.argmin(self.fit)
        self.gbest_score = self.fit[best_idx]
        self.gbest_pos = self.pos[best_idx].copy()

        # =====================================================
        # ITERASI
        # =====================================================

        for iteration in range(self.max_iter):

            # =================================================
            # 1. PSO UPDATE
            # =================================================

            r1 = np.random.rand(
                self.pop_size,
                self.dim
            )

            r2 = np.random.rand(
                self.pop_size,
                self.dim
            )

            self.vel = (
                self.w * self.vel
                + self.c1 * r1
                * (self.pbest_pos - self.pos)
                + self.c2 * r2
                * (self.gbest_pos - self.pos)
            )

            # Decay inertia weight
            self.w = self.rho * self.w

            # =================================================
            # 2. POSISI HASIL PSO
            # =================================================

            pos_pso = self.pos + self.vel

            # Boundary handling
            pos_pso = inisialisasi_ulang_acak(pos_pso)

            # =================================================
            # 3. EVALUASI KANDIDAT PSO
            # =================================================

            X_pso = (
                self.lb
                + pos_pso * (self.ub - self.lb)
            )

            scores_pso = np.asarray(
                self.objective_batch(X_pso),
                dtype=float
            )

            # =================================================
            # 4. SELEKSI PSO
            #    Dibandingkan dengan CURRENT FITNESS
            # =================================================

            better_pso = scores_pso < self.fit
            self.pos[better_pso] = pos_pso[better_pso]
            self.fit[better_pso] = scores_pso[better_pso]

            # =================================================
            # 5. UPDATE PBEST
            # =================================================

            better_pbest = self.fit < self.pbest_score
            self.pbest_score[better_pbest] = self.fit[better_pbest]
            self.pbest_pos[better_pbest] = self.pos[better_pbest]

            # =================================================
            # 6. DE/rand/1 (Fully Vectorized)
            #
            #    DE dilakukan terhadap POPULASI TERKINI
            #    setelah seleksi PSO.
            # =================================================

            r = np.empty((self.pop_size, 3), dtype=int)
            indices = np.arange(self.pop_size)

            for k in range(3):
                r[:, k] = np.random.randint(
                    0, self.pop_size, size=self.pop_size
                )

                same = r[:, k] == indices

                while np.any(same):
                    r[same, k] = np.random.randint(
                        0, self.pop_size, size=np.sum(same)
                    )
                    same = r[:, k] == indices

                if k > 0:
                    duplicate = np.zeros(
                        self.pop_size, dtype=bool
                    )

                    for previous in range(k):
                        duplicate |= r[:, k] == r[:, previous]

                    while np.any(duplicate):
                        r[duplicate, k] = np.random.randint(
                            0, self.pop_size, size=np.sum(duplicate)
                        )
                        duplicate = r[:, k] == indices

                        for previous in range(k):
                            duplicate |= r[:, k] == r[:, previous]

            r1_idx = r[:, 0]
            r2_idx = r[:, 1]
            r3_idx = r[:, 2]

            # DE mutation
            mutant = self.pos[r1_idx] + self.F * (
                self.pos[r2_idx] - self.pos[r3_idx]
            )

            # Binomial crossover
            crossover = (
                np.random.rand(self.pop_size, self.dim) < self.CR
            )
            j_rand = np.random.randint(
                0, self.dim, size=self.pop_size
            )
            crossover[np.arange(self.pop_size), j_rand] = True

            trial = np.where(crossover, mutant, self.pos)

            # =================================================
            # 7. BOUNDARY HANDLING DE
            # =================================================

            trial = inisialisasi_ulang_acak(trial)

            # =================================================
            # 8. EVALUASI DE
            # =================================================

            X_trial = (
                self.lb
                + trial * (self.ub - self.lb)
            )

            trial_scores = np.asarray(
                self.objective_batch(X_trial),
                dtype=float
            )

            # =================================================
            # 9. SELEKSI DE
            #    Dibandingkan dengan CURRENT FITNESS
            # =================================================

            better_de = trial_scores < self.fit
            self.pos[better_de] = trial[better_de]
            self.fit[better_de] = trial_scores[better_de]

            # =================================================
            # 10. UPDATE PBEST
            # =================================================

            better_pbest = self.fit < self.pbest_score
            self.pbest_score[better_pbest] = self.fit[better_pbest]
            self.pbest_pos[better_pbest] = self.pos[better_pbest]

            # =================================================
            # 11. UPDATE GBEST
            # =================================================

            best_idx = np.argmin(self.fit)

            if self.fit[best_idx] < self.gbest_score:
                self.gbest_score = self.fit[best_idx]
                self.gbest_pos = self.pos[best_idx].copy()

            # =================================================
            # 12. SIMPAN KONVERGENSI
            # =================================================

            convergence_curve.append(
                self.gbest_score
            )

        # =====================================================
        # WAKTU KOMPUTASI
        # =====================================================

        waktu_run = (
            time.perf_counter()
            - start_waktu
        )

        # =====================================================
        # BEST SOLUTION
        # =====================================================

        best_pos = (
            self.lb
            + self.gbest_pos
            * (self.ub - self.lb)
        )

        return (
            best_pos,
            self.gbest_score,
            convergence_curve,
            waktu_run
        )