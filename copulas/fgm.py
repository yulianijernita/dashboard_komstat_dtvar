"""Copula Farlie-Gumbel-Morgenstern (FGM)."""
import numpy as np
from scipy.optimize import minimize

from copulas.base import Copula


def cop(u, v, theta):
    """CDF FGM (fungsi murni, kompatibel dengan JAX untuk gradien)."""
    return u * v + theta * u * v * (1.0 - u) * (1.0 - v)


cop_fgm = cop  # alias yang dipakai pada backtesting


class FGMCopula(Copula):

    def __init__(self, theta):
        self.theta = theta
        self.loglik = None
        self.success = None

    def cdf(self, u, v):
        return cop(u, v, self.theta)

    # Hubungan teoritis FGM
    def spearman_rho(self):
        return self.theta / 3.0

    def kendall_tau(self):
        return 2.0 * self.theta / 9.0

    @classmethod
    def fit(cls, u, v, x0=0.1):
        """
        Maksimum pseudo-likelihood dengan L-BFGS-B:
        c(u,v) = 1 + theta (1-2u)(1-2v),  theta di [-1, 1].
        u, v = pseudo-observasi (sebaiknya sudah di-clip dari 0 dan 1).
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        def neg_loglik_fgm(params):
            theta = params[0]
            if theta < -1 or theta > 1:      # batas parameter FGM
                return np.inf
            density = 1 + theta * (1 - 2 * u) * (1 - 2 * v)
            if np.any(density <= 0):         # density tidak boleh <= 0
                return np.inf
            return -np.sum(np.log(density))

        result = minimize(neg_loglik_fgm, x0=[x0], bounds=[(-1, 1)], method="L-BFGS-B")

        copula = cls(float(result.x[0]))
        copula.loglik = float(-result.fun)
        copula.success = bool(result.success)
        return copula