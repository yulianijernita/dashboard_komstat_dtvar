"""Kendala h(x) = kappa * DCTV - CTV dan batas variabel optimasi."""
import numpy as np

import config
from distributions.marginal import ctv_weibull
from dtvar.model import dctv_weibull


def h_function(alpha1, delta1, kappa, alpha, delta, k, lamb, theta):
    dctv = dctv_weibull(alpha, delta, alpha1, delta1, k, lamb, theta)
    return kappa * dctv - ctv_weibull(alpha, k, lamb)


def get_bounds(alpha, delta):
    """
    Batas [alpha1, delta1, mu, kappa] -> (lower, upper), semuanya otomatis:
      alpha1 in [alpha, 1], delta1 in [delta, 1],
      mu dan kappa dari config.DEFAULT_BOUNDS.
    """
    mu_lo, mu_hi = config.DEFAULT_BOUNDS["mu"]
    ka_lo, ka_hi = config.DEFAULT_BOUNDS["kappa"]
    lower = np.array([alpha, delta, mu_lo, ka_lo])
    upper = np.array([0.9999, 0.9999, mu_hi, ka_hi])
    return lower, upper