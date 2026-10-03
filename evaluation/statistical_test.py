"""Uji statistik untuk backtesting."""
import numpy as np
from scipy.stats import chi2


def kupiec_pof(n_violation, n_obs, p0):
    """
    Uji Kupiec (Proportion of Failures): H0 proporsi violation = p0.
    Di sini p0 = joint significance dari copula.
    Return (LR, p-value); LR ~ chi-square(1).
    """
    x, n = int(n_violation), int(n_obs)
    if not (0.0 < p0 < 1.0) or n <= 0:
        return np.nan, np.nan

    phat = x / n
    ll0 = (n - x) * np.log(1.0 - p0) + x * np.log(p0)
    ll1 = 0.0
    if x > 0:
        ll1 += x * np.log(phat)
    if x < n:
        ll1 += (n - x) * np.log(1.0 - phat)

    lr = -2.0 * (ll0 - ll1)
    return float(lr), float(1.0 - chi2.cdf(lr, df=1))