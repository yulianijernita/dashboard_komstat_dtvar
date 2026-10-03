"""
Estimasi parameter marginal dan copula:
  X -> Weibull 2 parameter (loc = 0), Y -> Student-t, copula FGM (pseudo-likelihood).
"""
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, kendalltau

from copulas.fgm import FGMCopula

EPS = 1e-10  # hindari pseudo-observation tepat 0 atau 1


def fit_weibull(x):
    """Weibull 2 parameter: c (shape), loc = 0, scale (lambda) + uji KS."""
    x = np.asarray(x, dtype=float)
    if len(x) < 10:
        raise ValueError("Data terlalu sedikit untuk estimasi Weibull.")
    if np.any(x <= 0):
        raise ValueError("Weibull membutuhkan data X positif (> 0).")

    c, loc, scale = stats.weibull_min.fit(x, floc=0)
    ks = stats.kstest(x, "weibull_min", args=(c, loc, scale))
    return {"k": float(c), "loc": float(loc), "lamb": float(scale),
            "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue)}


def fit_student_t(y):
    """Student-t: df, loc (mu), scale (sigma) + uji KS."""
    y = np.asarray(y, dtype=float)
    if len(y) < 10:
        raise ValueError("Data terlalu sedikit untuk estimasi Student-t.")

    df, loc, scale = stats.t.fit(y)
    ks = stats.kstest(y, "t", args=(df, loc, scale))
    return {"df": float(df), "loc": float(loc), "scale": float(scale),
            "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue)}


def estimate_all(x, y):
    """
    Estimasi marginal X (Weibull), marginal Y (Student-t), lalu copula FGM
    dari pseudo-observation memakai parameter marginal hasil estimasi.
    x dan y harus berpasangan (panjang sama).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("Panjang X dan Y harus sama (data berpasangan).")

    w = fit_weibull(x)
    t = fit_student_t(y)

    # Pseudo-observation dari parameter marginal hasil estimasi
    u_hat = stats.weibull_min.cdf(x, c=w["k"], loc=w["loc"], scale=w["lamb"])
    v_hat = stats.t.cdf(y, df=t["df"], loc=t["loc"], scale=t["scale"])
    u_hat = np.clip(u_hat, EPS, 1 - EPS)
    v_hat = np.clip(v_hat, EPS, 1 - EPS)

    # Copula FGM
    copula = FGMCopula.fit(u_hat, v_hat)

    # Ukuran dependensi
    rho_sample, _ = spearmanr(u_hat, v_hat)
    tau_sample, _ = kendalltau(u_hat, v_hat)

    return {
        "weibull": w,
        "student_t": t,
        "fgm": {
            "theta": copula.theta,
            "loglik": copula.loglik,
            "spearman_sample": float(rho_sample),
            "kendall_sample": float(tau_sample),
            "spearman_fgm": copula.spearman_rho(),
            "kendall_fgm": copula.kendall_tau(),
        },
        "n": len(x),
    }